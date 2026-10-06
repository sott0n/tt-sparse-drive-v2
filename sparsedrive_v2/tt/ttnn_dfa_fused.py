# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fused deformable aggregation (sample + weight + reduce in one ``ttnn.generic_op``).

    out[a, g*32 + i] = sum over (cam, level, point, corner) inside the image of
                       ex[cam, a, g, level, point] * bilinear_w * value[cam, level][corner, g*32 + i]

Unnormalized: the caller divides by the softmax denominator. Samples never go to DRAM;
see ``kernels/reader_dfa.cpp`` and ``kernels/compute_dfa.cpp`` for the tile scheme.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_HW, TILE_WIDTH = 1024, 32
NUM_LEVELS = 4
MAX_SPATIAL = 8192  # keeps the reader's Q16.16 pixel math inside int32

CB_EX, CB_GRID, CB_V_RM, CB_V, CB_ET, CB_W, CB_A, CB_ACC = range(8)


def _cb(index, dtype, page_size, num_pages, cores):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)
    return ttnn.CBDescriptor(total_size=page_size * num_pages, core_ranges=cores, format_descriptors=[fmt])


def deformable_aggregation_fused(
    values: List[ttnn.Tensor], grid: ttnn.Tensor, ex: ttnn.Tensor, num_pts: int, num_groups: int
) -> ttnn.Tensor:
    """
    Args:
        values: per level (cams, H_l, W_l, C) bf16 row-major.
        grid: (cams, A, Pp * 2) fp32 row-major, (x, y) per key point in [-1, 1]; points to skip
            (outside the image or masked) must lie outside (-1, 1).
        ex: (cams, A, G * L * Pp) bf16 row-major, unnormalized softmax weights ordered
            (group, level, point).
        num_pts: real key points per anchor (<= Pp).
    Returns:
        (A, C) fp32 row-major.
    """
    device = grid.device()
    assert len(values) == NUM_LEVELS
    cams, num_anchors, grid_w = (int(d) for d in grid.shape)
    pp = grid_w // 2
    c = int(values[0].shape[-1])
    assert c == num_groups * TILE_WIDTH, "one 32-channel tile per group"
    assert tuple(int(d) for d in ex.shape) == (cams, num_anchors, num_groups * NUM_LEVELS * pp)
    assert grid.dtype == ttnn.float32 and ex.dtype == ttnn.bfloat16
    for t in (grid, ex, *values):
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT and not t.is_sharded()
    for v in values:
        assert v.dtype == ttnn.bfloat16 and int(v.shape[0]) == cams and int(v.shape[-1]) == c
        assert int(v.shape[1]) <= MAX_SPATIAL and int(v.shape[2]) <= MAX_SPATIAL

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_anchors, c]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    c_tiles = c // TILE_WIDTH
    ex_row = ex.buffer_aligned_page_size()
    grid_row = grid.buffer_aligned_page_size()
    stick = values[0].buffer_aligned_page_size()

    grid_size = device.compute_with_storage_grid_size()
    num_cores, all_cores, group_1, group_2, per_1, per_2 = ttnn.split_work_to_cores(grid_size, num_anchors)
    cores = ttnn.corerange_to_cores(all_cores, num_cores, True)

    cbs = [
        _cb(CB_EX, ttnn.bfloat16, cams * ex_row, 1, all_cores),
        _cb(CB_GRID, ttnn.float32, cams * grid_row, 1, all_cores),
        _cb(CB_V_RM, ttnn.bfloat16, TILE_HW * 2, 2 * c_tiles, all_cores),
        _cb(CB_V, ttnn.bfloat16, TILE_HW * 2, 2 * c_tiles, all_cores),
        _cb(CB_ET, ttnn.bfloat16, TILE_HW * 2, 2, all_cores),
        _cb(CB_W, ttnn.bfloat16, TILE_HW * 2, 2, all_cores),
        _cb(CB_A, ttnn.bfloat16, TILE_HW * 2, 2, all_cores),
        _cb(CB_ACC, ttnn.float32, TILE_HW * 4, 2 * c_tiles, all_cores),
    ]

    reader_ct = [
        CB_EX,
        CB_GRID,
        CB_V_RM,
        CB_ET,
        CB_W,
        cams,
        num_anchors,
        num_pts,
        pp,
        num_groups,
        stick,
        c_tiles,
        ex_row,
        grid_row,
        *(int(v.shape[1]) for v in values),
        *(int(v.shape[2]) for v in values),
    ]
    for t in (ex, grid, *values):
        reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    writer_ct = [CB_ACC, c_tiles] + ttnn.TensorAccessorArgs(output).get_compile_time_args()
    compute_ct = [CB_V_RM, CB_V, CB_ET, CB_W, CB_A, CB_ACC, c_tiles]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for core in cores:
        count = per_1 if group_1.contains(core) else per_2
        reader_rt[core.x][core.y] = [
            ex.buffer_address(),
            grid.buffer_address(),
            *(v.buffer_address() for v in values),
            start,
            count,
        ]
        writer_rt[core.x][core.y] = [output.buffer_address(), start, count]
        compute_rt[core.x][core.y] = [count]
        start += count
    assert start == num_anchors

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,  # 8 fp32 DEST tiles: one accumulator per channel tile
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("reader_dfa.cpp", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel("writer_dfa.cpp", writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
            kernel("compute_dfa.cpp", compute_ct, compute_rt, compute_cfg),
        ],
        semaphores=[],
        cbs=cbs,
    )
    return ttnn.generic_op([ex, grid, *values, output], program)
