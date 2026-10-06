# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bilinear grid_sample with a fixed-point reader, built on ``ttnn.generic_op``.

Same contract as ``ttnn.grid_sample(input, grid)`` for the case the DFA uses:
input (N, H, W, C) bf16 row-major, grid (N, Hg, Wg, 2) fp32 row-major in [-1, 1],
bilinear, zeros padding, ``align_corners=False``; output (N, Hg, Wg, C) bf16 row-major.

The program mirrors ttnn's GridSampleBilinearProgramFactory (interleaved path) and reuses its
compute and writer kernels unchanged. Only the reader is replaced
(``kernels/reader_grid_sample_fixed_point.cpp``): it computes the sampling geometry in Q16.16
integers instead of software float, and batches grid-stick reads behind one barrier.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

READER = str(Path(__file__).parent / "kernels" / "reader_grid_sample_fixed_point.cpp")
COMPUTE = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp"
WRITER = "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp"

# Constants of ttnn's bilinear grid_sample factory (pool/grid_sample, pool/rotate).
REDUCTION_SIZE = 4  # four bilinear corners reduced per output stick
BUFFERING_FACTOR = 2
MAX_TILES_PER_REDUCTION = 8  # fp32_dest_acc_en=False, half-sync DEST
MAX_ROWS_FOR_REDUCTION = 16
DUMMY_CB_ID = 32
TILE_WIDTH, FACE_WIDTH, TILE_HW = 32, 16, 1024

GRID_BLOCK = 16  # grid sticks fetched per read barrier
MAX_SPATIAL = 8192  # keeps (g + 1) * size inside int32 in the reader's Q16.16 math

CB_GRID, CB_INPUT, CB_SCALAR, CB_OUTPUT = 0, 1, 2, 3


def _cb(index, dtype, page_size, num_pages, cores, face_geometry=None):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)
    if face_geometry is not None:
        fmt.face_geometry = ttnn.FaceGeometry(*face_geometry)
    return ttnn.CBDescriptor(total_size=page_size * num_pages, core_ranges=cores, format_descriptors=[fmt])


def _compute_ct_args(in_ntiles_c, channels, sticks):
    return [
        in_ntiles_c,
        REDUCTION_SIZE,
        0,  # enable_split_reader: interleaved grid
        sticks,  # total_interpolations on this core group
        channels,
        1,  # in_nblocks_c
        MAX_ROWS_FOR_REDUCTION,
        CB_INPUT,
        DUMMY_CB_ID,  # input_cb_index_1
        CB_SCALAR,
        DUMMY_CB_ID,  # scalar_cb_index_1
        CB_OUTPUT,
        0,  # one scalar per core
        DUMMY_CB_ID,  # pre_tilize_cb_id: row-major output
        0,  # is_output_tiled
        0,  # is_output_block_format
        0,  # force_max_tiles_per_reduction_4
        *([DUMMY_CB_ID] * 7),  # max-pool-with-indices CBs (unused)
        *([1] * 7),  # stride, padded size, effective kernel, pad (unused by grid_sample)
        *([DUMMY_CB_ID] * 4),  # intra-kernel index CBs (unused)
        1,  # kernel_h
        1,  # kernel_w
        0,  # indexes_32_bit
        DUMMY_CB_ID,  # fast_tilize_cb_id (tiled output only)
    ]


def grid_sample_fixed_point(input_tensor: ttnn.Tensor, grid: ttnn.Tensor) -> ttnn.Tensor:
    device = input_tensor.device()
    n, h, w, c = (int(d) for d in input_tensor.shape)
    gn, gh, gw, two = (int(d) for d in grid.shape)
    assert input_tensor.dtype == ttnn.bfloat16 and input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    assert grid.dtype == ttnn.float32 and grid.layout == ttnn.ROW_MAJOR_LAYOUT and two == 2
    assert gn == n, "grid batch must match input batch"
    assert c % TILE_WIDTH == 0 and c // TILE_WIDTH <= MAX_TILES_PER_REDUCTION, "C must be a multiple of 32, <= 256"
    assert h <= MAX_SPATIAL and w <= MAX_SPATIAL
    for t in (input_tensor, grid):
        assert not t.is_sharded(), "interleaved tensors only"

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n, gh, gw, c]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    total_sticks = n * gh * gw
    grid_size = device.compute_with_storage_grid_size()
    num_cores, all_cores, group_1, group_2, sticks_1, sticks_2 = ttnn.split_work_to_cores(grid_size, total_sticks)
    cores = ttnn.corerange_to_cores(all_cores, num_cores, True)

    in_ntiles_c = c // TILE_WIDTH
    out_ntiles_c = c // FACE_WIDTH
    input_stick = input_tensor.buffer_aligned_page_size()
    grid_stick = grid.buffer_aligned_page_size()
    output_stick = output.buffer_aligned_page_size()
    elem = 2  # bf16

    cbs = [
        _cb(CB_GRID, ttnn.float32, grid_stick, GRID_BLOCK, all_cores),
        _cb(
            CB_INPUT,
            ttnn.bfloat16,
            in_ntiles_c * TILE_HW * elem,
            BUFFERING_FACTOR,
            all_cores,
            (REDUCTION_SIZE, 2),
        ),
        _cb(CB_SCALAR, ttnn.bfloat16, TILE_HW * elem, BUFFERING_FACTOR, all_cores, (1, 2)),
        _cb(CB_OUTPUT, ttnn.bfloat16, FACE_WIDTH * elem, out_ntiles_c * BUFFERING_FACTOR, all_cores, (1, 2)),
    ]

    reader_ct = [CB_INPUT, CB_GRID, CB_SCALAR, input_stick, grid_stick, h, w, gh * gw, GRID_BLOCK]
    reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(grid).get_compile_time_args()
    writer_ct = [CB_OUTPUT, output_stick, out_ntiles_c]
    writer_ct += ttnn.TensorAccessorArgs(output).get_compile_time_args()

    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for core in cores:
        sticks = sticks_1 if group_1.contains(core) else sticks_2
        reader_rt[core.x][core.y] = [input_tensor.buffer_address(), grid.buffer_address(), sticks, start]
        writer_rt[core.x][core.y] = [output.buffer_address(), sticks, start]
        start += sticks
    assert start == total_sticks

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False
    )
    for group, sticks in ((group_1, sticks_1), (group_2, sticks_2)):
        if group.num_cores() == 0:
            continue
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=group,
                compile_time_args=_compute_ct_args(in_ntiles_c, c, sticks),
                defines=[("REDUCE_OP", "PoolType::AVG"), ("REDUCE_DIM", "ReduceDim::REDUCE_COL")],
                config=compute_cfg,
            )
        )

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([input_tensor, grid, output], program)
