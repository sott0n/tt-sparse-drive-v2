# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fixed-point grid_sample (generic_op) vs ttnn.grid_sample and torch, at the DFA's shapes."""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.sparsedrive_v2.tt.ttnn_grid_sample import grid_sample_fixed_point


def _time_ms(device, fn, iters=5):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters * 1e3


@pytest.mark.parametrize(
    "n, h, w, c, gh, gw, inside",
    [
        (3, 64, 128, 256, 256, 512, 0.31),  # layer-0 path DFA chunk, level 0, realistic visibility
        (3, 8, 16, 256, 256, 512, 1.0),  # coarsest level, every point inside
        (3, 32, 64, 256, 144, 96, 0.5),  # trajectory DFA chunk shape
        (2, 5, 7, 64, 4, 9, 1.0),  # tiny odd sizes, partial work split
    ],
)
def test_grid_sample_fixed_point(device, n, h, w, c, gh, gw, inside):
    torch.manual_seed(0)
    feat = torch.randn(n, h, w, c)
    grid = torch.rand(n, gh, gw, 2) * 2 - 1
    grid.view(-1, 2)[:8] = torch.tensor([[-0.999, 0.0], [0.999, 0.0], [0.0, -0.999], [0.0, 0.999]] * 2)  # borders
    grid[torch.rand(n, gh, gw) > inside] = -3.0  # the DFA parks invisible points far outside

    feat_t = ttnn.from_torch(feat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_t = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ref = torch.nn.functional.grid_sample(
        feat.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).permute(0, 2, 3, 1)
    stock = ttnn.to_torch(ttnn.grid_sample(feat_t, grid_t)).float()
    fixed = ttnn.to_torch(grid_sample_fixed_point(feat_t, grid_t)).float()

    _, pcc_stock = comp_pcc(ref, stock)
    passed, pcc_fixed = comp_pcc(ref, fixed, 0.9999)
    diff = (fixed - stock).abs().max().item()
    t_stock = _time_ms(device, lambda: ttnn.grid_sample(feat_t, grid_t))
    t_fixed = _time_ms(device, lambda: grid_sample_fixed_point(feat_t, grid_t))
    print(
        f"[{n}x{h}x{w}x{c} grid {gh}x{gw} inside {inside}] pcc stock {pcc_stock:.6f} fixed {pcc_fixed:.6f} "
        f"max|fixed-stock| {diff:.4f} | stock {t_stock:.2f} ms, fixed {t_fixed:.2f} ms ({t_stock / t_fixed:.2f}x)"
    )
    assert passed, pcc_fixed
