# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fused deformable aggregation (generic_op) vs a torch reference of the same sum."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.sparsedrive_v2.tt.ttnn_dfa_fused import deformable_aggregation_fused


def _reference(values, grid, ex, num_pts, groups):
    """values: [(cams, H, W, C)], grid: (cams, A, Pp, 2), ex: (cams, A, G, L, Pp) -> (A, C)."""
    cams, num_anchors = grid.shape[:2]
    c = values[0].shape[-1]
    inside = (grid[..., 0].abs() < 1) & (grid[..., 1].abs() < 1)
    out = torch.zeros(num_anchors, groups, c // groups, dtype=torch.float64)
    for lvl, v in enumerate(values):
        fm = v.permute(0, 3, 1, 2).double()  # (cams, C, H, W)
        s = torch.nn.functional.grid_sample(fm, grid.double(), align_corners=False)  # (cams, C, A, Pp)
        s = s.permute(0, 2, 3, 1).reshape(cams, num_anchors, -1, groups, c // groups)
        w = ex[:, :, :, lvl].permute(0, 1, 3, 2).double() * inside[..., None]  # (cams, A, Pp, G)
        w[:, :, num_pts:] = 0
        out += (s * w[..., None]).sum(dim=(0, 2))
    return out.reshape(num_anchors, c).float()


@pytest.mark.parametrize(
    "num_anchors, pp, num_pts, sizes",
    [
        (7, 64, 50, [(8, 16), (4, 8), (2, 4), (1, 2)]),
        (128, 512, 500, [(64, 128), (32, 64), (16, 32), (8, 16)]),
    ],
)
def test_dfa_fused(device, num_anchors, pp, num_pts, sizes):
    torch.manual_seed(0)
    cams, groups, c = 3, 8, 256
    values = [torch.randn(cams, h, w, c) for h, w in sizes]
    grid = torch.rand(cams, num_anchors, pp, 2) * 2.4 - 1.2
    grid[torch.rand(cams, num_anchors, pp) > 0.5] = -3.0
    ex = torch.rand(cams, num_anchors, groups, 4, pp)
    ref = _reference(values, grid, ex, num_pts, groups)

    def rm(t, dtype):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out = deformable_aggregation_fused(
        [rm(v, ttnn.bfloat16) for v in values],
        rm(grid.permute(0, 1, 3, 2).reshape(cams, num_anchors, 2 * pp), ttnn.float32),
        rm(ex.permute(0, 1, 4, 3, 2).reshape(cams, num_anchors, -1), ttnn.bfloat16),
        num_pts,
        groups,
    )
    out = ttnn.to_torch(out).float()
    passed, pcc = comp_pcc(ref, out, 0.999)
    print(
        f"A={num_anchors} Pp={pp}: pcc {pcc}, max abs err {(out - ref).abs().max():.4f} (ref max {ref.abs().max():.2f})"
    )
    assert passed, pcc
