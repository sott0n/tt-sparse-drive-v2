# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN DeformableFeatureAggregation (Sparse4D-style multi-camera multi-level sampling).

Replaces upstream's CUDA ``deformable_aggregation`` with on-device ops, per anchor chunk:

  key points   X, Y = anchor_xy + learnable_fc(query)          (1, n, Pp) fp32
  projection   u, v, d = P[cam] @ [X, Y, z, 1]  (elementwise)  (cams, n, Pp) fp32
  weights      exp(weights_fc(query + camera_embed)) masked     (cams, n, Pp*L*G) fp32
  aggregate    fused sample + weight + reduce (ttnn_dfa_fused)  (n, C) fp32
  normalize    divide by the per-group softmax denominator     (1, n, C)

Points are padded from P to Pp (a multiple of 32) so every level slice is tile
aligned; padded points get a -3e4 weight logit and therefore zero weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

import ttnn
from models.experimental.sparsedrive_v2.tt.ttnn_dfa_fused import deformable_aggregation_fused
from models.experimental.sparsedrive_v2.tt.ttnn_ops import ACT_DTYPE, TtLayerNorm, TtLinear, to_device

PAD_LOGIT = -30000.0


def round_up(x: int, m: int = 32) -> int:
    return (x + m - 1) // m * m


@dataclass
class CameraContext:
    """Per-frame camera tensors shared by every DFA module.

    ``coef[r][k]``: (cams, 1, 1) fp32 entry (r, k) of lidar2img, rows 0/1 pre-divided by W/H so
    that u/d and v/d are already the normalized image coordinates.
    ``cam12``: (cams, 1, 12) first three rows of lidar2img (camera-encoder input).
    """

    coef: List[List[ttnn.Tensor]]
    cam12: ttnn.Tensor


def camera_host_tensors(projection_mat: torch.Tensor, image_wh: torch.Tensor) -> Dict[str, ttnn.Tensor]:
    """Host tensors of one frame's cameras; projection_mat (cams, 4, 4), image_wh (cams, 2)."""
    proj = projection_mat.float()
    norm = proj.clone()
    norm[:, 0] /= image_wh[:, 0:1]
    norm[:, 1] /= image_wh[:, 1:2]
    host = {
        f"coef_{r}{k}": ttnn.from_torch(norm[:, r, k].reshape(-1, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        for r in range(3)
        for k in range(4)
    }
    host["cam12"] = ttnn.from_torch(proj[:, :3].reshape(-1, 1, 12), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return host


def camera_context(tensors: Dict[str, ttnn.Tensor]) -> CameraContext:
    """CameraContext from the device copies of :func:`camera_host_tensors`."""
    coef = [[tensors[f"coef_{r}{k}"] for k in range(4)] for r in range(3)]
    return CameraContext(coef, tensors["cam12"])


def build_camera_context(device, projection_mat: torch.Tensor, image_wh: torch.Tensor) -> CameraContext:
    host = camera_host_tensors(projection_mat, image_wh)
    return camera_context({k: ttnn.to_device(v, device) for k, v in host.items()})


class TtDeformableFeatureAggregation:
    def __init__(self, device, ref, chunk: int):
        self.device = device
        self.chunk = chunk
        kps = ref.kps_generator
        self.G, self.L, self.C, self.cams = ref.num_groups, ref.num_levels, ref.embed_dims, ref.num_cams
        self.P = ref.num_pts
        self.Pp = round_up(self.P)
        nh, nl = len(kps.fix_height), kps.num_learnable_pts
        p = torch.arange(self.P)
        # key point p = (sample s, height h, learnable l), flattened s-major like upstream
        self.point_sample = p // (nh * nl)

        z = torch.zeros(self.Pp)
        z[: self.P] = torch.tensor(kps.fix_height)[(p // nl) % nh]
        self.z_dev = to_device(z.reshape(1, 1, -1), device, ttnn.float32)

        # learnable_fc rows are (point, xy) interleaved; regroup into [x(Pp) | y(Pp)].
        w, b = kps.learnable_fc.weight.detach(), kps.learnable_fc.bias.detach()
        wk = torch.zeros(2 * self.Pp, w.shape[1])
        bk = torch.zeros(2 * self.Pp)
        wk[: self.P], wk[self.Pp : self.Pp + self.P] = w[0::2], w[1::2]
        bk[: self.P], bk[self.Pp : self.Pp + self.P] = b[0::2], b[1::2]
        self.kps_fc = TtLinear(device, wk, bk)

        # weights_fc rows are (level, point, group); regroup into (Pp, level, group) so the fused
        # kernel reads a sample's G group weights contiguously.
        w, b = ref.weights_fc.weight.detach(), ref.weights_fc.bias.detach()
        w = w.view(self.L, self.P, self.G, -1).permute(1, 0, 2, 3)
        b = b.view(self.L, self.P, self.G).permute(1, 0, 2)
        wf = torch.zeros(self.Pp, self.L, self.G, w.shape[-1])
        bf = torch.full((self.Pp, self.L, self.G), PAD_LOGIT)
        wf[: self.P], bf[: self.P] = w, b
        # Extra inputs: one mask bias per key point, added to each of its L*G logits.
        expand = torch.zeros(self.Pp, self.L * self.G, self.Pp)
        expand[torch.arange(self.Pp), :, torch.arange(self.Pp)] = 1
        wf = torch.cat([wf.reshape(-1, w.shape[-1]), expand.reshape(-1, self.Pp)], dim=1)
        self.weights_fc = TtLinear(device, wf, bf.reshape(-1))

        enc = ref.camera_encoder
        self.cam_fc1, self.cam_ln1 = TtLinear.from_torch(device, enc[0]), TtLayerNorm(device, enc[2])
        self.cam_fc2, self.cam_ln2 = TtLinear.from_torch(device, enc[3]), TtLayerNorm(device, enc[5])
        self.output_proj = TtLinear.from_torch(device, ref.output_proj)

        gd = self.C // self.G
        mask = torch.zeros(self.G, self.C)
        for g in range(self.G):
            mask[g, g * gd : (g + 1) * gd] = 1
        self.group_expand = to_device(mask, device, ttnn.float32)  # (G, C)
        # column j of the (Pp, L, G)-ordered logits belongs to group j % G
        seg = torch.zeros(self.Pp * self.L * self.G, self.G)
        seg[torch.arange(seg.shape[0]), torch.arange(seg.shape[0]) % self.G] = 1
        self.group_sum = to_device(seg, device, ttnn.float32)

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

    # -- anchor tables -------------------------------------------------------

    def anchor_table(self, xy: torch.Tensor) -> torch.Tensor:
        """(N, num_sample, >=2) anchor polylines -> (N, 2*Pp) [x per key point | y per key point]."""
        table = torch.zeros(xy.shape[0], 2 * self.Pp)
        table[:, : self.P] = xy[:, self.point_sample, 0]
        table[:, self.Pp : self.Pp + self.P] = xy[:, self.point_sample, 1]
        return table

    # -- per frame -----------------------------------------------------------

    def prepare_frame(self, cam: CameraContext) -> Dict[str, object]:
        e = self.cam_ln1(self.cam_fc1(cam.cam12, activation="relu"))
        cam_embed = self.cam_ln2(self.cam_fc2(e, activation="relu"))  # (cams, 1, C)
        cam_embed = ttnn.typecast(cam_embed, ACT_DTYPE) if cam_embed.dtype != ACT_DTYPE else cam_embed
        zc = [ttnn.add(ttnn.multiply(self.z_dev, cam.coef[r][2]), cam.coef[r][3]) for r in range(3)]  # (cams, 1, Pp)
        return {"cam_embed": cam_embed, "zc": zc, "coef": cam.coef}

    # -- forward -------------------------------------------------------------

    def __call__(self, query, anchor_x, anchor_y, values: List[ttnn.Tensor], frame) -> ttnn.Tensor:
        """
        Args:
            query: (1, A, C) instance features.
            anchor_x, anchor_y: (1, A, Pp) fp32 anchor coordinates of every key point.
            values: per level (cams, H_l, W_l, C) row-major feature maps.
        Returns:
            (1, A, C) = output_proj(aggregated) + query
        """
        num_anchor = query.shape[1]
        chunks = []
        for a0 in range(0, num_anchor, self.chunk):
            a1 = min(a0 + self.chunk, num_anchor)
            chunks.append(self._aggregate(query[:, a0:a1], anchor_x[:, a0:a1], anchor_y[:, a0:a1], values, frame))
        feat = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=1)
        return ttnn.add(self.output_proj(feat), query)

    def _aggregate(self, query, anchor_x, anchor_y, values, frame):
        cams, G, L, Pp, C = self.cams, self.G, self.L, self.Pp, self.C
        n = query.shape[1]
        coef, zc = frame["coef"], frame["zc"]

        off = self.kps_fc(query, dtype=ttnn.float32)
        kx = ttnn.repeat(ttnn.add(off[:, :, :Pp], anchor_x), (cams, 1, 1))
        ky = ttnn.repeat(ttnn.add(off[:, :, Pp:], anchor_y), (cams, 1, 1))

        def project(r):
            return ttnn.add(ttnn.add(ttnn.multiply(kx, coef[r][0]), ttnn.multiply(ky, coef[r][1])), zc[r])

        u, v, depth = project(0), project(1), project(2)
        depth_c = ttnn.clamp(depth, min=1e-5)
        x, y = ttnn.div(u, depth_c), ttnn.div(v, depth_c)  # (cams, n, Pp) normalized image coords

        valid = ttnn.multiply(
            ttnn.multiply(ttnn.gt(x, 0.0), ttnn.lt(x, 1.0)), ttnn.multiply(ttnn.gt(y, 0.0), ttnn.lt(y, 1.0))
        )
        mask = ttnn.multiply(valid, ttnn.gt(depth, 1e-5))
        any_visible = ttnn.gt(ttnn.sum(mask, dim=0, keepdim=True), 0.0)
        keep = ttnn.rsub(ttnn.multiply(ttnn.rsub(mask, 1.0), any_visible), 1.0)  # upstream's -inf fill

        # Unnormalized softmax weights over (cams, levels, points) per (anchor, group).  One shift
        # per anchor is enough: per-group maxima of the released weights differ by < 6.
        feat = ttnn.add(ttnn.repeat(query, (cams, 1, 1)), frame["cam_embed"])
        mask_bias = ttnn.multiply(ttnn.subtract(keep, 1.0), -PAD_LOGIT)  # 0 kept, -3e4 dropped
        if mask_bias.dtype != feat.dtype:
            mask_bias = ttnn.typecast(mask_bias, feat.dtype)
        feat = ttnn.concat([feat, mask_bias], dim=-1)
        logits = self.weights_fc(feat, dtype=ttnn.float32)  # (cams, n, Pp*L*G)
        shift = ttnn.max(ttnn.max(logits, dim=-1, keepdim=True), dim=0, keepdim=True)
        ex = ttnn.exp(ttnn.subtract(logits, shift))
        den = ttnn.matmul(ex, self.group_sum, compute_kernel_config=self.kps_fc.cfg, core_grid=self.core_grid)
        den = ttnn.sum(den, dim=0, keepdim=True)
        ex = ttnn.to_layout(ttnn.typecast(ex, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)

        # Sampling grid in [-1, 1].  Upstream skips a camera unless the point lies strictly
        # inside the image; those points go far outside, where the fused kernel skips them.
        def to_grid(t):
            t = ttnn.clamp(ttnn.subtract(ttnn.multiply(t, 2.0), 1.0), min=-1.5, max=1.5)
            return ttnn.add(ttnn.multiply(t, valid), ttnn.multiply(ttnn.subtract(valid, 1.0), 3.0))

        grid = ttnn.to_layout(ttnn.concat([to_grid(x), to_grid(y)], dim=-1), ttnn.ROW_MAJOR_LAYOUT)  # (cams, n, 2*Pp)

        acc = deformable_aggregation_fused(values, grid, ex, self.P, G)  # (n, C)
        acc = ttnn.reshape(ttnn.to_layout(acc, ttnn.TILE_LAYOUT), (1, n, C))
        den = ttnn.matmul(den, self.group_expand, compute_kernel_config=self.kps_fc.cfg)  # (1, n, C)
        return ttnn.div(acc, den)
