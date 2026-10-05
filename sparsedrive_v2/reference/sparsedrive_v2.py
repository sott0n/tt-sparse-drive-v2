# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SparseDriveV2 reference model (NAVSIM agent) — inference-only pure PyTorch port.

Ported from https://github.com/swc-17/SparseDriveV2 (Apache-2.0), commit 696ef77,
``navsim/agents/sparsedrive/{sparsedrive_model,custom_decoder,blocks,sparsedrive_backbone}.py``.

Differences from upstream, none of which change inference outputs:
  * The CUDA ``deformable_aggregation`` extension is replaced by
    :func:`deformable_aggregation`, a torch implementation of the same kernel.
  * nuplan / navsim imports, GridMask (train-only) and all loss code are removed.
  * The model takes plain tensors instead of NAVSIM feature dicts.

Module and parameter names match upstream so the released Lightning checkpoint
loads directly (see :func:`load_checkpoint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_PREFIX = "agent._sparsedrive_model."


@dataclass
class SparseDriveV2Config:
    d_model: int = 256
    d_ffn: int = 1024
    num_head: int = 8

    num_levels: int = 4
    cams: Tuple[str, ...] = ("cam_l0", "cam_f0", "cam_r0")
    final_dim: Tuple[int, int] = (256, 512)  # (H, W) of each camera image fed to the backbone

    mode_path: int = 1024
    len_path: int = 50
    mode_vel: int = 256
    len_vel_seq: int = 8
    num_poses: int = 8

    decoder_num_layers: int = 2
    path_filter_num: Tuple[int, ...] = (128, 20)
    # NAVSIM v1 release; upstream's dataclass default (64, 10) is the v2 setting.
    velocity_filter_num: Tuple[int, ...] = (64, 20)

    fix_height: Tuple[float, ...] = (0.0, -0.25, -0.5, 0.25, 0.5)
    num_learnable_pts: int = 2
    num_groups: int = 8

    dataset_version: str = "v1"
    metrics: Tuple[str, ...] = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "time_to_collision_within_bound",
        "comfort",
        "ego_progress",
    )

    @property
    def num_cams(self) -> int:
        return len(self.cams)


# ---------------------------------------------------------------------------
# Deformable aggregation (torch implementation of the upstream CUDA kernel)
# ---------------------------------------------------------------------------


def deformable_aggregation(
    value: torch.Tensor,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Same semantics as ``deformable_aggregation_forward`` in upstream ``ops/src``.

    Args:
        value: (bs, num_cams, sum_l H_l*W_l, C) multi-camera multi-level features.
        spatial_shapes: [(H_l, W_l)] per level, in the order levels are concatenated.
        sampling_location: (bs, num_pts, num_cams, 2) normalized (x, y) in [0, 1].
        weights: (bs, num_pts, num_cams, num_levels, num_groups).

    Returns:
        (bs, num_pts, C)

    The kernel samples bilinearly at ``loc * size - 0.5`` with zero padding
    (== ``grid_sample(align_corners=False, padding_mode="zeros")``) and skips a
    camera entirely unless ``0 < x < 1 and 0 < y < 1``.
    """
    bs, num_cams, _, C = value.shape
    num_pts = sampling_location.shape[1]
    num_groups = weights.shape[-1]
    inside = (
        (sampling_location[..., 0] > 0)
        & (sampling_location[..., 0] < 1)
        & (sampling_location[..., 1] > 0)
        & (sampling_location[..., 1] < 1)
    )  # (bs, num_pts, num_cams)
    out = value.new_zeros(bs, num_pts, num_groups, C // num_groups)
    start = 0
    for lvl, (h, w) in enumerate(spatial_shapes):
        feat_l = value[:, :, start : start + h * w]
        start += h * w
        for cam in range(num_cams):
            fm = feat_l[:, cam].reshape(bs, h, w, C).permute(0, 3, 1, 2)
            grid = sampling_location[:, :, cam][:, :, None] * 2 - 1  # (bs, num_pts, 1, 2)
            sampled = F.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled[..., 0].permute(0, 2, 1).reshape(bs, num_pts, num_groups, C // num_groups)
            wgt = weights[:, :, cam, lvl] * inside[:, :, cam, None]
            out = out + sampled * wgt[..., None]
    return out.reshape(bs, num_pts, C)


def flatten_feature_maps(feature_maps: List[torch.Tensor]):
    """Upstream ``deformable_format``: [(bs, cams, C, H, W)] -> (bs, cams, sum HW, C), shapes."""
    shapes = [tuple(f.shape[-2:]) for f in feature_maps]
    value = torch.cat([f.flatten(start_dim=-2).transpose(-1, -2) for f in feature_maps], dim=-2)
    return value, shapes


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class SparseBackbone(nn.Module):
    """timm ResNet-34 (features_only, stages 1-4) + torchvision FPN (256 ch)."""

    def __init__(self, config: SparseDriveV2Config):
        super().__init__()
        import timm
        from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

        self.img_backbone = timm.create_model(
            "resnet34", pretrained=False, features_only=True, out_indices=(1, 2, 3, 4)[-config.num_levels :]
        )
        self.img_neck = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512][-config.num_levels :], out_channels=config.d_model
        )

    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        bs, num_cams = img.shape[:2]
        feature_maps = self.img_backbone(img.flatten(end_dim=1))
        feature_maps = list(self.img_neck({f"feat_{i}": f for i, f in enumerate(feature_maps)}).values())
        return [f.reshape((bs, num_cams) + f.shape[1:]) for f in feature_maps]


# ---------------------------------------------------------------------------
# Deformable feature aggregation block
# ---------------------------------------------------------------------------


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class SparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims: int, num_sample: int, num_learnable_pts: int, fix_height: Sequence[float]):
        super().__init__()
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        self.learnable_fc = nn.Linear(embed_dims, self.num_pts * 2)
        self.fix_height = tuple(fix_height)

    def forward(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        """anchor (bs, A, num_sample*2) xy -> key points (bs, A, num_pts, 3)."""
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = self.learnable_fc(instance_feature).reshape(
            bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2
        )
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat([key_points, key_points.new_zeros(key_points.shape[:-1] + (1,))], dim=-1)
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = torch.cat([key_points.new_zeros([len(fix_height), 2]), fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        return key_points.flatten(2, 4)


class DeformableFeatureAggregation(nn.Module):
    def __init__(self, config: SparseDriveV2Config, num_pts: int):
        super().__init__()
        embed_dims = config.d_model
        self.embed_dims = embed_dims
        self.num_groups = config.num_groups
        self.num_levels = config.num_levels
        self.num_cams = config.num_cams
        self.kps_generator = SparsePoint3DKeyPointsGenerator(
            embed_dims, num_pts, config.num_learnable_pts, config.fix_height
        )
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.camera_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
        self.weights_fc = nn.Linear(embed_dims, self.num_groups * self.num_levels * self.num_pts)

    @staticmethod
    def project_points(key_points, projection_mat, image_wh):
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        depth = points_2d[..., 2]
        mask = depth > 1e-5
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        mask = mask & (points_2d[..., 0] > 0) & (points_2d[..., 1] > 0)
        points_2d = points_2d / image_wh[:, :, None, None]
        mask = mask & (points_2d[..., 0] < 1) & (points_2d[..., 1] < 1)
        return points_2d, mask  # (bs, cams, A, pts, 2), (bs, cams, A, pts)

    def _get_weights(self, instance_feature, projection_mat, mask):
        bs, num_anchor = instance_feature.shape[:2]
        camera_embed = self.camera_encoder(projection_mat[:, :, :3].reshape(bs, self.num_cams, -1))
        feature = instance_feature[:, :, None] + camera_embed[:, None]
        weights = self.weights_fc(feature).reshape(
            bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups
        )
        mask = mask.permute(0, 2, 1, 3)[..., None, :, None]
        weights = weights.masked_fill(torch.logical_and(~mask, mask.sum(dim=2, keepdim=True) != 0), float("-inf"))
        return (
            weights.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
        )

    def forward(self, instance_feature, anchor, value, spatial_shapes, projection_mat, image_wh):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        points_2d, mask = self.project_points(key_points, projection_mat, image_wh)
        weights = self._get_weights(instance_feature, projection_mat, mask)
        points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(bs, num_anchor * self.num_pts, -1, 2)
        weights = (
            weights.permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .reshape(bs, num_anchor * self.num_pts, self.num_cams, self.num_levels, self.num_groups)
        )
        features = deformable_aggregation(value, spatial_shapes, points_2d, weights)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims).sum(dim=2)
        return self.output_proj(features) + instance_feature


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def _gather(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Gather along ``dim`` with a (bs, k) index broadcast over the trailing dims."""
    shape = list(x.shape)
    shape[dim] = index.shape[1]
    view = [index.shape[0]] + [1] * (x.dim() - 1)
    view[dim] = index.shape[1]
    return torch.gather(x, dim, index.view(view).expand(shape))


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, config: SparseDriveV2Config, decoder_idx: int):
        super().__init__()
        c = config
        self._config = c
        self.decoder_idx = decoder_idx
        self.is_last = decoder_idx == c.decoder_num_layers - 1

        def mha():
            return nn.MultiheadAttention(c.d_model, c.num_head, batch_first=True)

        self.p_deform_model = DeformableFeatureAggregation(c, c.len_path)
        self.p_attention = mha()
        self.p_ffn = _mlp(c.d_model, c.d_ffn, c.d_model)
        self.p_norm1 = nn.LayerNorm(c.d_model)
        self.p_norm2 = nn.LayerNorm(c.d_model)
        self.path_mlp = _mlp(c.d_model, c.d_ffn, 1)

        self.v_img_attention = mha()
        self.v_attention = mha()
        self.v_ffn = _mlp(c.d_model, c.d_ffn, c.d_model)
        self.v_norm1 = nn.LayerNorm(c.d_model)
        self.v_norm2 = nn.LayerNorm(c.d_model)
        self.vel_mlp = _mlp(c.d_model, c.d_ffn, 1)

        if self.is_last:
            self.t_deform_model = DeformableFeatureAggregation(c, c.num_poses)
            self.t_attention = mha()
            self.t_ffn = _mlp(c.d_model, c.d_ffn, c.d_model)
            self.t_norm1 = nn.LayerNorm(c.d_model)
            self.t_norm2 = nn.LayerNorm(c.d_model)
            self.traj_mlp = _mlp(c.d_model, c.d_ffn, 1)
            self.metric_heads = nn.ModuleDict({m: _mlp(c.d_model, c.d_ffn, 1) for m in c.metrics})

    def forward(self, state: Dict[str, torch.Tensor], ctx: Dict[str, torch.Tensor]):
        """``state`` holds the current candidate set; returns (new state, outputs)."""
        c = self._config
        status = ctx["status_encoding"].unsqueeze(1)
        dfa_args = (ctx["value"], ctx["spatial_shapes"], ctx["projection_mat"], ctx["image_wh"])

        path_vocab, vel_vocab = state["path_vocab"], state["vel_vocab"]

        # path
        path_embed = state["path_embed"] + status
        path_embed = self.p_deform_model(path_embed, path_vocab[..., :2].flatten(-2), *dfa_args)
        path_embed = self.p_norm1(path_embed + self.p_attention(path_embed, path_embed, path_embed)[0])
        path_embed = self.p_norm2(path_embed + self.p_ffn(path_embed))
        path_scores = self.path_mlp(path_embed).squeeze(-1)

        # velocity
        vel_embed = state["vel_embed"] + status
        img_value = ctx["img_value"]
        vel_embed = vel_embed + self.v_img_attention(vel_embed, img_value, img_value)[0]
        vel_embed = self.v_norm1(vel_embed + self.v_attention(vel_embed, vel_embed, vel_embed)[0])
        vel_embed = self.v_norm2(vel_embed + self.v_ffn(vel_embed))
        vel_scores = self.vel_mlp(vel_embed).squeeze(-1)

        # coarse filter (factorized top-k)
        path_idx = torch.topk(path_scores, c.path_filter_num[self.decoder_idx], dim=1).indices
        vel_idx = torch.topk(vel_scores, c.velocity_filter_num[self.decoder_idx], dim=1).indices
        new_state = {
            "path_embed": _gather(path_embed, 1, path_idx),
            "path_vocab": _gather(path_vocab, 1, path_idx),
            "traj_vocab": _gather(_gather(state["traj_vocab"], 1, path_idx), 2, vel_idx),
            "vel_embed": _gather(vel_embed, 1, vel_idx),
            "vel_vocab": _gather(vel_vocab, 1, vel_idx),
            "path_orig_idx": _gather(state["path_orig_idx"], 1, path_idx),
            "vel_orig_idx": _gather(state["vel_orig_idx"], 1, vel_idx),
        }
        out = {
            f"path_scores_{self.decoder_idx}": path_scores,
            f"vel_scores_{self.decoder_idx}": vel_scores,
            f"path_idx_{self.decoder_idx}": path_idx,
            f"vel_idx_{self.decoder_idx}": vel_idx,
        }
        if not self.is_last:
            return new_state, out

        # trajectory re-scoring on the composed candidates
        traj_vocab = new_state["traj_vocab"].flatten(1, 2)  # (bs, P*V, T, 3)
        traj_embed = (new_state["path_embed"].unsqueeze(2) + new_state["vel_embed"].unsqueeze(1)).flatten(1, 2)
        traj_embed = self.t_deform_model(traj_embed, traj_vocab[..., :2].flatten(-2), *dfa_args)
        traj_embed = self.t_norm1(traj_embed + self.t_attention(traj_embed, traj_embed, traj_embed)[0])
        traj_embed = self.t_norm2(traj_embed + self.t_ffn(traj_embed))
        out["traj_embed"] = traj_embed
        out["traj_scores"] = self.traj_mlp(traj_embed).squeeze(-1)
        logit = {m: self.metric_heads[m](traj_embed).squeeze(-1) for m in c.metrics}
        for m in c.metrics:
            out[f"metric_{m}"] = logit[m]
        out["scores"] = combine_metric_scores(logit, c.dataset_version)
        mode_idx = out["scores"].argmax(1)
        out["mode_idx"] = mode_idx
        out["trajectory"] = traj_vocab[torch.arange(traj_vocab.shape[0]), mode_idx]
        return new_state, out


def combine_metric_scores(logit: Dict[str, torch.Tensor], dataset_version: str) -> torch.Tensor:
    s = {k: v.sigmoid() for k, v in logit.items()}
    if dataset_version == "v1":
        return (s["no_at_fault_collisions"] * s["drivable_area_compliance"]) * (
            5 * s["time_to_collision_within_bound"] + 5 * s["ego_progress"] + 2 * s["comfort"]
        )
    return (
        s["no_at_fault_collisions"]
        * s["drivable_area_compliance"]
        * s["driving_direction_compliance"]
        * s["traffic_light_compliance"]
    ) * (
        5 * s["time_to_collision_within_bound"]
        + 5 * s["ego_progress"]
        + 2 * s["lane_keeping"]
        + 2 * s["history_comfort"]
    )


class CustomTransformerDecoder(nn.Module):
    def __init__(self, config: SparseDriveV2Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [CustomTransformerDecoderLayer(config, i) for i in range(config.decoder_num_layers)]
        )

    def forward(self, state, ctx):
        outputs = {}
        for layer in self.layers:
            state, out = layer(state, ctx)
            outputs.update(out)
        return outputs


class TrajectoryHead(nn.Module):
    def __init__(self, config: SparseDriveV2Config):
        super().__init__()
        c = config
        self.path_vocab = nn.Parameter(torch.zeros(c.mode_path, c.len_path, 3), requires_grad=False)
        self.vel_vocab = nn.Parameter(torch.zeros(c.mode_vel, c.len_vel_seq), requires_grad=False)
        self.traj_vocab = nn.Parameter(torch.zeros(c.mode_path, c.mode_vel, c.num_poses, 3), requires_grad=False)
        self.traj_mask = nn.Parameter(torch.zeros(c.mode_path, c.mode_vel, c.num_poses), requires_grad=False)
        self.path_pos_embed = _mlp(c.len_path * 3, c.d_ffn, c.d_model)
        self.vel_pos_embed = _mlp(c.len_vel_seq, c.d_ffn, c.d_model)
        self.decoder = CustomTransformerDecoder(c)

    def forward(self, ctx):
        bs = ctx["status_encoding"].shape[0]
        path_vocab = self.path_vocab[None].expand(bs, -1, -1, -1)
        vel_vocab = self.vel_vocab[None].expand(bs, -1, -1)
        dev = path_vocab.device
        state = {
            "path_embed": self.path_pos_embed(path_vocab.flatten(-2, -1)),
            "vel_embed": self.vel_pos_embed(vel_vocab),
            "path_vocab": path_vocab,
            "vel_vocab": vel_vocab,
            "traj_vocab": self.traj_vocab[None].expand(bs, -1, -1, -1, -1),
            "path_orig_idx": torch.arange(path_vocab.shape[1], device=dev)[None].expand(bs, -1),
            "vel_orig_idx": torch.arange(vel_vocab.shape[1], device=dev)[None].expand(bs, -1),
        }
        return self.decoder(state, ctx)


class SparseDriveV2Model(nn.Module):
    def __init__(self, config: Optional[SparseDriveV2Config] = None):
        super().__init__()
        self._config = config or SparseDriveV2Config()
        self._backbone = SparseBackbone(self._config)
        self._status_encoding = nn.Linear(4 + 2 + 2, self._config.d_model)
        self._trajectory_head = TrajectoryHead(self._config)

    def encode(self, imgs: torch.Tensor, projection_mat: torch.Tensor, image_wh: torch.Tensor, status_feature):
        """Everything the decoder consumes: backbone features and per-frame context."""
        feature_maps = self._backbone(imgs)
        value, spatial_shapes = flatten_feature_maps(feature_maps)
        return {
            "feature_maps": feature_maps,
            "value": value,
            "spatial_shapes": spatial_shapes,
            "img_value": feature_maps[-1].permute(0, 1, 3, 4, 2).flatten(1, 3),
            "projection_mat": projection_mat,
            "image_wh": image_wh,
            "status_encoding": self._status_encoding(status_feature),
        }

    def forward(self, imgs, projection_mat, image_wh, status_feature) -> Dict[str, torch.Tensor]:
        """
        Args:
            imgs: (bs, num_cams, 3, H, W) normalized images.
            projection_mat: (bs, num_cams, 4, 4) lidar2img including the resize/crop.
            image_wh: (bs, num_cams, 2) image (W, H).
            status_feature: (bs, 8) = driving_command(4) | velocity(2) | acceleration(2).
        """
        ctx = self.encode(imgs, projection_mat, image_wh, status_feature)
        out = self._trajectory_head(ctx)
        out["feature_maps"] = ctx["feature_maps"]
        return out


def load_checkpoint(model: SparseDriveV2Model, ckpt_path: str) -> SparseDriveV2Model:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k[len(CKPT_PREFIX) :]: v for k, v in sd.items() if k.startswith(CKPT_PREFIX)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()


def build_reference_model(ckpt_path: Optional[str], config: Optional[SparseDriveV2Config] = None):
    model = SparseDriveV2Model(config)
    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path)
    return model.eval()
