# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN SparseDriveV2 (NAVSIM v1 agent), batch size 1.

Data flow (all on device after the input upload):

  imgs (3 cams) -> ResNet-34 + FPN -> 4 levels (cams, H, W, 256)
  path queries (1024) / velocity queries (256): constant vocab embeddings + status
  decoder layer 0:  path DFA + self-attn + FFN -> score -> top-128
                    vel image cross-attn + self-attn + FFN -> score -> top-64
  decoder layer 1:  same on the survivors -> top-20 paths x top-20 velocities
                    400 composed trajectories -> DFA + self-attn + FFN -> metric heads
  scores = sigmoid(NC) * sigmoid(DAC) * (5 TTC + 5 EP + 2 C) -> argmax -> trajectory

Candidate bookkeeping uses indices into the original vocabularies: gathers of
rows use ``ttnn.embedding``; fp32 rows (coordinates, fp32 activations) are gathered as
a hi/lo pair of bf16 tables summed in fp32 (~1e-5 relative error).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

import ttnn
from models.experimental.sparsedrive_v2.reference.sparsedrive_v2 import SparseDriveV2Config
from models.experimental.sparsedrive_v2.tt.ttnn_backbone import FeatureMap, TtSparseBackbone
from models.experimental.sparsedrive_v2.tt.ttnn_dfa import (
    TtDeformableFeatureAggregation,
    camera_context,
    camera_host_tensors,
)
from models.experimental.sparsedrive_v2.tt.ttnn_ops import (
    ACT_DTYPE,
    TtLayerNorm,
    TtLinear,
    TtMLP,
    TtMultiheadAttention,
    to_device,
)

DFA_CHUNK = 256


class HiLoTable:
    """fp32 row table gathered with two bf16 ``ttnn.embedding`` lookups (hi + lo)."""

    def __init__(self, device, table: torch.Tensor):
        hi = table.float().to(torch.bfloat16).float()
        lo = table.float() - hi
        self.hi = to_device(hi, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.lo = to_device(lo, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, rows: ttnn.Tensor) -> ttnn.Tensor:
        """rows: (1, k) uint32 row-major -> (1, k, width) fp32 tile."""

        def lookup(t):
            return ttnn.typecast(ttnn.embedding(rows, t, layout=ttnn.TILE_LAYOUT), ttnn.float32)

        return ttnn.add(lookup(self.hi), lookup(self.lo))


def _rows(index: ttnn.Tensor) -> ttnn.Tensor:
    """(1, 1, 1, k) integer index (tile) -> (1, k) uint32 row-major for ``ttnn.embedding``."""
    k = index.shape[-1]
    if index.dtype != ttnn.uint32:
        index = ttnn.typecast(index, ttnn.uint32)
    return ttnn.to_layout(ttnn.reshape(index, (1, k)), ttnn.ROW_MAJOR_LAYOUT)


def _gather_rows(table: ttnn.Tensor, rows: ttnn.Tensor) -> ttnn.Tensor:
    """table (1, N, C) tile, rows (1, k) uint32 -> (1, k, C) tile in the table's dtype."""
    if table.dtype == ttnn.bfloat16:
        return ttnn.embedding(rows, ttnn.to_layout(table[0], ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.TILE_LAYOUT)
    hi = ttnn.typecast(table, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(table, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    parts = [_gather_rows(t, rows) for t in (hi, lo)]
    return ttnn.add(ttnn.typecast(parts[0], ttnn.float32), ttnn.typecast(parts[1], ttnn.float32))


def _as_row(scores: ttnn.Tensor) -> ttnn.Tensor:
    """(1, N, 1) -> (1, 1, 1, N)."""
    return ttnn.reshape(ttnn.transpose(scores, -2, -1), (1, 1, 1, scores.shape[1]))


class TtDecoderLayer:
    def __init__(self, device, ref, config: SparseDriveV2Config, decoder_idx: int):
        self.device = device
        self.config = config
        self.idx = decoder_idx
        self.is_last = ref.is_last
        self.path_k = config.path_filter_num[decoder_idx]
        self.vel_k = config.velocity_filter_num[decoder_idx]

        self.p_dfa = TtDeformableFeatureAggregation(device, ref.p_deform_model, DFA_CHUNK)
        self.p_attn = TtMultiheadAttention(device, ref.p_attention)
        self.p_ffn = TtMLP(device, ref.p_ffn)
        self.p_norm1, self.p_norm2 = TtLayerNorm(device, ref.p_norm1), TtLayerNorm(device, ref.p_norm2)
        self.path_mlp = TtMLP(device, ref.path_mlp)

        self.v_img_attn = TtMultiheadAttention(device, ref.v_img_attention)
        self.v_attn = TtMultiheadAttention(device, ref.v_attention)
        self.v_ffn = TtMLP(device, ref.v_ffn)
        self.v_norm1, self.v_norm2 = TtLayerNorm(device, ref.v_norm1), TtLayerNorm(device, ref.v_norm2)
        self.vel_mlp = TtMLP(device, ref.vel_mlp)

        if self.is_last:
            self.t_dfa = TtDeformableFeatureAggregation(device, ref.t_deform_model, DFA_CHUNK)
            self.t_attn = TtMultiheadAttention(device, ref.t_attention)
            self.t_ffn = TtMLP(device, ref.t_ffn)
            self.t_norm1, self.t_norm2 = TtLayerNorm(device, ref.t_norm1), TtLayerNorm(device, ref.t_norm2)
            self.traj_mlp = TtMLP(device, ref.traj_mlp)
            self.metric_heads = {m: TtMLP(device, ref.metric_heads[m]) for m in config.metrics}

    def prepare_frame(self, cam):
        frames = {"p": self.p_dfa.prepare_frame(cam)}
        if self.is_last:
            frames["t"] = self.t_dfa.prepare_frame(cam)
        return frames

    def _select(self, scores_row, k, forced):
        if forced is not None:
            return to_device(forced.reshape(1, 1, 1, -1).int(), self.device, ttnn.uint32)
        _, index = ttnn.topk(scores_row, k, dim=-1, largest=True, sorted=True)
        return ttnn.typecast(index, ttnn.uint32)

    def __call__(self, state, ctx, tables, forced: Optional[Dict[str, torch.Tensor]] = None):
        forced = forced or {}
        out = {}
        status = ctx["status"]

        # path
        e = ttnn.add(state["path_embed"], status)
        e = self.p_dfa(e, state["path_ax"], state["path_ay"], ctx["values"], ctx["frames"][self.idx]["p"])
        e = self.p_norm1(ttnn.add(e, self.p_attn(e)))
        path_embed = self.p_norm2(ttnn.add(e, self.p_ffn(e)))
        path_scores = _as_row(self.path_mlp(path_embed, dtype=ttnn.float32))

        # velocity
        e = ttnn.add(state["vel_embed"], status)
        e = ttnn.add(e, self.v_img_attn(e, ctx["img_value"]))
        e = self.v_norm1(ttnn.add(e, self.v_attn(e)))
        vel_embed = self.v_norm2(ttnn.add(e, self.v_ffn(e)))
        vel_scores = _as_row(self.vel_mlp(vel_embed, dtype=ttnn.float32))

        # factorized top-k
        path_idx = self._select(path_scores, self.path_k, forced.get(f"path_idx_{self.idx}"))
        vel_idx = self._select(vel_scores, self.vel_k, forced.get(f"vel_idx_{self.idx}"))
        path_orig = ttnn.gather(state["path_orig"], -1, path_idx)
        vel_orig = ttnn.gather(state["vel_orig"], -1, vel_idx)
        path_embed = _gather_rows(path_embed, _rows(path_idx))
        vel_embed = _gather_rows(vel_embed, _rows(vel_idx))

        out[f"path_scores_{self.idx}"] = path_scores
        out[f"vel_scores_{self.idx}"] = vel_scores
        out[f"path_idx_{self.idx}"] = path_idx
        out[f"vel_idx_{self.idx}"] = vel_idx

        new_state = {"path_embed": path_embed, "vel_embed": vel_embed, "path_orig": path_orig, "vel_orig": vel_orig}
        if not self.is_last:
            anchors = tables["path_anchor"](_rows(path_orig))
            pp = self.p_dfa.Pp
            new_state["path_ax"], new_state["path_ay"] = anchors[:, :, :pp], anchors[:, :, pp:]
            return new_state, out

        # compose P x V trajectories: index into the flattened (path, vel) vocabulary
        num_p, num_v = self.path_k, self.vel_k
        p_col = ttnn.repeat(ttnn.reshape(path_orig, (1, 1, num_p, 1)), (1, 1, 1, num_v))
        traj_orig = ttnn.add(ttnn.multiply(p_col, float(self.config.mode_vel)), vel_orig)
        traj_orig = ttnn.reshape(traj_orig, (1, 1, 1, num_p * num_v))
        traj_rows = _rows(traj_orig)

        p_rep = ttnn.repeat(ttnn.reshape(path_embed, (1, num_p, 1, path_embed.shape[-1])), (1, 1, num_v, 1))
        t = ttnn.add(p_rep, ttnn.reshape(vel_embed, (1, 1, num_v, vel_embed.shape[-1])))
        t = ttnn.reshape(t, (1, num_p * num_v, path_embed.shape[-1]))

        anchors = tables["traj_anchor"](traj_rows)
        tp = self.t_dfa.Pp
        t = self.t_dfa(t, anchors[:, :, :tp], anchors[:, :, tp:], ctx["values"], ctx["frames"][self.idx]["t"])
        t = self.t_norm1(ttnn.add(t, self.t_attn(t)))
        traj_embed = self.t_norm2(ttnn.add(t, self.t_ffn(t)))
        out["traj_embed"] = traj_embed
        out["traj_scores"] = self.traj_mlp(traj_embed, dtype=ttnn.float32)
        sig = {}
        for m, head in self.metric_heads.items():
            logit = head(traj_embed, dtype=ttnn.float32)
            out[f"metric_{m}"] = logit
            sig[m] = ttnn.sigmoid(logit)
        scores = self._combine(sig)
        out["scores"] = scores
        mode_idx = ttnn.argmax(_as_row(scores), dim=-1, keepdim=True)  # (1, 1, 1, 1)
        mode_idx = ttnn.to_layout(ttnn.typecast(mode_idx, ttnn.uint32), ttnn.TILE_LAYOUT)
        best = ttnn.gather(traj_orig, -1, mode_idx)
        out["mode_idx"] = mode_idx
        out["trajectory"] = tables["traj_out"](_rows(best))
        out["traj_orig"] = traj_orig
        return new_state, out

    def _combine(self, s):
        if self.config.dataset_version != "v1":
            raise NotImplementedError("only the NAVSIM v1 score is wired up")
        weighted = ttnn.add(
            ttnn.multiply(ttnn.add(s["time_to_collision_within_bound"], s["ego_progress"]), 5.0),
            ttnn.multiply(s["comfort"], 2.0),
        )
        return ttnn.multiply(ttnn.multiply(s["no_at_fault_collisions"], s["drivable_area_compliance"]), weighted)


class TtSparseDriveV2:
    def __init__(self, device, ref_model, config: Optional[SparseDriveV2Config] = None):
        self.device = device
        self.config = c = config or ref_model._config
        self.backbone = TtSparseBackbone(device, ref_model._backbone)
        self.status_fc = TtLinear.from_torch(device, ref_model._status_encoding)

        head = ref_model._trajectory_head
        self.layers = [TtDecoderLayer(device, layer, c, i) for i, layer in enumerate(head.decoder.layers)]
        p_dfa, t_dfa = self.layers[0].p_dfa, self.layers[-1].t_dfa

        with torch.no_grad():
            # vocabulary embeddings do not depend on the input: fold them at load time
            path_embed = head.path_pos_embed(head.path_vocab.flatten(-2))
            vel_embed = head.vel_pos_embed(head.vel_vocab)
            path_table = p_dfa.anchor_table(head.path_vocab)
            traj_flat = head.traj_vocab.reshape(-1, c.num_poses, 3)
            traj_table = t_dfa.anchor_table(traj_flat)
            traj_out = torch.nn.functional.pad(traj_flat.reshape(traj_flat.shape[0], -1), (0, 8))

        self.init_state = {
            "path_embed": to_device(path_embed[None], device, ACT_DTYPE),
            "vel_embed": to_device(vel_embed[None], device, ACT_DTYPE),
            "path_ax": to_device(path_table[None, :, : p_dfa.Pp], device, ttnn.float32),
            "path_ay": to_device(path_table[None, :, p_dfa.Pp :], device, ttnn.float32),
            "path_orig": to_device(torch.arange(c.mode_path).float().reshape(1, 1, 1, -1), device, ttnn.float32),
            "vel_orig": to_device(torch.arange(c.mode_vel).float().reshape(1, 1, 1, -1), device, ttnn.float32),
        }
        self.tables = {
            "path_anchor": HiLoTable(device, path_table),
            "traj_anchor": HiLoTable(device, traj_table),
            "traj_out": HiLoTable(device, traj_out),
        }

    # -- inputs ----------------------------------------------------------------

    def host_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, ttnn.Tensor]:
        """Host tensors of one frame (``common.sample_inputs`` layout, batch size 1)."""
        imgs = inputs["imgs"][0]
        self._img_shape = (imgs.shape[0], imgs.shape[2], imgs.shape[3])
        host = {
            "imgs": self.backbone.resnet.host_input(imgs),
            "status": ttnn.from_torch(
                inputs["status_feature"].reshape(1, 1, -1).float(), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT
            ),
        }
        host.update(camera_host_tensors(inputs["projection_mat"][0], inputs["image_wh"][0]))
        return host

    def to_device_inputs(self, host: Dict[str, ttnn.Tensor]) -> Dict[str, ttnn.Tensor]:
        return {k: ttnn.to_device(v, self.device) for k, v in host.items()}

    # -- device forward (trace-safe: no host transfers) --------------------------

    def encode(self, dev: Dict[str, ttnn.Tensor]):
        """Backbone + per-frame decoder context from device inputs."""
        n, h, w = self._img_shape
        stem_in = FeatureMap(dev["imgs"], n, h, w, self.backbone.resnet.STEM_IN_CHANNELS)
        feats = self.backbone.fpn(self.backbone.resnet(stem_in))
        values = []
        for f in feats:
            t = ttnn.to_layout(f.tensor, ttnn.ROW_MAJOR_LAYOUT)
            values.append(ttnn.reshape(t, (f.n, f.h, f.w, f.c)))
        last = feats[-1]
        img_value = ttnn.reshape(last.tensor, (1, last.n * last.h * last.w, last.c))
        cam = camera_context(dev)
        return {
            "feats": feats,
            "values": values,
            "img_value": img_value,
            "status": self.status_fc(dev["status"]),
            "frames": [layer.prepare_frame(cam) for layer in self.layers],
        }

    def decode(self, ctx, forced: Optional[Dict[str, torch.Tensor]] = None):
        state, outputs = dict(self.init_state), {}
        for layer in self.layers:
            state, out = layer(state, ctx, self.tables, forced)
            outputs.update(out)
        return outputs

    def forward(self, dev: Dict[str, ttnn.Tensor], forced=None):
        return self.decode(self.encode(dev), forced)

    def __call__(self, inputs: Dict[str, torch.Tensor], forced=None):
        return self.forward(self.to_device_inputs(self.host_inputs(inputs)), forced)


class TracedSparseDriveV2:
    """Captures the whole forward once and replays it; only the input copies run per frame.

    The device must be opened with a ``trace_region_size`` large enough for ~1500 ops.
    """

    def __init__(self, model: TtSparseDriveV2, example_inputs: Dict[str, torch.Tensor]):
        self.model = model
        self.device = model.device
        self.dev_inputs = model.to_device_inputs(model.host_inputs(example_inputs))
        model.forward(self.dev_inputs)  # compile + prepare conv weights outside the trace
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.outputs = model.forward(self.dev_inputs)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def __call__(self, inputs: Dict[str, torch.Tensor], blocking: bool = False) -> Dict[str, ttnn.Tensor]:
        for k, host in self.model.host_inputs(inputs).items():
            ttnn.copy_host_to_device_tensor(host, self.dev_inputs[k])
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=blocking)
        return self.outputs

    def release(self):
        ttnn.release_trace(self.device, self.trace_id)


def outputs_to_torch(out: Dict[str, ttnn.Tensor], config: SparseDriveV2Config) -> Dict[str, torch.Tensor]:
    """Device outputs -> torch tensors shaped like the reference outputs."""
    res = {}
    for k, v in out.items():
        t = ttnn.to_torch(v)
        if k.startswith(("path_scores", "vel_scores")):
            t = t.float().reshape(1, -1)
        elif k.startswith(("path_idx", "vel_idx")) or k == "traj_orig":
            t = t.long().reshape(1, -1)
        elif k == "mode_idx":
            t = t.long().reshape(1)
        elif k == "trajectory":
            t = t.float().reshape(1, -1)[:, : config.num_poses * 3].reshape(1, config.num_poses, 3)
        elif k == "traj_embed":
            t = t.float()
        else:
            t = t.float().reshape(1, -1)
        res[k] = t
    return res
