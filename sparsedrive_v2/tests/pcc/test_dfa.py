# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeformableFeatureAggregation on device vs the reference, fed the reference FPN features."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.sparsedrive_v2.tt.ttnn_dfa import TtDeformableFeatureAggregation, build_camera_context
from models.experimental.sparsedrive_v2.tt.ttnn_ops import to_device


def _values(device, feature_maps):
    return [to_device(f[0].permute(0, 2, 3, 1), device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for f in feature_maps]


@pytest.mark.parametrize("num_anchor, chunk", [(1024, 64), (128, 64)])
def test_path_dfa_layer0(device, ref_model, ref_outputs, num_anchor, chunk):
    ctx, _ = ref_outputs
    head = ref_model._trajectory_head
    ref_dfa = head.decoder.layers[0].p_deform_model
    with torch.no_grad():
        query = (head.path_pos_embed(head.path_vocab.flatten(-2)) + ctx["status_encoding"])[None, :num_anchor]
        anchor = head.path_vocab[None, :num_anchor, :, :2].flatten(-2)
        ref = ref_dfa(query, anchor, ctx["value"], ctx["spatial_shapes"], ctx["projection_mat"], ctx["image_wh"])

    tt_dfa = TtDeformableFeatureAggregation(device, ref_dfa, chunk=chunk)
    cam = build_camera_context(device, ctx["projection_mat"][0], ctx["image_wh"][0])
    frame = tt_dfa.prepare_frame(cam)
    table = tt_dfa.anchor_table(head.path_vocab[:num_anchor])
    ax = to_device(table[None, :, : tt_dfa.Pp], device, ttnn.float32)
    ay = to_device(table[None, :, tt_dfa.Pp :], device, ttnn.float32)
    out = tt_dfa(to_device(query, device), ax, ay, _values(device, ctx["feature_maps"]), frame)
    out = ttnn.to_torch(out).float()

    passed, pcc = comp_pcc(ref, out, 0.99)
    # The residual dominates the output; also check the aggregated part alone.
    passed_delta, pcc_delta = comp_pcc(ref - query, out - query, 0.98)
    print(f"A={num_anchor} pcc={pcc} delta_pcc={pcc_delta}")
    assert passed and passed_delta, (pcc, pcc_delta)
