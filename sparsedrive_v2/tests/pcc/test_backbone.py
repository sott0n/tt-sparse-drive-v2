# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ResNet-34 + FPN on device vs the reference, real checkpoint, 3 x 256x512 cameras."""

import pytest

from models.common.utility_functions import comp_pcc
from sparsedrive_v2.tt.ttnn_backbone import TtSparseBackbone, feature_map_to_torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_backbone_fpn(device, ref_model, inputs, ref_outputs):
    ctx, _ = ref_outputs
    tt_backbone = TtSparseBackbone(device, ref_model._backbone)
    feats = tt_backbone(inputs["imgs"][0])
    for lvl, (fm, ref) in enumerate(zip(feats, ctx["feature_maps"])):
        out = feature_map_to_torch(fm)
        passed, pcc = comp_pcc(ref[0], out, 0.99)
        print(f"level {lvl} {tuple(out.shape)} pcc={pcc}")
        assert passed, f"FPN level {lvl}: {pcc}"
