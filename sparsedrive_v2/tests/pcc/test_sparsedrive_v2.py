# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SparseDriveV2 on device vs the reference (real checkpoint).

The decoder narrows its candidates with top-k twice, so a tiny score difference
near the k-th place changes which candidates survive and every tensor after it
stops being comparable element-wise.  The test therefore runs twice:

* teacher-forced — the reference's top-k selections are injected, so every
  score tensor can be compared with PCC;
* free-running — the device makes its own selections; we check the overlap of
  the selected sets and that the chosen trajectory matches the reference.
"""

import pytest

from models.common.utility_functions import comp_pcc
from models.experimental.sparsedrive_v2.tt.ttnn_sparsedrive_v2 import TtSparseDriveV2, outputs_to_torch

SCORE_KEYS = ("path_scores_0", "vel_scores_0", "path_scores_1", "vel_scores_1", "traj_scores", "scores")
INDEX_KEYS = ("path_idx_0", "vel_idx_0", "path_idx_1", "vel_idx_1")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_teacher_forced(device, ref_model, ref_outputs, inputs):
    _, ref = ref_outputs
    tt_model = TtSparseDriveV2(device, ref_model)
    forced = {k: ref[k] for k in INDEX_KEYS}
    out = outputs_to_torch(tt_model(inputs, forced=forced), tt_model.config)
    metric_keys = [k for k in ref if k.startswith("metric_")]
    failures = []
    for k in SCORE_KEYS + tuple(metric_keys):
        passed, pcc = comp_pcc(ref[k], out[k], 0.99)
        print(f"{k:45s} pcc={pcc}")
        if not passed:
            failures.append((k, pcc))
    assert not failures, failures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_free_running(device, ref_model, ref_outputs, inputs):
    _, ref = ref_outputs
    tt_model = TtSparseDriveV2(device, ref_model)
    out = outputs_to_torch(tt_model(inputs), tt_model.config)
    # layer-1 indices point into the layer-0 survivor list; compare in vocabulary ids
    for name in ("path", "vel"):
        for res in (ref, out):
            res[f"{name}_vocab_0"] = res[f"{name}_idx_0"]
            res[f"{name}_vocab_1"] = res[f"{name}_idx_0"].gather(1, res[f"{name}_idx_1"])
        for i in range(2):
            k = f"{name}_vocab_{i}"
            overlap = len(set(out[k][0].tolist()) & set(ref[k][0].tolist())) / ref[k].shape[1]
            print(f"{k}: overlap {overlap:.3f}")
    l2 = (out["trajectory"][..., :2] - ref["trajectory"][..., :2]).norm(dim=-1)
    print("ref trajectory", ref["trajectory"][0, :, :2].tolist())
    print("tt  trajectory", out["trajectory"][0, :, :2].tolist())
    print(f"trajectory L2: mean {l2.mean():.4f} m, final {l2[0, -1]:.4f} m")
    ref_best = ref["scores"].max().item()
    print(f"ref best score {ref_best:.4f}, tt best score {out['scores'].max().item():.4f}")
    assert l2.mean() < 0.5, f"mean trajectory L2 {l2.mean():.3f} m"
