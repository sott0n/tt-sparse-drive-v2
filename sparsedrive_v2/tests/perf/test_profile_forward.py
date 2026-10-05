# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness: two eager forwards, signposts around the warm one.

    python -m tracy -p -r -v -m pytest models/experimental/sparsedrive_v2/tests/perf/test_profile_forward.py
    tt-perf-report <ops_perf_results.csv> --start-signpost iter1_start --end-signpost iter1_end
"""

import pytest

import ttnn
from models.experimental.sparsedrive_v2.tt.ttnn_sparsedrive_v2 import TtSparseDriveV2


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_profile_forward(device, ref_model, inputs):
    from tracy import signpost

    model = TtSparseDriveV2(device, ref_model)
    dev_inputs = model.to_device_inputs(model.host_inputs(inputs))
    for it in range(2):
        signpost(f"iter{it}_start")
        ctx = model.encode(dev_inputs)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)  # ~1500 ops per forward overflow the profiler buffer
        signpost(f"iter{it}_decode")
        state = dict(model.init_state)
        for i, layer in enumerate(model.layers):
            state, _ = layer(state, ctx, model.tables)
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            signpost(f"iter{it}_layer{i}_end")
        signpost(f"iter{it}_end")
