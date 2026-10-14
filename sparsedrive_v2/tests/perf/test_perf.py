# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Eager vs traced end-to-end latency (batch 1, warm program cache)."""

import time

import pytest
import torch

import ttnn
from sparsedrive_v2.tt.ttnn_sparsedrive_v2 import (
    TracedSparseDriveV2,
    TtSparseDriveV2,
    outputs_to_torch,
)

ITERS = 10


def _time_ms(device, fn, iters=ITERS):
    """(throughput ms/frame with frames pipelined, latency ms of one synchronized frame)."""
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    throughput = (time.perf_counter() - t0) / iters * 1e3
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        latencies.append((time.perf_counter() - t0) * 1e3)
    return throughput, sorted(latencies)[iters // 2]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 200_000_000}], indirect=True)
def test_perf(device, ref_model, inputs):
    model = TtSparseDriveV2(device, ref_model)
    eager_out = outputs_to_torch(model(inputs), model.config)
    eager_ms = _time_ms(device, lambda: model(inputs))

    traced = TracedSparseDriveV2(model, inputs)
    traced_out = outputs_to_torch(traced(inputs, blocking=True), model.config)
    traced_ms = _time_ms(device, lambda: traced(inputs))
    traced.release()

    for name, (thr, lat) in (("eager", eager_ms), ("traced", traced_ms)):
        print(f"{name:6s}: {thr:.1f} ms/frame pipelined ({1e3 / thr:.1f} FPS), single-frame latency {lat:.1f} ms")
    assert torch.equal(traced_out["mode_idx"], eager_out["mode_idx"])
    assert torch.allclose(traced_out["scores"], eager_out["scores"])
