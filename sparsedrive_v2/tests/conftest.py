# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the SparseDriveV2 TTNN tests.

The released checkpoint and vocabulary are not checked in; they are read from
``$SDV2_ASSET_DIR`` (default ``~/.cache/sparsedrive_v2``), see README.  Tests that need
them skip when absent, or fail under ``SDV2_REQUIRE_ASSETS=1``.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from sparsedrive_v2.common import checkpoint_path, sample_inputs
from sparsedrive_v2.reference.sparsedrive_v2 import build_reference_model


@pytest.fixture
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture
def device(device_params):
    dev = ttnn.open_device(device_id=0, **device_params)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def ref_model():
    path = checkpoint_path()
    if path is None:
        msg = "SparseDriveV2 checkpoint not found (set SDV2_ASSET_DIR or SDV2_CHECKPOINT_PATH)"
        if os.environ.get("SDV2_REQUIRE_ASSETS", "") in ("1", "true"):
            pytest.fail(msg, pytrace=False)
        pytest.skip(msg)
    return build_reference_model(path)


@pytest.fixture(scope="session")
def inputs():
    return sample_inputs()


@pytest.fixture(scope="session")
def ref_outputs(ref_model, inputs):
    """Reference forward plus the decoder context, computed once per session."""
    with torch.no_grad():
        ctx = ref_model.encode(**inputs)
        out = ref_model._trajectory_head(ctx)
    return ctx, out
