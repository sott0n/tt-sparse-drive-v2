# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HTTP server for SparseDriveV2 on one Blackhole chip (tt-model `tt-dit-server` kind).

POST /v1/plan    {"cameras": [{"image": <base64 PNG/JPEG, RGB>,
                               "intrinsics": 3x3,
                               "sensor2lidar_rotation": 3x3,
                               "sensor2lidar_translation": 3} x 3 in cam_l0, cam_f0, cam_r0 order],
                  "status": [8 floats: driving command one-hot (4), velocity xy, acceleration xy]}
                 -> {"trajectory": [[x, y, heading] x 8], "timing_ms": {...}}
GET  /v1/health  -> {"status": "ok" | "loading" | "error", ...}
GET  /v1/models  -> the served model id

The camera fields are NAVSIM's. Images may be any size; they are resized and cropped as in
upstream inference. Weights are not bundled: SDV2_CHECKPOINT_PATH names the upstream
checkpoint, otherwise it is fetched from the Hugging Face repo in HF_MODEL. The device opens
and the trace is captured at startup, so "Application startup complete" means ready.
Requests are serialised on the chip.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = "sparsedrive-v2"
NUM_CAMS = 3
STATUS_DIM = 8

STATE = {"status": "loading", "error": None, "model": None, "traced": None, "device": None}
LOCK = threading.Lock()


def _checkpoint() -> str:
    path = os.environ.get("SDV2_CHECKPOINT_PATH")
    if path:
        if not os.path.isfile(path):
            raise RuntimeError(f"SDV2_CHECKPOINT_PATH must name an existing file (got {path!r})")
        return path
    repo = os.environ.get("HF_MODEL")
    if not repo:
        raise RuntimeError("set SDV2_CHECKPOINT_PATH or HF_MODEL")
    from huggingface_hub import hf_hub_download

    from sparsedrive_v2.common import CKPT_NAME

    return hf_hub_download(repo, CKPT_NAME, revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION"))


def _default_mesh_descriptor() -> None:
    # One chip of a 2-chip p300 is detected as a CUSTOM cluster and needs a 1x1 mesh descriptor.
    # The server always runs on one chip, so the P150 descriptor fits every board. Its path is
    # inside the installed ttnn, which a user of a pulled bundle cannot easily name.
    if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
        return
    import importlib.util

    ttnn_dir = os.path.dirname(importlib.util.find_spec("ttnn").origin)
    desc = os.path.join(ttnn_dir, "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
    if os.path.isfile(desc):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = desc


def _load():
    _default_mesh_descriptor()
    import ttnn

    from sparsedrive_v2.common import FINAL_HW, frame_inputs, synthetic_camera_rig
    from sparsedrive_v2.reference.sparsedrive_v2 import build_reference_model
    from sparsedrive_v2.tt.ttnn_sparsedrive_v2 import TracedSparseDriveV2, TtSparseDriveV2

    ref = build_reference_model(_checkpoint())
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    model = TtSparseDriveV2(device, ref)
    # Every request reuses this trace, so capture it once with inputs of the served shape.
    blank = np.zeros((*FINAL_HW, 3), dtype=np.uint8)
    example = frame_inputs([blank] * NUM_CAMS, [np.eye(4)] * NUM_CAMS, np.zeros(STATUS_DIM))
    example["projection_mat"] = synthetic_camera_rig()[None]
    traced = TracedSparseDriveV2(model, example)
    STATE.update(model=model, traced=traced, device=device, status="ok")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _load()
    except Exception as e:
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    yield
    import ttnn

    with LOCK:
        if STATE["traced"] is not None:
            STATE["traced"].release()
        if STATE["device"] is not None:
            ttnn.close_device(STATE["device"])


app = FastAPI(title="SparseDriveV2", lifespan=lifespan)


class Camera(BaseModel):
    image: str = Field(description="base64 PNG/JPEG, RGB")
    intrinsics: List[List[float]] = Field(min_length=3, max_length=3)
    sensor2lidar_rotation: List[List[float]] = Field(min_length=3, max_length=3)
    sensor2lidar_translation: List[float] = Field(min_length=3, max_length=3)


class PlanRequest(BaseModel):
    cameras: List[Camera] = Field(min_length=NUM_CAMS, max_length=NUM_CAMS, description="cam_l0, cam_f0, cam_r0")
    status: List[float] = Field(min_length=STATUS_DIM, max_length=STATUS_DIM)


class PlanResponse(BaseModel):
    trajectory: List[List[float]]
    timing_ms: dict


def _decode_image(b64: str) -> np.ndarray:
    try:
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64, validate=True))).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"image is not a base64 PNG/JPEG: {e}")


def _matrix(rows: List[List[float]], name: str) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float64)
    if m.shape != (3, 3):
        raise HTTPException(status_code=422, detail=f"{name} must be 3x3, got {m.shape}")
    return m


@app.post("/v1/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    if STATE["status"] != "ok":
        raise HTTPException(status_code=503, detail=f"model {STATE['status']}")
    from sparsedrive_v2.common import frame_inputs, lidar2img
    from sparsedrive_v2.tt.ttnn_sparsedrive_v2 import outputs_to_torch

    t0 = time.perf_counter()
    images = [_decode_image(c.image) for c in req.cameras]
    mats = [
        lidar2img(
            _matrix(c.intrinsics, "intrinsics"),
            _matrix(c.sensor2lidar_rotation, "sensor2lidar_rotation"),
            c.sensor2lidar_translation,
        )
        for c in req.cameras
    ]
    inputs = frame_inputs(images, mats, req.status)
    t1 = time.perf_counter()
    with LOCK:
        out = STATE["traced"](inputs, blocking=True)
        trajectory = outputs_to_torch({"trajectory": out["trajectory"]}, STATE["model"].config)["trajectory"][0]
    t2 = time.perf_counter()
    return PlanResponse(
        trajectory=trajectory.tolist(),
        timing_ms={"preprocess": (t1 - t0) * 1e3, "device": (t2 - t1) * 1e3},
    )


@app.get("/v1/health")
def health() -> dict:
    return {"status": STATE["status"], "error": STATE["error"]}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}
