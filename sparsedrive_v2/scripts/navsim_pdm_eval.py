# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NAVSIM v1 PDM score of SparseDriveV2 on navtest, for the TTNN model and/or the CPU reference.

  infer:  navtest scenes -> <out>/trajectories_<backend>.npz  (token -> (8, 3) x, y, heading)
  score:  trajectories + metric cache -> <out>/pdm_<backend>.csv

Runs in a Python env that has the navsim devkit and nuplan-devkit next to ttnn. Expects the
standard NAVSIM variables: OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION,
NAVSIM_EXP_ROOT, NAVSIM_CACHE_ROOT. The metric cache comes from the devkit's
run_metric_caching.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from sparsedrive_v2.common import IMG_MEAN, IMG_STD, checkpoint_path

BACKENDS = ("ttnn", "torch")
CAMS = ("cam_l0", "cam_f0", "cam_r0")  # SparseDriveV2Config.cams order
FINAL_HW = (256, 512)


def _compose(overrides: List[str]):
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        return compose(config_name="default_run_pdm_score", overrides=overrides)


def _lidar2img(cam) -> np.ndarray:
    """Upstream SparseDriveFeatureBuilder.get_camera_params."""
    lidar2cam_r = np.linalg.inv(cam.sensor2lidar_rotation)
    lidar2cam_t = cam.sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:3, :3] = cam.intrinsics
    return viewpad @ lidar2cam_rt.T


def features(agent_input) -> Dict[str, torch.Tensor]:
    """Upstream test-mode pipeline: resize to 512 wide, keep the bottom 256 rows, normalize."""
    fh, fw = FINAL_HW
    cams = agent_input.cameras[-1]
    imgs, mats = [], []
    for name in CAMS:
        cam = getattr(cams, name)
        h, w = cam.image.shape[:2]
        resize = max(fh / h, fw / w)
        nw, nh = int(w * resize), int(h * resize)
        crop_w, crop_h = int(max(0, nw - fw) / 2), nh - fh
        img = Image.fromarray(cam.image).resize((nw, nh)).crop((crop_w, crop_h, crop_w + fw, crop_h + fh))
        imgs.append(((np.asarray(img, dtype=np.float32) - IMG_MEAN) / IMG_STD).astype(np.float32))
        # Image-plane affine on homogeneous (u*d, v*d, d): its translation multiplies depth.
        aug = np.eye(4)
        aug[:2, :2] *= resize
        aug[:2, 2] -= (crop_w, crop_h)
        mats.append(aug @ _lidar2img(cam))
    s = agent_input.ego_statuses[-1]
    status = np.concatenate([s.driving_command, s.ego_velocity, s.ego_acceleration]).astype(np.float32)
    return {
        "imgs": torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()[None],
        "projection_mat": torch.tensor(np.stack(mats), dtype=torch.float32)[None],
        "image_wh": torch.tensor([[fw, fh]] * len(CAMS), dtype=torch.float32)[None],
        "status_feature": torch.tensor(status)[None],
    }


def infer(cfg, args) -> None:
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    from sparsedrive_v2.reference.sparsedrive_v2 import build_reference_model

    sensors = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[],
        lidar_pc=[],
    )
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=sensors,
    )
    tokens = sorted(loader.tokens)[: args.limit or None]
    ref = build_reference_model(args.checkpoint)

    traced, device = None, None
    if "ttnn" in args.backends:
        import ttnn

        from sparsedrive_v2.tt.ttnn_sparsedrive_v2 import (
            TracedSparseDriveV2,
            TtSparseDriveV2,
            outputs_to_torch,
        )

        device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
        tt_model = TtSparseDriveV2(device, ref)

    results: Dict[str, Dict[str, np.ndarray]] = {b: {} for b in args.backends}
    t0 = time.time()
    try:
        for i, token in enumerate(tokens):
            inputs = features(loader.get_agent_input_from_token(token))
            if device is not None:
                if args.no_trace:
                    out = outputs_to_torch(tt_model(inputs), tt_model.config)
                else:
                    if traced is None:
                        traced = TracedSparseDriveV2(tt_model, inputs)
                    out = outputs_to_torch(traced(inputs, blocking=True), tt_model.config)
                results["ttnn"][token] = out["trajectory"][0].numpy()
            if "torch" in args.backends:
                with torch.no_grad():
                    results["torch"][token] = ref(**inputs)["trajectory"][0].numpy()
            if (i + 1) % 200 == 0:
                print(
                    f"[infer] {i + 1}/{len(tokens)} scenes, {(time.time() - t0) / (i + 1) * 1e3:.1f} ms/scene",
                    flush=True,
                )
    finally:
        # Keep what finished even if the device fails part-way.
        for backend, trajs in results.items():
            path = Path(args.out) / f"trajectories_{backend}.npz"
            np.savez(path, **{k: v.astype(np.float32) for k, v in trajs.items()})
            print(f"[infer] wrote {len(trajs)} trajectories to {path}")
        if device is not None:
            if traced is not None:
                traced.release()
            ttnn.close_device(device)


def score(cfg, args) -> None:
    import pandas as pd
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig, Trajectory
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    from navsim.planning.script.run_pdm_score_gpu_v1 import run_pdm_score
    from nuplan.planning.utils.multithreading.worker_utils import worker_map

    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=None,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    worker = build_worker(cfg)
    for backend in args.backends:
        data = np.load(Path(args.out) / f"trajectories_{backend}.npz")
        trajectories = {t: {"trajectory": Trajectory(data[t])} for t in data.files}
        data_points = [
            {
                "cfg": cfg,
                "log_file": log,
                "tokens": [t for t in toks if t in trajectories],
                "model_trajectory": trajectories,
            }
            for log, toks in loader.get_tokens_list_per_log().items()
        ]
        data_points = [d for d in data_points if d["tokens"]]
        df = pd.DataFrame(worker_map(worker, run_pdm_score, data_points))
        valid = df[df["valid"]]
        csv = Path(args.out) / f"pdm_{backend}.csv"
        df.to_csv(csv, index=False)
        metrics = valid.drop(columns=["token", "valid"]).mean()
        print(f"[score] {backend}: {len(valid)}/{len(df)} valid scenes, PDMS {metrics['score']:.4f} -> {csv}")
        print(metrics.to_string())


def main() -> None:
    import os

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("infer", "score", "all"))
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=checkpoint_path())
    p.add_argument("--split", default="navtest")
    p.add_argument("--limit", type=int, default=0, help="first N scenes only (0 = all)")
    p.add_argument("--worker", default="ray_distributed_no_torch")
    p.add_argument("--no-trace", action="store_true", help="run the TTNN model eagerly")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = _compose(
        [
            f"train_test_split={args.split}",
            f"worker={args.worker}",
            "experiment_name=sparsedrive_v2_pdm",
            f"metric_cache_path={os.environ['NAVSIM_CACHE_ROOT']}/{args.split}_v1_metric_cache",
        ]
    )
    if args.mode in ("infer", "all"):
        infer(cfg, args)
    if args.mode in ("score", "all"):
        score(cfg, args)


if __name__ == "__main__":
    main()
