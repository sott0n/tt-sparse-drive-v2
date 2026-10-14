# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Asset resolution and sample inputs shared by the reference, the TTNN model and the tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

HF_REPO = "https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main"
CKPT_NAME = "sparsedrive_navsimv1_92p2.ckpt"
ASSET_NAMES = (CKPT_NAME, "path_1024.npy", "velocity_256.npy", "trajectory_1024_256.npz")

IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)
CAMS = ("cam_l0", "cam_f0", "cam_r0")  # SparseDriveV2Config.cams order
FINAL_HW = (256, 512)

_SAMPLE_DIR = (
    Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[1] / "third_party" / "tt-metal"))
    / "models"
    / "experimental"
    / "petr"
    / "resources"
    / "sample_input"
)
# nuScenes front cameras standing in for NAVSIM cam_l0 / cam_f0 / cam_r0 (same order as the config).
_SAMPLE_IMAGES = (
    "n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg",
    "n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
    "n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg",
)


def asset_dir() -> Path:
    return Path(os.environ.get("SDV2_ASSET_DIR", Path.home() / ".cache" / "sparsedrive_v2"))


def checkpoint_path() -> Optional[str]:
    path = Path(os.environ.get("SDV2_CHECKPOINT_PATH", asset_dir() / CKPT_NAME))
    return str(path) if path.exists() else None


def synthetic_camera_rig(final_hw=(256, 512), orig_hw=(1080, 1920)) -> torch.Tensor:
    """NAVSIM-like lidar2img for cam_l0 / cam_f0 / cam_r0, including the test-time resize+crop.

    Stand-in until real NAVSIM calibration is wired in: nuPlan-like intrinsics, cameras
    1.5 m above the z=0 plane, l0/r0 yawed +/-55 deg.
    """
    fH, fW = final_hw
    H, W = orig_hw
    resize = max(fH / H, fW / W)
    crop_h = int(H * resize) - fH  # bot_pct_lim = 0 -> keep the bottom rows
    crop_w = int(max(0, int(W * resize) - fW) / 2)
    img_aug = np.eye(4)
    img_aug[:2, :2] *= resize
    # Acts on homogeneous (u*d, v*d, d): the crop shift must scale with depth (column 2).
    img_aug[:2, 2] -= np.array([crop_w, crop_h])

    K = np.eye(4)
    K[:3, :3] = [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]]
    # camera axes (x right, y down, z forward) expressed in the lidar frame (x fwd, y left, z up)
    cam_to_lidar_rot = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    mats = []
    for yaw_deg, t in ((55.0, (1.5, 0.5, 1.5)), (0.0, (1.7, 0.0, 1.5)), (-55.0, (1.5, -0.5, 1.5))):
        yaw = np.deg2rad(yaw_deg)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = rz @ cam_to_lidar_rot
        cam2lidar[:3, 3] = t
        mats.append(img_aug @ K @ np.linalg.inv(cam2lidar))
    return torch.tensor(np.stack(mats), dtype=torch.float32)


def _load_images(final_hw=(256, 512)) -> torch.Tensor:
    from PIL import Image

    fH, fW = final_hw
    imgs = []
    for name in _SAMPLE_IMAGES:
        img = Image.open(_SAMPLE_DIR / name).convert("RGB")
        w, h = img.size
        resize = max(fH / h, fW / w)
        nw, nh = int(w * resize), int(h * resize)
        crop_w = int(max(0, nw - fW) / 2)
        img = img.resize((nw, nh)).crop((crop_w, nh - fH, crop_w + fW, nh))
        imgs.append(np.asarray(img, dtype=np.float32))
    imgs = (np.stack(imgs) - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
    return torch.tensor(imgs).permute(0, 3, 1, 2).contiguous()


def sample_inputs(batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """One NAVSIM-shaped frame: 3 real road images, a synthetic camera rig and an ego status."""
    imgs = _load_images()
    proj = synthetic_camera_rig()
    image_wh = torch.tensor([[512.0, 256.0]] * 3)
    # driving_command one-hot (left, straight, right, unknown) | velocity (m/s) | acceleration
    status = torch.tensor([0.0, 1.0, 0.0, 0.0, 6.0, 0.1, 0.3, 0.0])
    return {
        "imgs": imgs[None].repeat(batch_size, 1, 1, 1, 1),
        "projection_mat": proj[None].repeat(batch_size, 1, 1, 1),
        "image_wh": image_wh[None].repeat(batch_size, 1, 1),
        "status_feature": status[None].repeat(batch_size, 1),
    }


def lidar2img(intrinsics, sensor2lidar_rotation, sensor2lidar_translation) -> np.ndarray:
    """Upstream SparseDriveFeatureBuilder.get_camera_params, for the raw image plane."""
    lidar2cam_r = np.linalg.inv(np.asarray(sensor2lidar_rotation, dtype=np.float64))
    lidar2cam_t = np.asarray(sensor2lidar_translation, dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:3, :3] = intrinsics
    return viewpad @ lidar2cam_rt.T


def frame_inputs(images: Sequence[np.ndarray], lidar2imgs: Sequence[np.ndarray], status) -> Dict[str, torch.Tensor]:
    """Upstream test-mode pipeline: resize to 512 wide, keep the bottom 256 rows, normalize.

    images: HxWx3 uint8 RGB per camera in CAMS order; lidar2imgs: their raw-image lidar2img;
    status: driving command one-hot (4), velocity xy, acceleration xy.
    """
    from PIL import Image

    fh, fw = FINAL_HW
    imgs, mats = [], []
    for image, l2i in zip(images, lidar2imgs):
        h, w = image.shape[:2]
        resize = max(fh / h, fw / w)
        nw, nh = int(w * resize), int(h * resize)
        crop_w, crop_h = int(max(0, nw - fw) / 2), nh - fh
        img = Image.fromarray(image).resize((nw, nh)).crop((crop_w, crop_h, crop_w + fw, crop_h + fh))
        imgs.append(((np.asarray(img, dtype=np.float32) - IMG_MEAN) / IMG_STD).astype(np.float32))
        # Image-plane affine on homogeneous (u*d, v*d, d): its translation multiplies depth.
        aug = np.eye(4)
        aug[:2, :2] *= resize
        aug[:2, 2] -= (crop_w, crop_h)
        mats.append(aug @ l2i)
    return {
        "imgs": torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()[None],
        "projection_mat": torch.tensor(np.stack(mats), dtype=torch.float32)[None],
        "image_wh": torch.tensor([[fw, fh]] * len(imgs), dtype=torch.float32)[None],
        "status_feature": torch.tensor(np.asarray(status, dtype=np.float32))[None],
    }
