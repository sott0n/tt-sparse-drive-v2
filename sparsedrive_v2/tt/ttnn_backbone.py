# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN ResNet-34 (timm, features_only stages 1-4) + torchvision-style FPN.

All activations stay on device in the conv2d-native flattened NHWC layout
``(1, 1, N*H*W, C)``.  For the 3 cameras of one frame N = 3, so every level's
rows are ordered (cam, h, w) — exactly the ``(cam, feat, C)`` layout the
deformable aggregation and the velocity cross-attention consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import ttnn

_RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale[:, None, None, None]
    bias = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    return weight.detach().float(), bias.detach().float()


@dataclass
class FeatureMap:
    """A device feature map in conv2d layout plus its logical NHWC shape."""

    tensor: ttnn.Tensor
    n: int
    h: int
    w: int
    c: int


class TtConv2d:
    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int = 1,
        padding: int = 0,
        activation=None,
    ):
        self.device = device
        self.keep_sharded = False  # True: return conv2d's L1-sharded output as is
        self.out_channels, self.in_channels, kh, kw = weight.shape
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
        self.bias = None if bias is None else ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=activation,
            deallocate_activation=False,
            reshard_if_not_optimal=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

    def __call__(self, x: FeatureMap) -> FeatureMap:
        if x.tensor.is_sharded():
            # Without an explicit layout, conv2d treats a sharded input's scheme as interleaved.
            self.conv_config.shard_layout = x.tensor.memory_config().memory_layout
            self.conv_config.reshard_if_not_optimal = False
        out, (oh, ow), (self.weight, self.bias) = ttnn.conv2d(
            input_tensor=x.tensor,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=x.n,
            input_height=x.h,
            input_width=x.w,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        if out.is_sharded() and not self.keep_sharded:
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return FeatureMap(out, x.n, oh, ow, self.out_channels)


def _to_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    return x if x.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(x, ttnn.TILE_LAYOUT)


class TtBasicBlock:
    def __init__(self, device, block):
        w1, b1 = fold_bn(block.conv1, block.bn1)
        w2, b2 = fold_bn(block.conv2, block.bn2)
        stride = block.conv1.stride[0]
        self.conv1 = TtConv2d(device, w1, b1, stride=stride, padding=1, activation=_RELU)
        self.conv2 = TtConv2d(device, w2, b2, stride=1, padding=1)
        self.downsample = None
        if block.downsample is not None:
            wd, bd = fold_bn(block.downsample[0], block.downsample[1])
            self.downsample = TtConv2d(device, wd, bd, stride=stride, padding=0)

    def __call__(self, x: FeatureMap) -> FeatureMap:
        out = self.conv2(self.conv1(x))
        identity = self.downsample(x) if self.downsample is not None else x
        y = ttnn.relu(ttnn.add(_to_tile(out.tensor), _to_tile(identity.tensor)))
        return FeatureMap(y, out.n, out.h, out.w, out.c)


class TtResNet34:
    """timm ResNet-34 with ``features_only=True, out_indices=(1, 2, 3, 4)``."""

    def __init__(self, device, ref, sharded_stages: int = 2):
        self.device = device
        w, b = fold_bn(ref.conv1, ref.bn1)
        self.stem = TtConv2d(device, w, b, stride=2, padding=3, activation=_RELU)
        self.layers = [[TtBasicBlock(device, blk) for blk in getattr(ref, f"layer{i}")] for i in range(1, 5)]
        # The first stages hand activations from conv to conv in L1 shards; deeper stages
        # go through DRAM so conv2d can pick its own (block) sharding.
        self.sharded_stages = sharded_stages
        for layer in self.layers[:sharded_stages]:
            for blk in layer:
                for conv in (blk.conv1, blk.conv2, blk.downsample):
                    if conv is not None:
                        conv.keep_sharded = True

    def host_input(self, imgs: torch.Tensor) -> ttnn.Tensor:
        """(N, 3, H, W) float torch -> host bf16 tensor, still NCHW with 3 channels.

        The NHWC permute runs on device (``stem_input``); a host permute is slower than the
        upload it would save.
        """
        return ttnn.from_torch(imgs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def stem_input(self, imgs: ttnn.Tensor) -> FeatureMap:
        """Device (N, 3, H, W) -> stem-conv input (1, 1, N*H*W, 3) in NHWC order."""
        n, c, h, w = imgs.shape
        # In L1 the stem conv shards the 6-byte rows ~3x faster than from DRAM.
        x = ttnn.permute(imgs, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 1, n * h * w, c))
        return FeatureMap(x, n, h, w, c)

    def preprocess(self, imgs: torch.Tensor) -> FeatureMap:
        """(N, 3, H, W) float torch -> device FeatureMap in conv2d layout."""
        return self.stem_input(ttnn.to_device(self.host_input(imgs), self.device))

    def __call__(self, x: FeatureMap) -> List[FeatureMap]:
        x = self.stem(x)
        t = ttnn.max_pool2d(
            x.tensor,
            batch_size=x.n,
            input_h=x.h,
            input_w=x.w,
            channels=x.c,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )
        if t.is_sharded() and self.sharded_stages == 0:
            t = ttnn.sharded_to_interleaved(t, ttnn.DRAM_MEMORY_CONFIG)
        x = FeatureMap(t, x.n, (x.h + 1) // 2, (x.w + 1) // 2, x.c)
        feats = []
        for i, layer in enumerate(self.layers):
            for blk in layer:
                x = blk(x)
            if x.tensor.is_sharded():
                dram = FeatureMap(ttnn.sharded_to_interleaved(x.tensor, ttnn.DRAM_MEMORY_CONFIG), x.n, x.h, x.w, x.c)
                feats.append(dram)
                x = x if i + 1 < self.sharded_stages else dram
            else:
                feats.append(x)
        return feats


class TtFPN:
    """torchvision FeaturePyramidNetwork (no extra block): lateral 1x1, nearest 2x top-down, 3x3 out."""

    def __init__(self, device, ref):
        self.device = device

        def conv(m, padding):
            return TtConv2d(device, m.weight.detach().float(), m.bias.detach().float(), padding=padding)

        self.inner = [conv(blk[0], 0) for blk in ref.inner_blocks]
        self.layer = [conv(blk[0], 1) for blk in ref.layer_blocks]

    def _upsample2x(self, x: FeatureMap) -> ttnn.Tensor:
        t = ttnn.to_layout(x.tensor, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.reshape(t, (x.n, x.h, x.w, x.c))
        t = ttnn.upsample(t, 2, mode="nearest")
        return ttnn.reshape(t, (1, 1, x.n * x.h * x.w * 4, x.c))

    def __call__(self, feats: List[FeatureMap]) -> List[FeatureMap]:
        last_inner = self.inner[-1](feats[-1])
        results = [self.layer[-1](last_inner)]
        for idx in range(len(feats) - 2, -1, -1):
            lateral = self.inner[idx](feats[idx])
            top_down = self._upsample2x(last_inner)
            t = ttnn.add(_to_tile(lateral.tensor), _to_tile(top_down))
            last_inner = FeatureMap(t, lateral.n, lateral.h, lateral.w, lateral.c)
            results.insert(0, self.layer[idx](last_inner))
        return results


class TtSparseBackbone:
    def __init__(self, device, ref_backbone):
        self.resnet = TtResNet34(device, ref_backbone.img_backbone)
        self.fpn = TtFPN(device, ref_backbone.img_neck)

    def __call__(self, imgs: torch.Tensor) -> List[FeatureMap]:
        """imgs: (num_cams, 3, H, W) torch -> 4 FPN levels (stride 4..32), 256 ch each."""
        return self.fpn(self.resnet(self.resnet.preprocess(imgs)))


def feature_map_to_torch(fm: FeatureMap) -> torch.Tensor:
    """FeatureMap -> (N, C, H, W) float torch."""
    t = ttnn.to_torch(fm.tensor).float().reshape(-1, fm.c)[: fm.n * fm.h * fm.w]
    return t.reshape(fm.n, fm.h, fm.w, fm.c).permute(0, 3, 1, 2)
