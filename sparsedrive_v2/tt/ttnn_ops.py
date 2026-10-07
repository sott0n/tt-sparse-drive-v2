# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Small TTNN building blocks (linear, layer norm, MLP, multi-head attention)."""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn

import ttnn

# Activation dtype of the decoder streams.  The decoder narrows candidates with top-k on
# scores that sit close together, so its activations are kept in fp32 by default.
ACT_DTYPE = ttnn.float32 if os.environ.get("SDV2_ACT_DTYPE", "fp32") == "fp32" else ttnn.bfloat16


def compute_config(device, fidelity=ttnn.MathFidelity.HiFi4, approx=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device)


class TtLinear:
    """y = x @ W^T + b with W in torch (out, in) convention."""

    def __init__(self, device, weight: torch.Tensor, bias: Optional[torch.Tensor], weight_dtype=ttnn.bfloat16):
        self.device = device
        self.out_features = weight.shape[0]
        self.weight = to_device(weight.detach().float().t(), device, weight_dtype)
        self.bias = None if bias is None else to_device(bias.detach().float().reshape(1, -1), device, weight_dtype)
        self.cfg = compute_config(device)

    @classmethod
    def from_torch(cls, device, lin: nn.Linear, **kw) -> "TtLinear":
        return cls(device, lin.weight, lin.bias, **kw)

    def __call__(self, x, activation: Optional[str] = None, dtype=None, memory_config=None):
        return ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            activation=activation,
            dtype=dtype or ACT_DTYPE,
            compute_kernel_config=self.cfg,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )


class TtLayerNorm:
    def __init__(self, device, ln: nn.LayerNorm):
        self.weight = to_device(ln.weight.detach().float().reshape(1, -1), device)
        self.bias = to_device(ln.bias.detach().float().reshape(1, -1), device)
        self.eps = ln.eps
        self.cfg = compute_config(device)

    def __call__(self, x):
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, compute_kernel_config=self.cfg)


class TtMLP:
    """Linear -> ReLU -> Linear (upstream ``nn.Sequential(Linear, ReLU, Linear)``)."""

    def __init__(self, device, seq: nn.Sequential):
        self.fc1 = TtLinear.from_torch(device, seq[0])
        self.fc2 = TtLinear.from_torch(device, seq[2])

    def __call__(self, x, dtype=None):
        return self.fc2(self.fc1(x, activation="relu"), dtype=dtype)


class TtMultiheadAttention:
    """``nn.MultiheadAttention(batch_first=True)`` forward (no masks, eval mode)."""

    def __init__(self, device, mha: nn.MultiheadAttention):
        d = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = d // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        w, b = mha.in_proj_weight.detach(), mha.in_proj_bias.detach()
        self.qkv = TtLinear(device, w, b)
        self.q = TtLinear(device, w[:d], b[:d])
        self.kv = TtLinear(device, w[d:], b[d:])
        self.out_proj = TtLinear.from_torch(device, mha.out_proj)
        self.cfg = compute_config(device)

    def _heads(self, x, n):
        x = ttnn.reshape(x, (1, n, self.num_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def __call__(self, query, key_value=None):
        """query: (1, Nq, d); key_value: (1, Nk, d) or None for self-attention."""
        d = self.num_heads * self.head_dim
        nq = query.shape[1]
        if key_value is None and nq % ttnn.TILE_SIZE == 0:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                self.qkv(query), num_heads=self.num_heads, transpose_key=False
            )
        elif key_value is None:
            # The fused split leaves the key's tile padding non-zero, which the softmax sees.
            qkv = self.qkv(query)
            q, k, v = (self._heads(qkv[:, :, i * d : (i + 1) * d], nq) for i in range(3))
        else:
            # The fused split needs equal query and key lengths.
            kv = self.kv(key_value)
            nk = key_value.shape[1]
            q = self._heads(self.q(query), nq)
            k, v = self._heads(kv[:, :, :d], nk), self._heads(kv[:, :, d:], nk)
        attn = ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=self.cfg)
        attn = ttnn.softmax(
            ttnn.multiply(attn, self.scale), dim=-1, numeric_stable=True, compute_kernel_config=self.cfg
        )
        out = ttnn.matmul(attn, v, compute_kernel_config=self.cfg)
        return self.out_proj(ttnn.transformer.concatenate_heads(out))
