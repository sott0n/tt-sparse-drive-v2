# SparseDriveV2 (TTNN)

TTNN port of [SparseDriveV2](https://github.com/swc-17/SparseDriveV2) (ECCV 2026), the
scoring-based end-to-end planner for NAVSIM. This port targets the released **NAVSIM v1**
checkpoint (`sparsedrive_navsimv1_92p2.ckpt`, 92.2 PDMS) at batch size 1.

**Status:** functional bring-up. The whole forward pass runs on device and matches the
reference; performance work is in progress (see [Performance](#performance)).

## Model

| Stage | What it does | Shapes |
|---|---|---|
| Backbone | timm ResNet-34 (stages 1-4) + FPN, 3 cameras | 3 x 3x256x512 -> 4 levels x 256 ch |
| Vocabulary | 1024 paths (50 pts) x 256 velocity profiles (8 steps) = 262,144 trajectories | constant |
| Decoder L0 | path queries: deformable aggregation (500 key points each) + self-attn + FFN -> top-128<br>velocity queries: image cross-attn + self-attn + FFN -> top-64 | 1024 / 256 queries |
| Decoder L1 | same on the survivors -> top-20 paths x top-20 velocities -> 400 composed trajectories -> deformable aggregation (80 pts) + self-attn + FFN -> 6 metric heads | 128 / 64 / 400 queries |
| Output | `sigmoid(NC)*sigmoid(DAC)*(5 TTC + 5 EP + 2 C)` -> argmax | trajectory 8 x (x, y, yaw) |

Files:

- `reference/sparsedrive_v2.py` — inference-only PyTorch reference. Upstream's CUDA
  `deformable_aggregation` is replaced by a torch implementation of the same kernel.
- `tt/ttnn_backbone.py` — ResNet-34 + FPN.
- `tt/ttnn_dfa.py` — deformable feature aggregation (projection, masked softmax weights,
  `ttnn.grid_sample`, batched-matmul reduction).
- `tt/ttnn_sparsedrive_v2.py` — decoder layers, on-device top-k / gather, trajectory composition.
- `common.py` — asset paths and the sample input.

## Setup

Assets are read from `$SDV2_ASSET_DIR` (default `~/.cache/sparsedrive_v2`):

```bash
mkdir -p ~/.cache/sparsedrive_v2 && cd ~/.cache/sparsedrive_v2
for f in sparsedrive_navsimv1_92p2.ckpt path_1024.npy velocity_256.npy trajectory_1024_256.npz; do
  curl -L -O https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/$f
done
```

The model reads only the checkpoint; the vocabulary is stored in it as buffers.
The `.npy`/`.npz` files are the upstream source of those buffers.

**Sample input.** NAVSIM data is not bundled. `common.sample_inputs()` uses three real
nuScenes front-camera images (from `models/experimental/petr/resources`) standing in for
`cam_l0 / cam_f0 / cam_r0`, with a synthetic NAVSIM-like camera rig. It exercises every
code path with realistic image statistics, but it is a single frame, not an accuracy
benchmark.

## Tests

```bash
pytest models/experimental/sparsedrive_v2/tests/pcc/test_backbone.py      # ResNet-34 + FPN
pytest models/experimental/sparsedrive_v2/tests/pcc/test_dfa.py           # deformable aggregation
pytest models/experimental/sparsedrive_v2/tests/pcc/test_sparsedrive_v2.py  # end to end
```

The decoder narrows its candidates with top-k twice. A tiny score difference near the k-th
place changes which candidates survive, and every tensor after that point can no longer be
compared element-wise. The end-to-end test therefore runs twice:

- **teacher-forced**: the reference's top-k selections are injected, and every score
  tensor is compared with PCC;
- **free-running**: the device makes its own selections, and the test reports the overlap
  of the selected sets and the distance between the chosen trajectories.

Results on the sample frame (Blackhole p300c, single chip):

| Check | Result |
|---|---|
| FPN levels 0-3 | PCC 0.9988 - 0.9996 |
| Path DFA, 1024 anchors (aggregated part) | PCC 0.99998 |
| Teacher-forced scores / 6 metric logits | PCC >= 0.997 (path L1), >= 0.9995 (others) |
| Free-running selection overlap (vocabulary ids) | paths 0.99 / 0.95, velocities 0.98 / 1.0 |
| Chosen trajectory | same as reference, or the neighbouring candidate (the top scores differ by < 0.2 %) |

The last row is the reason to evaluate on NAVSIM (PDMS over navtest) rather than on one
frame. On this frame the reference's top five final scores lie within 0.15 of 10.99.

`SDV2_ACT_DTYPE=bf16` switches the decoder activations from fp32 to bf16. On this frame
the two settings give the same PCC and the same latency.

## Performance

Eager forward on Blackhole p300c (one chip), batch 1, warm program cache:

| Part | Latency |
|---|---|
| Backbone + FPN | 26 ms |
| Decoder layer 0 | 255 ms (path DFA: 252 ms) |
| Decoder layer 1 | 63 ms |
| **Total** | **~344 ms** |

Tracing the decoder (it is trace-safe) gives the same time: the forward is device-bound.
Almost all of it is the layer-0 path DFA, which takes 512k key points x 3 cameras x
4 levels of bilinear samples. Of that time, `grid_sample` is ~90 ms and the batched-matmul
reduction ~80 ms. The rest is mostly layout conversions of the sampled tensor.

The next step is a fused deformable-aggregation op. It would sample, weight and
accumulate in one kernel without materializing the samples, and it would skip cameras a
point does not project into. That is the same strategy as upstream's CUDA kernel.
