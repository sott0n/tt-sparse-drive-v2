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
- `tt/ttnn_dfa.py` — deformable feature aggregation: projection, masked softmax weights,
  then the fused aggregation below.
- `tt/ttnn_dfa_fused.py` + `tt/kernels/{reader,compute}_dfa.cpp` — fused sample +
  weight + reduce in one `ttnn.generic_op`; samples never go to DRAM.
- `tt/ttnn_sparsedrive_v2.py` — decoder layers, on-device top-k / gather, trajectory composition.
- `common.py` — asset paths and the sample input.
- `scripts/navsim_pdm_eval.py` — NAVSIM navtest inference and PDM scoring.

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
pytest models/experimental/sparsedrive_v2/tests/pcc/test_dfa_fused.py     # fused aggregation kernel
pytest models/experimental/sparsedrive_v2/tests/pcc/test_dfa.py           # deformable aggregation
pytest models/experimental/sparsedrive_v2/tests/pcc/test_sparsedrive_v2.py  # end to end
pytest models/experimental/sparsedrive_v2/tests/perf/test_perf.py         # eager vs traced latency
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

## NAVSIM evaluation

`scripts/navsim_pdm_eval.py` runs navtest through the model and scores the trajectories
with the NAVSIM v1 PDM scorer. It needs the navsim and nuplan devkits, the navtest
logs and sensor blobs, and a v1 metric cache (see the script's docstring).

```bash
python models/experimental/sparsedrive_v2/scripts/navsim_pdm_eval.py all --backends ttnn \
    --worker single_machine_thread_pool --out <dir>
```

navtest, traced, fp32 activations (Blackhole p300c, single chip):

| | Scenes | PDMS | NC | DAC | TTC | Comfort | EP |
|---|---|---|---|---|---|---|---|
| Upstream (paper) | | 92.2 | | | | | |
| TTNN | 12146 / 12146 | 92.23 | 98.70 | 98.47 | 95.26 | 99.98 | 88.65 |

## Performance

Blackhole p300c (one chip), batch 1, warm program cache
(`tests/perf/test_perf.py`):

| Mode | Throughput (frames pipelined) | Single-frame latency |
|---|---|---|
| eager | 46 ms (22.0 FPS) | 49 ms |
| traced (`TracedSparseDriveV2`) | 45 ms (22.0 FPS) | 47 ms |

The forward is device-bound, so tracing barely changes the time. The decoder
dispatches far faster than the device runs it. The input pipeline costs ~4 ms:
images are uploaded as 3-channel NCHW bf16, and the NHWC permute plus the channel
padding for the stem conv run on device.

Device kernel time per stage, from the device profiler
(`tests/perf/test_profile_forward.py`, warm iteration):

| Stage | First port | Matmul/layout tuning | Fixed-point grid_sample | Fused aggregation | Fused tuning | Two readers |
|---|---|---|---|---|---|---|
| Backbone + FPN | 5 ms | 7 ms | 7 ms | 7 ms | 7 ms | 7 ms |
| Decoder layer 0 | 254 ms | 185 ms | 120 ms | 56 ms | 36 ms | 29 ms |
| Decoder layer 1 | 62 ms | 43 ms | 29 ms | 14 ms | 10 ms | 9 ms |
| **Total** | **321 ms** | **235 ms** | **156 ms** | **77 ms** | **53 ms** | **45 ms** |

What changed, in order:

- **DFA reduction matmul** — an explicit program config. The auto config ran it on 8 of
  110 cores; the change cut it from ~5 ms to ~1.3 ms per call.
- **Grid build** — the sampling grid is built with concat + transpose instead of `stack`.
  Dropping `grid_sample`'s grid batching was measured as a wash: it removed a 17 ms
  reshape copy, but plain `grid_sample` was ~17 ms slower.
- **Fixed-point grid_sample** — 128 ms -> 49 ms in total.
  Accumulating profiler zones placed inside `ttnn.grid_sample`'s kernels showed the op is
  reader-bound. ~60% of the reader's time went to the per-point sampling geometry. The
  data-movement cores have no FPU, so every float op compiles to a software routine
  (`__mulsf3`, `__addsf3`, ... in the ELF). The replacement reader does the same geometry
  in Q16.16 integers, as ttnn's `multi_scale_deformable_attn` reader does (#55133), and
  fetches grid sticks 16 at a time behind one barrier. Compute and writer are ttnn's own
  kernels, unchanged. On the layer-0 shape it runs 2.5x faster at the same PCC against
  torch (0.999993).

- **Fused aggregation** (`tt/ttnn_dfa_fused.py`) — sampling, weighting and the reduction
  run in one op, which removes the sampled tensor, its tilize and the reduction matmuls.
  Per anchor, the reader keeps only bilinear corners inside the feature map. Samples whose
  projection misses every image are not read: that is 69% of them at layer 0. Every 32
  kept samples become one tile. Compute forms `A = softmax weight (group x sample) *
  bilinear weight` and accumulates `A @ V` per 32-channel group in fp32 L1. The layer-0
  PCC against a torch reference of the same sum is 0.999996.
- **Fused tuning** — anchors go to cores strided (`i, i + cores, ...`) in one call per DFA.
  Visible-sample counts vary per anchor, and contiguous 2-3-anchor blocks left one core
  at 1.7x the median. A sample's 8 group weights are contiguous in `ex` and in the E tile,
  so the reader copies four words per sample instead of eight strided half-words. The
  sampling grid is passed as `[x row | y row]`, built with one concat + untilize instead of
  interleaving `(x, y)` per point.
- **Two readers** — NCRISC and BRISC each sample every other key point of an anchor, and
  compute takes their tiles in turn. The reader was the bound; giving each reader whole
  anchors made them wait on each other through compute's anchor order. The layer-0 path
  DFA is 12.6 ms; each reader spends ~30% of it waiting on NoC reads.

To profile:

```bash
python -m tracy -p -r -v -m pytest models/experimental/sparsedrive_v2/tests/perf/test_profile_forward.py
tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
    --start-signpost iter1_start --end-signpost iter1_end
```
