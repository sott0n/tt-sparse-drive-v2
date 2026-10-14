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
- `common.py` — asset paths, the NAVSIM camera preprocessing and the sample input.
- `server/app.py` — HTTP server, served by tt-model (see [Serving with tt-model](#serving-with-tt-model)).
- `scripts/navsim_pdm_eval.py` — NAVSIM navtest inference and PDM scoring.
- `scripts/package_tt_model.sh` — stage the tt-model bundle.

## Setup

```bash
git clone --recursive git@github.com:sott0n/tt-sparse-drive-v2.git
cd tt-sparse-drive-v2/third_party/tt-metal
./build_metal.sh && ./create_venv.sh     # see tt-metal docs
cd ../..
source third_party/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$PWD/third_party/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME
```

On one chip of a 2-chip p300 board, pick the chip and give it a 1x1 mesh:

```bash
export TT_VISIBLE_DEVICES=0   # /dev/tenstorrent/0, seen as device 0 in-process
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

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
nuScenes front-camera images (tt-metal's `models/experimental/petr/resources`) standing in for
`cam_l0 / cam_f0 / cam_r0`, with a synthetic NAVSIM-like camera rig. It exercises every
code path with realistic image statistics, but it is a single frame, not an accuracy
benchmark.

## Tests

```bash
pytest sparsedrive_v2/tests/pcc/test_backbone.py      # ResNet-34 + FPN
pytest sparsedrive_v2/tests/pcc/test_dfa_fused.py     # fused aggregation kernel
pytest sparsedrive_v2/tests/pcc/test_dfa.py           # deformable aggregation
pytest sparsedrive_v2/tests/pcc/test_sparsedrive_v2.py  # end to end
pytest sparsedrive_v2/tests/perf/test_perf.py         # eager vs traced latency
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
| Path DFA, 1024 anchors (aggregated part) | PCC 0.99999 |
| Teacher-forced scores / 6 metric logits | PCC >= 0.999 |
| Free-running selection overlap (vocabulary ids) | paths 0.98 / 1.0, velocities 0.98 / 1.0 |
| Chosen trajectory | same as reference (best score 10.4731 vs 10.4722) |

The last row is the reason to evaluate on NAVSIM (PDMS over navtest) rather than on one
frame. On this frame the reference's top five final scores lie within 0.02 of 10.47, so a
tiny score error can pick a neighbouring candidate.

`SDV2_ACT_DTYPE=bf16` switches the decoder activations from fp32 to bf16. On this frame
the two settings give the same PCC and the same latency.

## Serving with tt-model

`sparsedrive_v2/server/app.py` is an HTTP server for the model.
[tt-model](https://github.com/tenstorrent/tt-model-manager) serves it as a v6 thin bundle with
`kind: tt-dit-server` (uvicorn, no vLLM).

```
POST /v1/plan    {"cameras": [{"image": <base64 PNG/JPEG, RGB>, "intrinsics": 3x3,
                               "sensor2lidar_rotation": 3x3, "sensor2lidar_translation": 3}
                              x 3 in cam_l0, cam_f0, cam_r0 order],
                  "status": [command one-hot (4), velocity xy, acceleration xy]}
                 -> {"trajectory": [[x, y, heading] x 8], "timing_ms": {...}}
GET  /v1/health, GET /v1/models
```

The camera fields are NAVSIM's (`agent_input.cameras[-1].cam_*`). Images may be any size;
the server resizes and crops them as upstream inference does.

```bash
TT_VISIBLE_DEVICES=0 tt-model serve kyamaguchiTT/tt-sparsedrive-v2-p150x1 --port 8000
```

- The bundle carries no weights. The first start downloads `sparsedrive_navsimv1_92p2.ckpt`
  (536 MB) from [wenchaosun/SparseDriveV2](https://huggingface.co/wenchaosun/SparseDriveV2)
  at a pinned revision. `SDV2_CHECKPOINT_PATH=/path/to/ckpt` uses a local copy instead.
- `--port` picks the port (default 20000).
- The host needs SFPI 7.84.0 in `/opt/tenstorrent/sfpi`. ttnn wheels do not bundle SFPI.
- The first start compiles kernels, which takes ~14 min. Later starts reuse the cache.
- Ready when uvicorn prints "Application startup complete": the trace is captured at startup.

To build the bundle from this checkout (needs a built tt-metal, `uv` and `tt-model`):

```bash
sparsedrive_v2/scripts/package_tt_model.sh <out-dir>                       # stage only
sparsedrive_v2/scripts/package_tt_model.sh <out-dir> <org>/sparsedrive-v2  # stage and push
```

It bundles a ttnn wheel of the pinned tt-metal, since no index has one, and a `sparsedrive-v2`
wheel with the model and its kernels. `TT_METAL_SRC` points it at another built tt-metal
checkout of the same commit.

## NAVSIM evaluation

`scripts/navsim_pdm_eval.py` runs navtest through the model and scores the trajectories
with the NAVSIM v1 PDM scorer. It needs the navsim and nuplan devkits, the navtest
logs and sensor blobs, and a v1 metric cache (see the script's docstring).

```bash
python sparsedrive_v2/scripts/navsim_pdm_eval.py all --backends ttnn \
    --worker single_machine_thread_pool --out <dir>
```

navtest, traced, fp32 activations (Blackhole p300c, single chip):

| | Scenes | PDMS | NC | DAC | TTC | Comfort | EP |
|---|---|---|---|---|---|---|---|
| Upstream (paper) | | 92.2 | | | | | |
| TTNN | 12146 / 12146 | 92.22 | 98.70 | 98.46 | 95.27 | 99.98 | 88.65 |

## Performance

Blackhole p300c (one chip), batch 1, warm program cache
(`tests/perf/test_perf.py`):

| Mode | Throughput (frames pipelined) | Single-frame latency |
|---|---|---|
| eager | 23 ms (42.8 FPS) | 29 ms |
| traced (`TracedSparseDriveV2`) | 23 ms (43.6 FPS) | 25 ms |

The forward is device-bound, so tracing barely changes the time. The decoder
dispatches far faster than the device runs it. Images are uploaded as
3-channel NCHW bf16; the NHWC permute runs on device and the stem conv takes the
3 channels as they are.

Device kernel time per stage, from the device profiler
(`tests/perf/test_profile_forward.py`, warm iteration):

| Stage | First port | Matmul/layout tuning | Fixed-point grid_sample | Fused aggregation | Fused tuning | Two readers | Stem input | Mask in matmul | Fused exp | Sharded ResNet | Camera reductions | Head split | L1 features | Stem in L1, bf16 logits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Backbone + FPN | 5 ms | 7 ms | 7 ms | 7 ms | 7 ms | 7 ms | 5 ms | 5 ms | 5 ms | 5 ms | 5 ms | 5 ms | 5 ms | 4 ms |
| Decoder layer 0 | 254 ms | 185 ms | 120 ms | 56 ms | 36 ms | 29 ms | 30 ms | 25 ms | 23 ms | 23 ms | 22 ms | 21 ms | 14 ms | 13 ms |
| Decoder layer 1 | 62 ms | 43 ms | 29 ms | 14 ms | 10 ms | 9 ms | 9 ms | 8 ms | 7 ms | 7 ms | 7 ms | 7 ms | 6 ms | 5 ms |
| **Total** | **321 ms** | **235 ms** | **156 ms** | **77 ms** | **53 ms** | **45 ms** | **44 ms** | **38 ms** | **35 ms** | **35 ms** | **34 ms** | **33 ms** | **25 ms** | **23 ms** |

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
- **Stem input** — the stem conv reads the 3-channel image directly, without padding it to
  16 channels or slicing it from DRAM. The padding alone was 1.65 ms (6-byte rows); the
  backbone drops from 7.4 to 5.2 ms at the same PCC.
- **Mask in matmul** — the visibility mask enters the softmax-weight matmul as one extra
  input per key point (0 or -3e4) through a 0/1 expansion block in the weights, instead
  of a `repeat_interleave` to (cams, n, Pp*L*G) and a multiply. Device time 43.6 -> 38.5 ms;
  navtest PDMS 92.23 -> 92.22.
- **Fused exp** — `subtract(logits, shift)` applies `exp` in the same pass and writes the
  bf16 weights the fused kernel reads, removing an `exp` pass and a typecast over
  (cams, n, Pp*L*G); the denominator sums the bf16 weights. Device time 38.5 -> 35.4 ms.
- **Sharded ResNet** — in ResNet stages 1-2 the activations stay height-sharded in L1
  from conv to conv instead of going through DRAM; stages 3-4 keep conv2d's own block
  sharding, which height shards cannot fit. Backbone 5.2 -> 4.7 ms, identical outputs.
- **Camera reductions** — sums and maxima over the 3 cameras (dim 0) are slices plus
  2 binary ops; ttnn's dim-0 reduce transposed the tensor twice (0.8 ms for the layer-0
  visibility count alone). Device time 34.6 -> 33.6 ms, identical outputs.
- **Head split** — attention splits and merges heads with
  `ttnn.transformer.split_query_key_value_and_split_heads` / `concatenate_heads` and
  multiplies by K with `transpose_b`, instead of reshape + permute + transpose. The fused
  split is used only for tile-multiple lengths: it leaves K's tile padding non-zero, which
  the softmax over 400 trajectories would see. Device time 33.6 -> 32.9 ms, identical outputs.
- **L1 features** — the four FPN levels the fused aggregation samples sit in L1
  (interleaved) instead of DRAM. The three fused DFA calls drop from 15.4 to 7.1 ms. Their
  output is written by BRISC on NOC0: written by NCRISC on NOC1, next to its reads, the op
  hangs the chip. Device time 32.9 -> 24.8 ms.
- **Stem in L1, bf16 logits** — the NHWC permute writes the image to L1, where the stem
  conv shards its 6-byte rows ~3x faster than from DRAM. `weights_fc` outputs the softmax
  logits in bf16, the dtype the fused exp writes anyway. Device time 24.8 -> 22.6 ms.

To profile:

```bash
python -m tracy -p -r -v -m pytest sparsedrive_v2/tests/perf/test_profile_forward.py
tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
    --start-signpost iter1_start --end-signpost iter1_end
```
