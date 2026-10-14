#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stage SparseDriveV2 as a tt-model v6 thin bundle (kind tt-dit-server) from this checkout.
#
#   sparsedrive_v2/scripts/package_tt_model.sh <out-dir> [org/name to push] [more package-thin args]
#
# Needs a built tt-metal at the pinned commit (third_party/tt-metal, or TT_METAL_SRC), uv, and
# the tt-model CLI. The bundle carries a ttnn wheel of that tt-metal (no index has it) and a
# sparsedrive-v2 wheel. It carries no weights: the server fetches the checkpoint from the
# upstream HF repo on first start, or reads SDV2_CHECKPOINT_PATH.
set -euo pipefail

out=$(realpath -m "${1:?usage: $0 <out-dir> [package-thin args]}")
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
tt_metal=$(realpath "${TT_METAL_SRC:-$repo/third_party/tt-metal}")
wheels=$(mktemp -d)
trap 'rm -rf "$wheels" "$repo/build" "$repo/sparsedrive_v2.egg-info"' EXIT

# ttnn from the existing build tree; setup.py packages it without rebuilding.
TT_FROM_PRECOMPILED_DIR="$tt_metal" uv build --quiet --wheel --python 3.10 --out-dir "$wheels" "$tt_metal"
uv build --quiet --wheel --out-dir "$wheels" "$repo"

ttnn_whl=$(ls "$wheels"/ttnn-*.whl)
sdv2_whl=$(ls "$wheels"/sparsedrive_v2-*.whl)
ttnn_ver=$(basename "$ttnn_whl" | cut -d- -f2)
sdv2_ver=$(basename "$sdv2_whl" | cut -d- -f2)

# Versions validated together; torch must be the CPU build (tt-model adds its index).
cat > "$wheels/requirements.txt" <<REQ
ttnn==$ttnn_ver
sparsedrive-v2==$sdv2_ver
torch==2.11.0+cpu
torchvision==0.26.0+cpu
timm==1.0.30
huggingface_hub==1.33.0
numpy==1.26.4
pillow==12.3.0
fastapi==0.142.2
pydantic==2.9.2
uvicorn==0.54.0
REQ

tt-model package-thin \
    --model-py "$repo/sparsedrive_v2/server/app.py" \
    --kind tt-dit-server --app sparsedrive_v2.server.app:app \
    --requirements "$wheels/requirements.txt" \
    --models-wheel "$ttnn_whl" --models-wheel "$sdv2_whl" \
    --weights wenchaosun/SparseDriveV2 --weights-revision 42c15531aa6cb18c7360dd0377cfc1c221280597 \
    --arch blackhole --mesh P150 --device-count 1 --python 3.10 \
    --name sparsedrive-v2 --out "$out" "$@"
