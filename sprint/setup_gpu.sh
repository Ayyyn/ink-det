#!/bin/bash
# setup_gpu.sh - build the PHerc0343 ink environment on a fresh Linux NVIDIA box.
#
#   bash sprint/setup_gpu.sh [ROOT]            (default /data)
#
# Installs, in $ROOT: uv + CPython 3.14, villa at the commit Erwin Nieuwlaar's winning
# PHerc0343 maps were computed with (0fb38c45), his reproduction repo, the four public
# ink checkpoints (sha256-verified), and the N_LAYERS patch that lets his ensemble step
# read 31-layer renders.
#
# No VC3D and no spiral fit here: this path runs ink inference on surface volumes that
# are ALREADY published, so geometry is skipped entirely. Add VC3D only when moving to
# PHerc0268/PHerc1218, which have no published segments.
set -euo pipefail
ROOT=${1:-/data}
VILLA_COMMIT=0fb38c45b488e192566ea83ae36369cb6db92652
TORCH_BACKEND=${TORCH_BACKEND:-cu130}

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible"; exit 1; }

echo "== apt"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends curl ca-certificates git build-essential cmake ninja-build \
    pkg-config libgomp1 zstd jq python3 libgl1 libglib2.0-0 unzip

echo "== aws cli"
# The chain's dataset phase pulls the lasagna group with `aws s3 sync --no-sign-request`.
# Ubuntu's awscli package is v1 and drags in a second Python; take v2's static bundle.
command -v aws >/dev/null || {
    curl -LsS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update >/dev/null
    rm -rf /tmp/awscliv2.zip /tmp/aws
}
aws --version

echo "== uv + python 3.14 (vesuvius pins >=3.14,<3.15)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv python install 3.14

mkdir -p "$ROOT"
echo "== villa @ $VILLA_COMMIT"
if [[ ! -d $ROOT/villa/.git ]]; then
  mkdir -p "$ROOT/villa" && cd "$ROOT/villa" && git init -q
  git remote add origin https://github.com/ScrollPrize/villa.git
  git fetch -q --depth 1 origin "$VILLA_COMMIT" && git checkout -q FETCH_HEAD
fi
cd "$ROOT/villa"
test "$(git rev-parse HEAD)" = "$VILLA_COMMIT" || { echo "villa is not at the pinned commit"; exit 1; }

echo "== python env (villa's own bootstrap: torch + lasagna + vesuvius editable)"
if [[ ! -x $ROOT/venv/bin/python ]]; then
  cd "$ROOT/villa/lasagna"
  python3 scripts/bootstrap_venv.py --venv "$ROOT/venv" --backend "$TORCH_BACKEND"
fi
uv pip install --python "$ROOT/venv/bin/python" -q tifffile zarr numpy numcodecs opencv-python-headless safetensors s3fs
"$ROOT/venv/bin/python" -c "import vesuvius.ink_detection.inference.infer as m, cv2, zarr, torch; \
print('vesuvius', m.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== the winning reproduction repo"
[[ -d $ROOT/pherc343 ]] || git clone -q --depth 1 https://github.com/Nieuwlaar/pherc343-first-letters.git "$ROOT/pherc343"
chmod +x "$ROOT"/pherc343/ink/scripts/*.sh "$ROOT"/pherc343/ink/scripts/*.py 2>/dev/null || true

echo "== checkpoints (public, sha256-verified)"
PY="$ROOT/venv/bin/python" bash "$ROOT/pherc343/ink/scripts/fetch_checkpoints.sh" \
    "$ROOT/pherc343/ink/models.tsv" "$ROOT/ckpt"

echo "== N_LAYERS patch for 31-layer renders"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$ROOT/venv/bin/python" "$HERE/patch_n_layers.py" "$ROOT/pherc343/ink/scripts/ensemble_maps.py"

cat <<EOF

ready.
  PY=$ROOT/venv/bin/python
  CKPT_DIR=$ROOT/ckpt
  repo=$ROOT/pherc343

next:
  \$PY $HERE/pull_render.py --scroll PHerc0800 --segment 20251028222030 --out $ROOT/renders
  bash $HERE/sweep31.sh $ROOT/renders/20251028222030.zarr $ROOT/maps/20251028222030 -3 3
EOF
