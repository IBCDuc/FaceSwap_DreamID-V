#!/usr/bin/env bash
set -euo pipefail
export FS_ROOT=${FS_ROOT:-/data/faceswap/experiments}
export UV_CACHE_DIR=$FS_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$FS_ROOT/python
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=5
export TMPDIR=$FS_ROOT/tmp
mkdir -p "$FS_ROOT" "$TMPDIR"
if [ ! -x "$FS_ROOT/bootstrap/bin/uv" ]; then
 python3 -m venv "$FS_ROOT/bootstrap"
 "$FS_ROOT/bootstrap/bin/pip" install uv
fi
UV=$FS_ROOT/bootstrap/bin/uv
$UV python install 3.12
if [ ! -x "$FS_ROOT/dreamidv/.venv/bin/python" ]; then
 $UV venv --python 3.12 "$FS_ROOT/dreamidv/.venv"
fi
$UV pip install --python "$FS_ROOT/dreamidv/.venv/bin/python" torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
$UV pip install --python "$FS_ROOT/dreamidv/.venv/bin/python" -r "$(dirname "$0")/requirements.txt"
"$FS_ROOT/dreamidv/.venv/bin/python" -c 'import torch; print(torch.__version__,torch.cuda.get_device_name(0)); print(torch.ones(4,device="cuda").sum())'
