#!/usr/bin/env bash
#
# Clean-room install test for nav123d on Python 3.12.
#
# Recreates a dedicated conda env from scratch (so any previous nav123d install
# there is wiped), installs the local py123d + nav123d, and smoke-tests imports.
# Does NOT touch your working `nav123d` (3.9) env.
#
# Usage:
#   bash scripts/test_install_py312.sh [env_name]
#   PY123D_DIR=/path/to/py123d bash scripts/test_install_py312.sh
#
set -euo pipefail

ENV_NAME="${1:-nav123d-py312}"
PYTHON_VERSION="3.12"

# Repo root = parent of this script's directory.
NAV123D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# py123d is not on PyPI yet — install it from its local source.
PY123D_DIR="${PY123D_DIR:-$HOME/py123d_workspace/py123d}"

if [[ ! -f "$PY123D_DIR/pyproject.toml" && ! -f "$PY123D_DIR/setup.py" ]]; then
  echo "ERROR: local py123d source not found at '$PY123D_DIR'." >&2
  echo "       Set PY123D_DIR=/path/to/py123d and re-run." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ">>> Removing existing env '$ENV_NAME' (if any)..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

echo ">>> Creating env '$ENV_NAME' (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

conda activate "$ENV_NAME"
echo ">>> Using $(python --version) at $(which python)"

# Work around pip's occasional "Could not find a suitable TLS CA certificate
# bundle" error by pointing it at the system CA bundle when available.
if [[ -z "${PIP_CERT:-}" && -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export PIP_CERT=/etc/ssl/certs/ca-certificates.crt
fi

echo ">>> Installing local py123d (editable) from $PY123D_DIR ..."
pip install -e "$PY123D_DIR"

echo ">>> Installing nav123d (editable) from $NAV123D_DIR ..."
pip install -e "$NAV123D_DIR"

echo ">>> Smoke test: imports + resolved versions"
python - <<'PY'
import torch, torchvision, lightning, timm, tensorboard
import numpy, py123d, nav123d
print(f"  python       {__import__('sys').version.split()[0]}")
print(f"  numpy        {numpy.__version__}")
print(f"  torch        {torch.__version__}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  lightning    {lightning.__version__}")
print(f"  timm         {timm.__version__}")
print(f"  py123d       {getattr(py123d, '__version__', '?')}")
# Heaviest torch consumer — exercises torch + torchvision + timm together.
from nav123d.agents.transfuser.transfuser_agent import TransfuserAgent  # noqa: F401
print("  transfuser import OK")
PY

echo ""
echo ">>> SUCCESS: nav123d installs and imports on Python $PYTHON_VERSION (env: $ENV_NAME)."
echo ">>> Optional: run unit tests with"
echo "      conda activate $ENV_NAME && pytest tests/unit"
echo ">>> Remove the test env when done with"
echo "      conda env remove -n $ENV_NAME -y"
