#!/usr/bin/env bash
# Create .venv and install FoL training dependencies.
# Linux/CI: full requirements.txt (includes CUDA wheels when applicable).
# macOS: skips nvidia-* and triton lines (CPU PyTorch from PyPI).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "error: no python3 found" >&2
  exit 1
fi

echo "[setup_venv] using ${PY}"
"${PY}" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip

REQ_FILE="${ROOT}/requirements.txt"
if [[ "$(uname -s)" == Darwin ]]; then
  echo "[setup_venv] macOS: CPU PyTorch + requirements without NVIDIA CUDA / triton / torch pins"
  FILTERED="$(mktemp)"
  grep -vE '^(nvidia-|triton==|torch==)' "${REQ_FILE}" > "${FILTERED}"
  # PyPI macOS wheels for torch 2.5.x may be unavailable; 2.2.x is sufficient for smoke training.
  pip install 'torch==2.2.2'
  pip install -r "${FILTERED}"
  rm -f "${FILTERED}"
else
  pip install -r "${REQ_FILE}"
fi

if [[ "$(uname -s)" == Darwin ]]; then
  # AgileRL metadata asks for torch>=2.3.1; macOS PyPI often tops out at 2.2.x — smoke runs on 2.2.2.
  pip install -e ./AgileRL --no-deps
  pip install 'accelerate>=0.18.0,<0.19.0'
else
  pip install -e ./AgileRL
fi
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "${ROOT}/scripts" > "${SITE_PACKAGES}/fol_repo_scripts.pth"
echo "[setup_venv] added ${SITE_PACKAGES}/fol_repo_scripts.pth -> scripts/ (import fol.* without PYTHONPATH)"

echo "[setup_venv] done. Activate with: source .venv/bin/activate"
echo "[setup_venv] Then run: ./scripts/fol_smoke.sh check && ./scripts/fol_smoke.sh train-only"
