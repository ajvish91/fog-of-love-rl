#!/usr/bin/env bash
# FoL smoke: shell hygiene + YAML checks (+ optional short train).
# Usage (from repo root or via path):
#   ./scripts/fol_smoke.sh              # shell + YAML checks (default)
#   ./scripts/fol_smoke.sh check        # same
#   ./scripts/fol_smoke.sh train        # check + train (local convenience)
#   ./scripts/fol_smoke.sh train-only   # train only (CI after check)
#   ./scripts/fol_smoke.sh train-only --config configs/smoke_vanilla_global.yaml
#
# Interpreter: uses $PYTHON if set; else .venv/bin/python; else $VIRTUAL_ENV/bin/python;
# else python3. Override if your default python lacks deps: PYTHON=.venv/bin/python ./scripts/fol_smoke.sh train-only
#
# ShellCheck: skipped if not in PATH unless FOL_REQUIRE_SHELLCHECK=1 (CI sets this via apt install shellcheck).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

_fol_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if [[ -x "${PYTHON}" ]]; then
      printf '%s' "${PYTHON}"
      return
    fi
    local _w
    _w="$(command -v "${PYTHON}" 2>/dev/null || true)"
    if [[ -n "${_w}" ]]; then
      printf '%s' "${_w}"
      return
    fi
  fi
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    printf '%s' "${ROOT}/.venv/bin/python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s' "${VIRTUAL_ENV}/bin/python"
    return
  fi
  command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}

FOL_PY="$(_fol_python)"
if [[ -z "${FOL_PY}" ]]; then
  echo "[fol_smoke] error: no python found (set PYTHON=… or create .venv)" >&2
  exit 1
fi

_python_has_train_deps() {
  "$1" -c "import matplotlib, torch" 2>/dev/null
}

_print_train_setup() {
  cat >&2 <<EOF
[fol_smoke] Training needs torch + matplotlib in a project environment.
[fol_smoke] Selected interpreter lacks them: ${FOL_PY}

Create and use a venv in the repo root (recommended):

  cd "${ROOT}"
  ./scripts/setup_venv.sh
  source .venv/bin/activate
  ./scripts/fol_smoke.sh train-only

Or point at an existing env:

  PYTHON=/path/to/python ./scripts/fol_smoke.sh train-only
EOF
}

_require_train_python() {
  if _python_has_train_deps "${FOL_PY}"; then
    return 0
  fi
  # If user forced PYTHON without deps, still try .venv before failing.
  if [[ -x "${ROOT}/.venv/bin/python" ]] && _python_has_train_deps "${ROOT}/.venv/bin/python"; then
    FOL_PY="${ROOT}/.venv/bin/python"
    echo "[fol_smoke] using ${FOL_PY} (has torch/matplotlib)" >&2
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]] \
      && _python_has_train_deps "${VIRTUAL_ENV}/bin/python"; then
    FOL_PY="${VIRTUAL_ENV}/bin/python"
    echo "[fol_smoke] using ${FOL_PY} (has torch/matplotlib)" >&2
    return 0
  fi
  _print_train_setup
  return 1
}

_do_shell() {
  local f
  for f in scripts/fol_docker.inc.sh scripts/fol_smoke.sh run_*.sh; do
    echo "[fol_smoke] bash -n $f"
    bash -n "$f"
  done
  if command -v shellcheck >/dev/null 2>&1; then
    echo "[fol_smoke] shellcheck scripts/fol_docker.inc.sh scripts/fol_smoke.sh run_*.sh"
    shellcheck scripts/fol_docker.inc.sh scripts/fol_smoke.sh run_*.sh
  elif [[ "${FOL_REQUIRE_SHELLCHECK:-}" == "1" ]]; then
    echo "[fol_smoke] error: shellcheck not in PATH but FOL_REQUIRE_SHELLCHECK=1" >&2
    exit 1
  else
    echo "[fol_smoke] shellcheck not in PATH, skipping (brew install shellcheck / apt install shellcheck)" >&2
  fi
}

_do_yaml() {
  echo "[fol_smoke] ${FOL_PY} scripts/smoke_training.py check"
  "${FOL_PY}" scripts/smoke_training.py check
}

_do_train() {
  _require_train_python
  mkdir -p plots
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  echo "[fol_smoke] ${FOL_PY} scripts/smoke_training.py train" "$@"
  "${FOL_PY}" scripts/smoke_training.py train "$@"
}

cmd="${1:-check}"
case "$cmd" in
  check)
    _do_shell
    _do_yaml
    ;;
  train)
    shift
    _do_shell
    _do_yaml
    _do_train "$@"
    ;;
  train-only)
    shift
    _do_train "$@"
    ;;
  shell)
    _do_shell
    ;;
  yaml)
    _do_yaml
    ;;
  -h|--help)
    sed -n '1,22p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 [check|train|train-only|shell|yaml] [train args…]" >&2
    exit 2
    ;;
esac
