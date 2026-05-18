#!/usr/bin/env bash
# Shared Docker launcher for FoL training jobs.
#
# Source from a sweep script (scripts/sweeps/) or repo root:
#   REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # if cwd is scripts/
#   source "${REPO_ROOT}/scripts/fol_docker.inc.sh"
#
# Environment (override before calling fol_docker_run):
#   FOL_DOCKER_VOLUME   Host path mounted as /app (default below).
#   FOL_DOCKER_IMAGE    Image to run (default: python:3.12).
#   FOL_DOCKER_OPTIONS  Extra docker run args, e.g. "--gpus all" or "" for CPU-only.
#                       Defaults to "--gpus all". Use a quoted string for multiple flags.
#
# Example CPU-only smoke:
#   export FOL_DOCKER_OPTIONS=""
#   fol_docker_run "0-3" "python -m fol_training.run_fol arg_main 1 1 0 14 14 5.0"
#
# Same checks as CI (without Docker): from repo root run ./scripts/fol_smoke.sh check

_fol_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}
FOL_DOCKER_VOLUME="${FOL_DOCKER_VOLUME:-$(_fol_repo_root)}"
FOL_DOCKER_IMAGE="${FOL_DOCKER_IMAGE:-python:3.12}"
FOL_DOCKER_OPTIONS="${FOL_DOCKER_OPTIONS:---gpus all}"

# Usage: fol_docker_run <cpuset-cpus> '<inner shell command after cd /app>'
fol_docker_run() {
  local cpuset="$1"
  local inner_command="$2"
  # shellcheck disable=SC2086
  docker run -d --rm -e WANDB_API_KEY --cpuset-cpus="${cpuset}" ${FOL_DOCKER_OPTIONS} \
    -v "${FOL_DOCKER_VOLUME}:/app" -w /app "${FOL_DOCKER_IMAGE}" \
    bash -c "pip install -r requirements.txt && cd AgileRL/ && pip install -e . && cd .. && ${inner_command}"
}
