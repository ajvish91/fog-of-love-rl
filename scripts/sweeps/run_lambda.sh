#!/usr/bin/env bash
# Requires WANDB_API_KEY when running with debug=0.
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SWEEP_DIR}/../.." && pwd)"
# shellcheck source=../fol_docker.inc.sh disable=SC1091
source "${REPO_ROOT}/scripts/fol_docker.inc.sh"

start=14
end=20
gpu=0

for l in $(seq $start $end); do
  fol_docker_run "45-69" "python -m fol.training.run_fol arg_main 0 1 ${gpu} ${l} ${l} 5.0" &
  echo "$l" "$gpu" &
  gpu=$(( gpu + 1 ))
done
wait
