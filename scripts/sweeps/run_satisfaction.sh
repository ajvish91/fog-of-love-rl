#!/usr/bin/env bash
# Requires WANDB_API_KEY when running with debug=0.
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SWEEP_DIR}/../.." && pwd)"
# shellcheck source=../fol_docker.inc.sh disable=SC1091
source "${REPO_ROOT}/scripts/fol_docker.inc.sh"

sequence=(0.001 0.01 0.1 1 10 100 1000 10000 100000)
gpu=0

for value in "${sequence[@]}"; do
  echo "Processing value: $value on gpu $gpu"
  fol_docker_run "75-89" "python -m fol.training.run_fol arg_main 0 1 ${gpu} 20 20 ${value}" &
  gpu=$(( gpu + 1 ))
done
wait
