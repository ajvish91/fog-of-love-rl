#!/usr/bin/env bash
# Requires WANDB_API_KEY in the environment for Weights & Biases when debug=0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/fol_docker.inc.sh
source "${SCRIPT_DIR}/scripts/fol_docker.inc.sh"

start=0
end=2
gpu=0

for l in $(seq $start $end); do
  for k in $(seq $start $end); do
    fol_docker_run "75-89" "python -m fol_training.run_fol vanilla_main 0 1 ${gpu} ${l} ${k}" &
    echo "$l" "$k" "$gpu" &
    gpu=$(( gpu + 1 ))
  done
done
wait
