#!/usr/bin/env bash
# Requires WANDB_API_KEY when running with debug=0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/fol_docker.inc.sh
source "${SCRIPT_DIR}/scripts/fol_docker.inc.sh"

start=0
end=11

for gpu in $(seq $start $end); do
  fol_docker_run "75-89" "python -m fol_training.run_fol vanilla_main 0 1 ${gpu} 0 0" &
  echo "$gpu" &
done
wait
