#!/usr/bin/env bash
# Requires WANDB_API_KEY when running with debug=0.
#
# Intentionally launches two containers per (l, gpu): same inner command on
# disjoint CPU sets (45-69 vs 25-44) so two sweeps can progress in parallel
# on one host. Drop one fol_docker_run line if you only want a single pool.
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SWEEP_DIR}/../.." && pwd)"
# shellcheck source=../fol_docker.inc.sh disable=SC1091
source "${REPO_ROOT}/scripts/fol_docker.inc.sh"

start=14
end=20
num_gpus=7
gpu=6

for l in $(seq $start $end); do
  inner="python -m fol.training.run_fol basic_main 0 0 ${gpu} ${l} ${l}"
  fol_docker_run "45-69" "${inner}" &
  fol_docker_run "25-44" "${inner}" &
  echo "$l" "$gpu" &
  gpu=$(( 6 + (gpu - 5) % num_gpus ))
done
wait
