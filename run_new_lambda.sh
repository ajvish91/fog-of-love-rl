#!/usr/bin/env bash
# Requires WANDB_API_KEY when running with debug=0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/fol_docker.inc.sh
source "${SCRIPT_DIR}/scripts/fol_docker.inc.sh"

start=14
end=19
gpu_start=0
gpu_end=5

are_lists_unique() {
  for i in $(seq 0 $(( ${#shuffled_list1[@]} - 1 ))); do
    if [ "${shuffled_list1[$i]}" -eq "${shuffled_list2[$i]}" ]; then
      return 1
    fi
  done
  return 0
}

while true; do
  list1=($(seq $start $end))
  list2=($(seq $start $end))
  shuffled_list1=($(shuf -e "${list1[@]}"))
  shuffled_list2=($(shuf -e "${list2[@]}"))
  if are_lists_unique; then
    break
  fi
done

echo "Shuffled list1: ${shuffled_list1[*]}"
echo "Shuffled list2: ${shuffled_list2[*]}"

for gpu in $(seq $gpu_start $gpu_end); do
  a=${shuffled_list1[$gpu]}
  b=${shuffled_list2[$gpu]}
  echo "Running: python -m fol_training.run_fol arg_main 0 1 ${gpu} ${a} ${b} 5.0"
  fol_docker_run "45-69" "python -m fol_training.run_fol arg_main 0 1 ${gpu} ${a} ${b} 5.0" &
done
wait
