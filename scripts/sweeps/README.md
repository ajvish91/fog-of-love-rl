# Docker sweep launchers

Batch training on a host with Docker. Each script sources `scripts/fol_docker.inc.sh` and calls `fol_docker_run` with different `python -m fol_training.run_fol` arguments.

Run from the **repository root** (so the default `FOL_DOCKER_VOLUME` mount is the checkout):

```bash
./scripts/sweeps/run_lambda.sh
```

Override the mount if needed:

```bash
export FOL_DOCKER_VOLUME=/path/to/checkout
./scripts/sweeps/run_lambda.sh
```

| Script | Mode / purpose |
|--------|----------------|
| `run_lambda.sh` | `arg_main` over affinity index 14–20 |
| `run_new_lambda.sh` | `arg_main` with shuffled affinity pairs |
| `run_basic_lambda.sh` | `basic_main` (two CPU pools per iteration) |
| `run_equal.sh` | `vanilla_main` with reg indices `0 0` |
| `run_abrl.sh` | `vanilla_main` grid over reg indices |
| `run_satisfaction.sh` | `arg_main` at indices `20 20`, λ sweep |

Requires `WANDB_API_KEY` when jobs use `debug=0`. See `docs/training-cli.md`.
