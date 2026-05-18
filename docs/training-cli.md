# Training CLI quick reference

All training runs through **`fol_training.run_fol`**. For **background** (game vs simulator, ABRL math, module map), see **[`technical-overview.md`](technical-overview.md)**. For **how `AgileRL/` is stored in Git**, see **[`agilerl-layout.md`](agilerl-layout.md)**.

From the repository root (with `AgileRL` installed editable, as in `README.md`):

```bash
python -m fol_training.run_fol <MODE> <positional args …>
python -m fol_training.run_fol --config path/to/recipe.yaml
python -m fol_training.run_fol -c path/to/recipe.yaml
```

## Modes (positional)

| Mode | Arguments after mode | Notes |
|------|----------------------|--------|
| `arg_main` | `debug` `abrl` `gpu` `aff1` `aff2` `lambda` | `debug`/`abrl`: `0` or `1`. `gpu`: 0–11. Affinities: 14–20. |
| `basic_main` | `debug` `abrl` `gpu` `aff1` `aff2` | **`abrl` must be `0`** (plain MADDPG). Same index range 14–20. |
| `vanilla_main` | `debug` `abrl` `gpu` `reg1` `reg2` | `reg*`: 0–2 (global ABRL prior slots). `gpu`: 0–11. |
| `mul_main` | `debug` `abrl` `gpu` `<indices…> _ <indices…>` | `gpu`: 0–12. Each side: integers 14–20; `_` separates player 1 and player 2 lists. |

Example (localized ABRL smoke):

```bash
mkdir -p plots
python -m fol_training.run_fol arg_main 1 1 0 14 14 5.0
```

Example (multi-index):

```bash
python -m fol_training.run_fol mul_main 1 1 0 14 15 _ 14
```

## YAML (`--config` / `-c`)

`fol_train_config.py` maps `recipe:` to the same argv as above.

| `recipe` | Typical file | Maps to |
|----------|----------------|---------|
| `arg_localized` | `configs/smoke_arg_localized.yaml` | `arg_main …` |
| `basic_localized` | `configs/smoke_basic_localized.yaml` | `basic_main …` |
| `vanilla_global` | `configs/smoke_vanilla_global.yaml` | `vanilla_main …` |
| `mul_multi_affinity` | `configs/smoke_mul_multi_affinity.yaml` | `mul_main …` |

For `vanilla_global`, YAML may use **`reg_idx_1` / `reg_idx_2`** or **`aff_slot_1` / `aff_slot_2`** (same meaning; use one pair per player).

## Docker sweep helper (`scripts/fol_docker.inc.sh`)

`fol_docker_run` honors:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOL_DOCKER_VOLUME` | (see script) | Host directory mounted as `/app` |
| `FOL_DOCKER_IMAGE` | `python:3.12` | Image passed to `docker run` |
| `FOL_DOCKER_OPTIONS` | `--gpus all` | Extra `docker run` arguments; set to empty string for CPU-only |

**ShellCheck.** GitHub Actions runs **ShellCheck** (and `bash -n`) on `scripts/fol_docker.inc.sh`, `scripts/fol_smoke.sh`, and `run_*.sh`. For each `run_*.sh`, keep this immediately above the `source` line so ShellCheck resolves the helper:

`# shellcheck source=scripts/fol_docker.inc.sh`

## Smoke tests (`scripts/fol_smoke.sh`, `scripts/smoke_training.py`)

**First-time setup (recommended):**

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

On **macOS**, `setup_venv.sh` skips Linux-only NVIDIA CUDA pins, installs CPU **torch 2.2.2**, then editable **AgileRL** with `--no-deps` plus **accelerate** (AgileRL’s torch pin is newer than mac wheels allow).

From the repo root (with dependencies installed):

```bash
chmod +x scripts/fol_smoke.sh
./scripts/fol_smoke.sh check          # bash -n + ShellCheck (if installed) + YAML parse/expand
./scripts/fol_smoke.sh train          # same as check, then default YAML train smoke
./scripts/fol_smoke.sh train-only --config configs/smoke_mul_multi_affinity.yaml
```

**Python interpreter:** the wrapper picks, in order: **`$PYTHON`** (path or command name), **`.venv/bin/python`**, **`$VIRTUAL_ENV/bin/python`**, then **`python3`**. If `python` on your PATH is a bare conda base without project packages, activate the venv first or run e.g. `PYTHON=.venv/bin/python ./scripts/fol_smoke.sh train-only`.

**ShellCheck:** if `shellcheck` is not installed, the check step **skips** static analysis and prints a hint (install via `brew install shellcheck` or `apt install shellcheck`). CI sets **`FOL_REQUIRE_SHELLCHECK=1`** so the job fails if ShellCheck is missing after `apt install`.

Or call the Python helper directly:

```bash
python scripts/smoke_training.py check
python scripts/smoke_training.py train --config configs/smoke_arg_localized.yaml
```

CI runs `./scripts/fol_smoke.sh check` followed by `./scripts/fol_smoke.sh train-only` (see `.github/workflows/smoke.yml`).

## Docker image (repository `Dockerfile`)

Build:

```bash
docker build -t fol-rl .
```

The image uses `ENTRYPOINT ["python", "-m", "fol_training.run_fol"]` and a default **`CMD`** that points at a smoke YAML recipe (`configs/smoke_arg_localized.yaml` unless you change the Dockerfile).

Examples:

```bash
# Default CMD: smoke YAML inside the image
docker run --rm fol-rl

# Different YAML recipe
docker run --rm fol-rl --config configs/smoke_basic_localized.yaml

# Positional mode (same as local `python -m …`); pass W&B when debug=0
docker run --rm -e WANDB_API_KEY -e CUDA_VISIBLE_DEVICES=0 --gpus all \
  fol-rl arg_main 1 1 0 14 14 5.0
```

Do **not** bake `WANDB_API_KEY` into the image; use `-e WANDB_API_KEY` at run time when needed.

## Weights & Biases projects (typical)

When not in `debug` mode, runs may log to one of:

- `fol-local-abrl-agile-implement` — localized ABRL (`arg_main`; use **`arg_main`**, not **`basic_main`**, when **`abrl=1`**)
- `fol-multi-local-abrl-agile-implement` — multi-index localized (`mul_main` with ABRL)
- `fol-abrl-agile-implement` — global ABRL (`vanilla_main` with ABRL)
- `fol-agile-implement` — no ABRL

Exact project choice is implemented in `fol_training/run_fol.py` (`Run` / `FolTrainingConfig`).

## Artifact: `pareto_history.json`

Rich test-step logs (actions / observations per episode) are only emitted for the **`arg_main`** path when not in debug, as in the unified driver. Export or reconstruct `pareto_history.json` from W&B or local logs if your analysis notebook expects that filename; the schema is whatever your download script produced historically.
