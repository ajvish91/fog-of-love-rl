# Fog of Love RL

**A research playground for two AI “partners” learning to cooperate and compete at the same time.**

This repository trains **two cooperative–competitive agents** in a **Gym-style Markov game** inspired by the relationship board game [**Fog of Love**](https://www.floodgate.games/products/fog-of-love) (see disclaimer below). Each episode, both players pick among **three discrete actions** per “scene”; observations bundle **virtue-style trait channels**, **satisfaction**, and masked opponent goals so policies must generalize under **partial observability**. Learning uses **multi-agent deep deterministic policy gradients (MADDPG)** from an in-tree [**AgileRL**](https://github.com/AgileRL/AgileRL) fork, extended here with **affinity-based reinforcement learning (ABRL)**: an extra actor-side regularizer weighted by **λ** that biases action distributions toward state-dependent “role-model” priors—useful when pure reward shaping is brittle.

---

## What’s in the box

| Piece | Role |
|-------|------|
| **`fol_env.py`** | `FoLEnvironment`: simultaneous two-player steps, **Dict** observations, **Discrete(3)** actions, `match` / `no_match` satisfaction coupling—**no RL algorithm** inside, just transitions + rewards. |
| **`fol_training/`** | Unified driver **`python -m fol_training.run_fol`** switches **`arg_main` / `basic_main` / `vanilla_main` / `mul_main`** → `maddpg_local_abrl`, `maddpg_abrl`, or stock **`maddpg`**, plus W&B logging hooks. |
| **`AgileRL/`** | Vendored library **`pip install -e .`**; hosts the modified actor–critic stacks and ABRL losses (`maddpg_local_abrl.py`, `maddpg_abrl.py`, …). |
| **`configs/`** | Validated **YAML** `recipe:` files (smoke + sweeps) expanded by `fol_train_config.py` to the same argv as the CLI. |

Deeper write-ups live under **`docs/`**: how the tabletop game maps to the simulator, ABRL notation, and the full module map are in **[`technical-overview.md`](docs/technical-overview.md)**; day-to-day commands and Docker patterns are in **[`training-cli.md`](docs/training-cli.md)**.

---

## Try it locally

**You need:** Python **3.10+** (3.12 works well), a terminal, and about 5–10 minutes for a first install. A **GPU** speeds things up but is not required for the short “smoke” run.

The one-liner below runs **`arg_main`** with **`debug=1`**, **`abrl=1`**, **`gpu=0`**, affinity indices **`14`** for both players, and **λ = 5.0** (see **`docs/training-cli.md`** for other modes).

```bash
cd fol_latest_codebase
./scripts/setup_venv.sh            # Linux: full requirements.txt; macOS: skips CUDA pins
source .venv/bin/activate          # Windows: .venv\Scripts\activate
mkdir -p plots
python -m fol_training.run_fol arg_main 1 1 0 14 14 5.0
```

Or use the smoke wrapper: **`./scripts/fol_smoke.sh check`** then **`./scripts/fol_smoke.sh train-only`** (uses **`.venv/bin/python`** when present). Details: **`docs/training-cli.md`**.

**Automated tests:** with the same venv, run `pip install pytest`, then `MPLBACKEND=Agg FOL_TEST_MINIMAL=1 pytest tests/ -v`. `FOL_TEST_MINIMAL` shortens `debug=1` training loops only during tests (see `fol_training/run_fol.py`). CI runs `pytest` after installing `requirements.txt` and editable `AgileRL`.

The last line runs a **tiny** training job (`debug=1`) so you can confirm everything wires up. For longer runs and optional [Weights & Biases](https://wandb.ai) logging, copy **`.env.example`** to **`.env`**, add your API key, then see **`docs/training-cli.md`**.

---

## Docker (optional)

If you prefer containers:

```bash
docker build -t fol-rl .
docker run --rm fol-rl
```

The image sets **`ENTRYPOINT python -m fol_training.run_fol`** and a default **YAML** smoke **`CMD`**; append arguments after the image name to swap recipes or pass **`arg_main …`** positional tails. Full examples: **`docs/training-cli.md`**.

---

## Further reading

| Topic | Where to look |
|-------|----------------|
| Commands, YAML recipes, Docker, smoke tests | [`docs/training-cli.md`](docs/training-cli.md) |
| Game vs simulator, observation layout, ABRL math, CI & sweeps | [`docs/technical-overview.md`](docs/technical-overview.md) |
| Publishing `AgileRL/` (vendored copy vs submodule) | [`docs/agilerl-layout.md`](docs/agilerl-layout.md) |
| Index of all docs | [`docs/README.md`](docs/README.md) |

The **`AgileRL/`** directory is an in-tree fork of [AgileRL](https://github.com/AgileRL/AgileRL) extended for multi-agent training and ABRL on MADDPG—installed with `pip install -e ./AgileRL` from the repo root, same as the setup above. Git layout notes are in **`docs/agilerl-layout.md`**.

---

## Citation

If this code helps your work, cite your eventual paper or preprint when it exists, and cite **AgileRL** per [their project](https://github.com/AgileRL/AgileRL). Update **`CITATION.cff`** when author list and repo URL are final.

---

## License

Environment and training code at the repo root are **Apache License 2.0** (see **`LICENSE`**). The **`AgileRL/`** directory is upstream AgileRL **plus local modifications**, still Apache-2.0—see **`NOTICE`** and **`AgileRL/LICENSE`**.

---

## Disclaimer

**Fog of Love**® is a registered trademark of **Floodgate Games, LLC**. This software is **not** affiliated with, endorsed by, or sponsored by Floodgate Games. It is an independent research simulator inspired by relationship dynamics in tabletop role-playing.
