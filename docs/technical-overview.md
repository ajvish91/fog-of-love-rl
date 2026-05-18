# Technical overview

This page is for **engineers and researchers** who want the full picture: how the retail board game is abstracted, what the Markov game looks like, how **affinity-based reinforcement learning (ABRL)** is formulated here, and how the Python layout maps to training modes.

For day-to-day commands, see **[`training-cli.md`](training-cli.md)**. For Git layout of `AgileRL/`, see **[`agilerl-layout.md`](agilerl-layout.md)**.

---

## Research framing

Instilling virtuous behavior in artificial agents is an active topic in machine ethics. **ABRL** adds policy regularization so agents favor “role-model” action priors instead of relying only on the reward signal; prior work often used grid worlds and small action spaces. This codebase implements a **two-player multi-agent environment** inspired by the board game **Fog of Love**: agents pursue **individual virtues** (competitive pressure) while **cooperating** to raise **relationship satisfaction**. Standard **multi-agent deep deterministic policy gradient (MADDPG)** baselines struggle to jointly optimize those objectives in this setting; **localized** affinity regularization is implemented to improve competitive and cooperative outcomes while keeping behavior interpretable.

The sections below condense how the **retail game** is described in the research write-up versus what the **Gym-style simulator** actually implements. They are a formalization aid, not independent research into the commercial product.

---

## Fog of Love — board game (reference)

Fog of Love is described as a **two-player** experience mixing **cooperative** and **competitive** goals: the characters are in a relationship and resolve scenarios while balancing **personal objectives** against **shared relationship needs**.

**Setup.** Player 1 draws an **occupation** and three **Trait** goal cards; Player 2 picks a **feature** they liked about Player 1 from three random feature cards (the roles then repeat for Player 2). Occupations, traits, and features each specify one of six **virtues**—discipline, sincerity, sensitivity, extraversion, gentleness, curiosity—and an **integer** value. Trait cards encode targets the player must reach (for example discipline = 4, extraversion = −5, gentleness = 2). A **virtue value map** is initialized from occupation and features, **updates** whenever options are taken, and is compared to trait goals. **Satisfaction** measures how pleased each character is with the relationship and informs how much they weight cooperation.

**Scenes.** A **Scene card** states a situation or question. One or both players simultaneously choose among labeled options **(A, B, C, …)**. Each option lists up to **three** virtue deltas and can move **satisfaction**. A **match condition** usually adds **positive satisfaction** when both players pick the **same** option and applies a **negative** swing when they pick **different** options.

**Chapters and destiny.** A chapter may contain up to **10** scenes; at chapter boundaries, played scenes are discarded and new ones enter. The full game ends after all chapters when **destiny** resolves. Destiny cards stay **secret** until the end; examples include joint satisfaction thresholds or asymmetric virtue targets. Players may draw, keep, or discard destiny cards during play, and the relationship “survives” only if **both** destinies are fulfilled. A central **tension** is **incomplete information**: players **do not** know each other’s trait goals and must avoid sabotaging the partner while still pursuing their own traits and satisfaction.

---

## RL environment (simulator)

The Fog-of-Love-inspired world is implemented in **OpenAI Gym** with explicit state, action, reward, and episode dynamics. Evaluation uses metrics **ϱ** (goal success rate) and **δ_err** (goal error) over trait and satisfaction targets.

**Observations (88 numbers total, 44 per agent).** Each agent sees both players’ current **six** virtue totals and both **satisfaction** scores. It also sees **its own** trait and satisfaction goals for the episode (constants). Agents **do not** observe the **other** agent’s trait or satisfaction goals, yet **both** must still satisfy their satisfaction goal each episode. Each scene exposes **three** options; every option specifies all **six** virtues plus a satisfaction component (**21** numbers). Two scalars encode the **match** bonus and **no-match** penalty applied when actions coincide or diverge.

**Actions.** Each agent chooses among **three** discrete actions, each tied to one scene option.

**Rewards.** Rewards encourage alignment between the chosen option and each agent’s signed trait and satisfaction goals.

**Dynamics.** An episode lasts **100** simultaneous turns (“scenes”). After every joint decision, a **new** random scene is drawn with random virtue and satisfaction entries, yielding a **highly stochastic** setting intended to resist trivial reward hacking.

**Omissions and surrogate destiny.** Chapters, destiny machinery, and dramatic/narrative elements are **removed**. Cooperative pressure is approximated by fixing both agents’ **satisfaction goal at 30**, analogous to the **“Equal Partners”** destiny in the retail game. The intent is to preserve the **core competitive/cooperative coupling** while remaining tractable for RL.

---

## Tabletop vs. this simulation

| Aspect | Board game (as formalized in the write-up) | RL environment |
|--------|------------------------------------------|------------------|
| Narrative structure | Scene cards, chapters (≤10 scenes each), destiny deck | **100** procedural scenes; **no** chapters, destiny cards, or scripted drama |
| Option menus | Multiletter option lines (A, B, C, …) | **Three** discrete actions per scene |
| Information | Hidden opponent trait goals; destinies secret | Observation excludes the **other** agent’s goals while keeping numeric virtue/satisfaction state otherwise explicit |
| End conditions | Destiny cards with heterogeneous win tests | Shared satisfaction target **30** standing in for **Equal Partners** |
| Randomness | Card-driven distributions | Independent random scene generator each step |

**Keywords:** machine ethics; reinforcement learning; virtues; Fog of Love; multi-agent games.

---

## Affinity-based reinforcement learning (ABRL)

**Summary.** ABRL augments the usual RL objective with a regularizer $L$ that nudges each agent’s policy toward a prior $\pi_0$ over actions (a “role model” distribution). The scalar $\lambda$ trades off reward maximization versus staying close to that prior. *Global* affinities fix $\pi_0$ over the three scene options. *Localized* affinities make $\pi_0$ depend on the current **state** (and, in the generalized form, the sampled action). **Algorithm 1** in the write-up decides which option should receive high prior mass when a chosen **virtue goal** is still unmet and an option’s sign matches the goal’s sign. In this repository, that logic lives in **`check_affinity_condition`** in `scripts/fol/training/affinity_condition.py` (indices **14–20** pick which flattened observation entry is the goal channel used by the condition).

**Objective.**

$$
J(\theta) = \mathbb{E}_{S,A \sim \mathcal{D}}\bigl[R(S,A)\bigr] - \lambda L
$$

**Global affinity term.** The unlocalized loss compares the policy’s mass on each discrete action $a_j$ to a fixed prior $\pi_0(a_j \mid A)$ (in code: `reg_params` per player in `maddpg_abrl`):

$$
L = \frac{1}{M}\sum_{j=0}^{M-1}\Bigl[\mathbb{E}_{A \sim \pi_\theta}[a_j] - \pi_0(a_j \mid A)\Bigr]
$$

**Localized affinity term.** State-(and-action-)dependent prior $\pi_{0,i}(a_j \mid S, A)$ with a **squared** gap:

$$
L_s = \frac{1}{M}\sum_{j=0}^{M-1}\Bigl[\mathbb{E}_{S,A \sim \pi_\theta}[a_j] - \pi_{0,i}(a_j \mid S, A)\Bigr]^2
$$

Here $M$ is the number of discrete actions (this env: **three** scene options). **MADDPG** supplies separate actors and a shared critic; ABRL modifies the **actor** side through these terms.

**References (ABRL).**

1. Vishwanath, A., & Omlin, C. (2024). [Exploring Affinity-Based Reinforcement Learning for Designing Artificial Virtuous Agents in Stochastic Environments](https://link.springer.com/chapter/10.1007/978-981-99-9836-4_3). In *Frontiers of Artificial Intelligence, Ethics, and Multidisciplinary Applications* (FAIEMA 2023), pp. 25–38. Springer, Singapore. https://doi.org/10.1007/978-981-99-9836-4_3

2. Vishwanath, A., & Omlin, C. (2025). [Localized Affinity-Based Reinforcement Learning for Interpretable State-Specific Decision-Making](https://link.springer.com/chapter/10.1007/978-3-031-77915-2_16). In *Artificial Intelligence XLI* (SGAI 2024), Lecture Notes in Computer Science, vol. 15446, pp. 221–234. Springer, Cham. https://doi.org/10.1007/978-3-031-77915-2_16

---

## Which Python pieces configure what

| Piece | What to know |
|--------|----------------|
| `scripts/fol/env.py`, `scripts/fol/attributes.py` | Markov game: observations, three options per step, rewards, `match` / `no_match` satisfaction shifts. No ABRL here—only transition and reward semantics. |
| `scripts/fol/training/run_fol.py` | **Single entrypoint** (`python -m fol.training.run_fol …`). Parses CLI modes, builds `FolTrainingConfig`, selects the `MADDPG` class, and passes **`lambda_`**, **`affinity_indices`**, **`affinity_condition`**, and/or **`reg_params`** into the algorithm. |
| `scripts/fol/training/affinity_condition.py` → `check_affinity_condition` | **Localized prior gate:** maps `(state_vector, affinity_idx)` → `(option_1_to_3, bool)` for state-dependent conditioning. |
| `fol.train_config.py` | Optional **`--config path.yaml`**: YAML is validated and expanded to the same argv tail as the CLI (see `argv_suffix_from_yaml`). |
| `AgileRL/agilerl/algorithms/maddpg_local_abrl.py` | **Localized ABRL** implementation used by **`arg_main`** and **`mul_main`**: squared loss form, `affinity_indices`, callable `affinity_condition`, default $(0.2, 0.6, 0.2)$ `reg_params` over the three options when the condition fires. **`arg_main`:** CLI $\lambda$ is forwarded. **`mul_main`:** multiple indices per player (lists split by **`_`**). |
| `AgileRL/agilerl/algorithms/maddpg_abrl.py` | **Global** ABRL: prior mass over actions **`{0,1,2}`** built from **`vanilla_main`**’s two **`reg`** indices in **`[0,2]`** (each picks a “boosted” option; $1/3$ base mass is encoded in `run_fol.py`). **`lambda_`** is fixed at **5.0** in code for this branch. |
| `AgileRL/agilerl/algorithms/maddpg.py` | Vanilla **MADDPG** used when **`abrl=0`** (any mode) and for **`basic_main`**, which **must** use **`abrl=0`** in this driver (the CLI and YAML loader reject **`abrl=1`**). Use **`arg_main`** for localized ABRL with **`maddpg_local_abrl`**. |
| `configs/*.yaml` | Smoke / experiment notes; `recipe` field selects how YAML maps to `run_fol` arguments (see `fol.train_config.py`). |

**Mode → algorithm:** `arg_main` → `local_abrl` → `maddpg_local_abrl`; `mul_main` → `multi_local_abrl` → `maddpg_local_abrl`; `vanilla_main` → `maddpg_abrl`; `basic_main` → `standard_maddpg` → stock `maddpg`.

---

## Repository layout (detailed)

| Path | Role |
|------|------|
| `scripts/fol/env.py` | Gym-style `FoLEnvironment`: virtues, satisfaction, scenario options, rewards |
| `scripts/fol/attributes.py` | Domain objects (e.g. virtues) used by the environment |
| `scripts/fol/training/` | Python package for training: **`run_fol.py`** is the supported driver |
| `scripts/fol/training/run_fol.py` | **Unified training entrypoint.** `python -m fol.training.run_fol <MODE> …` or `--config` / `-c`. |
| `fol.train_config.py` | Loads and validates YAML recipes in `configs/`. |
| `configs/` | YAML recipes (`recipe:` + fields). |
| `docs/training-cli.md` | CLI / YAML / Docker cheat sheet. |
| `docs/agilerl-layout.md` | AgileRL Git policy (vendored vs submodule vs subtree). |
| `AgileRL/` | Local copy of [AgileRL](https://github.com/AgileRL/AgileRL) extended here for MADDPG + ABRL |
| `scripts/sweeps/run_*.sh` | Example Docker batch launchers; each `source`s `scripts/fol_docker.inc.sh`. |
| `scripts/fol_docker.inc.sh` | `fol_docker_run <cpuset> '<inner command>'` wraps `docker run`. |
| `scripts/fol_smoke.sh` | Local / CI smoke driver (ShellCheck, YAML checks, optional train). |
| `scripts/smoke_training.py` | Python smoke: YAML expansion + short training. |
| `archive/notebooks/get_history.ipynb` | Notebook for inspecting W&B exports under `archive/artifacts/` |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Image build; `ENTRYPOINT` is `python -m fol.training.run_fol` |

---

## CI (GitHub Actions)

Pushes and pull requests run `.github/workflows/smoke.yml`, which installs dependencies and editable **`AgileRL`**, then:

1. **`./scripts/fol_smoke.sh check`** — `bash -n` + **ShellCheck** on `scripts/fol_docker.inc.sh`, `scripts/fol_smoke.sh`, and `scripts/sweeps/run_*.sh`; then **`python scripts/smoke_training.py check`** (every `configs/smoke_*.yaml` through `fol.train_config.argv_suffix_from_yaml`).
2. **`./scripts/fol_smoke.sh train-only`** — one short training run via **`python scripts/smoke_training.py train`** (default `configs/smoke_arg_localized.yaml`).

ShellCheck sourcing conventions for sweep scripts are described in **`training-cli.md`**.

---

## Sweep scripts (`scripts/sweeps/`)

Scripts under **`scripts/sweeps/`** orchestrate **Docker** jobs. They `source` **`scripts/fol_docker.inc.sh`**, mount the repository root by default (override **`FOL_DOCKER_VOLUME`**—see `fol_docker.inc.sh`), and pass **`WANDB_API_KEY`** when `debug=0`. Run from the repo root, e.g. `./scripts/sweeps/run_lambda.sh`.

| Script | What it runs (mode / pattern) |
|--------|-------------------------------|
| `scripts/sweeps/run_lambda.sh` | `arg_main` over affinity index and GPU loops |
| `scripts/sweeps/run_new_lambda.sh` | `arg_main` with shuffled unique affinity pairs |
| `scripts/sweeps/run_basic_lambda.sh` | `basic_main` (two CPU sets per iteration) |
| `scripts/sweeps/run_equal.sh` | `vanilla_main` with fixed reg indices `0 0` |
| `scripts/sweeps/run_abrl.sh` | `vanilla_main` grid over reg indices |
| `scripts/sweeps/run_satisfaction.sh` | `arg_main` with fixed indices `20 20` and a λ sweep |

**CPU affinities** (`--cpuset-cpus`) in these files are **machine-specific**; edit when moving clusters.
