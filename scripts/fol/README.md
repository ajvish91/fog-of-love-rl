# `fol` Python package

Importable code for the Fog of Love RL project. **`setup_venv.sh`** adds `scripts/` to the venv path so you can run:

```bash
python -m fol.training.run_fol arg_main 1 1 0 14 14 5.0
```

| Module | Role |
|--------|------|
| `fol.env` | `FoLEnvironment` (Gym-style two-player game) |
| `fol.attributes` | Domain types (virtues, traits, satisfaction, …) |
| `fol.train_config` | YAML recipe validation → CLI argv |
| `fol.training.run_fol` | Training driver (`arg_main`, `basic_main`, …) |
| `fol.training.affinity_condition` | Localized ABRL prior gate |

Docker sets `PYTHONPATH=/app/scripts`. For a manual shell: `export PYTHONPATH="${PWD}/scripts"`.
