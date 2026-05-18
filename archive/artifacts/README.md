# Local analysis artifacts (not in Git)

Place exported run data here when using `archive/notebooks/get_history.ipynb`:

- `pareto_history.json` — W&B or custom export of test trajectories / metrics
- `player_1_obs.pkl`, `player_2_obs.pkl`, `player_1_act.pkl`, `player_2_act.pkl` — optional notebook outputs

These files are listed in the root `.gitignore`. Do not commit API keys or private W&B exports unless you intend to publish them.
