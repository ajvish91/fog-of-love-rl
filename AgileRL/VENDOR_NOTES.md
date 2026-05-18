# AgileRL vendored snapshot (FoL repository)

This directory is a **modified, vendored** copy of [AgileRL](https://github.com/AgileRL/AgileRL) used by the Fog of Love RL code. It is **not** a Git submodule in this layout (Option A — see `docs/agilerl-layout.md` in the parent repository).

## Upstream reference (recorded when nested `.git` was removed)

| Field | Value |
|-------|--------|
| Recorded upstream remote | `https://github.com/AgileRL/AgileRL.git` |
| Recorded upstream `HEAD` commit | `24e7af455673c45bd90c9abaff48961392e181d9` |

Your **local changes** (for example under `agilerl/algorithms/`) are not listed here; keep this file updated if you re-sync from upstream.

## Pruned paths (non-runtime)

To keep the repository smaller, the following **upstream-only** trees were removed from this snapshot. They are not required for `pip install -e .` or for the FoL training drivers, which import the **`agilerl`** package only:

- `tutorials/` (also excluded from the Poetry package list in `pyproject.toml`)
- `demos/`
- `benchmarking/`
- `docs/` (Sphinx sources for docs.agilerl.com)

The **`tests/`** tree is retained so you can still run AgileRL’s test suite locally if needed.

## License

See **`LICENSE`** in this directory (Apache-2.0). The parent FoL repository **`NOTICE`** / **`README.md`** describe attribution for this vendored dependency.
