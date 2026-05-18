#!/usr/bin/env python3
"""FoL smoke checks and short training runs (used by CI and scripts/fol_smoke.sh)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _ensure_fol_on_path() -> None:
    scripts = str(_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _train_env() -> dict[str, str]:
    env = os.environ.copy()
    scripts = str(_scripts_dir())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = scripts if not prev else f"{scripts}{os.pathsep}{prev}"
    return env


def _print_train_setup(python: str) -> None:
    root = _repo_root()
    print(
        f"error: training deps missing in {python}.\n\n"
        "Create a venv in the repo root and install dependencies:\n\n"
        f"  cd {root}\n"
        "  ./scripts/setup_venv.sh\n"
        "  source .venv/bin/activate\n"
        "  ./scripts/fol_smoke.sh train-only\n",
        file=sys.stderr,
    )


def _require_train_imports() -> bool:
    try:
        import matplotlib  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def cmd_check() -> int:
    root = _repo_root()
    configs = sorted((root / "configs").glob("smoke_*.yaml"))
    if not configs:
        print("error: no configs/smoke_*.yaml", file=sys.stderr)
        return 1

    import yaml

    for path in configs:
        with path.open(encoding="utf-8") as f:
            yaml.safe_load(f)
        print("ok parse:", path.relative_to(root))

    os.chdir(root)
    _ensure_fol_on_path()
    from fol.train_config import argv_suffix_from_yaml

    for path in configs:
        suffix = argv_suffix_from_yaml(path)
        print(path.name, "->", " ".join(suffix))
    print("ok:", len(configs), "YAML smoke recipes")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    root = _repo_root()
    cfg = (root / args.config).resolve()
    if not cfg.is_file():
        print(f"error: config not found: {cfg}", file=sys.stderr)
        return 1
    if not _require_train_imports():
        _print_train_setup(sys.executable)
        return 1
    (root / "plots").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(root)
    argv = [sys.executable, "-m", "fol.training.run_fol", "--config", str(cfg.relative_to(root))]
    print("run:", " ".join(argv))
    return subprocess.call(argv, env=_train_env())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Parse smoke YAML and validate argv expansion")

    t = sub.add_parser("train", help="Run one short training job via --config")
    t.add_argument(
        "--config",
        default="configs/smoke_arg_localized.yaml",
        help="YAML recipe path relative to repo root (default: %(default)s)",
    )

    ns = p.parse_args()
    if ns.command == "check":
        return cmd_check()
    if ns.command == "train":
        return cmd_train(ns)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
