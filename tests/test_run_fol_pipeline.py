"""Short end-to-end runs: debug=1 plus FOL_TEST_MINIMAL (see fol_training.run_fol)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from fol_train_config import argv_suffix_from_yaml
from fol_training.run_fol import main


@pytest.mark.parametrize(
    "rel_cfg",
    [
        "configs/smoke_arg_localized.yaml",
        "configs/smoke_basic_localized.yaml",
        "configs/smoke_vanilla_global.yaml",
        "configs/smoke_mul_multi_affinity.yaml",
    ],
)
def test_main_with_smoke_yaml(repo_root: Path, minimal_training_env: Path, rel_cfg: str) -> None:
    cfg = repo_root / rel_cfg
    assert cfg.is_file()
    main(["__main__", "--config", str(cfg)])


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["arg_main", "1", "1", "0", "14", "15", "2.5"],
        ["arg_main", "1", "0", "0", "20", "20", "0.0"],
        ["basic_main", "1", "0", "0", "16", "17"],
        ["vanilla_main", "1", "0", "0", "1", "2"],
        ["vanilla_main", "1", "1", "0", "0", "2"],
        ["mul_main", "1", "1", "0", "14", "15", "_", "16", "18"],
        ["mul_main", "1", "0", "11", "14", "_", "20"],
    ],
)
def test_main_direct_argv(
    repo_root: Path, minimal_training_env: Path, argv_tail: list[str]
) -> None:
    main(["__main__", *argv_tail])


def test_yaml_roundtrip_matches_direct_argv(repo_root: Path, minimal_training_env: Path) -> None:
    cfg = repo_root / "configs/smoke_arg_localized.yaml"
    suffix = argv_suffix_from_yaml(cfg)
    main(["__main__", *suffix])
