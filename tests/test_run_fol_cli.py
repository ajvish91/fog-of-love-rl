"""CLI argument validation for `fol.training.run_fol` (no training loop)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from fol.training.run_fol import (
    _split_mul_lists,
    _validate_mul_list_elements,
    _maddpg_class,
    main,
)


def test_maddpg_class_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown training variant"):
        _maddpg_class("not_a_variant")  # type: ignore[arg-type]


def test_split_mul_lists() -> None:
    assert _split_mul_lists([14, 15, "_", 16, 17], "_") == ([14, 15], [16, 17])


def test_split_mul_lists_missing_delimiter() -> None:
    with pytest.raises(ValueError, match="not found"):
        _split_mul_lists([14, 15], "_")


def test_validate_mul_list_elements() -> None:
    _validate_mul_list_elements([14, 20], 14, 20)
    with pytest.raises(ValueError, match="not between"):
        _validate_mul_list_elements([13], 14, 20)


@pytest.mark.parametrize(
    "argv",
    [
        ["prog"],
        ["prog", "unknown_mode"],
        ["prog", "arg_main", "1", "1", "0"],
        ["prog", "arg_main", "2", "1", "0", "14", "14", "1.0"],
        ["prog", "basic_main", "1", "1", "0", "14", "14", "extra"],
        ["prog", "basic_main", "1", "1", "0", "99", "14"],
        ["prog", "basic_main", "1", "1", "12", "14", "14"],
        ["prog", "basic_main", "1", "1", "0", "14", "14"],
        ["prog", "basic_main", "1", "1", "0", "14", "13"],
        ["prog", "basic_main", "1", "1", "0", "14", "21"],
        ["prog", "vanilla_main", "1", "1", "0", "3", "0"],
        ["prog", "vanilla_main", "1", "1", "0", "0", "-1"],
        ["prog", "mul_main", "1", "1", "0", "14", "15"],
    ],
)
def test_main_exits_nonzero(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 1


def test_basic_main_rejects_abrl_one() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["prog", "basic_main", "1", "1", "0", "14", "14"])
    assert exc.value.code == 1


def test_config_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as exc:
        main(["prog", "--config", str(missing)])
    assert exc.value.code == 1
