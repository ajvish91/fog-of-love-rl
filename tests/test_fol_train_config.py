"""YAML loader and argv expansion for unified training configs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from fol.train_config import (
    argv_suffix_from_yaml,
    load_yaml_mapping,
    parse_arg_localized,
    parse_basic_localized,
    parse_mul_multi_affinity,
    parse_vanilla_global,
)


def test_argv_suffix_all_smoke_configs(repo_root: Path) -> None:
    configs = sorted((repo_root / "configs").glob("smoke_*.yaml"))
    assert configs, "expected configs/smoke_*.yaml"
    for path in configs:
        suffix = argv_suffix_from_yaml(path)
        assert suffix[0] in ("arg_main", "basic_main", "vanilla_main", "mul_main")
        assert suffix[1] in ("0", "1")
        assert suffix[2] in ("0", "1")


def test_parse_arg_localized_gpu_string_and_bounds(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: true
            abrl: true
            gpu: "11"
            aff_idx_player_1: 20
            aff_idx_player_2: 14
            lambda: 0.5
            """
        ).strip(),
        encoding="utf-8",
    )
    d, a, gpu, a1, a2, lam = parse_arg_localized(load_yaml_mapping(p), p)
    assert d is True and a is True and gpu == "11" and a1 == 20 and a2 == 14 and lam == 0.5
    assert argv_suffix_from_yaml(p) == ["arg_main", "1", "1", "11", "20", "14", "0.5"]


def test_parse_basic_rejects_abrl_true(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "recipe": "basic_localized",
                "debug": True,
                "abrl": True,
                "gpu": 0,
                "aff_idx_player_1": 14,
                "aff_idx_player_2": 15,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="basic_main with abrl=true"):
        parse_basic_localized(load_yaml_mapping(p), p)


def test_vanilla_aff_slot_alias(tmp_path: Path) -> None:
    p = tmp_path / "v.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: vanilla_global
            debug: false
            abrl: true
            gpu: 1
            aff_slot_1: 2
            aff_slot_2: 1
            """
        ).strip(),
        encoding="utf-8",
    )
    d, a, gpu, aff1, aff2 = parse_vanilla_global(load_yaml_mapping(p), p)
    assert d is False and a is True and gpu == "1"
    assert aff1[0] == 2 and aff2[0] == 1
    assert argv_suffix_from_yaml(p) == ["vanilla_main", "0", "1", "1", "2", "1"]


def test_vanilla_rejects_both_aff_slot_and_reg_idx(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: vanilla_global
            debug: true
            abrl: false
            gpu: 0
            aff_slot_1: 0
            reg_idx_1: 1
            reg_idx_2: 0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only one of"):
        parse_vanilla_global(load_yaml_mapping(p), p)


def test_mul_argv_includes_delimiter(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: mul_multi_affinity
            debug: true
            abrl: true
            gpu: 12
            aff_indices_player_1: [14, 16]
            aff_indices_player_2: [15, 17, 18]
            """
        ).strip(),
        encoding="utf-8",
    )
    assert argv_suffix_from_yaml(p) == [
        "mul_main",
        "1",
        "1",
        "12",
        "14",
        "16",
        "_",
        "15",
        "17",
        "18",
    ]


def test_debug_must_be_boolean_not_integer(tmp_path: Path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: 1
            abrl: true
            gpu: 0
            aff_idx_player_1: 14
            aff_idx_player_2: 14
            lambda: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="must be a boolean"):
        argv_suffix_from_yaml(p)


def test_unknown_recipe_other(tmp_path: Path) -> None:
    p = tmp_path / "o.yaml"
    p.write_text("recipe: other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown recipe"):
        argv_suffix_from_yaml(p)


def test_invalid_debug_string(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: notbool
            abrl: true
            gpu: 0
            aff_idx_player_1: 14
            aff_idx_player_2: 14
            lambda: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="must be a boolean"):
        argv_suffix_from_yaml(p)


def test_gpu_99_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "g99.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: true
            abrl: true
            gpu: 99
            aff_idx_player_1: 14
            aff_idx_player_2: 14
            lambda: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 0 and 11"):
        argv_suffix_from_yaml(p)


def test_unknown_recipe_argv(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("recipe: not_a_recipe\ndebug: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown recipe"):
        argv_suffix_from_yaml(p)


def test_gpu_out_of_range_arg(tmp_path: Path) -> None:
    p = tmp_path / "g.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: true
            abrl: true
            gpu: 12
            aff_idx_player_1: 14
            aff_idx_player_2: 14
            lambda: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 0 and 11"):
        argv_suffix_from_yaml(p)


def test_affinity_below_range(tmp_path: Path) -> None:
    p = tmp_path / "a.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: arg_localized
            debug: true
            abrl: true
            gpu: 0
            aff_idx_player_1: 13
            aff_idx_player_2: 14
            lambda: 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 14 and 20"):
        argv_suffix_from_yaml(p)


def test_mul_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "m2.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: mul_multi_affinity
            debug: true
            abrl: true
            gpu: 0
            aff_indices_player_1: []
            aff_indices_player_2: [14]
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="non-empty list"):
        argv_suffix_from_yaml(p)


def test_yaml_root_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml_mapping(p)


def test_parse_mul_gpu_12_allowed(tmp_path: Path) -> None:
    p = tmp_path / "m3.yaml"
    p.write_text(
        textwrap.dedent(
            """
            recipe: mul_multi_affinity
            debug: true
            abrl: false
            gpu: 12
            aff_indices_player_1: [20]
            aff_indices_player_2: [14]
            """
        ).strip(),
        encoding="utf-8",
    )
    d, a, gpu, l1, l2 = parse_mul_multi_affinity(load_yaml_mapping(p), p)
    assert gpu == "12" and l1 == [20] and l2 == [14] and a is False
