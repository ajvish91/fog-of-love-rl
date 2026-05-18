"""Load and validate YAML training configs for FoL experiment drivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _as_bool(key: str, value: Any, path: Path) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{path}: {key} must be a boolean, got {type(value).__name__}")


def _as_int_in_range(key: str, value: Any, path: Path, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}: {key} must be an integer, got {value!r}")
    if not lo <= value <= hi:
        raise ValueError(f"{path}: {key} must be between {lo} and {hi}, got {value}")
    return value


def _as_float(key: str, value: Any, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}: {key} must be a number, got {value!r}")
    return float(value)


def _as_gpu_string(key: str, value: Any, path: Path, *, max_index: int) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        s = str(value)
    elif isinstance(value, str) and value.isdigit():
        s = value
    else:
        raise TypeError(f"{path}: {key} must be a non-negative integer or digit string, got {value!r}")
    iv = int(s)
    if iv < 0 or iv > max_index:
        raise ValueError(f"{path}: {key} must be between 0 and {max_index}, got {iv}")
    return s


def _as_int_list(key: str, value: Any, path: Path, lo: int, hi: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{path}: {key} must be a non-empty list of integers")
    out: list[int] = []
    for i, x in enumerate(value):
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"{path}: {key}[{i}] must be an integer, got {x!r}")
        if not lo <= x <= hi:
            raise ValueError(f"{path}: {key}[{i}] must be between {lo} and {hi}, got {x}")
        out.append(x)
    return out


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping, got {type(data).__name__}")
    return data


def parse_arg_localized(data: dict[str, Any], path: Path) -> tuple[bool, bool, str, int, int, float]:
    if data.get("recipe") != "arg_localized":
        raise ValueError(f"{path}: expected recipe: arg_localized, got {data.get('recipe')!r}")
    debug = _as_bool("debug", data["debug"], path)
    abrl = _as_bool("abrl", data["abrl"], path)
    gpu = _as_gpu_string("gpu", data["gpu"], path, max_index=11)
    aff_1 = _as_int_in_range("aff_idx_player_1", data["aff_idx_player_1"], path, 14, 20)
    aff_2 = _as_int_in_range("aff_idx_player_2", data["aff_idx_player_2"], path, 14, 20)
    lambda_ = _as_float("lambda", data["lambda"], path)
    return debug, abrl, gpu, aff_1, aff_2, lambda_


def parse_basic_localized(data: dict[str, Any], path: Path) -> tuple[bool, bool, str, int, int]:
    if data.get("recipe") != "basic_localized":
        raise ValueError(f"{path}: expected recipe: basic_localized, got {data.get('recipe')!r}")
    debug = _as_bool("debug", data["debug"], path)
    abrl = _as_bool("abrl", data["abrl"], path)
    if abrl:
        raise ValueError(
            f"{path}: basic_main with abrl=true is not supported in the unified driver "
            "(stock MADDPG is constructed without affinity arguments). "
            "Use recipe arg_localized for localized ABRL, or set abrl: false."
        )
    gpu = _as_gpu_string("gpu", data["gpu"], path, max_index=11)
    aff_1 = _as_int_in_range("aff_idx_player_1", data["aff_idx_player_1"], path, 14, 20)
    aff_2 = _as_int_in_range("aff_idx_player_2", data["aff_idx_player_2"], path, 14, 20)
    return debug, abrl, gpu, aff_1, aff_2


def _vanilla_reg_slot(data: dict[str, Any], path: Path, player: int) -> int:
    """README / CLI use reg_idx_*; YAML may use aff_slot_* or reg_idx_*."""
    aff_key, reg_key = f"aff_slot_{player}", f"reg_idx_{player}"
    if aff_key in data and reg_key in data:
        raise ValueError(f"{path}: specify only one of {aff_key!r} or {reg_key!r} for player {player}")
    if aff_key in data:
        return _as_int_in_range(aff_key, data[aff_key], path, 0, 2)
    if reg_key in data:
        return _as_int_in_range(reg_key, data[reg_key], path, 0, 2)
    raise ValueError(f"{path}: missing {aff_key!r} or {reg_key!r} for vanilla_global")


def parse_vanilla_global(data: dict[str, Any], path: Path) -> tuple[bool, bool, str, list[int], list[int]]:
    if data.get("recipe") != "vanilla_global":
        raise ValueError(f"{path}: expected recipe: vanilla_global, got {data.get('recipe')!r}")
    debug = _as_bool("debug", data["debug"], path)
    abrl = _as_bool("abrl", data["abrl"], path)
    gpu = _as_gpu_string("gpu", data["gpu"], path, max_index=11)
    s1 = _vanilla_reg_slot(data, path, 1)
    s2 = _vanilla_reg_slot(data, path, 2)
    aff_idx_1 = [s1, 1 / 3]
    aff_idx_2 = [s2, 1 / 3]
    return debug, abrl, gpu, aff_idx_1, aff_idx_2


def parse_mul_multi_affinity(data: dict[str, Any], path: Path) -> tuple[bool, bool, str, list[int], list[int]]:
    if data.get("recipe") != "mul_multi_affinity":
        raise ValueError(f"{path}: expected recipe: mul_multi_affinity, got {data.get('recipe')!r}")
    debug = _as_bool("debug", data["debug"], path)
    abrl = _as_bool("abrl", data["abrl"], path)
    gpu = _as_gpu_string("gpu", data["gpu"], path, max_index=12)
    aff_1 = _as_int_list("aff_indices_player_1", data["aff_indices_player_1"], path, 14, 20)
    aff_2 = _as_int_list("aff_indices_player_2", data["aff_indices_player_2"], path, 14, 20)
    return debug, abrl, gpu, aff_1, aff_2


def _b01(x: bool) -> str:
    return "1" if x else "0"


def argv_suffix_from_yaml(path: Path) -> list[str]:
    """Build `sys.argv[1:]` for `fol_training.run_fol.main` from a YAML file."""
    path = path.resolve()
    raw = load_yaml_mapping(path)
    recipe = raw.get("recipe")
    if recipe == "arg_localized":
        d, a, gpu, a1, a2, lam = parse_arg_localized(raw, path)
        return ["arg_main", _b01(d), _b01(a), gpu, str(a1), str(a2), str(lam)]
    if recipe == "basic_localized":
        d, a, gpu, a1, a2 = parse_basic_localized(raw, path)
        return ["basic_main", _b01(d), _b01(a), gpu, str(a1), str(a2)]
    if recipe == "vanilla_global":
        d, a, gpu, aff1, aff2 = parse_vanilla_global(raw, path)
        return ["vanilla_main", _b01(d), _b01(a), gpu, str(aff1[0]), str(aff2[0])]
    if recipe == "mul_multi_affinity":
        d, a, gpu, lst1, lst2 = parse_mul_multi_affinity(raw, path)
        tokens = ["mul_main", _b01(d), _b01(a), gpu]
        tokens.extend(str(x) for x in lst1)
        tokens.append("_")
        tokens.extend(str(x) for x in lst2)
        return tokens
    raise ValueError(f"{path}: unknown recipe {recipe!r}")
