"""Config loading: company YAML deep-merged over configs/defaults.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"


def _merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def load_config(path: str | Path, data_dir: str | Path | None = None) -> dict:
    path = Path(path).resolve()
    with open(DEFAULTS, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(path, encoding="utf-8") as f:
        comp = yaml.safe_load(f)
    # rule lists are additive: company rules run in addition to the defaults
    extra_rules = comp.pop("extra_exclusion_rules", []) or []
    cfg = _merge(base, comp)
    cfg["exclusion_rules"] = list(cfg.get("exclusion_rules") or []) + extra_rules

    if data_dir is not None:
        root = Path(data_dir).resolve()
    else:
        root = (path.parent / cfg.get("data_dir", ".")).resolve()
    cfg["_root"] = root

    def res(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    cfg["taxonomy"]["file"] = res(cfg["taxonomy"]["file"])
    for s in cfg["sources"]:
        s["file"] = res(s["file"])
    if cfg.get("fx"):
        cfg["fx"]["file"] = res(cfg["fx"]["file"])
    if cfg.get("labels"):
        lf = res(cfg["labels"]["file"])
        cfg["labels"]["file"] = lf
    return cfg
