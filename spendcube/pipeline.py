"""End-to-end: ingest -> clean -> count once -> exclusions -> suppliers/items -> classify -> output."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classify import classify
from .config import load_config
from .entities import item_keys, supplier_keys
from .exclusions import decide
from .ingest import load_all
from .matching import cross_system, within_source
from .taxonomy import Taxonomy

log = logging.getLogger(__name__)

OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def load_labels(cfg: dict) -> pd.DataFrame | None:
    lc = cfg.get("labels")
    if not lc or not Path(lc["file"]).exists():
        return None
    lab = pd.read_csv(lc["file"], dtype=str)
    lab = lab.rename(columns={lc["id"]: "record_id"})
    pref = lc.get("id_prefix")
    if pref:
        lab["record_id"] = pref + lab["record_id"]
    if lc.get("exclusion_reason") and lc["exclusion_reason"] in lab:
        lab["exclusion_reason"] = lab[lc["exclusion_reason"]]
    return lab


def run(config_path, out_path, data_dir=None, use_labels=True, diagnostics=True, holdout=0.0):
    t0 = time.time()
    cfg = load_config(config_path, data_dir)
    tax = Taxonomy(cfg)
    log.info("taxonomy: %d leaves, depth %d", len(tax.leaves), tax.depth)
    df = load_all(cfg)
    labels = load_labels(cfg) if use_labels else None
    if labels is not None and holdout > 0:
        # evaluation mode: hide a random share of labels and write their ids for scoring
        rng = np.random.default_rng(0)
        hide = rng.random(len(labels)) < holdout
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(str(out_path) + ".holdout_ids").write_text(chr(10).join(labels.loc[hide, "record_id"]))
        labels = labels[~hide].reset_index(drop=True)
        log.info("holdout: hid %d labels", hide.sum())

    # 1. de-duplicate suppliers first (matching and exclusion rules use them)
    df["supplier_key"] = supplier_keys(df, cfg)

    # 2. count once: cross-system links + repeated exports within a source
    cross = cross_system(df, cfg)
    dup_of = pd.Series(pd.NA, index=df.index, dtype="object")
    dup_of[cross.index] = cross
    within = within_source(df, cfg)
    m = dup_of.isna() & within.notna()
    dup_of[m] = within[m]

    # a secondary record that duplicates a primary takes the primary's supplier
    rid2idx = pd.Series(df.index, index=df["record_id"])
    linked = dup_of.notna()
    src_idx = rid2idx.reindex(dup_of[linked].values).values
    ok = ~pd.isna(src_idx)
    tgt = dup_of[linked].index[ok]
    src = src_idx[ok].astype(int)
    df.loc[tgt, "supplier_key"] = df.loc[src, "supplier_key"].values

    # 3. exclusions
    reason, detail = decide(df, cfg, dup_of, labels)
    df["exclusion_reason"] = reason
    df["exclusion_detail"] = detail
    df["include"] = (reason == "").astype(int)

    # 4. items
    ocr_src = [s["name"] for s in cfg["sources"] if s.get("ocr_text")]
    df["item_key"] = item_keys(df, cfg, df["supplier_key"], ocr_src)
    df.loc[tgt, "item_key"] = df.loc[src, "item_key"].values

    # 5. classify every line (excluded lines get a category too: useful for review;
    #    the output blanks categories for excluded lines)
    leaf, conf, method = classify(df, cfg, tax, labels, use_labels=use_labels)
    # linked duplicates take their primary's category
    leaf[tgt] = leaf[src]
    conf[tgt] = conf[src]
    df["leaf"] = leaf
    df["confidence"] = np.round(conf, 3)
    df["class_method"] = method
    paths = np.array([tax.path_columns(i, 4) for i in leaf], dtype=object)
    for k in range(4):
        df[f"l{k + 1}"] = paths[:, k]

    df["spend"] = np.where(df["include"] == 1, df["spend_calc"].fillna(0.0), 0.0).round(2)
    exc = df["include"] == 0
    for k in range(4):
        df.loc[exc, f"l{k + 1}"] = ""

    out = df[OUT_COLS].copy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    if diagnostics:
        diag = df[["record_id", "source", "include", "exclusion_reason", "exclusion_detail", "supplier_name",
                   "supplier_id", "supplier_key", "description", "item_key", "amount_raw", "currency", "fx_rate",
                   "fx_flag", "vat_rate", "spend_calc", "spend", "date", "date_flag", "l1", "l2", "l3", "l4",
                   "confidence", "class_method"]]
        diag.to_csv(out_path.with_name(out_path.stem + "_diagnostics.csv"), index=False)
    secs = time.time() - t0
    log.info("wrote %s: %d rows, included %d, spend %.2f %s, %.0fs",
             out_path, len(out), out["include"].sum(), out["spend"].sum(), cfg["reporting_currency"], secs)
    return df, cfg, tax
