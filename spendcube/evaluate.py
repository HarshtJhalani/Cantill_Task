"""Score an output file against a labels file.

python -m spendcube.evaluate --config configs/company_A.yaml --pred output/x__A.csv
Category accuracy per level (line-weighted and spend-weighted), supplier/item grouping
(adjusted Rand, homogeneity, completeness), and include/exclusion agreement when labelled.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from .config import load_config
from .pipeline import load_labels


def grouping(pred, true):
    m = true.notna() & (true.astype(str) != "")
    h, c, v = homogeneity_completeness_v_measure(true[m].astype(str), pred[m].astype(str))
    return {"n": int(m.sum()), "ARI": round(adjusted_rand_score(true[m].astype(str), pred[m].astype(str)), 4),
            "homogeneity": round(h, 4), "completeness": round(c, 4), "pred_groups": pred[m].nunique(),
            "true_groups": true[m].nunique()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--only", default=None, help="evaluate only record_ids in this file (one per line)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lab = load_labels(cfg)
    lc = cfg["labels"]
    p = pd.read_csv(a.pred, dtype=str, keep_default_na=False)
    p["spend"] = p["spend"].astype(float)
    lab = lab.rename(columns={c: "t_" + c for c in lab.columns if c != "record_id"})
    d = p.merge(lab, on="record_id", how="inner")
    if a.only:
        ids = set(open(a.only).read().split())
        d = d[d.record_id.isin(ids)]
    print(f"labelled rows evaluated: {len(d)}")
    if lc.get("include"):
        ti = d["t_" + lc["include"]].astype(str).str.strip()
        acc = (d["include"].astype(str) == ti).mean()
        print(f"include agreement: {acc:.4f}")
        print(pd.crosstab(d["exclusion_reason"].replace("", "-"), d["t_exclusion_reason"].fillna("-"), rownames=["pred"], colnames=["label"]))
    cats = lc.get("category") or []
    inc = d[d["include"] == "1"]
    if lc.get("include"):
        inc = inc[inc["t_" + lc["include"]].astype(str).str.strip() == "1"]
    w = inc["spend"].abs()
    for k, c in enumerate(cats):
        ok = inc[f"l{k + 1}"].fillna("") == inc["t_" + c].fillna("")
        print(f"L{k + 1}: line acc {ok.mean():.4f}   spend-weighted {(w * ok).sum() / max(w.sum(), 1e-9):.4f}")
    if lc.get("supplier_key"):
        print("supplier grouping:", grouping(d["supplier_key"], d["t_" + lc["supplier_key"]]))
    if lc.get("item_key"):
        print("item grouping:", grouping(d["item_key"], d["t_" + lc["item_key"]]))


if __name__ == "__main__":
    main()
