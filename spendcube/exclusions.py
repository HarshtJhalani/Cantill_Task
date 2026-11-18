"""Decide what counts as spend. Precedence: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def status_reason(status: pd.Series, cfg: dict) -> pd.Series:
    sc = cfg["status"]
    out = pd.Series("", index=status.index, dtype=object)
    has = status.notna() & (status.astype(str).str.strip() != "")
    s = status.fillna("").astype(str).str.strip().str.lower()
    decided = ~has  # no status column / blank status: nothing to say
    for reason in ("CANCELLED", "NOT_APPROVED", "INCLUDE"):
        pat = "|".join(f"(?:{p})" for p in sc.get(reason, []))
        if not pat:
            continue
        hit = ~decided & s.str.contains(pat, regex=True, flags=re.I)
        out[hit] = "" if reason == "INCLUDE" else reason
        decided |= hit
    out[~decided] = sc.get("unknown_status", "NOT_APPROVED")
    return out


def rule_hits(df: pd.DataFrame, cfg: dict):
    """Returns (reason Series, rule-name Series, vendor-propagation reason Series)."""
    reason = pd.Series("", index=df.index, dtype=object)
    rule = pd.Series("", index=df.index, dtype=object)
    prop = pd.Series(False, index=df.index)
    for r in cfg.get("exclusion_rules", []):
        hit = pd.Series(False, index=df.index)
        rx = re.compile(r["pattern"], re.I)
        for f in r["fields"]:
            if f in df:
                hit |= df[f].fillna("").astype(str).map(lambda x: bool(rx.search(x)))
        new = hit & (reason == "")
        reason[new] = r["reason"]
        rule[new] = r["name"]
        if r.get("propagate_vendor"):
            prop |= hit

    # vendor propagation: a supplier booked mostly to intercompany accounts is intercompany
    vp = cfg["vendor_propagation"]
    key = df["supplier_key"]
    share = prop.groupby(key).transform("mean")
    cnt = prop.groupby(key).transform("size")
    pv = (share >= vp["min_share"]) & (cnt >= vp["min_lines"]) & (reason == "") & (key != "UNKNOWN SUPPLIER")
    reason[pv] = "NON_PROCUREMENT"
    rule[pv] = "intercompany_vendor"
    return reason, rule


def label_vendor_reason(df: pd.DataFrame, labels: pd.DataFrame | None, cfg: dict) -> pd.Series:
    """Learn vendor-level exclusions from labelled include/exclusion_reason, if any."""
    out = pd.Series("", index=df.index, dtype=object)
    if labels is None or "exclusion_reason" not in labels:
        return out
    lv = cfg["label_vendor_exclusion"]
    lab = df[["record_id", "supplier_key"]].merge(labels[["record_id", "exclusion_reason"]], on="record_id")
    lab["exc"] = lab["exclusion_reason"].fillna("").astype(str).str.strip()
    g = lab.groupby("supplier_key")["exc"]
    stats = pd.DataFrame({
        "n": g.size(),
        "share": g.apply(lambda x: (x != "").mean()),
        "top": g.apply(lambda x: x[x != ""].value_counts().index[0] if (x != "").any() else ""),
    })
    bad = stats[(stats["n"] >= lv["min_lines"]) & (stats["share"] >= lv["min_share"])]
    m = df["supplier_key"].map(bad["top"]).fillna("")
    out[:] = m.values
    return out


def decide(df: pd.DataFrame, cfg: dict, dup_of: pd.Series, labels: pd.DataFrame | None):
    st = status_reason(df["status"], cfg)
    rr, rule = rule_hits(df, cfg)
    lv = label_vendor_reason(df, labels, cfg)
    reason = st.copy()
    detail = pd.Series(np.where(st != "", "status:" + df["status"].fillna("").astype(str), ""), index=df.index, dtype=object)
    m = (reason == "") & (rr != "")
    reason[m] = rr[m]
    detail[m] = "rule:" + rule[m]
    m = (reason == "") & (lv != "")
    reason[m] = lv[m]
    detail[m] = "rule:label_vendor"
    m = (reason == "") & dup_of.notna()
    reason[m] = "DUPLICATE"
    detail[m] = "dup_of:" + dup_of[m].astype(str)
    log.info("exclusions: %s", reason.replace("", "INCLUDED").value_counts().to_dict())
    return reason, detail
