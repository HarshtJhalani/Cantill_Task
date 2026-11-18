"""Count once: link secondary-system records to primary-system records, and find
records exported twice inside one system."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .textnorm import ocr_fold, supplier_core

log = logging.getLogger(__name__)


def _amount_close(a, b, tol):
    return np.abs(a - b) <= np.maximum(tol * np.maximum(np.abs(a), np.abs(b)), 0.02)


def cross_system(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return, for each secondary record, the record_id of the primary record it duplicates (or NA).

    Candidate primaries come from two blocks: same PO number, or same net amount (to the cent,
    +-tolerance). A candidate is accepted when the evidence agrees:
      * same PO and amount within tolerance and text not contradictory, or
      * same PO, text near-identical and amount within the looser PO tolerance
        (OCR'd quantities / rounding), or
      * no usable PO but amount within tolerance and both text and supplier agree.
    Several secondary lines may point to one primary (an invoice exported twice)."""
    m = cfg["matching"]
    tol, ptol = m["amount_tolerance"], m["po_amount_tolerance"]
    tmin, smin = m["min_text_similarity"], m["min_supplier_similarity"]
    suf = [s.lower() for s in cfg["suppliers"]["legal_suffixes"]]

    prim = df[df["role"] == "primary"].copy()
    sec = df[df["role"] == "secondary"].copy()
    result = pd.Series(pd.NA, index=sec.index, dtype="object")
    if prim.empty or sec.empty:
        return result

    for f in (prim, sec):
        f["_t"] = f["description"].map(ocr_fold)
        f["_s"] = f["supplier_name"].map(lambda s: ocr_fold(supplier_core(s, suf)))
        f["_a"] = f["spend_calc"].astype(float)
        f["_c"] = (f["_a"] * 100).round().astype("Int64")

    by_po = {k: g.index.values for k, g in prim.groupby("po") if pd.notna(k)}
    P = prim

    # plain arrays: per-row pandas access is far too slow here
    pos = {j: k for k, j in enumerate(P.index)}
    p_po = P["po"].astype(object).where(P["po"].notna(), None).tolist()
    p_pl = P["po_line"].values
    p_t, p_s, p_a = P["_t"].tolist(), P["_s"].tolist(), P["_a"].values
    p_id = P["record_id"].tolist()
    a_sorted_idx = np.argsort(p_a)
    a_sorted = p_a[a_sorted_idx]
    by_po_pos = {k: [pos[j] for j in v] for k, v in by_po.items()}

    for i, r_po, r_pl, r_t, r_s, a in zip(sec.index, sec["po"].astype(object).where(sec["po"].notna(), None),
                                          sec["po_line"].values, sec["_t"], sec["_s"], sec["_a"].values):
        if pd.isna(a):
            continue
        cands = set(by_po_pos.get(r_po, [])) if r_po else set()
        d = max(tol * abs(a), 0.02)
        lo, hi = np.searchsorted(a_sorted, a - d), np.searchsorted(a_sorted, a + d, side="right")
        if hi - lo <= 500:
            cands.update(a_sorted_idx[lo:hi].tolist())
        best, bsc = None, -1.0
        for j in cands:
            same_po = r_po is not None and r_po == p_po[j]
            ts = fuzz.ratio(r_t, p_t[j]) if r_t and p_t[j] else 0.0
            ss = 0.0
            if r_s and p_s[j]:
                ss = fuzz.ratio(r_s, p_s[j])
                if min(len(r_s), len(p_s[j])) >= 5:
                    ss = max(ss, fuzz.partial_ratio(r_s, p_s[j]))
            pa = p_a[j]
            close = abs(a - pa) <= max(tol * max(abs(a), abs(pa)), 0.02)
            loose = abs(a - pa) <= max(ptol * max(abs(a), abs(pa)), 0.02)
            line_ok = pd.isna(r_pl) or pd.isna(p_pl[j]) or r_pl == p_pl[j]
            ok = ((same_po and close and line_ok and (ts >= 40 or ss >= smin))
                  or (same_po and loose and line_ok and ts >= 85)
                  or (close and ts >= tmin and ss >= smin)
                  or (close and ts >= 92 and ss >= 50))
            if not ok:
                continue
            rel = abs(a - pa) / max(abs(a), abs(pa), 1)
            sc = ts + 0.5 * ss + (40 if same_po else 0) + (10 if line_ok and same_po else 0) - 200 * rel
            if sc > bsc:
                best, bsc = j, sc
        if best is not None:
            result.at[i] = p_id[best]
    log.info("cross-system: %d of %d secondary records matched to primary", result.notna().sum(), len(sec))
    return result


def within_source(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Records exported twice within one system. Returns the record_id of the first copy for
    each later copy (NA for originals). Keys: normalised invoice number + supplier + amount +
    description, or the configured key list."""
    dcfg = cfg.get("duplicates") or {}
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    suf = [s.lower() for s in cfg["suppliers"]["legal_suffixes"]]
    for src, g in df.groupby("source"):
        if g["invoice_number"].isna().all():
            continue
        k = pd.DataFrame(index=g.index)
        if dcfg.get("keys"):
            for c in dcfg["keys"]:
                k[c] = g[c].astype(str) if c in g else ""
        else:
            k["inv"] = g["invoice_number"].fillna("").map(lambda s: "".join(ch for ch in str(s) if ch.isdigit()) or ocr_fold(s))
            k["sup"] = g["supplier_name"].fillna("").map(lambda s: ocr_fold(supplier_core(s, suf)))
            k["amt"] = g["amount_raw"].round(2).astype(str)
            k["qty"] = g["quantity"].round(3).astype(str)
        k = k.astype(str)
        key = k.apply(lambda r: "|".join(r), axis=1)
        valid = g["invoice_number"].notna()
        first = pd.Series(g["record_id"].values, index=g.index).groupby(key).transform("first")
        dup = key.duplicated(keep="first") & valid
        out[dup[dup].index] = first[dup]
    log.info("within-source duplicates: %d", out.notna().sum())
    return out
