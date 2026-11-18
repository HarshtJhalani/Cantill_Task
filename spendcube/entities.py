"""Supplier and item de-duplication (union-find over ids, exact normalised names and fuzzy names)."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .textnorm import compact, is_generic, ocr_fold, supplier_core


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _h(s: str, prefix: str) -> str:
    return prefix + hashlib.md5(s.encode()).hexdigest()[:8]


def _fuzzy_union(uf: UF, keys: list[str], threshold: float, min_len: int = 5, block: int = 2):
    """Union keys whose strings are near-identical. Blocks on the first `block` chars
    (and on the OCR-folded form) to keep it fast and conservative."""
    keys = sorted(set(k for k in keys if k))
    by = defaultdict(list)
    for k in keys:
        by[k[:block]].append(k)
    for grp in by.values():
        if len(grp) < 2:
            continue
        sc = process.cdist(grp, grp, scorer=fuzz.ratio, workers=-1)
        ii, jj = np.where(np.triu(sc >= threshold, 1))
        for i, j in zip(ii, jj):
            a, b = grp[i], grp[j]
            if min(len(a), len(b)) >= min_len:
                uf.union("n:" + a, "n:" + b)


def supplier_keys(df: pd.DataFrame, cfg: dict) -> pd.Series:
    scfg = cfg["suppliers"]
    suf = [s.lower() for s in scfg["legal_suffixes"]]
    uf = UF()
    core = df["supplier_name"].fillna("").map(lambda s: supplier_core(s, suf))
    core_ocr = core.map(ocr_fold)
    sid = df["supplier_id"].fillna("").astype(str)
    nodes = []
    for c, co, i in zip(core, core_ocr, sid):
        a = ("n:" + co) if co else None
        b = ("i:" + i) if i else None
        if a and b:
            uf.union(a, b)                 # a vendor id and the name typed on it are one supplier
        nodes.append(a or b or "unknown")
        if a:
            uf.find(a)
    _fuzzy_union(uf, list(set(core_ocr)), scfg["fuzzy_threshold"])
    roots = [uf.find(n) if n != "unknown" else "unknown" for n in nodes]
    # readable key: the most common raw name (or id) in the cluster
    tmp = pd.DataFrame({"root": roots, "name": df["supplier_name"].fillna(""), "id": sid})
    tmp["label"] = np.where(tmp["name"] != "", tmp["name"].str.upper().str.replace(r"\s+", " ", regex=True), tmp["id"])
    lab = tmp.groupby("root")["label"].agg(lambda x: x.value_counts().index[0])
    return tmp["root"].map(lab).fillna("UNKNOWN SUPPLIER").rename("supplier_key")


def item_keys(df: pd.DataFrame, cfg: dict, supplier_key: pd.Series, ocr_sources=()) -> pd.Series:
    """One key per real item. Material numbers and normalised descriptions are linked:
    the same text under two codes, or one code with OCR-garbled text, collapse together.
    Generic/blank descriptions without a code fall back to supplier + context so that
    they are not all merged into one giant 'blank' item."""
    thr = cfg["items"]["fuzzy_threshold"]
    uf = UF()
    nodes = []
    desc = df["description"].fillna("")
    mat = df["material"].fillna("").astype(str)
    ocr = df["source"].isin(list(ocr_sources))
    texts = [ocr_fold(d) for d in desc]          # always OCR-fold so ERP and OCR text meet
    generic = desc.map(is_generic)
    for i in range(len(df)):
        t = texts[i]
        m = mat.iat[i]
        tn = None if generic.iat[i] or len(t) < 4 else "t:" + t
        mn = "m:" + m if m else None
        if tn and mn:
            uf.union(tn, mn)
        if tn or mn:
            nodes.append(tn or mn)
        else:
            nodes.append("g:" + str(supplier_key.iat[i]) + "|" + compact(df["context"].iat[i]) + "|" + compact(desc.iat[i]))
        uf.find(nodes[-1])
    # fuzzy-link OCR'd texts to clean texts (only texts from OCR sources need it)
    clean = sorted({texts[i] for i in range(len(df)) if not ocr.iat[i] and not generic.iat[i] and len(texts[i]) >= 6})
    noisy = sorted({texts[i] for i in range(len(df)) if ocr.iat[i] and not generic.iat[i] and len(texts[i]) >= 6} - set(clean))
    if clean and noisy:
        res = process.cdist(noisy, clean, scorer=fuzz.ratio, workers=-1, score_cutoff=thr)
        best = res.argmax(axis=1)
        for i, j in enumerate(best):
            if res[i, j] >= thr:
                uf.union("t:" + noisy[i], "t:" + clean[j])
    roots = [uf.find(n) for n in nodes]
    tmp = pd.DataFrame({"root": roots, "d": desc.str.upper().str.replace(r"\s+", " ", regex=True).str.strip()})
    lab = tmp.groupby("root")["d"].agg(lambda x: x.value_counts().index[0])
    keys = tmp["root"].map(lab)
    # keep keys unique per cluster even if two clusters share a display text
    keys = keys.str[:60] + "#" + tmp["root"].map(lambda r: hashlib.md5(r.encode()).hexdigest()[:6])
    return keys.rename("item_key")
