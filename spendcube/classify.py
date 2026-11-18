"""Classify every line into the company's tree, to its lowest level, with a confidence.

Signals, blended per line:
  1. supervised text model (TF-IDF word+char -> logistic regression) when labels exist,
  2. zero-shot: multilingual sentence embeddings of the line vs. descriptors built from the tree,
  3. optional LLM labels for the highest-spend unique texts (only if an API key is set),
  4. self-training: a text model fitted on the confident zero-shot/LLM labels, so lines with
     the same vocabulary inherit them,
  5. supplier smoothing: a supplier's lines are pulled towards that supplier's category mix.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .taxonomy import Taxonomy
from .textnorm import ascii_fold, is_generic

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- text building
def expand_abbrev(text: str, abbrev: dict) -> str:
    if not abbrev:
        return text
    def rep(m):
        w = m.group(0)
        return abbrev.get(w.lower(), w)
    return re.sub(r"[A-Za-z]+", rep, text)


def line_text(df: pd.DataFrame, cfg: dict, with_supplier=True) -> pd.Series:
    ab = {k.lower(): v for k, v in (cfg.get("classification", {}).get("abbreviations") or {}).items()}
    d = df["description"].fillna("")
    d = d.where(~d.map(is_generic), "")
    parts = [d.map(lambda s: expand_abbrev(s.replace(";", " ; ").replace(",", " , "), ab))]
    parts.append(df["context"].fillna(""))
    if with_supplier:
        parts.append(df["supplier_name"].fillna(""))
    t = parts[0]
    for p in parts[1:]:
        t = t + " | " + p
    return t.str.replace(r"(\s*\|\s*)+", " | ", regex=True).str.strip(" |")


def _vectorizers():
    w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b", lowercase=True, max_features=200000)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                        lowercase=True, max_features=300000)
    return w, c


class TextModel:
    def __init__(self, n_classes: int, C: float = 8.0):
        self.n = n_classes
        self.C = C

    def fit(self, texts, y, w=None):
        self.vw, self.vc = _vectorizers()
        X = sparse.hstack([self.vw.fit_transform(texts), self.vc.fit_transform(texts)]).tocsr()
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            self.const = self.classes_[0]
            return self
        self.const = None
        self.clf = LogisticRegression(C=self.C, max_iter=300, solver="saga", tol=1e-3)
        self.clf.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, texts):
        P = np.zeros((len(texts), self.n))
        if self.const is not None:
            P[:, self.const] = 1.0
            return P
        X = sparse.hstack([self.vw.transform(texts), self.vc.transform(texts)]).tocsr()
        P[:, self.clf.classes_] = self.clf.predict_proba(X)
        return P


# ---------------------------------------------------------------- embeddings
_MODEL = {}


def embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_name not in _MODEL:
        _MODEL[model_name] = SentenceTransformer(model_name)
    return _MODEL[model_name].encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)


def softmax(x, t):
    z = (x - x.max(axis=1, keepdims=True)) / t
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def lexical_scores(lines: list[str], tax: Taxonomy, cfg: dict) -> np.ndarray:
    """TF-IDF similarity between each line and a small document per leaf (names + hints).
    Word n-grams when the config gives keyword hints, character n-grams otherwise."""
    docs = [tax.leaf_text(i).lower() for i in range(len(tax.leaves))]
    if tax.hints:
        v = TfidfVectorizer(analyzer="word", sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z0-9&]+\b")
    else:
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    low = [x.lower() for x in lines]
    v.fit(docs + low)
    return (v.transform(low) @ v.transform(docs).T).toarray()


def zero_shot(texts: list[str], tax: Taxonomy, cfg: dict) -> np.ndarray:
    """Dense (multilingual embeddings) + lexical (TF-IDF vs leaf names/hints) match to the tree."""
    ccfg = cfg["classification"]
    uniq, inv = np.unique(np.array(texts, dtype=object), return_inverse=True)
    qp, pp = ccfg.get("query_prefix", ""), ccfg.get("passage_prefix", "")
    E = embed([qp + ascii_fold(u)[:300] for u in uniq], ccfg["embedding_model"])
    L = embed([pp + tax.leaf_text(i) for i in range(len(tax.leaves))], ccfg["embedding_model"])
    S = E @ L.T
    lw = ccfg.get("lexical_weight_with_hints", 1.0) if tax.hints else ccfg.get("lexical_weight", 0.3)
    if lw > 0:
        S = S + lw * lexical_scores(list(uniq), tax, cfg)
    P = softmax(S, ccfg.get("zero_shot_temperature", 0.03))
    return P[inv]


# ---------------------------------------------------------------- helpers
def supplier_smooth(P: np.ndarray, sup: pd.Series, weight: float, spend=None, min_lines: int = 3):
    if weight <= 0:
        return P
    codes, uniq = pd.factorize(sup)
    w = np.ones(len(P)) if spend is None else np.log1p(np.abs(np.nan_to_num(spend))) + 1.0
    S = np.zeros((len(uniq), P.shape[1]))
    np.add.at(S, codes, P * w[:, None])
    cnt = np.bincount(codes, minlength=len(uniq))
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-12)
    use = (cnt[codes] >= min_lines)[:, None]
    out = np.where(use, (1 - weight) * P + weight * S[codes], P)
    return out / out.sum(axis=1, keepdims=True)


def _labels_to_idx(labels: pd.DataFrame, cols, tax: Taxonomy):
    vals = labels[cols].fillna("").values.tolist()
    return np.array([(-1 if (i := tax.find_leaf(v)) is None else i) for v in vals])


# ---------------------------------------------------------------- main entry
def classify(df: pd.DataFrame, cfg: dict, tax: Taxonomy, labels: pd.DataFrame | None,
             use_labels: bool = True):
    """Returns (leaf index array, confidence array, method array)."""
    ccfg = cfg["classification"]
    n, K = len(df), len(tax.leaves)
    texts = line_text(df, cfg).tolist()
    texts_ns = line_text(df, cfg, with_supplier=False).tolist()
    method = np.array(["zero_shot"] * n, dtype=object)

    y = np.full(n, -1)
    if use_labels and labels is not None and cfg.get("labels", {}).get("category"):
        lab = df[["record_id"]].reset_index().merge(labels, on="record_id", how="inner")
        idx = _labels_to_idx(lab, cfg["labels"]["category"], tax)
        y[lab["index"].values] = idx
    have = y >= 0
    log.info("labelled lines usable for training: %d", have.sum())

    P_zs = zero_shot(texts_ns, tax, cfg)

    llm_y = np.full(n, -1)
    if ccfg.get("llm", {}).get("enabled") in (True, "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from .llm import llm_label
            llm_y = llm_label(df, texts, tax, cfg, skip=have)
            log.info("LLM labelled %d lines", (llm_y >= 0).sum())
        except Exception as e:  # never let the optional LLM step break a run
            log.warning("LLM step failed, continuing without it: %s", e)

    if have.sum() >= max(50, 2 * K) and len(np.unique(y[have])) > 1:
        # supervised: text model on labels (+ LLM labels where available)
        tr = have | (llm_y >= 0)
        ytr = np.where(have, y, llm_y)[tr]
        noisy = cfg.get("labels", {}).get("noisy", False)
        wtr = np.where(have[tr], 1.0, 0.7)
        tm = TextModel(K).fit([texts[i] for i in np.where(tr)[0]], ytr, wtr)
        P_sup = tm.predict_proba(texts)
        coverage = len(np.unique(ytr)) / K
        # the zero-shot prior matters more when labels are few/noisy or miss classes
        a = 0.15 if (not noisy and coverage > 0.9 and have.sum() > 5000) else 0.35
        P = (1 - a) * P_sup + a * P_zs
        method[:] = "supervised"
        if not noisy:
            # trusted labels are used as-is on the lines they cover
            P[have] = 0.0
            P[have, y[have]] = 1.0
            method[have] = "label"
        else:
            P[have] = 0.5 * P[have]
            P[have, y[have]] += 0.5
            method[have] = "seed_label"
    else:
        P = P_zs.copy()
        if ccfg.get("self_training", True):
            conf = P.max(axis=1)
            top = P.argmax(axis=1)
            q = np.quantile(conf, ccfg.get("pseudo_label_quantile", 0.7))
            # train on the description alone: shared codes in the context (GL account, supplier)
            # would otherwise turn one pseudo-label into a rule for thousands of unrelated lines
            dtexts = df["description"].fillna("").where(~df["description"].fillna("").map(is_generic), "").tolist()
            gen = np.array([t == "" for t in dtexts])
            sel = (conf >= q) & ~gen
            ytr = top.copy()
            ytr[llm_y >= 0] = llm_y[llm_y >= 0]
            sel = sel | (llm_y >= 0)
            w = np.where(llm_y >= 0, 2.0, conf / conf.max())
            if sel.sum() > 50 and len(np.unique(ytr[sel])) > 1:
                tm = TextModel(K, C=2.0).fit([dtexts[i] for i in np.where(sel)[0]], ytr[sel], w[sel])
                P_st = tm.predict_proba(dtexts)
                P_st[gen] = P_zs[gen]
                wst = ccfg.get("self_training_weight", 0.3)
                P = (1 - wst) * P_zs + wst * P_st
                method[:] = "zero_shot+self_train"
        if (llm_y >= 0).any():
            m = llm_y >= 0
            P[m] = 0.3 * P[m]
            P[m, llm_y[m]] += 0.7
            method[m] = "llm"

    P = supplier_smooth(P, df["supplier_key"], ccfg.get("supplier_smoothing", 0.35), df["spend_calc"].values)
    leaf = P.argmax(axis=1)
    conf = P.max(axis=1)
    # labelled lines keep their label regardless of smoothing
    if method[have].size and (method[have] == "label").all():
        leaf[have] = y[have]
        conf[have] = 1.0
    return leaf, np.clip(conf, 0, 1), method
