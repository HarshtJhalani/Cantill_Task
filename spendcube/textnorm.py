"""Text normalisation shared by matching, de-duplication and classification."""
from __future__ import annotations

import re
import unicodedata

_OCR = str.maketrans({"0": "O", "5": "S", "1": "I", "8": "B", "|": "I", "$": "S"})


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def compact(s: str, ocr: bool = False) -> str:
    """Upper-case alphanumerics only; optionally fold common OCR confusions."""
    s = ascii_fold(str(s or "")).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if ocr:
        s = s.translate(_OCR)
    return s


def ocr_fold(s: str) -> str:
    return compact(s, ocr=True)


def tokens(s: str) -> str:
    s = ascii_fold(str(s or "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


_GENERIC = {
    "", "product", "products", "parts", "part", "item", "items", "goods", "material", "materials",
    "service", "services", "misc", "miscellaneous", "various", "sundry", "sundries", "other", "na",
    "n a", "none", "no object", "see attached", "as per quote", "as per invoice", "invoice", "-", "general",
}


def is_generic(desc: str) -> bool:
    t = tokens(desc)
    return t in _GENERIC or len(re.sub(r"[^a-z]", "", t)) < 3


def supplier_core(name: str, suffixes: list[str]) -> str:
    """'OAK ENT MINING SA (PTY) LTD.' -> 'OAKENTMINING'"""
    t = tokens(name)
    words = t.split()
    suf = set(suffixes)
    # drop trailing legal-form words, and any standalone legal words in brackets
    while words and words[-1] in suf:
        words.pop()
    words = [w for w in words if w not in {"pty", "ltd", "proprietary", "limited"}]
    if words and words[0] == "the":
        words = words[1:]
    return "".join(words).upper()
