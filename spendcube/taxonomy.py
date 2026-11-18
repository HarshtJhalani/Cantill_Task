"""Category tree read at run time. No category names live in code."""
from __future__ import annotations

import re

import pandas as pd


class Taxonomy:
    def __init__(self, cfg: dict):
        tc = cfg["taxonomy"]
        t = pd.read_csv(tc["file"], dtype=str).fillna("")
        cols = tc.get("levels")
        if not cols:
            # detect level columns: anything ending in a level number, in numeric order
            lv = [(int(m.group(1)), c) for c in t.columns if (m := re.search(r"(\d+)\s*$", c))]
            cols = [c for _, c in sorted(lv)] or list(t.columns)
        self.level_cols = cols
        self.depth = len(cols)
        t = t[cols].apply(lambda s: s.str.strip())
        t = t[t[cols[0]] != ""].drop_duplicates().reset_index(drop=True)
        self.paths = [tuple(x for x in row if x) for row in t.itertuples(index=False)]
        # a path that is a strict prefix of another is not a leaf
        full = set(self.paths)
        self.leaves = [p for p in self.paths if not any(q != p and q[: len(p)] == p for q in full)]
        self.leaves = list(dict.fromkeys(self.leaves))
        self.index = {p: i for i, p in enumerate(self.leaves)}
        self.hints = {str(k).strip().lower(): v for k, v in (tc.get("hints") or {}).items()}

    def leaf_text(self, i: int) -> str:
        """Natural-language descriptor used for zero-shot matching."""
        p = self.leaves[i]
        names = list(dict.fromkeys(p))           # 'X > X' repeats collapse
        # expand acronyms that abbreviate an ancestor ('CM Repair Items' under 'Continuous Miners')
        inits = {"".join(w[0] for w in a.split() if w[:1].isalpha()).upper(): a for a in names[:-1]}
        names = [re.sub(r"\b([A-Z]{2,4})\b", lambda m: inits.get(m.group(1), m.group(1)), n) for n in names]
        txt = names[-1]
        if len(names) > 1:
            txt += " (" + ", ".join(reversed(names[:-1])) + ")"
        hint = self.hints.get(p[-1].lower()) or self.hints.get(" > ".join(p).lower())
        if hint:
            txt += ": " + (hint if isinstance(hint, str) else ", ".join(hint))
        return txt

    def find_leaf(self, values) -> int | None:
        vals = tuple(str(v).strip() for v in values if isinstance(v, str) and str(v).strip())
        if not vals:
            return None
        if vals in self.index:
            return self.index[vals]
        # match on the deepest given level name, prefer paths that share the prefix
        for i, p in enumerate(self.leaves):
            if p[: len(vals)] == vals and len(p) == len(vals):
                return i
        cands = [i for i, p in enumerate(self.leaves) if p[-1] == vals[-1]]
        if len(cands) == 1:
            return cands[0]
        cands = [i for i, p in enumerate(self.leaves) if p[: len(vals)] == vals]
        return cands[0] if len(cands) == 1 else None

    def path_columns(self, i: int, n_out: int = 4):
        p = list(self.leaves[i]) if i is not None and i >= 0 else []
        # repeat-names in the tree file are kept exactly as written (e.g. L2 == L3)
        row = [""] * n_out
        for k in range(min(len(p), n_out)):
            row[k] = p[k]
        return row
