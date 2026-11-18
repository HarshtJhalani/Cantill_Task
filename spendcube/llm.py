"""Optional LLM labelling of the highest-spend unique line texts (Claude API).

Runs only when ANTHROPIC_API_KEY is set and the `anthropic` package is installed; otherwise the
pipeline uses the local embedding + self-training path. The LLM sees the numbered leaf list from
the tree file and returns one leaf number per line, which becomes a strong training label for the
local text model (so the rest of the lines inherit the LLM's judgement for free).
Results are cached on disk by (model, tree, text) so re-runs cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent.parent / "cache"


class Pick(BaseModel):
    i: int          # line number in the request
    leaf: int       # chosen leaf number, or -1 when the text gives no clue


class Picks(BaseModel):
    picks: list[Pick]


SYSTEM = (
    "You classify company purchase lines into the company's own category tree for a spend analysis. "
    "Each line gives the purchase description and, after '|', context such as the supplier name, "
    "GL account text or material type. Descriptions may be abbreviated (SAP-style), OCR-garbled "
    "or in any language. Choose the single most specific category (leaf) that fits; use the supplier "
    "and context when the description is vague. Return -1 only if nothing at all indicates a category."
)


def _cache_load(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def llm_label(df: pd.DataFrame, texts: list[str], tax, cfg: dict, skip: np.ndarray) -> np.ndarray:
    import anthropic

    lc = cfg["classification"]["llm"]
    model = lc.get("model", "claude-opus-5-5")
    batch = int(lc.get("batch_size", 60))
    max_u = int(lc.get("max_unique_texts", 6000))
    leaves = "\n".join(f"{i}: " + " > ".join(dict.fromkeys(p)) for i, p in enumerate(tax.leaves))
    tree_id = hashlib.md5(leaves.encode()).hexdigest()[:10]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"llm_{tree_id}_{model}.json"
    cache = _cache_load(cpath)

    t = pd.DataFrame({"text": texts, "spend": df["spend_calc"].abs().fillna(0).values, "skip": skip})
    agg = t[~t["skip"]].groupby("text")["spend"].agg(["sum", "size"])
    agg["score"] = agg["sum"] + agg["size"]          # spend first, frequency breaks ties
    todo_all = list(agg.sort_values("score", ascending=False).index[:max_u])
    todo = [x for x in todo_all if x not in cache]
    log.info("LLM: %d unique texts selected, %d not cached (model %s)", len(todo_all), len(todo), model)

    client = anthropic.Anthropic(max_retries=4)
    sys_blocks = [{"type": "text", "text": SYSTEM + "\n\nCategory tree leaves:\n" + leaves,
                   "cache_control": {"type": "ephemeral"}}]
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        body = "\n".join(f"{k}: {x[:240]}" for k, x in enumerate(chunk))
        try:
            r = client.messages.parse(
                model=model, max_tokens=8000, system=sys_blocks,
                output_config={"effort": lc.get("effort", "low")},
                messages=[{"role": "user", "content": "Classify each line. Lines:\n" + body}],
                output_format=Picks,
            )
        except anthropic.RateLimitError as e:
            wait = int(e.response.headers.get("retry-after", "30"))
            log.warning("LLM rate limited; waiting %ss", wait)
            time.sleep(wait)
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            log.warning("LLM batch failed (%s); skipping batch", e)
            continue
        if r.stop_reason == "refusal" or r.parsed_output is None:
            continue
        for p in r.parsed_output.picks:
            if 0 <= p.i < len(chunk):
                cache[chunk[p.i]] = p.leaf if 0 <= p.leaf < len(tax.leaves) else -1
        cpath.write_text(json.dumps(cache), encoding="utf-8")
        log.info("LLM: %d/%d texts done", min(b + batch, len(todo)), len(todo))

    out = np.full(len(texts), -1)
    sel = set(todo_all)
    for i, x in enumerate(texts):
        if not skip[i] and x in sel:
            out[i] = cache.get(x, -1)
    return out
