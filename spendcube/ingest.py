"""Read each source by config into one canonical frame and clean amounts, dates, tax and currency."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CANON = [
    "record_id", "source", "role", "invoice_number", "invoice_line", "po", "po_line",
    "date", "date_flag", "supplier_name", "supplier_id", "description", "material",
    "quantity", "amount_raw", "currency", "fx_rate", "fx_flag", "vat_rate", "spend_calc",
    "status", "gl_text", "context", "system",
]


def _cols(spec):
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else list(spec)


def _join_text(df: pd.DataFrame, cols) -> pd.Series:
    cols = [c for c in _cols(cols) if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    parts = [df[c].fillna("").astype(str).str.strip() for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = np.where((out != "") & (p != "") & (out != p), out + " | " + p, np.where(out == "", p, out))
        out = pd.Series(out, index=df.index)
    return pd.Series(out, index=df.index).str.replace(r"\s+", " ", regex=True).str.strip()


# ---------------------------------------------------------------- amounts
_num_clean = re.compile(r"[^\d,.\-()]")


def parse_amount(s: pd.Series, decimal: str | None = None) -> pd.Series:
    """Parse '1,234.56', '1.234,56', '(12.30)', '12.30-', 'R 1 234,50' into floats."""
    def one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        t = str(x).strip()
        if not t:
            return np.nan
        neg = t.startswith("(") and t.endswith(")") or t.endswith("-") or t.startswith("-")
        t = _num_clean.sub("", t).replace("(", "").replace(")", "").replace("-", "")
        if not t:
            return np.nan
        if decimal == ",":
            t = t.replace(".", "").replace(",", ".")
        elif decimal == ".":
            t = t.replace(",", "")
        elif "," in t and "." in t:
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif "," in t:
            tail = t.split(",")[-1]
            if t.count(",") == 1 and len(tail) in (1, 2):
                t = t.replace(",", ".")       # decimal comma
            else:
                t = t.replace(",", "")        # thousands separators
        elif t.count(".") > 1:
            t = t.replace(".", "")            # 1.234.567
        try:
            v = float(t)
        except ValueError:
            return np.nan
        return -v if neg else v
    return s.map(one).astype(float)


def parse_rate(s: pd.Series, default: float | None) -> pd.Series:
    """VAT rate as '15', '15%', '0.15' -> 0.15; blank -> default."""
    v = parse_amount(s.astype(str).str.replace("%", "", regex=False).where(s.notna(), None))
    v = v.where(v.isna() | (v <= 1), v / 100.0)
    if default is not None:
        v = v.fillna(default)
    return v


# ---------------------------------------------------------------- dates
def parse_dates(s: pd.Series, formats, valid_range, dayfirst=True):
    """Try explicit formats in order, then Excel serials, then a lenient parse.
    Returns (dates, flag) where flag is '', 'missing', 'unparseable' or 'out_of_range'."""
    raw = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna() & raw.notna() & (raw != "")
        if not todo.any():
            break
        out[todo] = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
    todo = out.isna() & raw.notna() & raw.str.fullmatch(r"\d{5}(\.0+)?").fillna(False)
    if todo.any():  # Excel serial dates
        out[todo] = pd.to_datetime(raw[todo].astype(float), unit="D", origin="1899-12-30", errors="coerce")
    todo = out.isna() & raw.notna() & (raw != "")
    if todo.any():
        out[todo] = [pd.to_datetime(x, dayfirst=dayfirst, errors="coerce") for x in raw[todo]]
    flag = pd.Series("", index=s.index, dtype=object)
    flag[raw.isna() | (raw == "")] = "missing"
    flag[(flag == "") & out.isna()] = "unparseable"
    lo, hi = pd.Timestamp(valid_range[0]), pd.Timestamp(valid_range[1])
    bad = out.notna() & ((out < lo) | (out > hi))
    flag[bad] = "out_of_range"
    out[bad] = pd.NaT
    return out, flag


# ---------------------------------------------------------------- FX
class FxTable:
    def __init__(self, cfg):
        f = cfg["fx"]
        t = pd.read_csv(f["file"], dtype=str)
        t["cur"] = t[f["currency"]].str.strip().str.upper()
        t["m"] = pd.PeriodIndex(pd.to_datetime(t[f["month"]].str[:7], format="%Y-%m"), freq="M")
        t["r"] = parse_amount(t[f["rate"]])
        self.t = t.dropna(subset=["r"])
        self.by = {c: g.set_index("m")["r"].sort_index() for c, g in self.t.groupby("cur")}

    def rate(self, cur: str, month) -> float:
        g = self.by.get(cur)
        if g is None or len(g) == 0:
            return np.nan
        if month is None or pd.isna(month):
            return float(g.median())
        if month in g.index:
            return float(g.loc[month])
        # nearest available month
        d = np.abs(np.array([(m - month).n for m in g.index]))
        return float(g.iloc[int(d.argmin())])


def load_source(src: dict, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(src["file"], dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", "nan", "NaN", "NULL", "null", "None"])
    c = src["columns"]
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out["record_id"] = src.get("id_prefix", "") + df[src["id"]].astype(str)
    out["source"] = src["name"]
    out["role"] = src.get("role", "primary")

    def col(key):
        name = c.get(key)
        if isinstance(name, list):
            return _join_text(df, name)
        if name and name in df.columns:
            return df[name]
        return pd.Series([None] * n, index=df.index, dtype=object)

    out["invoice_number"] = col("invoice_number")
    out["invoice_line"] = col("invoice_line")
    po = col("po").astype("string")
    if src.get("po_pattern"):
        po_line = po.str.extract(src["po_line_pattern"])[0] if src.get("po_line_pattern") else None
        po = po.str.extract(src["po_pattern"])[0]
    else:
        po_line = None
    out["po"] = po.str.strip().str.lstrip("0").replace("", pd.NA)
    pl = col("po_line") if c.get("po_line") else po_line
    if pl is None:
        pl = pd.Series([None] * n, index=df.index)
    out["po_line"] = pd.to_numeric(pd.Series(pl, index=df.index), errors="coerce")

    dcfg = cfg["dates"]
    d, flag = parse_dates(col("date"), dcfg["formats"], dcfg["valid_range"], dcfg.get("dayfirst", True))
    if c.get("fallback_date"):
        d2, _ = parse_dates(col("fallback_date"), dcfg["formats"], dcfg["valid_range"], dcfg.get("dayfirst", True))
        use = d.isna() & d2.notna()
        d[use] = d2[use]
        flag[use] = flag[use].replace("", "missing") + "->fallback"
    out["date"] = d
    out["date_flag"] = flag

    out["supplier_name"] = col("supplier_name").fillna("").astype(str).str.strip()
    out["supplier_id"] = col("supplier_id").astype("string").str.strip()
    out["description"] = _join_text(df, c.get("description"))
    out["material"] = col("material").astype("string").str.strip()
    out["quantity"] = parse_amount(col("quantity"))
    out["amount_raw"] = parse_amount(col("amount"), src.get("decimal"))

    cur = col("currency").astype("string").str.strip().str.upper()
    fixed = src.get("currency")
    if fixed:
        cur = pd.Series(fixed, index=df.index, dtype="string")
    out["currency"] = cur.fillna(cfg["reporting_currency"])

    if src.get("amount_is_gross"):
        out["vat_rate"] = parse_rate(col("vat_rate"), src.get("default_vat_rate", 0.0))
        net = out["amount_raw"] / (1.0 + out["vat_rate"])
    else:
        out["vat_rate"] = 0.0
        net = out["amount_raw"]

    # currency conversion to reporting currency (rates are units of currency per 1 reporting unit)
    rep = cfg["reporting_currency"].upper()
    given = parse_amount(col("fx_rate")) if c.get("fx_rate") else pd.Series(np.nan, index=df.index)
    fx = cfg.get("_fx")
    cache = {}
    rate = pd.Series(1.0, index=df.index)
    fxflag = pd.Series("", index=df.index, dtype=object)
    months = out["date"].dt.to_period("M")
    for i in np.where(out["currency"] != rep)[0]:
        idx = df.index[i]
        cu = out.at[idx, "currency"]
        key = (cu, months.iat[i])
        if key not in cache:
            cache[key] = fx.rate(cu, months.iat[i]) if fx is not None else np.nan
        table = cache[key]
        g = given.iat[i]
        if pd.notna(g) and g > 0:
            ratio = cfg.get("fx", {}).get("sanity_ratio", 3)
            if pd.notna(table) and (g / table > ratio or table / g > ratio):
                rate.at[idx] = table
                fxflag.at[idx] = "given_rate_implausible->table"
            else:
                rate.at[idx] = g
        elif pd.notna(table):
            rate.at[idx] = table
            fxflag.at[idx] = "table"
        else:
            rate.at[idx] = np.nan
            fxflag.at[idx] = "no_rate"
    out["fx_rate"] = rate
    out["fx_flag"] = fxflag
    out["spend_calc"] = (net / rate).round(2)

    out["status"] = col("status").astype("string").str.strip()
    out["gl_text"] = _join_text(df, c.get("gl_text"))
    out["context"] = _join_text(df, c.get("context"))
    out["system"] = col("system").astype("string")
    return out[CANON]


def load_all(cfg: dict) -> pd.DataFrame:
    if cfg.get("fx"):
        cfg["_fx"] = FxTable(cfg)
    frames = []
    for src in cfg["sources"]:
        f = load_source(src, cfg)
        log.info("loaded %s: %d rows", src["name"], len(f))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df
