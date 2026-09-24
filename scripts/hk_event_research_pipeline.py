import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Missing dependency `yfinance`. Install with: pip install yfinance") from e


@dataclass(frozen=True)
class Config:
    # Default to master sheet; you can still point to smaller files.
    event_csv: str = "data/hang_seng_events_2016_2026_master.csv"
    eligible_csv: str = "data/eligible_securities.csv"

    output_dir: str = "data/research_outputs"
    cache_dir: str = "data/yfinance_cache_research"
    charts_dir: str = "data/research_outputs/charts"

    pre_window: int = 60
    post_window: int = 60

    # Conservative: static shares outstanding can be used only as approximation.
    allow_static_shares_approx: bool = True


INDEX_SYNONYMS = {
    "HSTECH": "Hang Seng TECH Index",
    "HSI": "Hang Seng Index",
    "Stock Connect": "HKEX Stock Connect China Enterprises Index",
    "Stock Connect China Enterprises": "HKEX Stock Connect China Enterprises Index",
    "Hang Seng Biotech": "Hang Seng Biotech Index",
}


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.charts_dir, exist_ok=True)
    for sub in ("core", "diagnostics", "coverage", "features", "event_study", "insights"):
        os.makedirs(os.path.join(cfg.output_dir, sub), exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def normalize_change_type(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s.startswith("add"):
        return "Inclusion"
    if s.startswith("remove"):
        return "Removal"
    if "inclusion" in s:
        return "Inclusion"
    if "exclusion" in s:
        return "Exclusion"
    if "removal" in s:
        return "Removal"
    return None


def parse_mixed_date(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%m/%d/%y")
    miss = out.isna()
    out.loc[miss] = pd.to_datetime(s.loc[miss], errors="coerce", format="%Y-%m-%d")
    miss = out.isna()
    out.loc[miss] = pd.to_datetime(s.loc[miss], errors="coerce")
    return out


def extract_ticker_parts(ticker_normalized: object) -> Tuple[Optional[str], Optional[str]]:
    if ticker_normalized is None or (isinstance(ticker_normalized, float) and math.isnan(ticker_normalized)):
        return None, None
    s = str(ticker_normalized).strip()
    if "." not in s:
        return None, None
    a, b = s.split(".", 1)
    try:
        code = str(int(a))
    except Exception:
        code = None
    return code, b.upper()


def map_event_ticker_to_yf(tn: str) -> Optional[str]:
    code, exch = extract_ticker_parts(tn)
    if code is None or exch is None:
        return None
    if exch == "HK":
        c = int(code)
        c_str = f"{c:04d}" if c < 10000 else str(c)
        return f"{c_str}.HK"
    if exch in {"SH", "SZ"}:
        # Yahoo often uses .SS for Shanghai and .SZ for Shenzhen
        if exch == "SH":
            return f"{int(code):06d}.SS"
        return f"{int(code):06d}.SZ"
    return None


def _tokenize_symbol_candidates(raw: object) -> List[str]:
    """Split combined symbol strings into clean tokens.

    Examples:
      "1211.HK / 002594.SZ" -> ["1211.HK", "002594.SZ"]
      "2333 / 601633" -> ["2333", "601633"]
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"\s*/\s*|,|;", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


def _map_symbol_token_to_yf(token: str) -> Optional[str]:
    s = str(token).strip().upper()
    if not s:
        return None

    if "." in s:
        code, exch = s.split(".", 1)
        try:
            code_int = int(code)
        except Exception:
            return None
        if exch == "HK":
            c = f"{code_int:04d}" if code_int < 10000 else str(code_int)
            return f"{c}.HK"
        if exch == "SZ":
            return f"{code_int:06d}.SZ"
        if exch == "SH":
            return f"{code_int:06d}.SS"
        if exch == "SS":
            return f"{code_int:06d}.SS"
        return None

    # No exchange suffix -> cannot map robustly here.
    return None


def _normalize_index_name(idx: str) -> Tuple[str, str]:
    if idx in INDEX_SYNONYMS:
        return INDEX_SYNONYMS[idx], "mapped_synonym"
    return idx, "unchanged"


def normalize_master_events(df_events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Expand rows with multi-symbol entries into one row per parsed symbol.
    Marks one `is_primary_symbol=1` per source row (HK preferred, else first parsed).
    """
    df = df_events.copy()
    expected = ["date", "index", "change_type", "ticker", "ticker_normalized", "company"]
    miss = [c for c in expected if c not in df.columns]
    if miss:
        raise KeyError(f"Event CSV missing columns: {miss}")

    df["source_row_id"] = np.arange(len(df)).astype(int)
    df["date"] = parse_mixed_date(df["date"]).dt.normalize()
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), expected].head(10)
        raise ValueError("Unparsed dates detected:\n" + bad.to_string(index=False))

    for c in ["index", "change_type", "ticker", "ticker_normalized", "company"]:
        df[c] = df[c].astype(str).str.strip()

    df["change_type_raw"] = df["change_type"]
    df["change_type"] = df["change_type"].map(normalize_change_type)
    unknown_change_type_audit = df.loc[
        df["change_type"].isna(),
        ["source_row_id", "date", "index", "change_type_raw", "ticker", "ticker_normalized", "company"],
    ].copy()
    if len(unknown_change_type_audit) > 0:
        # Keep pipeline robust: audit and drop non-target label types.
        df = df[df["change_type"].notna()].copy()

    idx_norm = df["index"].map(_normalize_index_name)
    df["index_raw"] = df["index"]
    df["index"] = idx_norm.map(lambda x: x[0])
    df["index_normalization_status"] = idx_norm.map(lambda x: x[1])

    expanded_rows: List[Dict[str, object]] = []
    parse_audit_rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        sid = int(r["source_row_id"])
        tnorm_tokens = _tokenize_symbol_candidates(r["ticker_normalized"])
        ticker_tokens = _tokenize_symbol_candidates(r["ticker"])
        tokens = tnorm_tokens if len(tnorm_tokens) > 0 else ticker_tokens

        mapped: List[Tuple[str, Optional[str]]] = []
        for t in tokens:
            mapped.append((t, _map_symbol_token_to_yf(t)))

        valid = [(tok, yf_tkr) for tok, yf_tkr in mapped if yf_tkr is not None]
        if len(valid) == 0:
            parse_audit_rows.append(
                {
                    "source_row_id": sid,
                    "ticker_normalized_raw": r["ticker_normalized"],
                    "ticker_raw": r["ticker"],
                    "ticker_parse_status": "parse_failed",
                    "parsed_tokens": " | ".join(tokens),
                    "mapped_tickers": None,
                }
            )
            continue

        # primary: HK preferred; else first mapped token
        primary_idx = 0
        for j, (_, yf_tkr) in enumerate(valid):
            if str(yf_tkr).endswith(".HK"):
                primary_idx = j
                break

        parse_audit_rows.append(
            {
                "source_row_id": sid,
                "ticker_normalized_raw": r["ticker_normalized"],
                "ticker_raw": r["ticker"],
                "ticker_parse_status": "parse_ok",
                "parsed_tokens": " | ".join([x[0] for x in valid]),
                "mapped_tickers": " | ".join([x[1] for x in valid]),
            }
        )

        for j, (tok, yf_tkr) in enumerate(valid, start=1):
            row = dict(r)
            row["symbol_rank_within_row"] = j
            row["is_primary_symbol"] = 1 if (j - 1) == primary_idx else 0
            row["symbol_token_raw"] = tok
            row["ticker_yf_normalized"] = yf_tkr
            row["ticker_parse_status"] = "parse_ok"
            expanded_rows.append(row)

    expanded = pd.DataFrame(expanded_rows)
    parse_audit = pd.DataFrame(parse_audit_rows)
    index_audit = (
        df[["source_row_id", "index_raw", "index", "index_normalization_status"]]
        .drop_duplicates()
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )

    # Validation: exactly one primary symbol per source row among parse_ok rows
    primary_check = (
        expanded.groupby("source_row_id")["is_primary_symbol"].sum().reset_index(name="primary_count")
    )
    primary_violations = primary_check[primary_check["primary_count"] != 1].copy()

    # Deduplicate normalized events safely
    dedup_key = ["date", "index", "change_type", "ticker_yf_normalized"]
    duplicate_audit = expanded[expanded.duplicated(subset=dedup_key, keep=False)].copy()
    expanded_dedup = expanded.drop_duplicates(subset=dedup_key, keep="first").copy()

    # Provide `ticker_yf` alias for downstream compatibility
    expanded_dedup["ticker_yf"] = expanded_dedup["ticker_yf_normalized"]

    audits = {
        "master_ticker_parse_audit": parse_audit,
        "master_index_normalization_audit": index_audit,
        "master_events_duplicate_audit": duplicate_audit,
        "master_primary_symbol_violations": primary_violations,
        "master_unknown_change_type_audit": unknown_change_type_audit,
    }
    return expanded_dedup.reset_index(drop=True), audits


# ==================================================
# PHASE 1 — LOAD, CLEAN, AND VALIDATE DATA
# ==================================================
def load_event_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.event_csv, dtype=str)
    print(f"[load_event_data] loaded rows={len(df)} cols={list(df.columns)}")
    return df


def load_eligible_universe(cfg: Config) -> pd.DataFrame:
    df = safe_read_csv(cfg.eligible_csv)
    print(f"[load_eligible_universe] loaded rows={len(df)} cols={list(df.columns)}")
    return df


def clean_event_data(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    expected = ["date", "index", "change_type", "ticker", "ticker_normalized", "company"]
    miss = [c for c in expected if c not in df.columns]
    if miss:
        raise KeyError(f"Event CSV missing columns: {miss}")

    df["date"] = parse_mixed_date(df["date"]).dt.normalize()
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), expected].head(10)
        raise ValueError("Unparsed dates detected:\n" + bad.to_string(index=False))

    for c in ["index", "change_type", "ticker", "ticker_normalized", "company"]:
        df[c] = df[c].astype(str).str.strip()

    df["change_type"] = df["change_type"].map(normalize_change_type)
    if df["change_type"].isna().any():
        bad = df.loc[df["change_type"].isna(), ["date", "index", "change_type", "ticker_normalized"]].head(10)
        raise ValueError("Unknown change_type values:\n" + bad.to_string(index=False))

    # Normalize index synonyms
    df["index_raw"] = df["index"]
    df["index"] = df["index"].map(lambda x: INDEX_SYNONYMS.get(x, x))

    # Standardized yfinance mapping from event ticker
    df["ticker_yf"] = df["ticker_normalized"].map(map_event_ticker_to_yf)
    return df


def clean_eligible_data(df_eligible: pd.DataFrame) -> pd.DataFrame:
    df = df_eligible.copy()
    if "证券代码" not in df.columns:
        raise KeyError("eligible_securities.csv must include column `证券代码`")

    df["证券代码"] = df["证券代码"].astype(str).str.strip()
    code_num = pd.to_numeric(df["证券代码"], errors="coerce")
    df["code_num"] = code_num
    df = df[df["code_num"].notna()].copy()
    df["code_num"] = df["code_num"].astype(int)
    def _hk_fmt(x: int) -> str:
        return f"{x:04d}.HK" if int(x) < 10000 else f"{int(x)}.HK"

    df["ticker_yf"] = df["code_num"].map(_hk_fmt)
    return df


def validate_event_data(df_events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dup_key = ["date", "index", "change_type", "ticker_normalized"]
    duplicates = df_events[df_events.duplicated(subset=dup_key, keep=False)].copy()

    missing_counts = df_events[["date", "index", "change_type", "ticker", "ticker_normalized", "company"]].isna().sum()
    label_dist = df_events["change_type"].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index": "change_type"})
    idx_dist = df_events["index"].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index": "index"})

    ticker_pattern_invalid = df_events.loc[
        ~df_events["ticker_normalized"].astype(str).str.match(r"^\d+\.(HK|SZ|SH)$"),
        ["date", "index", "change_type", "ticker_normalized"],
    ].copy()

    diag = {
        "duplicates": duplicates,
        "missing_counts": missing_counts.to_frame("missing_count").reset_index().rename(columns={"index": "column"}),
        "label_dist": label_dist,
        "index_dist": idx_dist,
        "ticker_pattern_invalid": ticker_pattern_invalid,
    }
    return diag


def summarize_event_data(df_events: pd.DataFrame, df_eligible: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df_events.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    by_index = df.groupby("index").size().reset_index(name="event_count").sort_values("event_count", ascending=False)
    by_change = df.groupby("change_type").size().reset_index(name="event_count").sort_values("event_count", ascending=False)
    by_year_month = (
        df.groupby(["year_month", "change_type"]).size().reset_index(name="event_count").sort_values(["year_month", "change_type"])
    )

    eligible_tickers = set(df_eligible["ticker_yf"].dropna().unique().tolist())
    event_tickers = set(df["ticker_yf"].dropna().unique().tolist())
    overlap = sorted(event_tickers.intersection(eligible_tickers))
    overlap_table = pd.DataFrame({"ticker_yf": overlap})

    overlap_summary = pd.DataFrame(
        [
            {"metric": "eligible_tickers", "value": len(eligible_tickers)},
            {"metric": "event_tickers", "value": len(event_tickers)},
            {"metric": "overlap_tickers", "value": len(overlap)},
        ]
    )

    return {
        "event_counts_by_index": by_index,
        "event_counts_by_change_type": by_change,
        "event_counts_by_year_month": by_year_month,
        "ticker_overlap_table": overlap_table,
        "ticker_overlap_summary": overlap_summary,
    }


# ==================================================
# PHASE 2 — BUILD YFINANCE ENRICHMENT LAYER
# ==================================================
def build_enrichment_universe(df_events: pd.DataFrame, df_eligible: pd.DataFrame) -> pd.DataFrame:
    event_part = pd.DataFrame({"ticker_yf": sorted(df_events["ticker_yf"].dropna().unique().tolist()), "source": "event"})
    eligible_part = pd.DataFrame({"ticker_yf": sorted(df_eligible["ticker_yf"].dropna().unique().tolist()), "source": "eligible"})
    u = pd.concat([event_part, eligible_part], ignore_index=True)
    out = (
        u.groupby("ticker_yf")["source"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"source": "sources"})
    )
    return out


def build_dual_enrichment_universes(df_events_norm: pd.DataFrame, df_eligible: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    hk_only = (
        pd.DataFrame({"ticker_yf": sorted(df_eligible["ticker_yf"].dropna().astype(str).unique().tolist())})
        .assign(universe_flag="hk_only", sources="eligible")
    )
    all_symbols = (
        pd.DataFrame({"ticker_yf": sorted(df_events_norm["ticker_yf_normalized"].dropna().astype(str).unique().tolist())})
        .assign(universe_flag="all_symbols", sources="master_events")
    )
    combined = pd.concat([hk_only, all_symbols], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker_yf", "universe_flag"]).copy()
    return {
        "enrichment_universe_hk_only": hk_only,
        "enrichment_universe_all_symbols": all_symbols,
        "enrichment_universe_combined": combined,
    }


def cache_price_history(ticker: str, hist: pd.DataFrame, cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", ticker)
    path = os.path.join(cfg.cache_dir, f"{safe}.pkl")
    payload = {
        "ticker": ticker,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "history": hist,
    }
    pd.to_pickle(payload, path)
    return path


def load_cached_price_history(ticker: str, cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", ticker)
    path = os.path.join(cfg.cache_dir, f"{safe}.pkl")
    if not os.path.exists(path):
        return None
    try:
        payload = pd.read_pickle(path)
        c_start = pd.to_datetime(payload.get("start"))
        c_end = pd.to_datetime(payload.get("end"))
        if c_start <= start and c_end >= end:
            hist = payload.get("history", None)
            if isinstance(hist, pd.DataFrame):
                return hist
    except Exception:
        return None
    return None


def _hk_yahoo_symbol_variants(ticker_yf: str) -> List[str]:
    """
    Yahoo Finance HK tickers are usually 4-digit (e.g. 0005.HK), but some datasets
    carry codes where Yahoo expects a leading zero variant (e.g. 09666.HK).

    We try a small set of safe, deterministic variants to reduce avoidable 404s.
    """
    t = str(ticker_yf).strip()
    if not t:
        return []
    out = [t]
    m = re.match(r"^(\d+)\.HK$", t, flags=re.IGNORECASE)
    if m:
        code = m.group(1)
        # If provided as 4 digits, also try 5 digits with a leading zero.
        if len(code) == 4:
            out.append(f"0{code}.HK")
    # De-dup while preserving order
    dedup: List[str] = []
    seen = set()
    for s in out:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


def download_yfinance_history(universe: pd.DataFrame, cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    audit_rows: List[Dict[str, object]] = []
    tickers = universe["ticker_yf"].tolist()

    for i, tkr in enumerate(tickers, 1):
        print(f"[download_yfinance_history] ({i}/{len(tickers)}) {tkr}")

        cached = load_cached_price_history(tkr, cfg, start, end)
        if cached is not None and not cached.empty:
            histories[tkr] = cached
            audit_rows.append(
                {
                    "ticker_yf": tkr,
                    "status": "success_cached",
                    "n_rows": len(cached),
                    "min_date": str(pd.to_datetime(cached.index.min()).date()),
                    "max_date": str(pd.to_datetime(cached.index.max()).date()),
                    "yahoo_symbol_used": tkr,
                    "error": None,
                }
            )
            continue

        try:
            hist = None
            yahoo_used: Optional[str] = None
            last_err: Optional[str] = None

            variants = _hk_yahoo_symbol_variants(tkr) if str(tkr).upper().endswith(".HK") else [tkr]
            for sym in variants:
                try:
                    h = yf.Ticker(sym).history(
                        start=str(start.date()),
                        end=str(end.date()),
                        auto_adjust=False,
                        actions=False,
                    )
                    if h is None or h.empty:
                        last_err = "empty_history"
                        continue
                    hist = h
                    yahoo_used = sym
                    break
                except Exception as e:
                    last_err = str(e)
                    continue

            if hist is None or hist.empty:
                audit_rows.append(
                    {
                        "ticker_yf": tkr,
                        "status": "failed_empty",
                        "n_rows": 0,
                        "min_date": None,
                        "max_date": None,
                        "yahoo_symbol_used": yahoo_used,
                        "error": last_err or "empty_history",
                    }
                )
                continue

            hist = hist.copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.sort_index()
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            hist = hist[keep]
            cache_price_history(tkr, hist, cfg, start, end)
            histories[tkr] = hist
            audit_rows.append(
                {
                    "ticker_yf": tkr,
                    "status": "success_downloaded",
                    "n_rows": len(hist),
                    "min_date": str(pd.to_datetime(hist.index.min()).date()),
                    "max_date": str(pd.to_datetime(hist.index.max()).date()),
                    "yahoo_symbol_used": yahoo_used or tkr,
                    "error": None,
                }
            )
        except Exception as e:
            audit_rows.append(
                {
                    "ticker_yf": tkr,
                    "status": "failed_exception",
                    "n_rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "yahoo_symbol_used": None,
                    "error": str(e),
                }
            )

    audit_df = pd.DataFrame(audit_rows)
    return histories, audit_df


def build_price_coverage_report(universe: pd.DataFrame, audit_df: pd.DataFrame, histories: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    merged = universe.merge(audit_df, on="ticker_yf", how="left")

    summary = (
        merged.groupby("status", dropna=False).size().reset_index(name="ticker_count").sort_values("ticker_count", ascending=False)
    )

    missing = merged[~merged["status"].isin(["success_cached", "success_downloaded"])].copy()

    coverage_stats = pd.DataFrame(
        [
            {"metric": "total_universe_tickers", "value": len(universe)},
            {"metric": "successful_tickers", "value": int((merged["status"].isin(["success_cached", "success_downloaded"])).sum())},
            {"metric": "failed_tickers", "value": int((~merged["status"].isin(["success_cached", "success_downloaded"])).sum())},
        ]
    )

    return {
        "price_coverage_by_status": summary,
        "price_coverage_universe_audit": merged,
        "price_missing_or_failed_tickers": missing,
        "price_coverage_stats": coverage_stats,
    }


# ==================================================
# PHASE 3 — BUILD LEAKAGE-SAFE FEATURES
# ==================================================
def _align_to_prev_trading_day(hist: pd.DataFrame, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    if hist is None or hist.empty:
        return None
    idx = pd.to_datetime(hist.index).normalize()
    pos = np.searchsorted(idx.values.astype("datetime64[ns]"), np.datetime64(dt.normalize()), side="left") - 1
    if pos < 0:
        return None
    return pd.Timestamp(idx[pos])


def compute_pre_event_returns(hist: pd.DataFrame, anchor_date: pd.Timestamp) -> Dict[str, float]:
    close = hist["Close"].astype(float)
    c_t = close.loc[anchor_date] if anchor_date in close.index else np.nan

    def ret_n(n: int) -> float:
        idx = close.index.get_loc(anchor_date)
        j = idx - n
        if j < 0:
            return np.nan
        prev = float(close.iloc[j])
        cur = float(close.iloc[idx])
        if prev <= 0:
            return np.nan
        return cur / prev - 1.0

    # distance from rolling highs
    idx = close.index.get_loc(anchor_date)
    h60 = close.iloc[max(0, idx - 59) : idx + 1].max()
    h252 = close.iloc[max(0, idx - 251) : idx + 1].max()

    d60 = np.nan if pd.isna(h60) or h60 == 0 else c_t / h60 - 1.0
    d252 = np.nan if pd.isna(h252) or h252 == 0 else c_t / h252 - 1.0

    return {
        "close_at_event_anchor": float(c_t) if pd.notna(c_t) else np.nan,
        "ret_5d_pre": ret_n(5),
        "ret_20d_pre": ret_n(20),
        "ret_60d_pre": ret_n(60),
        "dist_from_60d_high": float(d60) if pd.notna(d60) else np.nan,
        "dist_from_252d_high": float(d252) if pd.notna(d252) else np.nan,
    }


def compute_liquidity_features(hist: pd.DataFrame, anchor_date: pd.Timestamp) -> Dict[str, float]:
    close = hist["Close"].astype(float)
    vol = hist["Volume"].astype(float)
    dv = close * vol
    i = close.index.get_loc(anchor_date)
    win20 = slice(max(0, i - 19), i + 1)
    win60 = slice(max(0, i - 59), i + 1)

    return {
        "volume_at_event_anchor": float(vol.iloc[i]) if pd.notna(vol.iloc[i]) else np.nan,
        "dollar_volume_at_event_anchor": float(dv.iloc[i]) if pd.notna(dv.iloc[i]) else np.nan,
        "avg_volume_20d_pre": float(vol.iloc[win20].mean()) if i >= 0 else np.nan,
        "avg_volume_60d_pre": float(vol.iloc[win60].mean()) if i >= 0 else np.nan,
        "avg_dollar_volume_20d_pre": float(dv.iloc[win20].mean()) if i >= 0 else np.nan,
        "avg_dollar_volume_60d_pre": float(dv.iloc[win60].mean()) if i >= 0 else np.nan,
    }


def compute_volatility_features(hist: pd.DataFrame, anchor_date: pd.Timestamp) -> Dict[str, float]:
    close = hist["Close"].astype(float)
    lr = np.log(close).diff()
    i = close.index.get_loc(anchor_date)
    win20 = slice(max(0, i - 19), i + 1)
    win60 = slice(max(0, i - 59), i + 1)
    return {
        "realized_vol_20d_pre": float(lr.iloc[win20].std()) if i >= 0 else np.nan,
        "realized_vol_60d_pre": float(lr.iloc[win60].std()) if i >= 0 else np.nan,
    }


def _fetch_static_shares_outstanding(ticker_yf: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        info = yf.Ticker(ticker_yf).info or {}
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            return None, None
        shares = float(shares)
        if not np.isfinite(shares) or shares <= 0:
            return None, None
        return shares, "yfinance_info_static_sharesOutstanding"
    except Exception:
        return None, None


def compute_market_cap_proxy(
    row_features: Dict[str, float], shares_outstanding: Optional[float], shares_source: Optional[str], cfg: Config
) -> Dict[str, object]:
    close_px = row_features.get("close_at_event_anchor", np.nan)
    out = {
        "shares_source": shares_source,
        "market_cap_available": 0,
        "market_cap_is_approx": 0,
        "turnover_available": 0,
        "market_cap_approx": np.nan,
        "share_turnover_at_event_anchor": np.nan,
    }
    if shares_outstanding is None or not cfg.allow_static_shares_approx:
        return out
    if pd.isna(close_px):
        return out
    out["market_cap_available"] = 1
    out["market_cap_is_approx"] = 1
    out["turnover_available"] = 1
    out["market_cap_approx"] = float(close_px) * float(shares_outstanding)
    vol = row_features.get("volume_at_event_anchor", np.nan)
    out["share_turnover_at_event_anchor"] = np.nan if pd.isna(vol) else float(vol) / float(shares_outstanding)
    return out


def attach_event_features(df_events: pd.DataFrame, histories: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    shares_cache: Dict[str, Tuple[Optional[float], Optional[str]]] = {}

    for i, r in df_events.iterrows():
        tkr = r["ticker_yf"]
        evt_dt = pd.Timestamp(r["date"])
        hist = histories.get(tkr)
        if hist is None or hist.empty:
            audit_rows.append(
                {
                    "event_idx": i,
                    "ticker_yf": tkr,
                    "date": str(evt_dt.date()),
                    "status": "no_history",
                    "anchor_date": None,
                    "notes": "ticker history unavailable",
                }
            )
            continue

        h = hist.copy()
        h.index = pd.to_datetime(h.index).normalize()
        h = h[~h.index.duplicated(keep="last")].sort_index()
        anchor = _align_to_prev_trading_day(h, evt_dt)
        if anchor is None or anchor not in h.index:
            audit_rows.append(
                {
                    "event_idx": i,
                    "ticker_yf": tkr,
                    "date": str(evt_dt.date()),
                    "status": "no_anchor",
                    "anchor_date": None,
                    "notes": "no prior trading day in history",
                }
            )
            continue

        f1 = compute_pre_event_returns(h, anchor)
        f2 = compute_liquidity_features(h, anchor)
        f3 = compute_volatility_features(h, anchor)

        if tkr not in shares_cache:
            shares_cache[tkr] = _fetch_static_shares_outstanding(tkr)
        shares_out, shares_src = shares_cache[tkr]
        f4 = compute_market_cap_proxy({**f1, **f2, **f3}, shares_out, shares_src, cfg)

        row = dict(r)
        row.update(
            {
                "event_anchor_date": anchor,
            }
        )
        row.update(f1)
        row.update(f2)
        row.update(f3)
        row.update(f4)
        feature_rows.append(row)

        audit_rows.append(
            {
                "event_idx": i,
                "ticker_yf": tkr,
                "date": str(evt_dt.date()),
                "status": "features_ok",
                "anchor_date": str(anchor.date()),
                "notes": None,
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_audit = pd.DataFrame(audit_rows)
    return feature_df, feature_audit


# ==================================================
# PHASE 4 — EVENT STUDY ENGINE
# ==================================================
def extract_event_window(hist: pd.DataFrame, anchor_date: pd.Timestamp, pre: int, post: int) -> pd.DataFrame:
    """
    Return event window dataframe with event_day in [-pre, +post].
    Allows partial windows.
    """
    h = hist.copy()
    h.index = pd.to_datetime(h.index).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    if anchor_date not in h.index:
        return pd.DataFrame()
    i0 = h.index.get_loc(anchor_date)
    start = max(0, i0 - pre)
    end = min(len(h) - 1, i0 + post)

    w = h.iloc[start : end + 1].copy()
    w["ret_1d"] = w["Close"].astype(float).pct_change()
    w["event_day"] = np.arange(start - i0, end - i0 + 1)

    # Cumulative returns relative to anchor close
    anchor_close = float(h.iloc[i0]["Close"])
    w["cum_ret_from_anchor"] = w["Close"].astype(float) / anchor_close - 1.0
    return w.reset_index().rename(columns={"index": "trade_date"})


def build_event_windows(df_events: pd.DataFrame, histories: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit = []
    for i, r in df_events.iterrows():
        tkr = r["ticker_yf"]
        evt_dt = pd.Timestamp(r["date"])
        hist = histories.get(tkr)
        if hist is None or hist.empty:
            audit.append({"event_idx": i, "status": "no_history", "ticker_yf": tkr, "event_date": str(evt_dt.date())})
            continue
        h = hist.copy()
        h.index = pd.to_datetime(h.index).normalize()
        anchor = _align_to_prev_trading_day(h, evt_dt)
        if anchor is None:
            audit.append({"event_idx": i, "status": "no_anchor", "ticker_yf": tkr, "event_date": str(evt_dt.date())})
            continue
        ew = extract_event_window(h, anchor, cfg.pre_window, cfg.post_window)
        if ew.empty:
            audit.append({"event_idx": i, "status": "empty_window", "ticker_yf": tkr, "event_date": str(evt_dt.date())})
            continue
        ew["event_idx"] = i
        ew["ticker_yf"] = tkr
        ew["event_date"] = evt_dt
        ew["event_anchor_date"] = anchor
        ew["index"] = r["index"]
        ew["change_type"] = r["change_type"]
        ew["year"] = pd.Timestamp(r["date"]).year
        rows.append(ew)
        audit.append({"event_idx": i, "status": "window_ok", "ticker_yf": tkr, "event_date": str(evt_dt.date())})
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit)


def aggregate_event_study(event_windows: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    if event_windows.empty:
        return pd.DataFrame()
    gcols = by + ["event_day"]
    out = (
        event_windows.groupby(gcols)
        .agg(
            n_events=("event_idx", "nunique"),
            avg_daily_ret=("ret_1d", "mean"),
            med_daily_ret=("ret_1d", "median"),
            avg_cum_ret=("cum_ret_from_anchor", "mean"),
            med_cum_ret=("cum_ret_from_anchor", "median"),
        )
        .reset_index()
        .sort_values(gcols)
    )
    return out


def summarize_pre_post_returns(event_windows: pd.DataFrame) -> pd.DataFrame:
    if event_windows.empty:
        return pd.DataFrame()
    w = event_windows.copy()
    # Extract anchor-relative cumulative returns at key points
    wanted = [-20, -1, 1, 20, 60]
    snap = w[w["event_day"].isin(wanted)].copy()
    piv = (
        snap.pivot_table(index=["event_idx", "ticker_yf", "index", "change_type", "year"], columns="event_day", values="cum_ret_from_anchor")
        .reset_index()
        .rename(columns={-20: "cum_ret_t_minus_20", -1: "cum_ret_t_minus_1", 1: "cum_ret_t_plus_1", 20: "cum_ret_t_plus_20", 60: "cum_ret_t_plus_60"})
    )
    # Period returns derived from cumulative snapshots where available
    piv["pre_t20_to_t1"] = piv["cum_ret_t_minus_1"] - piv["cum_ret_t_minus_20"]
    piv["post_t1_to_t20"] = piv["cum_ret_t_plus_20"] - piv["cum_ret_t_plus_1"]
    piv["post_t1_to_t60"] = piv["cum_ret_t_plus_60"] - piv["cum_ret_t_plus_1"]
    return piv


def plot_event_study_curves(agg_df: pd.DataFrame, group_col: str, out_path: str) -> None:
    if agg_df.empty:
        return
    plt.figure(figsize=(10, 6))
    for g, d in agg_df.groupby(group_col):
        dd = d.sort_values("event_day")
        plt.plot(dd["event_day"], dd["avg_cum_ret"], label=f"{g} (n~{int(dd['n_events'].max())})")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Average Cumulative Return Around Events by {group_col}")
    plt.xlabel("Event Day")
    plt.ylabel("Average Cumulative Return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ==================================================
# PHASE 5 — INSIGHT ANALYSIS
# ==================================================
def compare_change_types(feature_df: pd.DataFrame, pre_post_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    a = (
        feature_df.groupby("change_type")
        .agg(
            n_events=("ticker_yf", "size"),
            mean_ret_20d_pre=("ret_20d_pre", "mean"),
            median_ret_20d_pre=("ret_20d_pre", "median"),
            mean_avg_dollar_volume_20d_pre=("avg_dollar_volume_20d_pre", "mean"),
            median_avg_dollar_volume_20d_pre=("avg_dollar_volume_20d_pre", "median"),
            mean_realized_vol_20d_pre=("realized_vol_20d_pre", "mean"),
            median_realized_vol_20d_pre=("realized_vol_20d_pre", "median"),
        )
        .reset_index()
    )
    if pre_post_df.empty:
        return a
    b = (
        pre_post_df.groupby("change_type")
        .agg(
            mean_post_t1_t20=("post_t1_to_t20", "mean"),
            median_post_t1_t20=("post_t1_to_t20", "median"),
            mean_post_t1_t60=("post_t1_to_t60", "mean"),
            median_post_t1_t60=("post_t1_to_t60", "median"),
        )
        .reset_index()
    )
    return a.merge(b, on="change_type", how="left")


def compare_indices(feature_df: pd.DataFrame, pre_post_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    a = (
        feature_df.groupby(["index", "change_type"])
        .agg(
            n_events=("ticker_yf", "size"),
            mean_ret_20d_pre=("ret_20d_pre", "mean"),
            median_ret_20d_pre=("ret_20d_pre", "median"),
            mean_avg_dollar_volume_20d_pre=("avg_dollar_volume_20d_pre", "mean"),
            median_avg_dollar_volume_20d_pre=("avg_dollar_volume_20d_pre", "median"),
            mean_realized_vol_20d_pre=("realized_vol_20d_pre", "mean"),
            median_realized_vol_20d_pre=("realized_vol_20d_pre", "median"),
        )
        .reset_index()
    )
    if pre_post_df.empty:
        return a
    b = (
        pre_post_df.groupby(["index", "change_type"])
        .agg(
            mean_post_t1_t20=("post_t1_to_t20", "mean"),
            median_post_t1_t20=("post_t1_to_t20", "median"),
            mean_post_t1_t60=("post_t1_to_t60", "mean"),
            median_post_t1_t60=("post_t1_to_t60", "median"),
        )
        .reset_index()
    )
    return a.merge(b, on=["index", "change_type"], how="left")


def analyze_multi_index_overlap(df_events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    per_ticker_indices = (
        df_events.groupby("ticker_yf")["index"].nunique().reset_index(name="n_unique_indices").sort_values("n_unique_indices", ascending=False)
    )
    per_ticker_events = (
        df_events.groupby("ticker_yf").size().reset_index(name="n_events").sort_values("n_events", ascending=False)
    )
    event_gaps = df_events.sort_values(["ticker_yf", "date"]).copy()
    event_gaps["prev_date"] = event_gaps.groupby("ticker_yf")["date"].shift(1)
    event_gaps["days_since_prev_event"] = (event_gaps["date"] - event_gaps["prev_date"]).dt.days

    multi_index_tickers = per_ticker_indices[per_ticker_indices["n_unique_indices"] > 1].copy()
    return {
        "ticker_index_overlap": per_ticker_indices,
        "ticker_event_counts": per_ticker_events,
        "ticker_multi_index_only": multi_index_tickers,
        "ticker_event_gaps": event_gaps[["ticker_yf", "date", "index", "change_type", "days_since_prev_event"]].copy(),
    }


def rank_event_outcomes(pre_post_df: pd.DataFrame) -> pd.DataFrame:
    if pre_post_df.empty:
        return pd.DataFrame()
    out = pre_post_df.copy()
    out["rank_post_t20"] = out["post_t1_to_t20"].rank(ascending=False, method="min")
    out = out.sort_values("post_t1_to_t20", ascending=False)
    return out


def generate_key_findings(feature_df: pd.DataFrame, pre_post_df: pd.DataFrame, compare_ct: pd.DataFrame, compare_idx: pd.DataFrame) -> List[str]:
    findings: List[str] = []
    n_events = len(feature_df)
    findings.append(f"Feature-enriched events available: {n_events}")

    if n_events < 50:
        findings.append("Sample size is limited; treat all pattern claims as descriptive, not statistically definitive.")

    if not compare_ct.empty:
        try:
            inc = compare_ct.loc[compare_ct["change_type"] == "Inclusion"].iloc[0]
            exc = compare_ct.loc[compare_ct["change_type"] == "Exclusion"].iloc[0]
            findings.append(
                f"Pre-event 20d momentum (mean): Inclusion={inc['mean_ret_20d_pre']:.4f}, Exclusion={exc['mean_ret_20d_pre']:.4f}."
            )
            findings.append(
                f"Pre-event 20d dollar-volume (mean): Inclusion={inc['mean_avg_dollar_volume_20d_pre']:.2f}, Exclusion={exc['mean_avg_dollar_volume_20d_pre']:.2f}."
            )
        except Exception:
            pass

    if not compare_idx.empty:
        idx_post = (
            compare_idx.groupby("index")["mean_post_t1_t20"].mean().reset_index().sort_values("mean_post_t1_t20", ascending=False)
        )
        if len(idx_post) > 0:
            top = idx_post.iloc[0]
            findings.append(f"Highest average post-event T+1 to T+20 drift in current sample: {top['index']} ({top['mean_post_t1_t20']:.4f}).")

    if not pre_post_df.empty:
        sell_news_rate = (pre_post_df["post_t1_to_t20"] < 0).mean()
        findings.append(f"Share of events with negative post-event T+1 to T+20 return: {sell_news_rate:.2%}.")

    findings.append(
        "To improve reliability, add candidate-universe non-event rows and richer point-in-time fundamentals/liquidity data."
    )
    return findings


# ==================================================
# PHASE 6 — OUTPUTS (tables + charts + final summary)
# ==================================================
def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[output] saved: {path} ({len(df)} rows)")


def plot_counts_by_year_month(df_events: pd.DataFrame, out_path: str) -> None:
    d = df_events.copy()
    d["year_month"] = d["date"].dt.to_period("M").astype(str)
    c = d.groupby("year_month").size().reset_index(name="count")
    plt.figure(figsize=(11, 4))
    plt.plot(c["year_month"], c["count"], marker="o")
    plt.xticks(rotation=60)
    plt.title("Event Count by Year-Month")
    plt.xlabel("Year-Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_change_type_by_index(df_events: pd.DataFrame, out_path: str) -> None:
    c = (
        df_events.groupby(["index", "change_type"]).size().reset_index(name="count").pivot(index="index", columns="change_type", values="count").fillna(0)
    )
    c = c.sort_values(c.columns.tolist(), ascending=False)
    c.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title("Inclusion/Removal/Exclusion Counts by Index")
    plt.xlabel("Index")
    plt.ylabel("Event Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pre_event_momentum_box(feature_df: pd.DataFrame, out_path: str) -> None:
    if feature_df.empty:
        return
    data = []
    labels = []
    for ct in ["Inclusion", "Removal", "Exclusion"]:
        x = feature_df.loc[feature_df["change_type"] == ct, "ret_20d_pre"].dropna().values
        if len(x) > 0:
            data.append(x)
            labels.append(ct)
    if not data:
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Pre-event 20d Momentum by Change Type")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_liquidity_by_event_type(feature_df: pd.DataFrame, out_path: str) -> None:
    if feature_df.empty:
        return
    d = (
        feature_df.groupby("change_type")["avg_dollar_volume_20d_pre"]
        .median()
        .reset_index()
        .sort_values("avg_dollar_volume_20d_pre", ascending=False)
    )
    plt.figure(figsize=(7, 5))
    plt.bar(d["change_type"], d["avg_dollar_volume_20d_pre"])
    plt.title("Median Pre-event 20d Dollar Volume by Change Type")
    plt.ylabel("Median Dollar Volume")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run_pipeline(cfg: Config) -> None:
    ensure_dirs(cfg)
    root = cfg.output_dir
    core = os.path.join(root, "core")
    diagnostics = os.path.join(root, "diagnostics")
    coverage = os.path.join(root, "coverage")
    features = os.path.join(root, "features")
    event_study = os.path.join(root, "event_study")
    insights = os.path.join(root, "insights")

    # ---------------- PHASE 1 ----------------
    df_events_raw = load_event_data(cfg)
    df_eligible_raw = load_eligible_universe(cfg)

    # Normalize master events with dual-symbol expansion and primary flag.
    df_events, master_audits = normalize_master_events(df_events_raw)
    df_eligible = clean_eligible_data(df_eligible_raw)

    diag = validate_event_data(df_events)
    summary = summarize_event_data(df_events, df_eligible)

    save_df(df_events, os.path.join(core, "master_events_normalized.csv"))
    save_df(df_eligible, os.path.join(core, "eligible_universe_cleaned.csv"))
    for k, d in master_audits.items():
        save_df(d, os.path.join(diagnostics, f"{k}.csv"))
    for k, d in diag.items():
        save_df(d, os.path.join(diagnostics, f"diag_{k}.csv"))
    for k, d in summary.items():
        save_df(d, os.path.join(core, f"summary_{k}.csv"))

    # ---------------- PHASE 2 ----------------
    universes = build_dual_enrichment_universes(df_events, df_eligible)
    for name, u in universes.items():
        save_df(u, os.path.join(core, f"{name}.csv"))

    # Date bounds for download (enough for pre/post windows)
    min_dt = pd.Timestamp(df_events["date"].min()) - pd.Timedelta(days=450)
    max_dt = pd.Timestamp(df_events["date"].max()) + pd.Timedelta(days=450)
    # Enrichment per universe with explicit audit files
    universe_histories: Dict[str, Dict[str, pd.DataFrame]] = {}
    universe_audits: Dict[str, pd.DataFrame] = {}
    for flag, u in [("hk_only", universes["enrichment_universe_hk_only"]), ("all_symbols", universes["enrichment_universe_all_symbols"])]:
        print(f"\n[phase2] downloading yfinance for universe={flag} tickers={len(u)}")
        h, price_audit = download_yfinance_history(u[["ticker_yf"]], cfg, min_dt, max_dt)
        universe_histories[flag] = h
        universe_audits[flag] = price_audit
        cov = build_price_coverage_report(u[["ticker_yf"]], price_audit, h)
        save_df(price_audit, os.path.join(coverage, f"yfinance_download_audit_{flag}.csv"))
        # Backward-compatible default file uses all_symbols
        if flag == "all_symbols":
            save_df(price_audit, os.path.join(coverage, "yfinance_download_audit.csv"))
        for k, d in cov.items():
            save_df(d, os.path.join(coverage, f"{k}_{flag}.csv"))

    # Main histories for downstream event features/windows: all symbols universe
    histories = universe_histories["all_symbols"]
    price_audit = universe_audits["all_symbols"]
    universe = universes["enrichment_universe_all_symbols"][["ticker_yf"]].copy()

    # ---------------- PHASE 3 ----------------
    feature_df, feature_audit = attach_event_features(df_events, histories, cfg)
    save_df(feature_df, os.path.join(features, "event_features_enriched.csv"))
    save_df(feature_audit, os.path.join(features, "event_feature_audit.csv"))

    # ---------------- PHASE 4 ----------------
    event_windows, window_audit = build_event_windows(df_events, histories, cfg)
    save_df(event_windows, os.path.join(event_study, "event_windows_daily.csv"))
    save_df(window_audit, os.path.join(event_study, "event_windows_audit.csv"))

    agg_by_change = aggregate_event_study(event_windows, ["change_type"])
    agg_by_index = aggregate_event_study(event_windows, ["index"])
    agg_by_year = aggregate_event_study(event_windows, ["year"])
    agg_by_ticker = aggregate_event_study(event_windows, ["ticker_yf"])
    pre_post = summarize_pre_post_returns(event_windows)

    save_df(agg_by_change, os.path.join(event_study, "event_study_agg_by_change_type.csv"))
    save_df(agg_by_index, os.path.join(event_study, "event_study_agg_by_index.csv"))
    save_df(agg_by_year, os.path.join(event_study, "event_study_agg_by_year.csv"))
    save_df(agg_by_ticker, os.path.join(event_study, "event_study_agg_by_ticker.csv"))
    save_df(pre_post, os.path.join(event_study, "event_study_pre_post_summary.csv"))

    # ---------------- PHASE 5 ----------------
    compare_ct = compare_change_types(feature_df, pre_post)
    compare_idx = compare_indices(feature_df, pre_post)
    overlap = analyze_multi_index_overlap(df_events)
    ranked = rank_event_outcomes(pre_post)
    findings = generate_key_findings(feature_df, pre_post, compare_ct, compare_idx)

    save_df(compare_ct, os.path.join(insights, "insight_compare_change_types.csv"))
    save_df(compare_idx, os.path.join(insights, "insight_compare_indices.csv"))
    for k, d in overlap.items():
        save_df(d, os.path.join(insights, f"insight_{k}.csv"))
    save_df(ranked, os.path.join(insights, "insight_ranked_event_outcomes.csv"))

    # ---------------- PHASE 6 charts ----------------
    plot_counts_by_year_month(df_events, os.path.join(cfg.charts_dir, "event_count_by_year_month.png"))
    plot_change_type_by_index(df_events, os.path.join(cfg.charts_dir, "change_type_count_by_index.png"))
    plot_event_study_curves(agg_by_change, "change_type", os.path.join(cfg.charts_dir, "avg_cumret_by_change_type.png"))
    plot_event_study_curves(agg_by_index, "index", os.path.join(cfg.charts_dir, "avg_cumret_by_index.png"))
    plot_pre_event_momentum_box(feature_df, os.path.join(cfg.charts_dir, "pre_event_20d_momentum_boxplot.png"))
    plot_liquidity_by_event_type(feature_df, os.path.join(cfg.charts_dir, "liquidity_by_event_type.png"))

    # Final printed summary
    print("\n========================")
    print("FINAL RESEARCH SUMMARY")
    print("========================")
    total_evt = len(df_events)
    feat_evt = len(feature_df)
    succ_tickers = int((price_audit["status"].isin(["success_cached", "success_downloaded"])).sum()) if len(price_audit) else 0
    total_tickers = len(universe)
    print(f"Events after normalization/expansion: {total_evt}")
    print(f"Events with attached features: {feat_evt}")
    print(f"Ticker enrichment success: {succ_tickers}/{total_tickers}")
    # Validation checks requested in plan
    raw_n = len(df_events_raw)
    norm_n = len(df_events)
    print(f"Master row expansion check: raw_rows={raw_n}, normalized_rows={norm_n}")
    if norm_n < raw_n:
        print("WARNING: normalized rows < raw rows; expected >= due to symbol expansion.")
    prim_viol = master_audits["master_primary_symbol_violations"]
    print(f"Primary-symbol validation violations: {len(prim_viol)}")
    parse_audit = master_audits["master_ticker_parse_audit"]
    parse_ok = int((parse_audit["ticker_parse_status"] == "parse_ok").sum()) if len(parse_audit) else 0
    parse_fail = int((parse_audit["ticker_parse_status"] != "parse_ok").sum()) if len(parse_audit) else 0
    print(f"Ticker parse status: parse_ok={parse_ok}, parse_failed={parse_fail}")

    if len(feature_df):
        cov_cols = [
            "ret_20d_pre",
            "avg_dollar_volume_20d_pre",
            "realized_vol_20d_pre",
            "market_cap_approx",
            "share_turnover_at_event_anchor",
        ]
        cov = feature_df[cov_cols].notna().mean().sort_values(ascending=False)
        print("\nFeature coverage rates:")
        for c, v in cov.items():
            print(f"  - {c}: {v:.1%}")

    print("\nKey empirical findings (descriptive, not causal):")
    for f in findings:
        print(f"  - {f}")

    print("\nSample-size and robustness caveats:")
    print("  - This pipeline is descriptive event research; it does not claim predictive alpha.")
    print("  - Event counts are limited; many subgroup slices can become small.")
    print("  - Post-event drift summaries should be interpreted with caution when n is small.")

    print("\nHighest-value next data additions:")
    print("  - Candidate non-event universe panel (for future controlled comparisons).")
    print("  - Point-in-time shares/float history for stronger turnover/market-cap estimates.")
    print("  - Point-in-time fundamentals with filing-date alignment.")
    print("  - Sector / industry histories with as-of timestamps.")
    print("  - Index eligibility rule histories and policy-change calendars.")

    # Save findings text
    with open(os.path.join(insights, "key_findings.txt"), "w", encoding="utf-8") as f:
        for line in findings:
            f.write(f"- {line}\n")

    print(f"\nAll outputs saved under: {cfg.output_dir}")
    print("  core/          — normalized events, eligible list, enrichment universes, event summaries")
    print("  diagnostics/   — parse/normalization audits, validation tables")
    print("  coverage/      — yfinance download audits, price coverage reports")
    print("  features/      — event_features_enriched, feature audit")
    print("  event_study/   — daily windows, aggregated event-study tables")
    print("  insights/      — comparison tables, key_findings.txt")
    print("  charts/        — PNG figures")


def main() -> None:
    cfg = Config()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

