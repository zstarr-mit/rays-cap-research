"""
Build a post-inclusion "keep vs sell" training table for Stock Connect inclusions.

Definition (configurable):
- Universe: Stock Connect Inclusion events from normalized master events.
- Decision as-of date: decision_asof_date = event_date - lag_days (calendar days).
- Label: y_keep = 1 if forward total return over next N TRADING days >= threshold, else 0.

Features:
- Point-in-time-ish trailing valuation metrics (best-effort):
  - market_cap_approx_asof = close_asof * sharesOutstanding_static (approx; flagged)
  - ps_ttm = market_cap_approx_asof / revenue_ttm_asof
  - pe_ttm = market_cap_approx_asof / net_income_ttm_asof
- Fundamentals pulled from yfinance quarterly statements and aligned as-of decision_asof_date
  using a staleness gate (max_staleness_days).
- Optional forward PE snapshot (NOT point-in-time; explicitly flagged as snapshot).
- Peer grouping: hybrid sector/industry auto + manual overrides.
  - peer_group resolved: override if present else (sector||industry) if available.
  - peer-relative z-scores computed within peer_group.

Outputs (under out_dir):
- keep_sell_training_table.csv
- label_diagnostics.csv
- valuation_feature_coverage.csv
- peer_group_coverage.csv
- price_audit.csv
- meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Missing dependency `yfinance`. Install with: pip install yfinance") from e


# Local imports (scripts/ on path when run from repo root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hk_event_research_pipeline import (  # noqa: E402
    Config as HKConfig,
    _align_to_prev_trading_day,
    _fetch_static_shares_outstanding,
    download_yfinance_history,
)


@dataclass(frozen=True)
class Config:
    events_path: str = "data/research_outputs/core/master_events_normalized.csv"
    out_dir: str = "data/research_outputs/stock_connect_keep_sell"

    # Label spec
    horizon_trading_days: int = 60
    keep_return_threshold: float = 0.0
    lag_days: int = 0

    # Fundamentals
    fundamentals_cache_dir: str = "data/yfinance_fundamentals_cache"
    max_staleness_days: int = 540

    # Peer grouping overrides (optional)
    peer_overrides_csv: str = "data/peer_overrides.csv"
    peer_info_cache_dir: str = "data/yfinance_peer_info_cache"

    # Forward PE snapshot (NOT point-in-time)
    include_forward_pe_snapshot: bool = False

    # Filters
    primary_only: bool = True
    index_contains: str = "Stock Connect"

    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.fundamentals_cache_dir, exist_ok=True)
    os.makedirs(cfg.peer_info_cache_dir, exist_ok=True)


def _load_events(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.events_path):
        raise FileNotFoundError(cfg.events_path)
    df = pd.read_csv(cfg.events_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "ticker_yf" not in df.columns and "ticker_yf_normalized" in df.columns:
        df["ticker_yf"] = df["ticker_yf_normalized"]
    df = df[df["ticker_yf"].notna()].copy()
    if cfg.primary_only and "is_primary_symbol" in df.columns:
        df = df[df["is_primary_symbol"].astype(int) == 1].copy()
    if cfg.index_contains:
        m = df["index"].astype(str).str.contains(str(cfg.index_contains), case=False, na=False)
        df = df.loc[m].copy()
    return df.reset_index(drop=True)


def _filter_stock_connect_inclusions(events: pd.DataFrame) -> pd.DataFrame:
    d = events.copy()
    d = d[d["change_type"].isin(["Inclusion"])].copy()
    d = d.sort_values(["date", "ticker_yf"]).reset_index(drop=True)
    return d


def _fund_cache_path(ticker: str, cfg: Config) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(ticker))
    return os.path.join(cfg.fundamentals_cache_dir, f"{safe}.pkl")


def fetch_fundamental_statements_cached(ticker: str, cfg: Config) -> Dict[str, pd.DataFrame]:
    path = _fund_cache_path(ticker, cfg)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass

    payload: Dict[str, pd.DataFrame] = {}
    try:
        t = yf.Ticker(ticker)
        payload["quarterly_financials"] = getattr(t, "quarterly_financials", pd.DataFrame())
        payload["quarterly_balance_sheet"] = getattr(t, "quarterly_balance_sheet", pd.DataFrame())
        payload["quarterly_cashflow"] = getattr(t, "quarterly_cashflow", pd.DataFrame())
    except Exception:
        payload["quarterly_financials"] = pd.DataFrame()
        payload["quarterly_balance_sheet"] = pd.DataFrame()
        payload["quarterly_cashflow"] = pd.DataFrame()

    pd.to_pickle(payload, path)
    return payload


def _pick_metric(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return None


def build_quarterly_feature_frame(payload: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    qf = payload.get("quarterly_financials", pd.DataFrame())
    qb = payload.get("quarterly_balance_sheet", pd.DataFrame())
    qc = payload.get("quarterly_cashflow", pd.DataFrame())

    if qf is None or qf.empty:
        return pd.DataFrame()

    qf = qf.copy()
    qf.columns = pd.to_datetime(qf.columns, errors="coerce")
    qf = qf.loc[:, qf.columns.notna()]

    qb = qb.copy() if qb is not None else pd.DataFrame()
    qb.columns = pd.to_datetime(qb.columns, errors="coerce")
    qb = qb.loc[:, qb.columns.notna()]

    qc = qc.copy() if qc is not None else pd.DataFrame()
    qc.columns = pd.to_datetime(qc.columns, errors="coerce")
    qc = qc.loc[:, qc.columns.notna()]

    cols = sorted(qf.columns)
    out = pd.DataFrame(index=cols)

    rev = _pick_metric(qf, ["Total Revenue", "Revenue", "Total Revenues", "TotalRevenue"])
    ni = _pick_metric(qf, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    eq = _pick_metric(qb, ["Total Stockholder Equity", "Stockholders Equity", "Total equity", "TotalEquity"])
    ocf = _pick_metric(qc, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"])

    def align(s: Optional[pd.Series]) -> pd.Series:
        if s is None:
            return pd.Series(index=cols, dtype=float)
        z = pd.to_numeric(s, errors="coerce")
        z.index = pd.to_datetime(z.index, errors="coerce")
        return z.reindex(cols)

    out["revenue_q"] = align(rev)
    out["net_income_q"] = align(ni)
    out["equity_q"] = align(eq)
    out["ocf_q"] = align(ocf)

    out["revenue_ttm"] = out["revenue_q"].rolling(4, min_periods=4).sum()
    out["net_income_ttm"] = out["net_income_q"].rolling(4, min_periods=4).sum()
    out["ocf_ttm"] = out["ocf_q"].rolling(4, min_periods=4).sum()

    out["roe"] = out["net_income_ttm"] / out["equity_q"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "period_end"
    return out.sort_index()


def asof_fundamental_features(qframe: pd.DataFrame, asof_date: pd.Timestamp, cfg: Config) -> Dict[str, float]:
    base = {
        "fundamentals_available": 0,
        "fundamental_staleness_days": np.nan,
        "revenue_ttm_asof": np.nan,
        "net_income_ttm_asof": np.nan,
        "ocf_ttm_asof": np.nan,
        "roe_asof": np.nan,
        "rev_ttm_qoq_trend": np.nan,
        "ni_ttm_qoq_trend": np.nan,
        "ocf_ttm_qoq_trend": np.nan,
    }
    if qframe is None or qframe.empty:
        return base

    q = qframe[qframe.index <= pd.Timestamp(asof_date)].copy()
    if q.empty:
        return base

    last = q.iloc[-1]
    prev = q.iloc[-2] if len(q) >= 2 else None
    period_end = pd.Timestamp(q.index[-1])
    staleness_days = float((pd.Timestamp(asof_date) - period_end).days)
    if staleness_days > cfg.max_staleness_days:
        out = dict(base)
        out["fundamental_staleness_days"] = staleness_days
        return out

    def pct(a: float, b: Optional[float]) -> float:
        if b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return (a / b) - 1.0

    out = dict(base)
    out["fundamentals_available"] = 1
    out["fundamental_staleness_days"] = staleness_days
    out["revenue_ttm_asof"] = float(last.get("revenue_ttm", np.nan))
    out["net_income_ttm_asof"] = float(last.get("net_income_ttm", np.nan))
    out["ocf_ttm_asof"] = float(last.get("ocf_ttm", np.nan))
    out["roe_asof"] = float(last.get("roe", np.nan))
    out["rev_ttm_qoq_trend"] = pct(out["revenue_ttm_asof"], None if prev is None else float(prev.get("revenue_ttm", np.nan)))
    out["ni_ttm_qoq_trend"] = pct(out["net_income_ttm_asof"], None if prev is None else float(prev.get("net_income_ttm", np.nan)))
    out["ocf_ttm_qoq_trend"] = pct(out["ocf_ttm_asof"], None if prev is None else float(prev.get("ocf_ttm", np.nan)))
    return out


def _peer_info_cache_path(ticker: str, cfg: Config) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(ticker))
    return os.path.join(cfg.peer_info_cache_dir, f"{safe}.json")


def fetch_peer_info_cached(ticker: str, cfg: Config) -> Dict[str, object]:
    path = _peer_info_cache_path(ticker, cfg)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    out: Dict[str, object] = {"ticker_yf": ticker, "sector": None, "industry": None, "forwardPE": None}
    try:
        info = yf.Ticker(ticker).info or {}
        out["sector"] = info.get("sector", None)
        out["industry"] = info.get("industry", None)
        out["forwardPE"] = info.get("forwardPE", None)
    except Exception:
        pass

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def load_peer_overrides(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["ticker_yf", "peer_group_override", "is_commodity_like"])
    df = pd.read_csv(path, dtype=str)
    if "ticker_yf" not in df.columns:
        raise KeyError("peer_overrides CSV must include column `ticker_yf`")
    if "peer_group_override" not in df.columns:
        df["peer_group_override"] = ""
    if "is_commodity_like" not in df.columns:
        df["is_commodity_like"] = ""
    df["ticker_yf"] = df["ticker_yf"].astype(str).str.strip()
    df["peer_group_override"] = df["peer_group_override"].astype(str).str.strip()
    df["is_commodity_like"] = df["is_commodity_like"].astype(str).str.strip()
    return df.drop_duplicates(subset=["ticker_yf"]).reset_index(drop=True)


def _resolve_peer_group(sector: Optional[str], industry: Optional[str], override: Optional[str]) -> Optional[str]:
    ov = (override or "").strip()
    if ov:
        return ov
    s = (sector or "").strip()
    i = (industry or "").strip()
    if not s and not i:
        return None
    return f"{s}||{i}"


def _peer_zscores(df: pd.DataFrame, metric_col: str, group_col: str = "peer_group", min_group_n: int = 5) -> pd.DataFrame:
    d = df.copy()
    z_col = f"{metric_col}_peer_z"
    n_col = f"{metric_col}_peer_n"
    pct_col = f"{metric_col}_peer_pctile"
    d[z_col] = np.nan
    d[n_col] = 0
    d[pct_col] = np.nan

    valid = d[d[group_col].notna() & d[metric_col].notna()].copy()
    if valid.empty:
        return d

    stats = (
        valid.groupby(group_col)[metric_col]
        .agg(peer_mean="mean", peer_std="std", peer_n="size")
        .reset_index()
    )
    d = d.merge(stats, on=group_col, how="left")
    ok = (d["peer_n"] >= min_group_n) & d["peer_std"].notna() & (d["peer_std"] > 0) & d[metric_col].notna()
    d.loc[ok, z_col] = (d.loc[ok, metric_col] - d.loc[ok, "peer_mean"]) / d.loc[ok, "peer_std"]
    d[n_col] = d["peer_n"].fillna(0).astype(int)

    # Percentile within peer group (ties averaged)
    def _pct_rank(s: pd.Series) -> pd.Series:
        # rank(pct=True) returns 0..1. Use average for ties.
        return s.rank(pct=True, method="average")

    pct = valid.groupby(group_col)[metric_col].transform(_pct_rank)
    d.loc[valid.index, pct_col] = pct

    d = d.drop(columns=[c for c in ["peer_mean", "peer_std", "peer_n"] if c in d.columns])
    return d


def compute_forward_return_label(
    hist: pd.DataFrame,
    decision_asof_date: pd.Timestamp,
    horizon_trading_days: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[float]]:
    """
    Returns (anchor_date, horizon_date, total_return) where:
      anchor_date = last trading day strictly before decision_asof_date (consistent with _align_to_prev_trading_day)
      horizon_date = trading day N sessions after anchor_date
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, None, None
    h = hist.copy()
    h.index = pd.to_datetime(h.index).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()

    anchor = _align_to_prev_trading_day(h, pd.Timestamp(decision_asof_date).normalize())
    if anchor is None or anchor not in h.index:
        return None, None, None

    i0 = int(h.index.get_loc(anchor))
    i1 = i0 + int(horizon_trading_days)
    if i1 >= len(h.index):
        return anchor, None, None

    p0 = float(h.loc[anchor, "Close"])
    t1 = pd.Timestamp(h.index[i1])
    p1 = float(h.iloc[i1]["Close"])
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1):
        return anchor, t1, None
    return anchor, t1, (p1 / p0) - 1.0


def build_dataset(cfg: Config) -> Dict[str, pd.DataFrame]:
    ensure_dirs(cfg)
    events = _load_events(cfg)
    incl = _filter_stock_connect_inclusions(events)
    if incl.empty:
        raise ValueError("No Stock Connect Inclusion rows found after filters.")

    incl = incl.copy()
    incl["event_date"] = pd.to_datetime(incl["date"]).dt.normalize()
    incl["decision_asof_date"] = incl["event_date"] - pd.Timedelta(days=int(cfg.lag_days))

    tickers = sorted(incl["ticker_yf"].astype(str).unique().tolist())
    universe = pd.DataFrame({"ticker_yf": tickers})

    hk_cfg = HKConfig()
    # Download bounds: give plenty of history for anchor and future horizon.
    min_dt = pd.Timestamp(incl["decision_asof_date"].min()) - pd.Timedelta(days=600)
    max_dt = pd.Timestamp(incl["event_date"].max()) + pd.Timedelta(days=600)
    histories, price_audit = download_yfinance_history(universe, hk_cfg, min_dt, max_dt)

    overrides = load_peer_overrides(cfg.peer_overrides_csv)
    ov_map = overrides.set_index("ticker_yf")["peer_group_override"].to_dict() if len(overrides) else {}
    commodity_map = overrides.set_index("ticker_yf")["is_commodity_like"].to_dict() if len(overrides) else {}

    # Fundamentals cache (quarterly frames)
    qcache: Dict[str, pd.DataFrame] = {}
    shares_cache: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    peer_cache: Dict[str, Dict[str, object]] = {}

    for t in tickers:
        payload = fetch_fundamental_statements_cached(t, cfg)
        qcache[t] = build_quarterly_feature_frame(payload)
        shares_cache[t] = _fetch_static_shares_outstanding(t)
        peer_cache[t] = fetch_peer_info_cached(t, cfg)

    rows: List[Dict[str, object]] = []
    skip: List[Dict[str, object]] = []

    for _, r in incl.iterrows():
        t = str(r["ticker_yf"])
        hist = histories.get(t)
        event_date = pd.Timestamp(r["event_date"])
        asof = pd.Timestamp(r["decision_asof_date"])

        anchor, horizon_dt, fwd_ret = compute_forward_return_label(hist, asof, cfg.horizon_trading_days)
        if anchor is None:
            skip.append({"ticker_yf": t, "event_date": str(event_date.date()), "reason": "no_anchor"})
            continue

        y_keep = None if fwd_ret is None else int(float(fwd_ret) >= float(cfg.keep_return_threshold))

        sector = peer_cache.get(t, {}).get("sector", None)
        industry = peer_cache.get(t, {}).get("industry", None)
        forward_pe = peer_cache.get(t, {}).get("forwardPE", None) if cfg.include_forward_pe_snapshot else None
        peer_group = _resolve_peer_group(sector, industry, ov_map.get(t, ""))

        # Market cap approx at anchor close
        close_anchor = np.nan
        if hist is not None and not hist.empty and anchor in pd.to_datetime(hist.index).normalize():
            h = hist.copy()
            h.index = pd.to_datetime(h.index).normalize()
            h = h[~h.index.duplicated(keep="last")].sort_index()
            try:
                close_anchor = float(h.loc[anchor, "Close"])
            except Exception:
                close_anchor = np.nan

        sh, sh_src = shares_cache.get(t, (None, None))
        market_cap_approx = np.nan
        market_cap_available = 0
        market_cap_is_approx = 0
        if sh is not None and np.isfinite(close_anchor):
            market_cap_approx = float(close_anchor) * float(sh)
            market_cap_available = 1
            market_cap_is_approx = 1

        f = asof_fundamental_features(qcache.get(t, pd.DataFrame()), asof, cfg)
        revenue_ttm = f.get("revenue_ttm_asof", np.nan)
        net_income_ttm = f.get("net_income_ttm_asof", np.nan)

        ps_ttm = np.nan
        pe_ttm = np.nan
        if market_cap_available and pd.notna(revenue_ttm) and float(revenue_ttm) != 0:
            ps_ttm = float(market_cap_approx) / float(revenue_ttm)
        if market_cap_available and pd.notna(net_income_ttm) and float(net_income_ttm) != 0:
            pe_ttm = float(market_cap_approx) / float(net_income_ttm)

        out = {
            "ticker_yf": t,
            "event_date": event_date,
            "decision_asof_date": asof,
            "label_anchor_trade_date": anchor,
            "label_horizon_trade_date": horizon_dt,
            "horizon_trading_days": int(cfg.horizon_trading_days),
            "keep_return_threshold": float(cfg.keep_return_threshold),
            "forward_total_return": fwd_ret,
            "y_keep": y_keep,
            "index": r.get("index", None),
            "company": r.get("company", None),
            "sector_snapshot": sector,
            "industry_snapshot": industry,
            "peer_group": peer_group,
            "is_commodity_like": str(commodity_map.get(t, "")).strip(),
            "shares_outstanding_static": float(sh) if sh is not None else np.nan,
            "shares_source": sh_src,
            "market_cap_available": int(market_cap_available),
            "market_cap_is_approx": int(market_cap_is_approx),
            "market_cap_approx_asof": market_cap_approx,
            "ps_ttm": ps_ttm,
            "pe_ttm": pe_ttm,
            "forward_pe_snapshot": forward_pe,
            "forward_pe_is_snapshot": int(1 if (cfg.include_forward_pe_snapshot and forward_pe is not None) else 0),
        }
        out.update(f)
        rows.append(out)

    df = pd.DataFrame(rows)
    skip_df = pd.DataFrame(skip)

    if df.empty:
        raise RuntimeError("No rows produced; check price coverage and event filters.")

    # Peer-relative z-scores
    for col in ["ps_ttm", "pe_ttm"]:
        df = _peer_zscores(df, col, group_col="peer_group", min_group_n=5)
    if cfg.include_forward_pe_snapshot:
        df = _peer_zscores(df, "forward_pe_snapshot", group_col="peer_group", min_group_n=5)

    # Diagnostics
    label_diag = pd.DataFrame(
        [
            {"metric": "rows_total", "value": int(len(df))},
            {"metric": "rows_skipped_no_anchor", "value": int(len(skip_df))},
            {"metric": "y_keep_known", "value": int(df["y_keep"].notna().sum())},
            {"metric": "keep_rate", "value": float(df["y_keep"].dropna().mean()) if df["y_keep"].notna().any() else np.nan},
        ]
    )

    cov = pd.DataFrame(
        [
            {"metric": "market_cap_available", "value": float(df["market_cap_available"].mean())},
            {"metric": "fundamentals_available", "value": float(df["fundamentals_available"].mean())},
            {"metric": "ps_ttm_available", "value": float(df["ps_ttm"].notna().mean())},
            {"metric": "pe_ttm_available", "value": float(df["pe_ttm"].notna().mean())},
            {"metric": "peer_group_available", "value": float(df["peer_group"].notna().mean())},
        ]
    )

    peer_cov = (
        df.assign(peer_group_missing=df["peer_group"].isna().astype(int))
        .groupby("peer_group_missing")
        .size()
        .reset_index(name="rows")
    )

    # Persist
    df_out_path = os.path.join(cfg.out_dir, "keep_sell_training_table.csv")
    df.to_csv(df_out_path, index=False)
    skip_df.to_csv(os.path.join(cfg.out_dir, "skipped_rows.csv"), index=False)
    price_audit.to_csv(os.path.join(cfg.out_dir, "price_audit.csv"), index=False)
    label_diag.to_csv(os.path.join(cfg.out_dir, "label_diagnostics.csv"), index=False)
    cov.to_csv(os.path.join(cfg.out_dir, "valuation_feature_coverage.csv"), index=False)
    peer_cov.to_csv(os.path.join(cfg.out_dir, "peer_group_coverage.csv"), index=False)

    meta = {
        "events_path": cfg.events_path,
        "out_dir": cfg.out_dir,
        "horizon_trading_days": cfg.horizon_trading_days,
        "keep_return_threshold": cfg.keep_return_threshold,
        "lag_days": cfg.lag_days,
        "max_staleness_days": cfg.max_staleness_days,
        "peer_overrides_csv": cfg.peer_overrides_csv,
        "include_forward_pe_snapshot": cfg.include_forward_pe_snapshot,
        "rows": int(len(df)),
        "skipped_rows": int(len(skip_df)),
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "keep_sell_training_table": df,
        "skipped_rows": skip_df,
        "label_diagnostics": label_diag,
        "valuation_feature_coverage": cov,
        "peer_group_coverage": peer_cov,
        "price_audit": price_audit,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build Stock Connect keep/sell dataset (post-inclusion).")
    p.add_argument("--events-path", default="data/research_outputs/core/master_events_normalized.csv")
    p.add_argument("--out-dir", default="data/research_outputs/stock_connect_keep_sell")
    p.add_argument("--horizon-trading-days", type=int, default=60)
    p.add_argument("--keep-return-threshold", type=float, default=0.0)
    p.add_argument("--lag-days", type=int, default=0)
    p.add_argument("--max-staleness-days", type=int, default=540)
    p.add_argument("--peer-overrides-csv", default="data/peer_overrides.csv")
    p.add_argument("--include-forward-pe-snapshot", action="store_true")
    p.add_argument("--no-primary-only", action="store_true")
    p.add_argument("--index-contains", default="Stock Connect")
    args = p.parse_args()

    cfg = Config(
        events_path=args.events_path,
        out_dir=args.out_dir,
        horizon_trading_days=args.horizon_trading_days,
        keep_return_threshold=args.keep_return_threshold,
        lag_days=args.lag_days,
        max_staleness_days=args.max_staleness_days,
        peer_overrides_csv=args.peer_overrides_csv,
        include_forward_pe_snapshot=bool(args.include_forward_pe_snapshot),
        primary_only=not bool(args.no_primary_only),
        index_contains=str(args.index_contains or "").strip(),
    )

    build_dataset(cfg)
    print("[done] wrote dataset to:", cfg.out_dir)


if __name__ == "__main__":
    main()

