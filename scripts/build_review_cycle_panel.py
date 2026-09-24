"""
Build Option A training skeleton: review-cycle labels + pre-announcement as-of dates.

**Source rows**
- Default input is `data/hang_seng_events_2016_2026_master.csv`. Raw sheets are passed through
  `normalize_master_events` (same as hk_event_research_pipeline): multi-symbol cells become
  one row per yfinance ticker, `is_primary_symbol` picks the HK line when present.
- You can also pass `master_events_normalized.csv` to skip re-normalization.

**Announcement batch**
- Every distinct `date` in the filtered event table is one "batch" (quarterly rebalance days
  are shared across many index rows — that is why many rows share the same date).

**Labels (event tickers — from your data)**
- `included`  — `change_type` Inclusion
- `excluded`  — `change_type` Exclusion (distinct from Removal)
- `removed`   — `change_type` Removal
- Priority if a (ticker, date) has multiple rows: Inclusion > Exclusion > Removal.

**`no_change` (NOT in the master CSV)**
- Synthetic label for **eligible HK securities** (`eligible_securities.csv`) that have **no**
  index membership change row on that announcement `date` after the same filters
  (optional index substring, primary-only). Used as the negative / "no event" class for ML.

**Event rows (included / excluded / removed)** use **every** normalized primary `(date, ticker_yf)`
from the master; `in_eligible_universe` flags HK eligible-list membership (A-shares etc. are still kept).

`asof_date` is the last trading day on or before `announcement_date` minus K HK sessions
(^HSI history as calendar), i.e. features should use data through that close.

Outputs: CSV for downstream feature engineering (prices/fundamentals as-of `asof_date`).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Install yfinance: pip install yfinance") from e


@dataclass(frozen=True)
class Config:
    # Raw master CSV (normalized on load) or pre-built master_events_normalized.csv
    events_path: str = "data/hang_seng_events_2016_2026_master.csv"
    eligible_path: str = "data/eligible_securities.csv"
    out_dir: str = "data/research_outputs/review_cycle"
    # If None or empty: use every row in the events table (all indices). Else substring filter.
    index_contains: Optional[str] = None
    primary_only: bool = True
    # Trading days before the reference session (see script docstring).
    trading_days_before: int = 5
    # If set, emit one row per k in this list (long table). Else single `trading_days_before`.
    k_list: Optional[Tuple[int, ...]] = None
    include_no_change: bool = True
    max_no_change_per_announcement: Optional[int] = None
    random_state: int = 42
    # Reference index for HK trading calendar (Hang Seng Index).
    hk_calendar_symbol: str = "^HSI"


def _ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)


def _load_normalized_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_events_auto(path: str) -> pd.DataFrame:
    """
    If CSV already has `ticker_yf` (e.g. master_events_normalized.csv), load as-is.
    Otherwise treat as raw master and run normalize_master_events.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    peek = pd.read_csv(path, nrows=1)
    if "ticker_yf" in peek.columns:
        return _load_normalized_events(path)
    from hk_event_research_pipeline import Config as HKConfig, load_event_data, normalize_master_events

    cfg = HKConfig(event_csv=path)
    raw = load_event_data(cfg)
    df, _ = normalize_master_events(raw)
    return df


def _load_eligible_tickers(path: str) -> List[str]:
    """HK tickers from eligible_securities (same convention as hk_event_research_pipeline)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "证券代码" not in df.columns:
        raise KeyError("eligible_securities.csv must include column `证券代码`")
    df["证券代码"] = df["证券代码"].astype(str).str.strip()
    code_num = pd.to_numeric(df["证券代码"], errors="coerce")
    df = df[code_num.notna()].copy()
    df["code_num"] = code_num.dropna().astype(int)
    out: List[str] = []
    for c in df["code_num"].astype(int):
        if c < 10000:
            out.append(f"{c:04d}.HK")
        else:
            out.append(f"{int(c)}.HK")
    return sorted(set(out))


def _apply_index_filter(df: pd.DataFrame, index_contains: Optional[str], primary_only: bool) -> pd.DataFrame:
    d = df.copy()
    if index_contains is not None and str(index_contains).strip() != "":
        m = d["index"].astype(str).str.contains(str(index_contains).strip(), case=False, na=False)
        d = d.loc[m].copy()
    if primary_only and "is_primary_symbol" in d.columns:
        d = d[d["is_primary_symbol"].astype(int) == 1].copy()
    if "ticker_yf" not in d.columns:
        raise KeyError("Events must include `ticker_yf` after load (normalize raw master).")
    return d


def _priority_label(change_types: List[str]) -> str:
    """Single label for (ticker, date). Inclusion beats Exclusion beats Removal."""
    s = set(change_types)
    if "Inclusion" in s:
        return "included"
    if "Exclusion" in s:
        return "excluded"
    if "Removal" in s:
        return "removed"
    return "unknown"


def _build_event_labels(sc_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (announcement_date, ticker_yf) with outcome for tickers that changed."""
    rows = []
    for (dt, tkr), g in sc_df.groupby(["date", "ticker_yf"]):
        lab = _priority_label(g["change_type"].astype(str).tolist())
        rows.append(
            {
                "announcement_date": dt,
                "ticker_yf": tkr,
                "y_review": lab,
            }
        )
    return pd.DataFrame(rows)


def _fetch_hk_trading_days(cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    t = yf.Ticker(cfg.hk_calendar_symbol)
    h = t.history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False)
    if h is None or h.empty:
        raise RuntimeError(f"No history for {cfg.hk_calendar_symbol}; cannot build HK trading calendar.")
    idx = pd.to_datetime(h.index).tz_localize(None).normalize()
    return np.sort(idx.unique())


def _last_trading_day_on_or_before(trading_days: np.ndarray, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    dt64 = np.datetime64(pd.Timestamp(dt).normalize())
    pos = int(np.searchsorted(trading_days.astype("datetime64[ns]"), dt64, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(trading_days[pos])


def _asof_trading_days_before(
    trading_days: np.ndarray, announcement_date: pd.Timestamp, k: int
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
    """
    ref = last HK session on or before announcement_date.
    asof = k trading sessions before ref (not including ref).
    """
    ref = _last_trading_day_on_or_before(trading_days, announcement_date)
    if ref is None:
        return None, None, False
    pos = int(np.searchsorted(trading_days.astype("datetime64[ns]"), np.datetime64(ref), side="left"))
    if pos - k < 0:
        return ref, None, False
    asof = pd.Timestamp(trading_days[pos - k])
    return ref, asof, True


def build_panel(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = load_events_auto(cfg.events_path)
    sc = _apply_index_filter(events, cfg.index_contains, cfg.primary_only)

    if sc.empty:
        hint = f"index contains '{cfg.index_contains}'" if cfg.index_contains else "no index filter"
        raise ValueError(f"No rows after filter ({hint}). Check events_path.")

    eligible = _load_eligible_tickers(cfg.eligible_path)
    elig_set = set(eligible)

    ann_dates = sorted(sc["date"].dropna().unique())
    ann_dates = [pd.Timestamp(d).normalize() for d in ann_dates]

    min_d = min(ann_dates) - pd.Timedelta(days=400)
    max_d = max(ann_dates) + pd.Timedelta(days=5)
    trading_days = _fetch_hk_trading_days(cfg, min_d, max_d)

    event_labels = _build_event_labels(sc)

    k_values: List[int]
    if cfg.k_list is not None:
        k_values = list(cfg.k_list)
    else:
        k_values = [cfg.trading_days_before]

    rng = np.random.default_rng(cfg.random_state)

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for ann in ann_dates:
        sub = event_labels.loc[event_labels["announcement_date"] == ann]
        changed = {str(r["ticker_yf"]): str(r["y_review"]) for _, r in sub.iterrows()}

        # Per-announcement summary
        n_inc = int((sub["y_review"] == "included").sum())
        n_exc = int((sub["y_review"] == "excluded").sum())
        n_rem = int((sub["y_review"] == "removed").sum())
        summary_rows.append(
            {
                "announcement_date": ann,
                "n_event_rows_primary": int(len(sc.loc[sc["date"] == ann])),
                "n_tickers_included": n_inc,
                "n_tickers_excluded": n_exc,
                "n_tickers_removed": n_rem,
                "n_eligible_universe": len(eligible),
            }
        )

        for k in k_values:
            ref, asof, ok = _asof_trading_days_before(trading_days, ann, k)
            base = {
                "announcement_date": ann,
                "ref_trading_day": ref,
                "cutoff_trading_days_before": k,
                "asof_date": asof if ok else pd.NaT,
                "asof_calendar_ok": bool(ok),
            }

            # Every ticker with an event that day (all listings from master, not only HK eligible)
            for tk, lab in changed.items():
                r = dict(base)
                r["ticker_yf"] = tk
                r["y_review"] = lab
                r["in_eligible_universe"] = tk in elig_set
                r["row_kind"] = "event"
                rows.append(r)

            if not cfg.include_no_change:
                continue

            no_change_tickers = [t for t in eligible if t not in changed]
            if cfg.max_no_change_per_announcement is not None and len(no_change_tickers) > cfg.max_no_change_per_announcement:
                pick = rng.choice(
                    len(no_change_tickers), size=cfg.max_no_change_per_announcement, replace=False
                )
                no_change_tickers = [no_change_tickers[i] for i in sorted(pick)]

            for tk in no_change_tickers:
                r = dict(base)
                r["ticker_yf"] = tk
                r["y_review"] = "no_change"
                r["in_eligible_universe"] = True
                r["row_kind"] = "eligible_no_change"
                rows.append(r)

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel, pd.DataFrame(summary_rows)

    code_map = {"no_change": 0, "included": 1, "excluded": 2, "removed": 3, "unknown": -1}
    panel["y_review_code"] = panel["y_review"].map(code_map).astype(int)

    panel = panel.sort_values(["announcement_date", "cutoff_trading_days_before", "ticker_yf"]).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)
    return panel, summary_df


def main() -> None:
    p = argparse.ArgumentParser(description="Build Hang Seng review-cycle label panel (Option A).")
    p.add_argument(
        "--events-path",
        default="data/hang_seng_events_2016_2026_master.csv",
        help="Raw master CSV (normalized on load) or master_events_normalized.csv",
    )
    p.add_argument("--eligible-path", default="data/eligible_securities.csv")
    p.add_argument("--out-dir", default="data/research_outputs/review_cycle")
    p.add_argument(
        "--index-contains",
        default=None,
        metavar="SUBSTRING",
        help="Keep only rows whose index name contains this substring. Default: no filter (all indices).",
    )
    p.add_argument("--trading-days-before", type=int, default=5, help="K sessions before ref day (default 5).")
    p.add_argument(
        "--k-list",
        default=None,
        help="Comma-separated K values, e.g. 1,2,3,4,5. If set, overrides --trading-days-before.",
    )
    p.add_argument(
        "--exclude-no-change",
        action="store_true",
        help="Only emit rows for tickers with an index change that day (no synthetic no_change rows).",
    )
    p.add_argument("--max-no-change-per-announcement", type=int, default=None)
    p.add_argument("--no-primary-filter", action="store_true", help="Use all symbol rows (not recommended).")
    args = p.parse_args()

    k_list: Optional[Tuple[int, ...]] = None
    if args.k_list:
        k_list = tuple(int(x.strip()) for x in args.k_list.split(",") if x.strip())

    ic_arg = args.index_contains
    if ic_arg is None or (isinstance(ic_arg, str) and ic_arg.strip() == ""):
        index_contains_resolved: Optional[str] = None
    else:
        index_contains_resolved = str(ic_arg).strip()

    cfg = Config(
        events_path=args.events_path,
        eligible_path=args.eligible_path,
        out_dir=args.out_dir,
        index_contains=index_contains_resolved,
        primary_only=not args.no_primary_filter,
        trading_days_before=args.trading_days_before,
        k_list=k_list,
        include_no_change=not args.exclude_no_change,
        max_no_change_per_announcement=args.max_no_change_per_announcement,
    )

    _ensure_dirs(cfg)

    panel, summary = build_panel(cfg)

    panel_path = os.path.join(cfg.out_dir, "review_cycle_panel.csv")
    summary_path = os.path.join(cfg.out_dir, "review_cycle_announcement_summary.csv")
    meta_path = os.path.join(cfg.out_dir, "review_cycle_panel_meta.json")

    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "events_path": cfg.events_path,
        "eligible_path": cfg.eligible_path,
        "index_contains": cfg.index_contains,
        "primary_only": cfg.primary_only,
        "k_list": list(k_list) if k_list else [cfg.trading_days_before],
        "include_no_change": cfg.include_no_change,
        "max_no_change_per_announcement": cfg.max_no_change_per_announcement,
        "rows": int(len(panel)),
        "asof_ok_rate": float(panel["asof_calendar_ok"].mean()) if len(panel) else None,
        "label_counts": panel["y_review"].value_counts().to_dict() if len(panel) else {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] wrote {panel_path} rows={len(panel)}")
    print(f"       wrote {summary_path}")
    print(f"       wrote {meta_path}")
    if len(panel):
        print("\nLabel distribution:")
        print(panel["y_review"].value_counts().to_string())
        print(f"\nasof_calendar_ok rate: {panel['asof_calendar_ok'].mean():.3f}")


if __name__ == "__main__":
    main()
