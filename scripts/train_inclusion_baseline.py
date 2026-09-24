"""
Binary baseline: inclusion vs (exclusion + removal), features as-of ~120 calendar days pre-event.

For each primary (event_date, ticker_yf) in the normalized master:
  - y_inclusion_binary = 1 if Inclusion (wins ties if multiple indices same day), else 0 for Exclusion/Removal.
  - feature_anchor_date = last trading day in that ticker's history on or before (event_date - calendar_days_before).

Uses the same market feature block as hk_event_research_pipeline (returns, liquidity, vol, market cap proxy).
Walk-forward CV is by calendar month of event_date (no random shuffle).

Requires: run from repo root or PYTHONPATH including scripts; yfinance; cached prices under hk cache_dir.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Local imports (scripts/ on path when run as python scripts/train_inclusion_baseline.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from build_review_cycle_panel import load_events_auto  # noqa: E402
from hk_event_research_pipeline import (  # noqa: E402
    Config as HKConfig,
    _align_to_prev_trading_day,
    _fetch_static_shares_outstanding,
    compute_liquidity_features,
    compute_market_cap_proxy,
    compute_pre_event_returns,
    compute_volatility_features,
    download_yfinance_history,
)


@dataclass(frozen=True)
class Config:
    # Raw master or master_events_normalized.csv
    events_path: str = "data/hang_seng_events_2016_2026_master.csv"
    out_dir: str = "data/research_outputs/ml_baseline"
    calendar_days_before: int = 120
    target_col: str = "y_inclusion_binary"
    # Optional substring filter on index column; None = all indices
    index_contains: Optional[str] = None

    n_folds: int = 4
    min_train_months: int = 12
    test_months_per_fold: int = 6
    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)


def _apply_index_filter(df: pd.DataFrame, index_contains: Optional[str]) -> pd.DataFrame:
    if index_contains is None or str(index_contains).strip() == "":
        return df
    m = df["index"].astype(str).str.contains(str(index_contains).strip(), case=False, na=False)
    return df.loc[m].copy()


def build_dedup_event_table(events: pd.DataFrame, index_contains: Optional[str]) -> pd.DataFrame:
    d = events.copy()
    if "is_primary_symbol" in d.columns:
        d = d[d["is_primary_symbol"].astype(int) == 1].copy()
    d = d[d["change_type"].isin(["Inclusion", "Exclusion", "Removal"])].copy()
    d = _apply_index_filter(d, index_contains)
    if d.empty:
        raise ValueError("No events after filters (primary + Inclusion/Exclusion/Removal + optional index filter).")

    rows: List[Dict[str, object]] = []
    for (evt_dt, tkr), g in d.groupby(["date", "ticker_yf"]):
        ct = set(g["change_type"].astype(str))
        y = 1 if "Inclusion" in ct else 0
        idx_name = str(g["index"].iloc[0])
        if y == 1:
            neg_type = ""
        else:
            neg_type = "Exclusion" if "Exclusion" in ct else "Removal"
        rows.append(
            {
                "event_date": pd.Timestamp(evt_dt).normalize(),
                "ticker_yf": str(tkr),
                "y_inclusion_binary": y,
                "index": idx_name,
                "neg_subtype": neg_type,
            }
        )
    out = pd.DataFrame(rows).sort_values(["event_date", "ticker_yf"]).reset_index(drop=True)
    return out


def compute_features_at_pre_event_anchor(
    tkr: str,
    hist: pd.DataFrame,
    event_date: pd.Timestamp,
    calendar_days_before: int,
    hk_cfg: HKConfig,
    shares_cache: Dict[str, Tuple[Optional[float], Optional[str]]],
) -> Optional[Dict[str, object]]:
    if hist is None or hist.empty:
        return None
    boundary = pd.Timestamp(event_date).normalize() - pd.Timedelta(days=int(calendar_days_before))
    h = hist.copy()
    h.index = pd.to_datetime(h.index).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    anchor = _align_to_prev_trading_day(h, boundary)
    if anchor is None or anchor not in h.index:
        return None

    f1 = compute_pre_event_returns(h, anchor)
    f2 = compute_liquidity_features(h, anchor)
    f3 = compute_volatility_features(h, anchor)
    if tkr not in shares_cache:
        shares_cache[tkr] = _fetch_static_shares_outstanding(tkr)
    sh, src = shares_cache[tkr]
    f4 = compute_market_cap_proxy({**f1, **f2, **f3}, sh, src, hk_cfg)

    return {
        "feature_anchor_date": anchor,
        "feature_boundary_calendar": boundary,
        "calendar_days_before": int(calendar_days_before),
        **f1,
        **f2,
        **f3,
        **f4,
    }


def build_training_frame(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = load_events_auto(cfg.events_path)
    base = build_dedup_event_table(events, cfg.index_contains)

    hk_cfg = HKConfig()
    min_dt = pd.Timestamp(base["event_date"].min()) - pd.Timedelta(days=450)
    max_dt = pd.Timestamp(base["event_date"].max()) + pd.Timedelta(days=5)
    universe = pd.DataFrame({"ticker_yf": sorted(base["ticker_yf"].astype(str).unique())})
    histories, audit = download_yfinance_history(universe, hk_cfg, min_dt, max_dt)
    audit_path = os.path.join(cfg.out_dir, "binary_baseline_price_audit.csv")
    audit.to_csv(audit_path, index=False)

    shares_cache: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    feat_rows: List[Dict[str, object]] = []
    skip_rows: List[Dict[str, object]] = []

    for _, r in base.iterrows():
        tkr = str(r["ticker_yf"])
        evt = pd.Timestamp(r["event_date"])
        hist = histories.get(tkr)
        feats = compute_features_at_pre_event_anchor(
            tkr, hist, evt, cfg.calendar_days_before, hk_cfg, shares_cache
        )
        if feats is None:
            skip_rows.append({"ticker_yf": tkr, "event_date": str(evt.date()), "reason": "no_history_or_anchor"})
            continue
        row = {**r.to_dict(), **feats}
        feat_rows.append(row)

    train_df = pd.DataFrame(feat_rows)
    if train_df.empty:
        raise RuntimeError("No rows with valid features; check price download and calendar_days_before.")

    train_df["event_date"] = pd.to_datetime(train_df["event_date"]).dt.normalize()
    train_df["year"] = train_df["event_date"].dt.year
    train_df["month"] = train_df["event_date"].dt.month
    train_df = train_df.sort_values(["ticker_yf", "event_date"]).reset_index(drop=True)
    train_df["prior_inclusion_count_ticker"] = (
        train_df.groupby("ticker_yf")[cfg.target_col].transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    skip_df = pd.DataFrame(skip_rows)
    return train_df, skip_df


def build_walk_forward_splits(df: pd.DataFrame, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    d = df.copy()
    d["ym"] = d["event_date"].dt.to_period("M").astype(str)
    months = sorted(d["ym"].unique().tolist())
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    if len(months) < (cfg.min_train_months + cfg.test_months_per_fold):
        return splits
    start_test = cfg.min_train_months
    fold = 0
    while start_test + cfg.test_months_per_fold <= len(months) and fold < cfg.n_folds:
        train_months = months[:start_test]
        test_months = months[start_test : start_test + cfg.test_months_per_fold]
        tr_idx = d.index[d["ym"].isin(train_months)].to_numpy()
        te_idx = d.index[d["ym"].isin(test_months)].to_numpy()
        label = f"train<= {train_months[-1]} | test={test_months[0]}..{test_months[-1]}"
        splits.append((tr_idx, te_idx, label))
        start_test += cfg.test_months_per_fold
        fold += 1
    return splits


def get_feature_columns(work: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Numeric and categorical columns passed to the logistic baseline."""
    num_features = [
        c
        for c in [
            "year",
            "month",
            "prior_inclusion_count_ticker",
            "ret_5d_pre",
            "ret_20d_pre",
            "ret_60d_pre",
            "dist_from_60d_high",
            "dist_from_252d_high",
            "avg_volume_20d_pre",
            "avg_volume_60d_pre",
            "avg_dollar_volume_20d_pre",
            "avg_dollar_volume_60d_pre",
            "realized_vol_20d_pre",
            "realized_vol_60d_pre",
            "market_cap_approx",
            "share_turnover_at_event_anchor",
        ]
        if c in work.columns
    ]
    cat_features = [c for c in ["index"] if c in work.columns]
    return num_features, cat_features


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(np.floor(n * k_frac)))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top]))


def train_and_evaluate(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = df.copy()
    work[cfg.target_col] = work[cfg.target_col].fillna(0).astype(int)

    num_features, cat_features = get_feature_columns(work)

    splits = build_walk_forward_splits(work, cfg)
    if len(splits) == 0:
        raise ValueError("Not enough month coverage for walk-forward splits.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_state)),
        ]
    )

    rows = []
    for fold_i, (tr_idx, te_idx, label) in enumerate(splits, 1):
        tr = work.iloc[tr_idx].copy()
        te = work.iloc[te_idx].copy()
        y_tr = tr[cfg.target_col].values
        y_te = te[cfg.target_col].values

        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            rows.append(
                {
                    "fold": fold_i,
                    "split_label": label,
                    "train_rows": len(tr),
                    "test_rows": len(te),
                    "train_pos_rate": float(np.mean(y_tr)),
                    "test_pos_rate": float(np.mean(y_te)),
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "precision_at_1pct": np.nan,
                    "precision_at_5pct": np.nan,
                    "precision_at_10pct": np.nan,
                    "status": "skipped_one_class",
                }
            )
            continue

        X_tr = tr[num_features + cat_features]
        X_te = te[num_features + cat_features]
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]

        rows.append(
            {
                "fold": fold_i,
                "split_label": label,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_pos_rate": float(np.mean(y_tr)),
                "test_pos_rate": float(np.mean(y_te)),
                "pr_auc": float(average_precision_score(y_te, p)),
                "roc_auc": float(roc_auc_score(y_te, p)),
                "precision_at_1pct": precision_at_k(y_te, p, 0.01),
                "precision_at_5pct": precision_at_k(y_te, p, 0.05),
                "precision_at_10pct": precision_at_k(y_te, p, 0.10),
                "status": "ok",
            }
        )

    fold_df = pd.DataFrame(rows)
    summary = {
        "task": "binary_inclusion_vs_exclusion_removal",
        "target_col": cfg.target_col,
        "calendar_days_before_feature_anchor": cfg.calendar_days_before,
        "events_path": cfg.events_path,
        "rows_total": int(len(work)),
        "pos_rate_total": float(work[cfg.target_col].mean()),
        "n_folds": int(len(fold_df)),
        "n_ok_folds": int((fold_df["status"] == "ok").sum()) if len(fold_df) else 0,
        "avg_pr_auc_ok_folds": float(fold_df.loc[fold_df["status"] == "ok", "pr_auc"].mean()) if len(fold_df) else np.nan,
        "avg_roc_auc_ok_folds": float(fold_df.loc[fold_df["status"] == "ok", "roc_auc"].mean()) if len(fold_df) else np.nan,
    }
    return fold_df, summary


def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)

    train_df, skip_df = build_training_frame(cfg)
    train_path = os.path.join(cfg.out_dir, "train_table_snapshot.csv")
    train_df.to_csv(train_path, index=False)
    skip_path = os.path.join(cfg.out_dir, "binary_baseline_skipped_rows.csv")
    skip_df.to_csv(skip_path, index=False)

    diag = pd.DataFrame(
        [
            {"metric": "rows_with_features", "value": len(train_df)},
            {"metric": "rows_skipped_no_anchor", "value": len(skip_df)},
            {"metric": "target_positive_count", "value": int(train_df[cfg.target_col].sum())},
            {"metric": "target_positive_rate", "value": float(train_df[cfg.target_col].mean())},
        ]
    )
    diag_path = os.path.join(cfg.out_dir, "training_label_diagnostics.csv")
    diag.to_csv(diag_path, index=False)

    fold_df, summary = train_and_evaluate(train_df, cfg)
    fold_path = os.path.join(cfg.out_dir, "walk_forward_fold_metrics.csv")
    summary_path = os.path.join(cfg.out_dir, "walk_forward_summary.json")
    fold_df.to_csv(fold_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] binary inclusion baseline (120d pre-event features)")
    print(f"  - {train_path}")
    print(f"  - {skip_path}")
    print(f"  - {diag_path}")
    print(f"  - {fold_path}")
    print(f"  - {summary_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
