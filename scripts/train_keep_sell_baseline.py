"""
Train a simple baseline model for Stock Connect keep/sell labels.

Input:
  data/research_outputs/stock_connect_keep_sell/keep_sell_training_table.csv

Target:
  y_keep (1 = keep, 0 = sell) based on forward return over next N trading days.

Evaluation:
  Walk-forward by event month (same split style as train_inclusion_baseline.py).

Outputs:
  data/research_outputs/stock_connect_keep_sell/
    - keep_sell_fold_metrics.csv
    - keep_sell_summary.json
    - keep_sell_oof_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Config:
    dataset_path: str = "data/research_outputs/stock_connect_keep_sell/keep_sell_training_table.csv"
    out_dir: str = "data/research_outputs/stock_connect_keep_sell"
    target_col: str = "y_keep"

    n_folds: int = 4
    min_train_months: int = 12
    test_months_per_fold: int = 6
    fallback_test_frac: float = 0.2
    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)


def build_walk_forward_splits(df: pd.DataFrame, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    d = df.copy()
    d["ym"] = pd.to_datetime(d["event_date"]).dt.to_period("M").astype(str)
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


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(np.floor(n * k_frac)))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top]))


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Numeric features (valuation + fundamentals + peer z-scores)
    num = [
        c
        for c in [
            "year",
            "month",
            "market_cap_approx_asof",
            "ps_ttm",
            "pe_ttm",
            "ps_ttm_peer_z",
            "pe_ttm_peer_z",
            "ps_ttm_peer_pctile",
            "pe_ttm_peer_pctile",
            "revenue_ttm_asof",
            "net_income_ttm_asof",
            "ocf_ttm_asof",
            "roe_asof",
            "rev_ttm_qoq_trend",
            "ni_ttm_qoq_trend",
            "ocf_ttm_qoq_trend",
            "fundamental_staleness_days",
            "forward_pe_snapshot",
            "forward_pe_snapshot_peer_z",
            "forward_pe_snapshot_peer_pctile",
        ]
        if c in df.columns
    ]
    cat = [c for c in ["index", "peer_group", "sector_snapshot", "industry_snapshot", "is_commodity_like"] if c in df.columns]
    return num, cat


def train_and_evaluate(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    work = df.copy()
    work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize()
    work = work[work[cfg.target_col].notna()].copy()
    work[cfg.target_col] = work[cfg.target_col].astype(int)
    work["year"] = work["event_date"].dt.year
    work["month"] = work["event_date"].dt.month
    work = work.sort_values(["event_date", "ticker_yf"]).reset_index(drop=True)

    num_features, cat_features = get_feature_columns(work)
    splits = build_walk_forward_splits(work, cfg)

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
    oof = np.full(len(work), np.nan, dtype=float)

    if len(splits) == 0:
        # Fallback: single time split by date (keeps the project usable on small samples).
        unique_dates = np.array(sorted(work["event_date"].unique()))
        if len(unique_dates) < 5:
            raise ValueError("Not enough unique dates to evaluate (need at least 5).")
        cut_idx = int((1.0 - cfg.fallback_test_frac) * len(unique_dates)) - 1
        cut_idx = max(0, min(cut_idx, len(unique_dates) - 2))
        cutoff = pd.Timestamp(unique_dates[cut_idx])
        tr_idx = work.index[work["event_date"] <= cutoff].to_numpy()
        te_idx = work.index[work["event_date"] > cutoff].to_numpy()
        splits = [(tr_idx, te_idx, f"fallback_time_split cutoff={str(cutoff.date())}")]

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
                    "train_pos_rate": float(np.mean(y_tr)) if len(y_tr) else np.nan,
                    "test_pos_rate": float(np.mean(y_te)) if len(y_te) else np.nan,
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
        oof[te_idx] = p

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
    ok = fold_df[fold_df["status"] == "ok"]

    summary: Dict[str, object] = {
        "task": "stock_connect_keep_sell",
        "dataset_path": cfg.dataset_path,
        "rows_total_with_label": int(len(work)),
        "pos_rate_total": float(work[cfg.target_col].mean()) if len(work) else np.nan,
        "split_mode": "walk_forward_by_month" if ("fallback_time_split" not in str(fold_df["split_label"].iloc[0]) if len(fold_df) else True) else "fallback_time_split",
        "n_folds": int(len(fold_df)),
        "n_ok_folds": int(len(ok)),
        "avg_pr_auc_ok_folds": float(ok["pr_auc"].mean()) if len(ok) else np.nan,
        "avg_roc_auc_ok_folds": float(ok["roc_auc"].mean()) if len(ok) else np.nan,
        "features_numeric": num_features,
        "features_categorical": cat_features,
    }

    oof_df = work[["ticker_yf", "event_date", cfg.target_col]].copy()
    oof_df["y_score_oof"] = oof
    return fold_df, summary, oof_df


def main() -> None:
    p = argparse.ArgumentParser(description="Train baseline keep/sell classifier for Stock Connect inclusions.")
    p.add_argument("--dataset-path", default="data/research_outputs/stock_connect_keep_sell/keep_sell_training_table.csv")
    p.add_argument("--out-dir", default="data/research_outputs/stock_connect_keep_sell")
    args = p.parse_args()

    cfg = Config(dataset_path=args.dataset_path, out_dir=args.out_dir)
    ensure_dirs(cfg)

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(cfg.dataset_path)
    df = pd.read_csv(cfg.dataset_path)

    fold_df, summary, oof_df = train_and_evaluate(df, cfg)

    fold_path = os.path.join(cfg.out_dir, "keep_sell_fold_metrics.csv")
    summary_path = os.path.join(cfg.out_dir, "keep_sell_summary.json")
    oof_path = os.path.join(cfg.out_dir, "keep_sell_oof_predictions.csv")

    fold_df.to_csv(fold_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    oof_df.to_csv(oof_path, index=False)

    print("[done] keep/sell baseline complete")
    print(f"  - {fold_path}")
    print(f"  - {summary_path}")
    print(f"  - {oof_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

