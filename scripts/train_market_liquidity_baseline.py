import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
    features_path: str = "data/research_outputs/features/event_features_enriched.csv"
    out_dir: str = "data/research_outputs/ml_market_liquidity_baseline"

    n_folds: int = 4
    min_train_months: int = 12
    test_months_per_fold: int = 6
    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(np.floor(n * k_frac)))
    top = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top]))


def load_training_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.features_path):
        raise FileNotFoundError(cfg.features_path)
    df = pd.read_csv(cfg.features_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["change_type"].isin(["Inclusion", "Exclusion", "Removal"])].copy()
    df["y_inclusion"] = (df["change_type"] == "Inclusion").astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["ticker_yf", "date"]).copy()
    d["prior_events_ticker"] = d.groupby("ticker_yf")["y_inclusion"].transform(lambda s: s.shift(1).notna().cumsum())
    d["prior_inclusion_ticker"] = d.groupby("ticker_yf")["y_inclusion"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    d["prior_inclusion_rate_ticker"] = np.where(
        d["prior_events_ticker"] > 0, d["prior_inclusion_ticker"] / d["prior_events_ticker"], 0.0
    )

    d = d.sort_values(["index", "date"]).copy()
    d["prior_events_index"] = d.groupby("index")["y_inclusion"].transform(lambda s: s.shift(1).notna().cumsum())
    d["prior_inclusion_index"] = d.groupby("index")["y_inclusion"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    d["prior_inclusion_rate_index"] = np.where(
        d["prior_events_index"] > 0, d["prior_inclusion_index"] / d["prior_events_index"], 0.0
    )
    return d


def build_walk_forward_splits(df: pd.DataFrame, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    months = sorted(d["ym"].unique().tolist())
    splits = []
    if len(months) < (cfg.min_train_months + cfg.test_months_per_fold):
        return splits
    start = cfg.min_train_months
    fold = 0
    while start + cfg.test_months_per_fold <= len(months) and fold < cfg.n_folds:
        tr_m = months[:start]
        te_m = months[start : start + cfg.test_months_per_fold]
        tr_idx = d.index[d["ym"].isin(tr_m)].to_numpy()
        te_idx = d.index[d["ym"].isin(te_m)].to_numpy()
        label = f"train<= {tr_m[-1]} | test={te_m[0]}..{te_m[-1]}"
        splits.append((tr_idx, te_idx, label))
        start += cfg.test_months_per_fold
        fold += 1
    return splits


def train_evaluate(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    num_features = [
        c
        for c in [
            "year",
            "month",
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
            "prior_events_ticker",
            "prior_inclusion_ticker",
            "prior_inclusion_rate_ticker",
            "prior_events_index",
            "prior_inclusion_index",
            "prior_inclusion_rate_index",
        ]
        if c in df.columns
    ]
    cat_features = [c for c in ["index", "ticker_parse_status"] if c in df.columns]

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[("prep", prep), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_state))]
    )

    splits = build_walk_forward_splits(df, cfg)
    if len(splits) == 0:
        raise ValueError("Not enough temporal coverage for walk-forward splits.")

    rows = []
    for i, (tr_idx, te_idx, label) in enumerate(splits, 1):
        tr = df.iloc[tr_idx].copy()
        te = df.iloc[te_idx].copy()
        y_tr = tr["y_inclusion"].values
        y_te = te["y_inclusion"].values
        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            rows.append(
                {
                    "fold": i,
                    "split": label,
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

        model.fit(tr[num_features + cat_features], y_tr)
        p = model.predict_proba(te[num_features + cat_features])[:, 1]
        rows.append(
            {
                "fold": i,
                "split": label,
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
    summary = {
        "rows_total": int(len(df)),
        "pos_rate_total": float(df["y_inclusion"].mean()),
        "n_folds": int(len(fold_df)),
        "n_ok_folds": int(len(ok)),
        "avg_pr_auc": float(ok["pr_auc"].mean()) if len(ok) else np.nan,
        "avg_roc_auc": float(ok["roc_auc"].mean()) if len(ok) else np.nan,
        "avg_precision_at_5pct": float(ok["precision_at_5pct"].mean()) if len(ok) else np.nan,
        "avg_precision_at_10pct": float(ok["precision_at_10pct"].mean()) if len(ok) else np.nan,
    }
    return fold_df, summary


def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)

    df = load_training_data(cfg)
    df = add_history_features(df)

    fold_df, summary = train_evaluate(df, cfg)

    fold_path = os.path.join(cfg.out_dir, "walk_forward_metrics.csv")
    summary_path = os.path.join(cfg.out_dir, "summary.json")
    df_path = os.path.join(cfg.out_dir, "training_dataset_snapshot.csv")

    fold_df.to_csv(fold_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(df_path, index=False)

    print("[done] market/liquidity baseline complete")
    print(f"  - {df_path}")
    print(f"  - {fold_path}")
    print(f"  - {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

