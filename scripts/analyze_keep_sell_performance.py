"""
Analyze Stock Connect keep/sell model outputs: labels, outcomes, and prediction quality.

Reads (default paths under data/research_outputs/stock_connect_keep_sell/):
  - keep_sell_training_table.csv
  - keep_sell_oof_predictions.csv  (merge on ticker_yf + event_date)
  - keep_sell_fold_metrics.csv, keep_sell_summary.json, meta.json (optional context)

Writes:
  - performance_report.txt
  - performance_decile_lift.csv
  - performance_by_year.csv
  - performance_by_index.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as e:
    raise ImportError("Install scikit-learn for metrics.") from e


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    try:
        p2 = np.clip(p, 1e-15, 1 - 1e-15)
        return float(log_loss(y, p2))
    except Exception:
        return float("nan")


def _majority_baseline_acc(y: np.ndarray) -> float:
    y = y.astype(int)
    maj = int(np.argmax(np.bincount(y, minlength=2)))
    return float(np.mean(y == maj))


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decile_lift(
    df: pd.DataFrame, score_col: str, label_col: str, ret_col: str, n_bins: int = 10
) -> pd.DataFrame:
    d = df[[score_col, label_col, ret_col]].dropna(subset=[score_col, label_col]).copy()
    if len(d) < n_bins:
        return pd.DataFrame()
    try:
        d["decile"] = pd.qcut(d[score_col], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()
    g = (
        d.groupby("decile", observed=True)
        .agg(
            n=(label_col, "size"),
            mean_score=(score_col, "mean"),
            keep_rate=(label_col, "mean"),
            mean_forward_return=(ret_col, "mean"),
            median_forward_return=(ret_col, "median"),
        )
        .reset_index()
        .sort_values("decile")
    )
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze keep/sell performance and write report + CSVs.")
    p.add_argument(
        "--out-dir",
        default="data/research_outputs/stock_connect_keep_sell",
        help="Directory with training table, OOF preds, fold metrics, summary.",
    )
    args = p.parse_args()
    out_dir = args.out_dir

    table_path = os.path.join(out_dir, "keep_sell_training_table.csv")
    oof_path = os.path.join(out_dir, "keep_sell_oof_predictions.csv")
    meta_path = os.path.join(out_dir, "meta.json")
    summary_path = os.path.join(out_dir, "keep_sell_summary.json")

    if not os.path.exists(table_path):
        raise FileNotFoundError(table_path)
    if not os.path.exists(oof_path):
        raise FileNotFoundError(oof_path)

    df = pd.read_csv(table_path)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    oof = pd.read_csv(oof_path)
    oof["event_date"] = pd.to_datetime(oof["event_date"]).dt.normalize()

    merged = df.merge(
        oof[["ticker_yf", "event_date", "y_score_oof"]],
        on=["ticker_yf", "event_date"],
        how="inner",
    )
    merged = merged[merged["y_keep"].notna()].copy()
    merged["y_keep"] = merged["y_keep"].astype(int)

    lines: List[str] = []
    lines.append("Stock Connect keep/sell — performance analysis")
    lines.append("=" * 60)

    meta = load_json(meta_path)
    if meta:
        lines.append("\nRun configuration (meta.json)")
        lines.append("-" * 40)
        for k in (
            "horizon_trading_days",
            "keep_return_threshold",
            "lag_days",
            "max_staleness_days",
            "include_forward_pe_snapshot",
            "rows",
            "skipped_rows",
        ):
            if k in meta:
                lines.append(f"  {k}: {meta[k]}")

    summary = load_json(summary_path)
    if summary:
        lines.append("\nModel training summary (keep_sell_summary.json)")
        lines.append("-" * 40)
        lines.append(f"  split_mode: {summary.get('split_mode', 'n/a')}")
        lines.append(f"  rows_total_with_label: {summary.get('rows_total_with_label', 'n/a')}")
        lines.append(f"  pos_rate_total: {summary.get('pos_rate_total', 'n/a')}")
        feats = summary.get("features_numeric", []) or []
        if "forward_total_return" in feats:
            lines.append("")
            lines.append(
                "  WARNING: forward_total_return was used as a model feature — this is label leakage."
            )
            lines.append(
                "  Re-train with train_keep_sell_baseline.py after removing that column from features."
            )

    # Label & outcome distribution
    lines.append("\nLabels and forward returns (full training table, y_keep known)")
    lines.append("-" * 40)
    labeled = df[df["y_keep"].notna()].copy()
    labeled["y_keep"] = labeled["y_keep"].astype(int)
    lines.append(f"  rows_with_label: {len(labeled)}")
    lines.append(f"  keep_rate (y_keep=1): {labeled['y_keep'].mean():.4f}")
    if "forward_total_return" in labeled.columns:
        r = labeled["forward_total_return"].dropna()
        if len(r):
            lines.append(f"  forward_total_return: mean={r.mean():.4f} median={r.median():.4f} std={r.std():.4f}")
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                lines.append(f"    q{int(q*100)}: {r.quantile(q):.4f}")

    labeled["year"] = labeled["event_date"].dt.year
    by_year = (
        labeled.groupby("year")
        .agg(
            n=("y_keep", "size"),
            keep_rate=("y_keep", "mean"),
            mean_fwd_ret=("forward_total_return", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    by_year_path = os.path.join(out_dir, "performance_by_year.csv")
    by_year.to_csv(by_year_path, index=False)

    lines.append("\nBy year (see performance_by_year.csv)")
    lines.append(by_year.to_string(index=False))

    if "index" in labeled.columns:
        by_idx = (
            labeled.groupby("index")
            .agg(n=("y_keep", "size"), keep_rate=("y_keep", "mean"), mean_fwd_ret=("forward_total_return", "mean"))
            .reset_index()
            .sort_values("n", ascending=False)
        )
        by_idx_path = os.path.join(out_dir, "performance_by_index.csv")
        by_idx.to_csv(by_idx_path, index=False)
        lines.append("\nBy index (top 15 by count; full table in performance_by_index.csv)")
        lines.append(by_idx.head(15).to_string(index=False))

    # Feature coverage from table
    lines.append("\nFeature coverage (fraction non-null)")
    lines.append("-" * 40)
    cov_cols = [
        "market_cap_approx_asof",
        "ps_ttm",
        "pe_ttm",
        "fundamentals_available",
        "peer_group",
        "sector_snapshot",
    ]
    for c in cov_cols:
        if c not in labeled.columns:
            continue
        s = labeled[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            cov = float(s.notna().mean())
        else:
            non_empty = s.astype(str).str.strip().replace({"nan": ""}) != ""
            cov = float((s.notna() & non_empty).mean())
        lines.append(f"  {c}: {cov:.3f}")

    # Scored subset
    scored = merged[merged["y_score_oof"].notna()].copy()
    lines.append("\nPrediction subset (rows with y_score_oof)")
    lines.append("-" * 40)
    lines.append(f"  n_scored: {len(scored)} / {len(labeled)} labeled rows")

    if len(scored) == 0:
        lines.append("  No scores to evaluate.")
        report_path = os.path.join(out_dir, "performance_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"\n[done] wrote {report_path}")
        return

    y = scored["y_keep"].values.astype(int)
    pr = scored["y_score_oof"].values.astype(float)
    pred = (pr >= 0.5).astype(int)

    lines.append("\nClassification metrics @ threshold 0.5 (scored rows only)")
    lines.append("-" * 40)
    lines.append(f"  accuracy: {accuracy_score(y, pred):.4f}")
    lines.append(f"  balanced_accuracy: {balanced_accuracy_score(y, pred):.4f}")
    lines.append(f"  precision (keep): {precision_score(y, pred, zero_division=0):.4f}")
    lines.append(f"  recall (keep): {recall_score(y, pred, zero_division=0):.4f}")
    lines.append(f"  f1 (keep): {f1_score(y, pred, zero_division=0):.4f}")
    mcc = matthews_corrcoef(y, pred) if np.unique(y).size > 1 and np.unique(pred).size > 1 else float("nan")
    lines.append(f"  matthews_corrcoef: {mcc:.4f}")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    lines.append("  confusion_matrix [rows actual 0,1; cols pred 0,1]:")
    lines.append(f"    TN={cm[0,0]} FP={cm[0,1]}")
    lines.append(f"    FN={cm[1,0]} TP={cm[1,1]}")

    lines.append("\nRanking / probability quality (scored rows only)")
    lines.append("-" * 40)
    lines.append(f"  roc_auc: {_safe_auc(y, pr):.4f}")
    lines.append(f"  pr_auc (average_precision): {_safe_pr_auc(y, pr):.4f}")
    lines.append(f"  brier_score: {brier_score_loss(y, pr):.4f}")
    lines.append(f"  log_loss: {_safe_logloss(y, pr):.4f}")

    lines.append("\nBaselines (same scored rows)")
    lines.append("-" * 40)
    lines.append(f"  majority_class_accuracy: {_majority_baseline_acc(y):.4f}")
    lines.append(f"  prevalence P(keep): {y.mean():.4f}")

    def p_at_k(kf: float) -> float:
        n = len(y)
        k = max(1, int(np.floor(n * kf)))
        order = np.argsort(-pr)
        return float(np.mean(y[order[:k]]))

    for kf, name in [(0.01, "1%"), (0.05, "5%"), (0.10, "10%")]:
        lines.append(f"  precision@{name} (by score): {p_at_k(kf):.4f}")

    # Decile lift
    ret_col = "forward_total_return"
    if ret_col in scored.columns:
        lift = decile_lift(scored, "y_score_oof", "y_keep", ret_col, n_bins=10)
        lift_path = os.path.join(out_dir, "performance_decile_lift.csv")
        lift.to_csv(lift_path, index=False)
        lines.append("\nDecile lift by predicted score (high decile = higher score; see performance_decile_lift.csv)")
        lines.append("-" * 40)
        if not lift.empty:
            lines.append(lift.to_string(index=False))
        else:
            lines.append("  (insufficient data for deciles)")

    lines.append("\nNotes")
    lines.append("-" * 40)
    lines.append("  - Metrics apply only to rows with non-null y_score_oof (e.g. test fold in time split).")
    lines.append("  - If ROC-AUC is very high, verify the model did not use forward_total_return as a feature.")
    lines.append("  - Decile lift should show monotone-ish keep_rate if the score ranks inclusions usefully.")

    report_path = os.path.join(out_dir, "performance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[done] wrote {report_path}")
    print(f"       {by_year_path}")
    if "index" in labeled.columns:
        print(f"       {os.path.join(out_dir, 'performance_by_index.csv')}")
    if ret_col in scored.columns:
        print(f"       {os.path.join(out_dir, 'performance_decile_lift.csv')}")


if __name__ == "__main__":
    main()
