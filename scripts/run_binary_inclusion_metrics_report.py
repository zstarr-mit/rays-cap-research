"""
Train and time-split-test the binary inclusion baseline, then write a plain-text metrics report.

Walk-forward by event month (same as train_inclusion_baseline). Collects out-of-fold
probabilities on all test windows, then reports pooled ranking metrics (ROC-AUC, PR-AUC),
thresholded classification metrics at 0.5, baselines, and per-fold tables.

Usage (from repo root):
  python3 scripts/run_binary_inclusion_metrics_report.py
  python3 scripts/run_binary_inclusion_metrics_report.py --tag inc_vs_exc_v2
  python3 scripts/run_binary_inclusion_metrics_report.py --reuse-train-table data/research_outputs/ml_baseline/train_table_snapshot.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_inclusion_baseline import (  # noqa: E402
    Config,
    build_training_frame,
    build_walk_forward_splits,
    ensure_dirs,
    get_feature_columns,
    precision_at_k,
)


def _build_pipeline(num_features: List[str], cat_features: List[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
        ]
    )


def _safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    return float(matthews_corrcoef(y_true, y_pred))


def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        p = np.clip(proba, 1e-15, 1 - 1e-15)
        return float(log_loss(y_true, p))
    except Exception:
        return float("nan")


def run_walk_forward_with_oof(work: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, object]]]:
    work = work.copy()
    work[cfg.target_col] = work[cfg.target_col].fillna(0).astype(int)
    num_features, cat_features = get_feature_columns(work)
    splits = build_walk_forward_splits(work, cfg)
    if len(splits) == 0:
        raise ValueError("Not enough month coverage for walk-forward splits.")

    n = len(work)
    oof_proba = np.full(n, np.nan, dtype=float)
    fold_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    model = _build_pipeline(num_features, cat_features, cfg.random_state)

    for fold_i, (tr_idx, te_idx, label) in enumerate(splits, 1):
        tr = work.iloc[tr_idx]
        te = work.iloc[te_idx]
        y_tr = tr[cfg.target_col].values
        y_te = te[cfg.target_col].values

        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            fold_rows.append(
                {
                    "fold": fold_i,
                    "split_label": label,
                    "train_rows": len(tr),
                    "test_rows": len(te),
                    "status": "skipped_one_class",
                }
            )
            continue

        X_tr = tr[num_features + cat_features]
        X_te = te[num_features + cat_features]
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        oof_proba[te_idx] = p

        y_hat = (p >= 0.5).astype(int)
        cm = confusion_matrix(y_te, y_hat, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = (int(x) for x in cm.ravel())
        else:
            tn = fp = fn = tp = -1

        fold_rows.append(
            {
                "fold": fold_i,
                "split_label": label,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_pos_rate": float(np.mean(y_tr)),
                "test_pos_rate": float(np.mean(y_te)),
                "accuracy": float(accuracy_score(y_te, y_hat)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te, y_hat)),
                "precision_pos": float(precision_score(y_te, y_hat, zero_division=0)),
                "recall_pos": float(recall_score(y_te, y_hat, zero_division=0)),
                "f1_pos": float(f1_score(y_te, y_hat, zero_division=0)),
                "mcc": _safe_mcc(y_te, y_hat),
                "brier": float(brier_score_loss(y_te, p)),
                "log_loss": _safe_log_loss(y_te, p),
                "roc_auc": float(roc_auc_score(y_te, p)),
                "pr_auc": float(average_precision_score(y_te, p)),
                "precision_at_1pct": precision_at_k(y_te, p, 0.01),
                "precision_at_5pct": precision_at_k(y_te, p, 0.05),
                "precision_at_10pct": precision_at_k(y_te, p, 0.10),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "status": "ok",
            }
        )

        for j, idx in enumerate(te_idx):
            detail_rows.append(
                {
                    "fold": fold_i,
                    "row_index": int(idx),
                    "ticker_yf": work.iloc[idx]["ticker_yf"],
                    "event_date": str(work.iloc[idx]["event_date"].date()),
                    "y_true": int(y_te[j]),
                    "y_score": float(p[j]),
                    "y_pred_0.5": int(y_hat[j]),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, oof_proba, detail_rows


def _pooled_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_hat = (p >= 0.5).astype(int)
    out: Dict[str, float] = {
        "n": float(len(y)),
        "prevalence_positive": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "precision_positive": float(precision_score(y, y_hat, zero_division=0)),
        "recall_positive": float(recall_score(y, y_hat, zero_division=0)),
        "f1_positive": float(f1_score(y, y_hat, zero_division=0)),
        "mcc": _safe_mcc(y, y_hat),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": _safe_log_loss(y, p),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_avg_precision": float(average_precision_score(y, p)),
        "precision_at_1pct": precision_at_k(y, p, 0.01),
        "precision_at_5pct": precision_at_k(y, p, 0.05),
        "precision_at_10pct": precision_at_k(y, p, 0.10),
    }
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        out["tn"] = float(tn)
        out["fp"] = float(fp)
        out["fn"] = float(fn)
        out["tp"] = float(tp)
    return out


def _majority_baseline_accuracy(y: np.ndarray) -> float:
    y = y.astype(int)
    counts = np.bincount(y, minlength=2)
    maj = int(np.argmax(counts))
    return float(np.mean(y == maj))


def build_report_text(
    cfg: Config,
    work: pd.DataFrame,
    fold_df: pd.DataFrame,
    oof_proba: np.ndarray,
    skip_n: int,
) -> str:
    lines: List[str] = []
    lines.append("Binary inclusion baseline — evaluation report")
    lines.append("=" * 60)
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 40)
    lines.append(f"  events_path: {cfg.events_path}")
    lines.append(f"  calendar_days_before (feature anchor): {cfg.calendar_days_before}")
    lines.append(f"  target: {cfg.target_col} (1=Inclusion, 0=Exclusion or Removal)")
    lines.append(f"  walk-forward: min_train_months={cfg.min_train_months}, test_months_per_fold={cfg.test_months_per_fold}, n_folds cap={cfg.n_folds}")
    lines.append(f"  index_contains filter: {cfg.index_contains!r}")
    lines.append("")

    y_all = work[cfg.target_col].fillna(0).astype(int).values
    lines.append("Dataset (full table after feature build)")
    lines.append("-" * 40)
    lines.append(f"  rows_with_features: {len(work)}")
    lines.append(f"  rows_skipped (no price/anchor): {skip_n if skip_n >= 0 else 'n/a (reused table)'}")
    lines.append(f"  positive_count (inclusion): {int(y_all.sum())}")
    lines.append(f"  positive_rate: {float(y_all.mean()):.4f}")
    lines.append("")

    mask = ~np.isnan(oof_proba)
    n_oof = int(mask.sum())
    lines.append("Walk-forward test coverage")
    lines.append("-" * 40)
    lines.append(f"  rows_with_out-of-fold_prediction: {n_oof} / {len(work)}")
    lines.append(f"  (Rows in months never used as a test fold have no OOF score.)")
    lines.append("")

    if n_oof > 0:
        y_p = y_all[mask]
        p_p = oof_proba[mask]
        pm = _pooled_metrics(y_p, p_p)
        maj = _majority_baseline_accuracy(y_p)

        lines.append("Pooled test windows (concatenated walk-forward test folds)")
        lines.append("-" * 40)
        lines.append("  Ranking / calibration (probability scores)")
        lines.append(f"    ROC-AUC:              {pm['roc_auc']:.4f}")
        lines.append(f"    PR-AUC (avg prec):    {pm['pr_auc_avg_precision']:.4f}")
        lines.append(f"    Brier score:          {pm['brier_score']:.4f}  (lower is better)")
        lines.append(f"    Log loss:             {pm['log_loss']:.4f}  (lower is better)")
        lines.append("")
        lines.append("  Classification at threshold 0.5 on predicted P(inclusion)")
        lines.append(f"    Accuracy:             {pm['accuracy']:.4f}")
        lines.append(f"    Balanced accuracy:    {pm['balanced_accuracy']:.4f}")
        lines.append(f"    Precision (class 1):  {pm['precision_positive']:.4f}")
        lines.append(f"    Recall (class 1):     {pm['recall_positive']:.4f}")
        lines.append(f"    F1 (class 1):         {pm['f1_positive']:.4f}")
        lines.append(f"    Matthews correlation: {pm['mcc']:.4f}")
        lines.append("")
        lines.append("  Confusion matrix @0.5  [rows actual 0,1 ; cols pred 0,1]")
        lines.append(f"    TN={int(pm['tn'])}  FP={int(pm['fp'])}")
        lines.append(f"    FN={int(pm['fn'])}  TP={int(pm['tp'])}")
        lines.append("")
        lines.append("  Top-k precision on test pool (fraction positive in top k% by score)")
        lines.append(f"    precision@1%:  {pm['precision_at_1pct']:.4f}")
        lines.append(f"    precision@5%:  {pm['precision_at_5pct']:.4f}")
        lines.append(f"    precision@10%: {pm['precision_at_10pct']:.4f}")
        lines.append("")
        lines.append("  Baselines on same pooled test rows")
        lines.append(f"    Majority-class accuracy (always predict mode): {maj:.4f}")
        lines.append("")
        buf = StringIO()
        y_hat_p = (p_p >= 0.5).astype(int)
        buf.write(classification_report(y_p, y_hat_p, target_names=["neg_excl_or_rem", "pos_inclusion"], digits=4))
        lines.append("  sklearn classification_report @0.5")
        lines.append(buf.getvalue().rstrip())
        lines.append("")

    lines.append("Per-fold metrics (each test window)")
    lines.append("-" * 40)
    ok = fold_df[fold_df["status"] == "ok"]
    if len(ok):
        lines.append(ok.to_string(index=False))
        lines.append("")
        lines.append("  Mean (ok folds only):")
        for col in ["accuracy", "balanced_accuracy", "f1_pos", "roc_auc", "pr_auc", "mcc", "brier"]:
            if col in ok.columns:
                lines.append(f"    {col}: {ok[col].mean():.4f}  (std {ok[col].std():.4f})")
    else:
        lines.append("  (no completed folds)")
    lines.append("")

    lines.append("Interpretation notes")
    lines.append("-" * 40)
    lines.append("  - ROC-AUC / PR-AUC measure ranking quality on held-out months, not live PnL.")
    lines.append("  - High accuracy can track class balance; use balanced accuracy, MCC, and PR-AUC for imbalanced views.")
    lines.append("  - precision@k is noisy when the test fold has few rows.")
    lines.append("")

    return "\n".join(lines)


def _sanitize_tag(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]", "", s)
    return s[:80] if s else ""


def _default_ml_paths(tag: Optional[str]) -> Tuple[str, str, str, str]:
    """Default report, OOF preds, fold metrics, meta json under ml_baseline/."""
    base = "data/research_outputs/ml_baseline"
    suf = f"_{_sanitize_tag(tag)}" if tag and _sanitize_tag(tag) else ""
    return (
        os.path.join(base, f"binary_inclusion_metrics_report{suf}.txt"),
        os.path.join(base, f"binary_inclusion_oof_predictions{suf}.csv"),
        os.path.join(base, f"binary_inclusion_fold_metrics_full{suf}.csv"),
        os.path.join(base, f"binary_inclusion_report_meta{suf}.json"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train/test binary inclusion model and write metrics report.")
    p.add_argument("--reuse-train-table", default=None, help="CSV from a prior train_table_snapshot.csv (skip yfinance download).")
    p.add_argument(
        "--tag",
        default=None,
        metavar="LABEL",
        help="Appended to default output filenames (e.g. --tag run1 -> binary_inclusion_metrics_report_run1.txt). "
        "Ignored for any path you set explicitly with --out-report / --out-predictions / --out-folds / --out-meta.",
    )
    p.add_argument("--out-report", default=None, help="Override report .txt path (default: ml_baseline/binary_inclusion_metrics_report[ _TAG].txt).")
    p.add_argument("--out-predictions", default=None, help="Override OOF predictions CSV.")
    p.add_argument("--out-folds", default=None, help="Override per-fold metrics CSV.")
    p.add_argument("--out-meta", default=None, help="Override meta JSON path.")
    args = p.parse_args()

    d_rep, d_pred, d_folds, d_meta = _default_ml_paths(args.tag)
    out_report = args.out_report or d_rep
    out_predictions = args.out_predictions or d_pred
    out_folds = args.out_folds or d_folds
    out_meta = args.out_meta or d_meta

    cfg = Config()
    ensure_dirs(cfg)

    if args.reuse_train_table:
        if not os.path.exists(args.reuse_train_table):
            raise FileNotFoundError(args.reuse_train_table)
        work = pd.read_csv(args.reuse_train_table)
        work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize()
        skip_n = -1
    else:
        work, skip_df = build_training_frame(cfg)
        skip_n = len(skip_df)
        snap_suf = f"_{_sanitize_tag(args.tag)}" if args.tag and _sanitize_tag(args.tag) else ""
        snap = os.path.join(cfg.out_dir, f"train_table_snapshot{snap_suf}.csv")
        work.to_csv(snap, index=False)
        skip_path = os.path.join(cfg.out_dir, f"binary_baseline_skipped_rows{snap_suf}.csv")
        skip_df.to_csv(skip_path, index=False)

    fold_df, oof_proba, detail_rows = run_walk_forward_with_oof(work, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(out_report)) or ".", exist_ok=True)
    report_text = build_report_text(cfg, work, fold_df, oof_proba, skip_n)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_text)

    fold_df.to_csv(out_folds, index=False)
    pd.DataFrame(detail_rows).to_csv(out_predictions, index=False)

    summary = {
        "tag": args.tag,
        "report_path": out_report,
        "oof_predictions_path": out_predictions,
        "fold_metrics_path": out_folds,
        "n_folds_recorded": int(len(fold_df)),
        "n_oof_scored": int(np.sum(~np.isnan(oof_proba))),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(report_text)
    print(f"\n[done] Wrote {out_report}")
    print(f"       {out_predictions}")
    print(f"       {out_folds}")
    print(f"       {out_meta}")


if __name__ == "__main__":
    main()
