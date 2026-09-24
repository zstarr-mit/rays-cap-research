import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Missing dependency `yfinance`. Install with: pip install yfinance") from e


@dataclass(frozen=True)
class Config:
    events_path: str = "data/research_outputs/core/master_events_normalized.csv"
    out_dir: str = "data/research_outputs/stock_connect_fundamental_trends"
    cache_dir: str = "data/yfinance_fundamentals_cache"
    # "all_indices" uses all rows; "stock_connect_only" restricts scope.
    event_scope: str = "all_indices"

    # Point-in-time cut: fundamentals are taken as known on (event_date - asof_lag_days).
    # Larger lag = earlier snapshot = more conservative vs announcement/filing leakage.
    asof_lag_days: int = 120
    # Maximum acceptable staleness between asof_date and statement period_end.
    max_staleness_days: int = 540
    min_rows_for_model: int = 60
    min_minority: int = 12
    test_frac: float = 0.2
    min_fundamental_coverage_per_split: float = 0.05
    # Auto-eligible subset controls
    enable_coverage_subset_mode: bool = True
    min_ticker_coverage_for_subset: float = 0.25
    min_year_coverage_for_subset: float = 0.10
    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)


def load_events_for_scope(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.events_path):
        raise FileNotFoundError(cfg.events_path)
    df = pd.read_csv(cfg.events_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if cfg.event_scope == "stock_connect_only":
        mask_sc = df["index"].astype(str).str.contains("Stock Connect", case=False, na=False)
        df = df[mask_sc].copy()
    elif cfg.event_scope == "all_indices":
        pass
    else:
        raise ValueError(f"Unsupported event_scope: {cfg.event_scope}")
    df = df[df["change_type"].isin(["Inclusion", "Exclusion", "Removal"])].copy()
    if "ticker_yf" not in df.columns and "ticker_yf_normalized" in df.columns:
        df["ticker_yf"] = df["ticker_yf_normalized"]
    df = df[df["ticker_yf"].notna()].copy()
    df = df.sort_values(["date", "ticker_yf"]).reset_index(drop=True)

    # target: event is an Inclusion (entry-like event) vs non-inclusion events.
    df["y_enter_stock_connect"] = (df["change_type"] == "Inclusion").astype(int)
    df["asof_date"] = df["date"] - pd.Timedelta(days=cfg.asof_lag_days)
    return df


def _fund_cache_path(ticker: str, cfg: Config) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(ticker))
    return os.path.join(cfg.cache_dir, f"{safe}.pkl")


def _pick_metric(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return None


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


def build_quarterly_feature_frame(payload: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    qf = payload.get("quarterly_financials", pd.DataFrame())
    qb = payload.get("quarterly_balance_sheet", pd.DataFrame())
    qc = payload.get("quarterly_cashflow", pd.DataFrame())

    if qf is None or qf.empty:
        return pd.DataFrame()

    qf = qf.copy()
    qf.columns = pd.to_datetime(qf.columns, errors="coerce")
    qf = qf.loc[:, qf.columns.notna()]

    if qb is None:
        qb = pd.DataFrame()
    else:
        qb = qb.copy()
        qb.columns = pd.to_datetime(qb.columns, errors="coerce")
        qb = qb.loc[:, qb.columns.notna()]

    if qc is None:
        qc = pd.DataFrame()
    else:
        qc = qc.copy()
        qc.columns = pd.to_datetime(qc.columns, errors="coerce")
        qc = qc.loc[:, qc.columns.notna()]

    cols = sorted(qf.columns)
    out = pd.DataFrame(index=cols)

    rev = _pick_metric(qf, ["Total Revenue", "Revenue", "Total Revenues", "TotalRevenue"])
    ni = _pick_metric(qf, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    eq = _pick_metric(qb, ["Total Stockholder Equity", "Stockholders Equity", "Total equity", "TotalEquity"])
    ta = _pick_metric(qb, ["Total Assets", "TotalAssets", "Total assets"])
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
    out["assets_q"] = align(ta)
    out["ocf_q"] = align(ocf)

    out["revenue_ttm"] = out["revenue_q"].rolling(4, min_periods=4).sum()
    out["net_income_ttm"] = out["net_income_q"].rolling(4, min_periods=4).sum()
    out["ocf_ttm"] = out["ocf_q"].rolling(4, min_periods=4).sum()

    out["roe"] = out["net_income_ttm"] / out["equity_q"]
    out["roa"] = out["net_income_ttm"] / out["assets_q"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "period_end"
    return out.sort_index()


def asof_features(qframe: pd.DataFrame, asof_date: pd.Timestamp, cfg: Config) -> Dict[str, float]:
    if qframe is None or qframe.empty:
        return {
            "fundamentals_available": 0,
            "revenue_ttm_asof": np.nan,
            "net_income_ttm_asof": np.nan,
            "ocf_ttm_asof": np.nan,
            "roe_asof": np.nan,
            "roa_asof": np.nan,
            "rev_ttm_qoq_trend": np.nan,
            "ni_ttm_qoq_trend": np.nan,
            "ocf_ttm_qoq_trend": np.nan,
        }

    q = qframe[qframe.index <= pd.Timestamp(asof_date)].copy()
    if q.empty:
        return {
            "fundamentals_available": 0,
            "revenue_ttm_asof": np.nan,
            "net_income_ttm_asof": np.nan,
            "ocf_ttm_asof": np.nan,
            "roe_asof": np.nan,
            "roa_asof": np.nan,
            "rev_ttm_qoq_trend": np.nan,
            "ni_ttm_qoq_trend": np.nan,
            "ocf_ttm_qoq_trend": np.nan,
        }

    last = q.iloc[-1]
    prev = q.iloc[-2] if len(q) >= 2 else None
    period_end = pd.Timestamp(q.index[-1])
    staleness_days = float((pd.Timestamp(asof_date) - period_end).days)
    if staleness_days > cfg.max_staleness_days:
        return {
            "fundamentals_available": 0,
            "fundamental_staleness_days": staleness_days,
            "revenue_ttm_asof": np.nan,
            "net_income_ttm_asof": np.nan,
            "ocf_ttm_asof": np.nan,
            "roe_asof": np.nan,
            "roa_asof": np.nan,
            "rev_ttm_qoq_trend": np.nan,
            "ni_ttm_qoq_trend": np.nan,
            "ocf_ttm_qoq_trend": np.nan,
        }

    def pct(a: float, b: Optional[float]) -> float:
        if b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return (a / b) - 1.0

    return {
        "fundamentals_available": 1,
        "fundamental_staleness_days": staleness_days,
        "revenue_ttm_asof": float(last.get("revenue_ttm", np.nan)),
        "net_income_ttm_asof": float(last.get("net_income_ttm", np.nan)),
        "ocf_ttm_asof": float(last.get("ocf_ttm", np.nan)),
        "roe_asof": float(last.get("roe", np.nan)),
        "roa_asof": float(last.get("roa", np.nan)),
        "rev_ttm_qoq_trend": pct(last.get("revenue_ttm", np.nan), None if prev is None else prev.get("revenue_ttm", np.nan)),
        "ni_ttm_qoq_trend": pct(last.get("net_income_ttm", np.nan), None if prev is None else prev.get("net_income_ttm", np.nan)),
        "ocf_ttm_qoq_trend": pct(last.get("ocf_ttm", np.nan), None if prev is None else prev.get("ocf_ttm", np.nan)),
    }


def build_stock_connect_fundamental_dataset(df_events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    tickers = sorted(df_events["ticker_yf"].dropna().astype(str).unique().tolist())
    qcache: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        payload = fetch_fundamental_statements_cached(t, cfg)
        qcache[t] = build_quarterly_feature_frame(payload)

    for _, r in df_events.iterrows():
        t = str(r["ticker_yf"])
        qf = qcache.get(t, pd.DataFrame())
        feats = asof_features(qf, pd.Timestamp(r["asof_date"]), cfg)
        out = dict(r)
        out.update(feats)
        rows.append(out)

    data = pd.DataFrame(rows)
    return data


def summarize_trends(data: pd.DataFrame) -> pd.DataFrame:
    grp = (
        data.groupby("change_type")
        .agg(
            n=("ticker_yf", "size"),
            fundamentals_coverage=("fundamentals_available", "mean"),
            mean_revenue_ttm=("revenue_ttm_asof", "mean"),
            mean_net_income_ttm=("net_income_ttm_asof", "mean"),
            mean_roe=("roe_asof", "mean"),
            mean_roa=("roa_asof", "mean"),
            mean_rev_ttm_qoq=("rev_ttm_qoq_trend", "mean"),
            mean_ni_ttm_qoq=("ni_ttm_qoq_trend", "mean"),
        )
        .reset_index()
    )
    return grp


def build_coverage_report(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    d = data.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    by_year = (
        d.groupby("year")
        .agg(
            n_events=("ticker_yf", "size"),
            fundamentals_coverage=("fundamentals_available", "mean"),
            median_staleness_days=("fundamental_staleness_days", "median"),
        )
        .reset_index()
    )
    by_ticker = (
        d.groupby("ticker_yf")
        .agg(
            n_events=("ticker_yf", "size"),
            fundamentals_coverage=("fundamentals_available", "mean"),
            median_staleness_days=("fundamental_staleness_days", "median"),
        )
        .reset_index()
        .sort_values("fundamentals_coverage", ascending=True)
    )
    overall = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(d)},
            {"metric": "fundamentals_coverage_overall", "value": float(d["fundamentals_available"].mean()) if len(d) else np.nan},
            {"metric": "median_staleness_days", "value": float(d["fundamental_staleness_days"].median()) if len(d) else np.nan},
        ]
    )
    return {
        "fundamental_coverage_overall": overall,
        "fundamental_coverage_by_year": by_year,
        "fundamental_coverage_by_ticker": by_ticker,
    }


def select_coverage_eligible_subset(data: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = data.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year

    t_cov = d.groupby("ticker_yf")["fundamentals_available"].mean().reset_index(name="ticker_coverage")
    y_cov = d.groupby("year")["fundamentals_available"].mean().reset_index(name="year_coverage")

    good_tickers = set(t_cov.loc[t_cov["ticker_coverage"] >= cfg.min_ticker_coverage_for_subset, "ticker_yf"].tolist())
    good_years = set(y_cov.loc[y_cov["year_coverage"] >= cfg.min_year_coverage_for_subset, "year"].tolist())

    sub = d[d["ticker_yf"].isin(good_tickers) & d["year"].isin(good_years)].copy()
    sub = sub.reset_index(drop=True)

    subset_diag = pd.DataFrame(
        [
            {"metric": "rows_full", "value": len(d)},
            {"metric": "rows_subset", "value": len(sub)},
            {"metric": "subset_fraction", "value": float(len(sub) / len(d)) if len(d) else np.nan},
            {"metric": "full_fund_coverage", "value": float(d["fundamentals_available"].mean()) if len(d) else np.nan},
            {"metric": "subset_fund_coverage", "value": float(sub["fundamentals_available"].mean()) if len(sub) else np.nan},
            {"metric": "n_good_tickers", "value": len(good_tickers)},
            {"metric": "n_good_years", "value": len(good_years)},
            {"metric": "min_ticker_coverage_for_subset", "value": cfg.min_ticker_coverage_for_subset},
            {"metric": "min_year_coverage_for_subset", "value": cfg.min_year_coverage_for_subset},
        ]
    )
    return sub, subset_diag


def train_baseline_entry_model(data: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Keep rows with known target
    y = df["y_enter_stock_connect"].astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    if len(df) < cfg.min_rows_for_model:
        return pd.DataFrame(), {"status": "skip_small_sample", "rows": int(len(df))}

    pos = int(y.sum())
    neg = int((1 - y).sum())
    if min(pos, neg) < cfg.min_minority:
        return pd.DataFrame(), {"status": "skip_class_imbalance", "rows": int(len(df)), "pos": pos, "neg": neg}

    # Time split
    unique_dates = np.array(sorted(df["date"].unique()))
    cut_idx = int((1.0 - cfg.test_frac) * len(unique_dates)) - 1
    cut_idx = max(0, min(cut_idx, len(unique_dates) - 2))
    cutoff = pd.Timestamp(unique_dates[cut_idx])

    tr = df[df["date"] <= cutoff].copy()
    te = df[df["date"] > cutoff].copy()
    y_tr = tr["y_enter_stock_connect"].values
    y_te = te["y_enter_stock_connect"].values

    tr_cov = float(tr["fundamentals_available"].mean()) if len(tr) else 0.0
    te_cov = float(te["fundamentals_available"].mean()) if len(te) else 0.0
    if tr_cov < cfg.min_fundamental_coverage_per_split or te_cov < cfg.min_fundamental_coverage_per_split:
        return pd.DataFrame(), {
            "status": "skip_low_fundamental_coverage",
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_fund_coverage": tr_cov,
            "test_fund_coverage": te_cov,
            "min_required": cfg.min_fundamental_coverage_per_split,
        }

    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return pd.DataFrame(), {
            "status": "skip_one_class_split",
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_pos_rate": float(np.mean(y_tr)) if len(y_tr) else np.nan,
            "test_pos_rate": float(np.mean(y_te)) if len(y_te) else np.nan,
        }

    num_features = [
        c
        for c in [
            "year",
            "month",
            "revenue_ttm_asof",
            "net_income_ttm_asof",
            "ocf_ttm_asof",
            "roe_asof",
            "roa_asof",
            "rev_ttm_qoq_trend",
            "ni_ttm_qoq_trend",
            "ocf_ttm_qoq_trend",
        ]
        if c in df.columns
    ]
    cat_features = [c for c in ["index"] if c in df.columns]

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
    model.fit(tr[num_features + cat_features], y_tr)
    p = model.predict_proba(te[num_features + cat_features])[:, 1]
    pred = (p >= 0.5).astype(int)

    metrics = {
        "status": "ok",
        "asof_lag_days": cfg.asof_lag_days,
        "max_staleness_days": cfg.max_staleness_days,
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "train_fund_coverage": tr_cov,
        "test_fund_coverage": te_cov,
        "train_pos_rate": float(np.mean(y_tr)),
        "test_pos_rate": float(np.mean(y_te)),
        "pr_auc": float(average_precision_score(y_te, p)),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "accuracy": float(accuracy_score(y_te, pred)),
    }
    return te.assign(pred_proba_enter=p, pred_label=pred), metrics


def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)

    events = load_events_for_scope(cfg)
    data = build_stock_connect_fundamental_dataset(events, cfg)
    trends = summarize_trends(data)
    coverage = build_coverage_report(data)
    pred_df, metrics = train_baseline_entry_model(data, cfg)

    # Coverage-eligible subset mode
    subset_df = pd.DataFrame()
    subset_diag = pd.DataFrame()
    subset_pred = pd.DataFrame()
    subset_metrics: Dict[str, object] = {"status": "subset_mode_disabled"}
    if cfg.enable_coverage_subset_mode:
        subset_df, subset_diag = select_coverage_eligible_subset(data, cfg)
        if len(subset_df) > 0:
            subset_pred, subset_metrics = train_baseline_entry_model(subset_df, cfg)
        else:
            subset_metrics = {"status": "subset_empty"}

    data_path = os.path.join(cfg.out_dir, "stock_connect_events_with_fundamentals.csv")
    trends_path = os.path.join(cfg.out_dir, "fundamental_trend_summary_by_change_type.csv")
    metrics_path = os.path.join(cfg.out_dir, "stock_connect_entry_model_metrics.json")
    pred_path = os.path.join(cfg.out_dir, "stock_connect_entry_test_predictions.csv")
    subset_data_path = os.path.join(cfg.out_dir, "stock_connect_events_with_fundamentals_subset.csv")
    subset_diag_path = os.path.join(cfg.out_dir, "fundamental_coverage_subset_diagnostics.csv")
    subset_metrics_path = os.path.join(cfg.out_dir, "stock_connect_entry_model_metrics_subset.json")
    subset_pred_path = os.path.join(cfg.out_dir, "stock_connect_entry_test_predictions_subset.csv")
    cov_overall_path = os.path.join(cfg.out_dir, "fundamental_coverage_overall.csv")
    cov_year_path = os.path.join(cfg.out_dir, "fundamental_coverage_by_year.csv")
    cov_ticker_path = os.path.join(cfg.out_dir, "fundamental_coverage_by_ticker.csv")

    data.to_csv(data_path, index=False)
    trends.to_csv(trends_path, index=False)
    coverage["fundamental_coverage_overall"].to_csv(cov_overall_path, index=False)
    coverage["fundamental_coverage_by_year"].to_csv(cov_year_path, index=False)
    coverage["fundamental_coverage_by_ticker"].to_csv(cov_ticker_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if len(pred_df) > 0:
        pred_df.to_csv(pred_path, index=False)
    if len(subset_df) > 0:
        subset_df.to_csv(subset_data_path, index=False)
    if len(subset_diag) > 0:
        subset_diag.to_csv(subset_diag_path, index=False)
    with open(subset_metrics_path, "w", encoding="utf-8") as f:
        json.dump(subset_metrics, f, indent=2)
    if len(subset_pred) > 0:
        subset_pred.to_csv(subset_pred_path, index=False)

    print("[done] stock connect fundamentals trend workflow")
    print(f"  - {data_path}")
    print(f"  - {trends_path}")
    print(f"  - {metrics_path}")
    print(f"  - {cov_overall_path}")
    print(f"  - {cov_year_path}")
    print(f"  - {cov_ticker_path}")
    if len(subset_df) > 0:
        print(f"  - {subset_data_path}")
    if len(subset_diag) > 0:
        print(f"  - {subset_diag_path}")
    print(f"  - {subset_metrics_path}")
    if len(subset_pred) > 0:
        print(f"  - {subset_pred_path}")
    if len(pred_df) > 0:
        print(f"  - {pred_path}")
    print("\nModel/trend summary:")
    print(json.dumps(metrics, indent=2))
    print("\nSubset model summary (coverage-eligible subset):")
    print(json.dumps(subset_metrics, indent=2))
    print(f"\nEvent scope used: {cfg.event_scope}")
    print(f"Rows used after scope + label/ticker filters: {len(events)}")
    print(f"As-of lag days: {cfg.asof_lag_days}")
    print(f"Max statement staleness days: {cfg.max_staleness_days}")
    print("\nNote:")
    print(
        f"  This baseline uses point-in-time fundamentals as-of "
        f"{cfg.asof_lag_days} calendar days before each event date."
    )
    print("  It is descriptive/research-oriented and should not be treated as production alpha.")


if __name__ == "__main__":
    main()

