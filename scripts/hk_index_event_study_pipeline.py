import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Config:
    input_csv: str = "data/full_hk_index_dataset.csv"
    cleaned_csv: str = "data/full_hk_index_dataset_cleaned.csv"

    # Modeling subset outputs (optional but helpful)
    modeling_subset_csv: str = "data/full_hk_index_modeling_subset.csv"

    # Descriptive + ML timing split
    test_date_fraction: float = 0.2

    # Minimum size constraints for attempting a baseline model
    min_total_rows_for_model: int = 80
    min_minority_rows_for_model: int = 15

    # Prior-history features: only from Inclusion/Exclusion events
    prior_event_types_for_features: Tuple[str, ...] = ("Inclusion", "Exclusion")

    # Use up to these many unique (index) categories in one-hot to keep things stable.
    # If more, we still one-hot everything (can be sparse), but the script warns.
    max_reasonable_index_categories: int = 50

    random_state: int = 42


def _normalize_change_type(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s_low = s.lower()
    if "inclusion" in s_low:
        return "Inclusion"
    if "exclusion" in s_low:
        return "Exclusion"
    if "removal" in s_low:
        return "Removal"
    return None


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """
    Parse a date column that may contain formats like:
      - MM/DD/YY (e.g., 11/21/25)
      - YYYY-MM-DD (e.g., 2023-09-01)
      - M/D/YY (e.g., 2/13/26)
    """
    raw = series.astype(str).str.strip()

    # Parse using explicit formats to avoid pandas' inference warnings.
    out_md_y = pd.to_datetime(raw, errors="coerce", format="%m/%d/%y")
    out_y_m_d = pd.to_datetime(raw, errors="coerce", format="%Y-%m-%d")
    out = out_md_y
    mask = out.isna()
    out.loc[mask] = out_y_m_d.loc[mask]

    # Fallback: only if anything still unparsed.
    if out.isna().any():
        remaining = raw[out.isna()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.loc[out.isna()] = pd.to_datetime(remaining, errors="coerce")

    return out


def load_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")

    df = pd.read_csv(cfg.input_csv, dtype=str)

    print("Loaded CSV")
    print(f"  path: {cfg.input_csv}")
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")

    print("\nDtype summary (as-loaded):")
    print(df.dtypes.to_string())

    expected_cols = ["date", "index", "change_type", "ticker", "ticker_normalized", "company"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing expected columns: {missing}")

    # Report change_type distribution (raw)
    raw_change_type_counts = df["change_type"].value_counts(dropna=False)
    print("\nRaw `change_type` unique values:")
    print(raw_change_type_counts.to_string())

    print("\nCounts by `index` (top 15):")
    print(df["index"].value_counts().head(15).to_string())

    # Basic missingness
    print("\nMissing values (count):")
    print(df[expected_cols].isna().sum().sort_values(ascending=False).to_string())

    # Formatting checks (pre-clean)
    str_cols = ["index", "ticker", "ticker_normalized", "company", "change_type"]
    print("\nFormatting inconsistencies (pre-clean):")
    for col in str_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        n_leadtrail = int(s.str.match(r"^\s|\s$").sum())
        n_internal_multi_ws = int(s.str.contains(r"\s{2,}", regex=True).sum())
        print(f"  - {col}: leading/trailing whitespace rows={n_leadtrail}, internal-multi-space rows={n_internal_multi_ws}")

    # Validate ticker_normalized pattern: expected digits + . + (HK|SZ|SH)
    if "ticker_normalized" in df.columns:
        tn = df["ticker_normalized"].astype(str).str.strip()
        valid_pat = tn.str.match(r"^\d+\.(HK|SZ|SH)$")
        n_invalid = int((~valid_pat).sum())
        print(f"  - ticker_normalized pattern rows invalid vs ^digits\\.(HK|SZ|SH)$: {n_invalid}")

    return df


def clean_events(df_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_raw.copy()

    # Parse date
    df["date"] = _parse_mixed_dates(df["date"])
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), ["date", "index", "change_type", "ticker_normalized"]].head(10)
        raise ValueError("Failed to parse some dates. Example rows:\n" + bad.to_string(index=False))
    df["date"] = df["date"].dt.normalize()

    # Standardize change_type
    df["change_type"] = df["change_type"].map(_normalize_change_type)
    if df["change_type"].isna().any():
        bad = df.loc[df["change_type"].isna(), ["change_type", "index", "ticker_normalized", "date"]].head(10)
        raise ValueError("Found unknown `change_type` values. Example rows:\n" + bad.to_string(index=False))

    # Standardize string fields
    for col in ["index", "ticker", "ticker_normalized", "company"]:
        df[col] = df[col].astype(str).str.strip()

    print("\nDtype summary (post date/change_type standardization):")
    print(df[["date", "index", "change_type", "ticker_normalized"]].dtypes.to_string())

    # Report canonical change_type counts (post clean)
    print("\nCanonical `change_type` counts:")
    print(df["change_type"].value_counts(dropna=False).to_string())

    # Duplicate check (exact duplicates on requested key)
    dup_key = ["date", "index", "change_type", "ticker_normalized"]
    n_dups = int(df.duplicated(subset=dup_key).sum())
    print(f"\nExact duplicates on {tuple(dup_key)}: {n_dups}")

    # Remove exact duplicates only (do not collapse distinct events)
    before = len(df)
    df = df.drop_duplicates(subset=dup_key, keep="first").copy()
    after = len(df)
    print(f"Rows before exact-dedupe: {before} -> after: {after} (dropped {before - after})")

    # Missing values after clean
    expected_cols = ["date", "index", "change_type", "ticker", "ticker_normalized", "company"]
    miss_after = df[expected_cols].isna().sum().sort_values(ascending=False)
    print("\nMissing values after cleaning (count):")
    print(miss_after.to_string())

    df = df.sort_values(["date", "index", "ticker_normalized"]).reset_index(drop=True)

    df.to_csv(cfg.cleaned_csv, index=False)
    print(f"\nSaved cleaned CSV: {cfg.cleaned_csv}")

    return df


def summarize_events(df: pd.DataFrame) -> None:
    print("\n====================")
    print("Descriptive analytics")
    print("====================")

    # Count by year, index, change_type
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    print("\nEvent counts by change_type:")
    print(df["change_type"].value_counts().to_string())

    print("\nEvent counts by year and change_type:")
    counts_year = df.groupby(["year", "change_type"]).size().reset_index(name="count")
    # Print compact view
    print(counts_year.pivot(index="year", columns="change_type", values="count").fillna(0).astype(int).to_string())

    print("\nEvent counts by index (top 15):")
    top_idx = df["index"].value_counts().head(15)
    print(top_idx.to_string())

    # Most frequent tickers
    top_tickers = df["ticker_normalized"].value_counts().head(15)
    print("\nMost frequent tickers (by ticker_normalized):")
    print(top_tickers.to_string())

    # Clustering summary
    # Inclusion/Exclusion share by index (top-N)
    df_ie = df[df["change_type"].isin(["Inclusion", "Exclusion"])].copy()
    if len(df_ie) == 0:
        print("\nNo Inclusion/Exclusion events available for clustering summary.")
        return

    idx_ie = df_ie.pivot_table(index="index", columns="change_type", values="ticker_normalized", aggfunc="count", fill_value=0)
    idx_ie["total_ie"] = idx_ie.sum(axis=1)
    idx_ie = idx_ie.sort_values("total_ie", ascending=False).head(10)

    print("\nTop 10 indices by Inclusion+Exclusion events:")
    for idx, row in idx_ie.iterrows():
        inc = int(row.get("Inclusion", 0))
        exc = int(row.get("Exclusion", 0))
        tot = int(row.get("total_ie", 0))
        print(f"  - {idx}: Inclusion={inc}, Exclusion={exc}, total={tot}")

    # Inclusion/Exclusion share by time periods (year-month)
    ym = df_ie.groupby(["year", "month", "change_type"]).size().reset_index(name="count")
    ym_piv = ym.pivot_table(index=["year", "month"], columns="change_type", values="count", fill_value=0).reset_index()
    ym_piv["total_ie"] = ym_piv.get("Inclusion", 0) + ym_piv.get("Exclusion", 0)
    ym_piv = ym_piv.sort_values("total_ie", ascending=False).head(12)

    print("\nTop 12 year-month buckets by Inclusion+Exclusion events:")
    for _, r in ym_piv.iterrows():
        print(f"  - {int(r['year'])}-{int(r['month']):02d}: Inclusion={int(r.get('Inclusion',0))}, Exclusion={int(r.get('Exclusion',0))}, total={int(r['total_ie'])}")


def build_modeling_subset(df_clean: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_clean.copy()
    df = df[df["change_type"].isin(["Inclusion", "Exclusion"])].copy()

    # Preserve original categorical label
    df["change_type_label"] = df["change_type"].copy()

    # Binary target for inclusion model
    df["y_inclusion"] = np.where(df["change_type"] == "Inclusion", 1, 0).astype(int)

    # Ensure no missing modeling keys
    df = df.dropna(subset=["date", "index", "ticker_normalized", "y_inclusion"]).copy()

    # Save subset for reproducibility
    df_out = df.sort_values(["date", "ticker_normalized", "index"]).reset_index(drop=True)
    df_out.to_csv(cfg.modeling_subset_csv, index=False)
    print(f"\nSaved modeling subset CSV: {cfg.modeling_subset_csv}")

    return df_out


def time_based_split(df_model: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.array(sorted(df_model["date"].unique()))
    if len(dates) < 5:
        raise ValueError("Not enough unique dates to create a time-based split.")

    cutoff_idx = int((1.0 - cfg.test_date_fraction) * len(dates)) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(dates) - 2))
    cutoff_date = pd.Timestamp(dates[cutoff_idx])

    train = df_model[df_model["date"] <= cutoff_date].copy()
    test = df_model[df_model["date"] > cutoff_date].copy()

    return train, test, cutoff_date


def add_prior_event_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Adds prior-history features computed using only rows with date < current date.

    This is leakage-safe because:
      - all rolling/cumulative features are shifted by 1 within each group
      - thus the current row's label is never included in its own feature values
    """
    df = df.copy()
    df = df.sort_values(["ticker_normalized", "date"]).reset_index(drop=True)

    # Prior counts by ticker
    df["prior_any_events_ticker"] = (
        df.groupby("ticker_normalized")["y_inclusion"].transform(lambda s: s.shift(1).notna().cumsum())
    )
    df["prior_incl_events_ticker"] = df.groupby("ticker_normalized")["y_inclusion"].transform(lambda s: s.shift(1).cumsum())
    df["prior_excl_events_ticker"] = df.groupby("ticker_normalized")["y_inclusion"].transform(
        lambda s: (1 - s).shift(1).cumsum()
    )
    denom = df["prior_incl_events_ticker"] + df["prior_excl_events_ticker"]
    df["prior_inclusion_rate_ticker"] = np.where(denom > 0, df["prior_incl_events_ticker"] / denom, 0.0)

    # Time since last event (in calendar days) computed using previous event date
    prev_date = df.groupby("ticker_normalized")["date"].shift(1)
    df["days_since_prev_event_ticker"] = (df["date"] - prev_date).dt.days

    # Prior counts by index
    df = df.sort_values(["index", "date"]).reset_index(drop=True)

    df["prior_any_events_index"] = df.groupby("index")["y_inclusion"].transform(lambda s: s.shift(1).notna().cumsum())
    df["prior_incl_events_index"] = df.groupby("index")["y_inclusion"].transform(lambda s: s.shift(1).cumsum())
    df["prior_excl_events_index"] = df.groupby("index")["y_inclusion"].transform(lambda s: (1 - s).shift(1).cumsum())
    denom_i = df["prior_incl_events_index"] + df["prior_excl_events_index"]
    df["prior_inclusion_rate_index"] = np.where(denom_i > 0, df["prior_incl_events_index"] / denom_i, 0.0)

    return df


def train_baseline_model(train: pd.DataFrame, cfg: Config):
    # Feature set derived only from event history available before each event.
    # Note: we compute these features earlier using shifted cumulative counts.
    feature_num = [
        "year",
        "month",
        "prior_any_events_ticker",
        "prior_incl_events_ticker",
        "prior_excl_events_ticker",
        "prior_inclusion_rate_ticker",
        "days_since_prev_event_ticker",
        "prior_any_events_index",
        "prior_incl_events_index",
        "prior_excl_events_index",
        "prior_inclusion_rate_index",
    ]
    feature_cat = ["index"]

    for c in feature_num + feature_cat:
        if c not in train.columns:
            raise KeyError(f"Missing feature column in train: {c}")

    # Class presence check
    y = train["y_inclusion"].values
    unique_y = np.unique(y)
    if len(unique_y) < 2:
        raise ValueError(f"Training has only one class (y_inclusion={unique_y.tolist()}).")

    minority = min((train["y_inclusion"] == 0).sum(), (train["y_inclusion"] == 1).sum())
    if minority < cfg.min_minority_rows_for_model:
        raise ValueError(f"Training minority class too small: minority_rows={minority}.")

    # Pipelines: impute + scale numeric, one-hot + impute categorical
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_num),
            ("cat", categorical_transformer, feature_cat),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=cfg.random_state,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    model.fit(train[feature_num + feature_cat], y)
    return model, feature_num, feature_cat


def evaluate_model(model, train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    feature_num = [
        "year",
        "month",
        "prior_any_events_ticker",
        "prior_incl_events_ticker",
        "prior_excl_events_ticker",
        "prior_inclusion_rate_ticker",
        "days_since_prev_event_ticker",
        "prior_any_events_index",
        "prior_incl_events_index",
        "prior_excl_events_index",
        "prior_inclusion_rate_index",
    ]
    feature_cat = ["index"]

    train_y = train["y_inclusion"].values
    test_y = test["y_inclusion"].values

    # Probabilities for AUC
    test_proba = model.predict_proba(test[feature_num + feature_cat])[:, 1]
    train_proba = model.predict_proba(train[feature_num + feature_cat])[:, 1]

    # Basic metrics
    test_pred = (test_proba >= 0.5).astype(int)
    acc = accuracy_score(test_y, test_pred)
    cm = confusion_matrix(test_y, test_pred)

    print("\n====================")
    print("Model evaluation")
    print("====================")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(f"Train class balance: {train['y_inclusion'].value_counts().to_dict()}")
    print(f"Test class balance: {test['y_inclusion'].value_counts().to_dict()}")
    print("Confusion matrix (test, pred vs true) [rows=true, cols=pred]:")
    print(cm)

    # AUC only if both classes appear in test
    metrics: Dict[str, float] = {"accuracy": float(acc)}
    if np.unique(test_y).size < 2:
        print("WARNING: AUC is meaningless because test has only one class.")
        metrics["auc_roc"] = float("nan")
        return metrics

    auc = roc_auc_score(test_y, test_proba)
    metrics["auc_roc"] = float(auc)
    print(f"ROC AUC (test): {auc:.4f}")

    # Sanity: probability variance
    proba_std = float(np.std(test_proba))
    print(f"Predicted probability std (test): {proba_std:.6f}")
    if proba_std < 1e-3:
        print("WARNING: Model outputs nearly constant probabilities; AUC may be unstable/uninformative.")

    # Leakage suspicion diagnostics
    # If performance is extremely high on a small sample, run canary feature leakage test.
    if (len(test) < 200) and (np.isfinite(auc) and auc > 0.95):
        print("\nWARNING: Suspiciously high AUC detected. Running a leakage canary:")
        print("  Canary: recompute prior features using same-date counts (intentionally leaking current label).")
        canary_metrics = evaluate_leakage_canary(train, test, cfg)
        metrics["auc_canary_leaky_features"] = canary_metrics.get("auc_roc_leaky", float("nan"))

        print("  If canary AUC is dramatically higher than baseline, that suggests the baseline may still be impacted by time alignment issues.")

    return metrics


def evaluate_leakage_canary(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    # Intentionally build leaky features: do NOT shift cumulative counts,
    # so the current row's y_inclusion is included in its own feature values.
    def build_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.sort_values(["ticker_normalized", "date"]).reset_index(drop=True)
        d["leaky_any_events_ticker"] = d.groupby("ticker_normalized")["y_inclusion"].transform(lambda s: s.notna().cumsum())
        d["leaky_incl_events_ticker"] = d.groupby("ticker_normalized")["y_inclusion"].transform(lambda s: s.cumsum())
        d["leaky_excl_events_ticker"] = d.groupby("ticker_normalized")["y_inclusion"].transform(lambda s: (1 - s).cumsum())
        denom = d["leaky_incl_events_ticker"] + d["leaky_excl_events_ticker"]
        d["leaky_inclusion_rate_ticker"] = np.where(denom > 0, d["leaky_incl_events_ticker"] / denom, 0.0)

        d = d.sort_values(["index", "date"]).reset_index(drop=True)
        d["leaky_any_events_index"] = d.groupby("index")["y_inclusion"].transform(lambda s: s.notna().cumsum())
        d["leaky_incl_events_index"] = d.groupby("index")["y_inclusion"].transform(lambda s: s.cumsum())
        d["leaky_excl_events_index"] = d.groupby("index")["y_inclusion"].transform(lambda s: (1 - s).cumsum())
        denom_i = d["leaky_incl_events_index"] + d["leaky_excl_events_index"]
        d["leaky_inclusion_rate_index"] = np.where(denom_i > 0, d["leaky_incl_events_index"] / denom_i, 0.0)
        return d

    leaky_train = build_leaky_features(train)
    leaky_test = build_leaky_features(test)

    feature_num = [
        "year",
        "month",
        "leaky_any_events_ticker",
        "leaky_incl_events_ticker",
        "leaky_excl_events_ticker",
        "leaky_inclusion_rate_ticker",
        "prior_any_events_index",
        "leaky_any_events_index",
        "leaky_inclusion_rate_index",
    ]

    # For simplicity, reuse baseline numeric scaling/imputation with a smaller set.
    # Also include index one-hot like baseline.
    feature_cat = ["index"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_num),
            ("cat", categorical_transformer, feature_cat),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=cfg.random_state,
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    model.fit(leaky_train[feature_num + feature_cat], leaky_train["y_inclusion"].values)

    if np.unique(leaky_test["y_inclusion"].values).size < 2:
        return {"auc_roc_leaky": float("nan")}
    proba = model.predict_proba(leaky_test[feature_num + feature_cat])[:, 1]
    auc = roc_auc_score(leaky_test["y_inclusion"].values, proba)
    return {"auc_roc_leaky": float(auc)}


def build_and_run_baseline(df_model: pd.DataFrame, cfg: Config) -> None:
    # Add calendar features
    df = df_model.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Add prior-history features (leakage-safe: uses only date < current date)
    df_feat = add_prior_event_features(df, cfg)

    # Time-based split
    train, test, cutoff_date = time_based_split(df_feat, cfg)
    print("\nTime split:")
    print(f"  cutoff_date: {cutoff_date.date()}")
    print(f"  train rows: {len(train)}")
    print(f"  test rows: {len(test)}")

    # Global sample size warnings
    if len(df_feat) < cfg.min_total_rows_for_model:
        print(
            f"\nWARNING: Total modeling subset rows ({len(df_feat)}) < {cfg.min_total_rows_for_model}."
            " Skipping baseline model; prioritize descriptive analysis."
        )
        return

    minority_train = min((train["y_inclusion"] == 0).sum(), (train["y_inclusion"] == 1).sum())
    minority_test = min((test["y_inclusion"] == 0).sum(), (test["y_inclusion"] == 1).sum())
    if minority_train < cfg.min_minority_rows_for_model or minority_test < cfg.min_minority_rows_for_model:
        print(
            "\nWARNING: Minority-class rows too small for a meaningful classifier "
            f"(minority_train={minority_train}, minority_test={minority_test}). Skipping baseline model."
        )
        return

    if np.unique(train["y_inclusion"].values).size < 2:
        print("\nWARNING: Only one class appears in train; skipping baseline model.")
        return
    if np.unique(test["y_inclusion"].values).size < 2:
        print("\nWARNING: Only one class appears in test; AUC would be meaningless; skipping baseline model.")
        return

    # Train
    try:
        model, feature_num, feature_cat = train_baseline_model(train, cfg)
    except ValueError as e:
        print("\nBaseline model not trained:")
        print("  Reason:", str(e))
        return

    # Evaluate
    _ = evaluate_model(model, train, test, cfg)


def main() -> None:
    cfg = Config()

    df_raw = load_data(cfg)
    df_clean = clean_events(df_raw, cfg)

    # Step 5 descriptive analytics
    summarize_events(df_clean)

    # Step 3 modeling subset build
    df_model = build_modeling_subset(df_clean, cfg)
    print("\nModeling subset class balance:")
    print(df_model["y_inclusion"].value_counts().to_string())

    # Step 6-8 baseline model (only if justified)
    print("\n====================")
    print("Leakage-safe baseline model")
    print("====================")
    build_and_run_baseline(df_model, cfg)

    # Final conclusion
    print("\n====================")
    print("Final Conclusion")
    print("====================")
    n = len(df_model)
    if n < cfg.min_total_rows_for_model:
        print(
            f"Event dataset is small for ML (n={n}). "
            "Prioritize descriptive event-study analysis and next-step feature augmentation."
        )
    else:
        # Use conservative thresholds to judge sufficiency.
        vc = df_model["y_inclusion"].value_counts()
        minority = int(vc.min()) if len(vc) == 2 else 0
        if minority < cfg.min_minority_rows_for_model:
            print(
                f"Even with n={n}, minority class is too small (minority={minority}) for a reliable classifier. "
                "Prefer descriptive analysis for now."
            )
        else:
            print(
                f"Data size (n={n}) and class balance look sufficient to attempt a conservative baseline. "
                "Interpret model results cautiously and ensure time-split diagnostics remain stable."
            )

    print("\nNext best dataset enhancements (to build a real inclusion prediction model):")
    print(" - Candidate universe of non-included stocks (time-varying eligibility, not just eligible-in-universe).")
    print(" - Market cap and liquidity/turnover as-of date (point-in-time shares/float; avoid snapshot leakage).")
    print(" - Fundamentals and valuation aligned to filing/period-end dates (point-in-time financials).")
    print(" - Sector/industry with as-of dates (or filing-based alignment).")
    print(" - Index eligibility criteria / historical index rules if available.")
    print(" - Historical price/volume panel with correct event-date alignment (effective date vs announcement date).")
    print("\nIf you want, the next stage can extend this script by adding leakage-safe yfinance-based OHLCV features and (best-effort) time-aligned fundamentals.")


if __name__ == "__main__":
    main()

