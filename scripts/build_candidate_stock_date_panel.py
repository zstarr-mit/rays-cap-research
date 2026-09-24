import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    events_path: str = "data/research_outputs/core/master_events_normalized.csv"
    eligible_path: str = "data/research_outputs/core/eligible_universe_cleaned.csv"
    all_symbols_universe_path: str = "data/research_outputs/core/enrichment_universe_all_symbols.csv"
    out_dir: str = "data/research_outputs/candidate_panels"

    # Label horizon for forward-looking targets.
    # Uses calendar days in this step-1 panel builder.
    horizon_days: int = 120

    # Date-grid design:
    # - "event_dates": use unique event dates only (compact panel)
    # - "month_end": use month-end dates between min and max event date (broader panel)
    date_grid_mode: str = "event_dates"


def load_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(cfg.events_path):
        raise FileNotFoundError(f"Missing events file: {cfg.events_path}")
    if not os.path.exists(cfg.eligible_path):
        raise FileNotFoundError(f"Missing eligible file: {cfg.eligible_path}")
    if not os.path.exists(cfg.all_symbols_universe_path):
        raise FileNotFoundError(f"Missing all-symbol universe file: {cfg.all_symbols_universe_path}")

    events = pd.read_csv(cfg.events_path)
    eligible = pd.read_csv(cfg.eligible_path)
    all_symbols = pd.read_csv(cfg.all_symbols_universe_path)

    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    return events, eligible, all_symbols


def build_observation_dates(events: pd.DataFrame, cfg: Config) -> pd.DatetimeIndex:
    dmin = pd.Timestamp(events["date"].min()).normalize()
    dmax = pd.Timestamp(events["date"].max()).normalize()

    if cfg.date_grid_mode == "event_dates":
        obs = pd.DatetimeIndex(sorted(events["date"].dropna().unique()))
    elif cfg.date_grid_mode == "month_end":
        obs = pd.date_range(start=dmin, end=dmax, freq="M")
    else:
        raise ValueError(f"Unsupported date_grid_mode: {cfg.date_grid_mode}")
    return obs


def _build_panel_for_universe(events: pd.DataFrame, ticker_universe: pd.DataFrame, cfg: Config, universe_flag: str) -> pd.DataFrame:
    obs_dates = build_observation_dates(events, cfg)

    if "ticker_yf" not in ticker_universe.columns:
        raise KeyError("ticker universe must contain `ticker_yf`")
    if "ticker_yf" not in events.columns and "ticker_yf_normalized" not in events.columns:
        raise KeyError("events file must contain `ticker_yf`")
    if "ticker_yf" not in events.columns and "ticker_yf_normalized" in events.columns:
        events = events.copy()
        events["ticker_yf"] = events["ticker_yf_normalized"]

    eligible_tickers = sorted(ticker_universe["ticker_yf"].dropna().astype(str).unique().tolist())
    print(f"[panel] universe={universe_flag} tickers: {len(eligible_tickers)}")
    print(f"[panel] observation dates: {len(obs_dates)} ({cfg.date_grid_mode})")

    panel = pd.MultiIndex.from_product(
        [eligible_tickers, obs_dates],
        names=["ticker_yf", "obs_date"],
    ).to_frame(index=False)

    # Forward labels by horizon:
    # y_inclusion_horizon = 1 if next Inclusion occurs within horizon_days calendar days after obs_date
    # (same pattern for exclusion / any change)
    incl = events.loc[events["change_type"] == "Inclusion", ["ticker_yf", "date"]].copy()
    excl = events.loc[events["change_type"] == "Exclusion", ["ticker_yf", "date"]].copy()
    any_chg = events.loc[events["change_type"].isin(["Inclusion", "Exclusion", "Removal"]), ["ticker_yf", "date"]].copy()

    incl_groups: Dict[str, np.ndarray] = {
        t: np.sort(g["date"].values.astype("datetime64[ns]")) for t, g in incl.groupby("ticker_yf")
    }
    excl_groups: Dict[str, np.ndarray] = {
        t: np.sort(g["date"].values.astype("datetime64[ns]")) for t, g in excl.groupby("ticker_yf")
    }
    any_groups: Dict[str, np.ndarray] = {
        t: np.sort(g["date"].values.astype("datetime64[ns]")) for t, g in any_chg.groupby("ticker_yf")
    }

    horizon_ns = np.timedelta64(cfg.horizon_days, "D")

    def next_within(arr: np.ndarray, dt: np.datetime64) -> Tuple[int, float]:
        if arr is None or len(arr) == 0:
            return 0, np.nan
        j = int(np.searchsorted(arr, dt, side="right"))
        if j >= len(arr):
            return 0, np.nan
        delta = arr[j] - dt
        days = float(delta / np.timedelta64(1, "D"))
        y = 1 if delta <= horizon_ns else 0
        return y, days

    y_inc: List[int] = []
    y_exc: List[int] = []
    y_any: List[int] = []
    d_inc: List[float] = []
    d_exc: List[float] = []
    d_any: List[float] = []

    for _, r in panel.iterrows():
        t = r["ticker_yf"]
        dt = np.datetime64(pd.Timestamp(r["obs_date"]).normalize())

        yi, di = next_within(incl_groups.get(t), dt)
        ye, de = next_within(excl_groups.get(t), dt)
        ya, da = next_within(any_groups.get(t), dt)

        y_inc.append(yi)
        y_exc.append(ye)
        y_any.append(ya)
        d_inc.append(di)
        d_exc.append(de)
        d_any.append(da)

    panel["y_inclusion_horizon"] = y_inc
    panel["y_exclusion_horizon"] = y_exc
    panel["y_any_change_horizon"] = y_any

    panel["days_to_next_inclusion"] = d_inc
    panel["days_to_next_exclusion"] = d_exc
    panel["days_to_next_any_change"] = d_any

    panel["horizon_days"] = cfg.horizon_days
    panel["date_grid_mode"] = cfg.date_grid_mode

    # Keep only fully-observed rows for model training targets:
    # if obs_date > max_event_date - horizon, we don't know full future horizon from this dataset.
    max_event_date = pd.Timestamp(events["date"].max()).normalize()
    panel["label_fully_observed"] = (
        pd.to_datetime(panel["obs_date"]).dt.normalize() <= (max_event_date - pd.Timedelta(days=cfg.horizon_days))
    ).astype(int)

    # Attach eligible metadata
    keep_cols = [c for c in ["ticker_yf", "证券代码", "中文简称", "英文简称", "code_num", "sources"] if c in ticker_universe.columns]
    panel = panel.merge(ticker_universe[keep_cols].drop_duplicates("ticker_yf"), on="ticker_yf", how="left")
    panel["universe_flag"] = universe_flag

    return panel


def build_panels(cfg: Config) -> Dict[str, pd.DataFrame]:
    events, eligible, all_symbols = load_inputs(cfg)

    hk_uni = eligible[["ticker_yf"] + [c for c in ["证券代码", "中文简称", "英文简称", "code_num"] if c in eligible.columns]].drop_duplicates(
        "ticker_yf"
    )
    hk_uni["sources"] = "eligible"

    all_uni = all_symbols[["ticker_yf"] + [c for c in ["sources"] if c in all_symbols.columns]].drop_duplicates("ticker_yf")
    if "sources" not in all_uni.columns:
        all_uni["sources"] = "master_events"

    panel_hk = _build_panel_for_universe(events, hk_uni, cfg, "hk_only")
    panel_all = _build_panel_for_universe(events, all_uni, cfg, "all_symbols")
    panel_combined = pd.concat([panel_hk, panel_all], ignore_index=True)

    return {
        "candidate_panel_hk_only": panel_hk,
        "candidate_panel_all_symbols": panel_all,
        "candidate_panel_combined": panel_combined,
    }


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    fully = panel[panel["label_fully_observed"] == 1].copy()
    return pd.DataFrame(
        [
            {"metric": "panel_rows_total", "value": len(panel)},
            {"metric": "unique_tickers", "value": panel["ticker_yf"].nunique()},
            {"metric": "unique_obs_dates", "value": panel["obs_date"].nunique()},
            {"metric": "rows_fully_observed", "value": len(fully)},
            {"metric": "positive_inclusion_fully_observed", "value": int(fully["y_inclusion_horizon"].sum())},
            {"metric": "positive_exclusion_fully_observed", "value": int(fully["y_exclusion_horizon"].sum())},
            {"metric": "positive_any_change_fully_observed", "value": int(fully["y_any_change_horizon"].sum())},
            {
                "metric": "inclusion_rate_fully_observed",
                "value": float(fully["y_inclusion_horizon"].mean()) if len(fully) else np.nan,
            },
            {
                "metric": "exclusion_rate_fully_observed",
                "value": float(fully["y_exclusion_horizon"].mean()) if len(fully) else np.nan,
            },
            {
                "metric": "any_change_rate_fully_observed",
                "value": float(fully["y_any_change_horizon"].mean()) if len(fully) else np.nan,
            },
        ]
    )


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    panels = build_panels(cfg)
    summaries = []

    for name, panel in panels.items():
        out_path = os.path.join(cfg.out_dir, f"{name}.csv")
        panel.to_csv(out_path, index=False)
        print(f"\n[done] saved: {out_path} ({len(panel)} rows)")

        s = build_summary(panel)
        if name == "candidate_panel_hk_only":
            flag = "hk_only"
        elif name == "candidate_panel_all_symbols":
            flag = "all_symbols"
        else:
            flag = "combined"
        s["universe_flag"] = flag
        summaries.append(s)

    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary_path = os.path.join(cfg.out_dir, "candidate_panel_summary.csv")
    summary_all.to_csv(summary_path, index=False)
    # Backward compatibility
    if "candidate_panel_hk_only" in panels:
        panels["candidate_panel_hk_only"].to_csv(os.path.join(cfg.out_dir, "candidate_stock_date_panel.csv"), index=False)
    print(f"\n[done] saved: {summary_path}")
    print("\nSummary:")
    print(summary_all.to_string(index=False))

    print(
        "\nWhat this does:\n"
        "  1) Creates stock-date rows for all eligible tickers x observation dates.\n"
        f"  2) Adds forward horizon labels (inclusion/exclusion/any change within next {cfg.horizon_days} calendar days).\n"
        "  3) Marks rows with fully observed future horizon (`label_fully_observed`) for leakage-safe training/evaluation.\n"
        "  4) Keeps mostly negative rows, which are essential for meaningful ML classification later."
    )


if __name__ == "__main__":
    main()

