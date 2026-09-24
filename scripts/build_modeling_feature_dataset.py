import os
import re
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Missing dependency `yfinance`. Install with: pip install yfinance") from e


# -------------------------
# Configuration
# -------------------------
INPUT_EVENT_CSV = "data/full_hk_index_dataset.csv"
INPUT_ELIGIBLE_CSV = "data/eligible_securities.csv"

OUTPUT_FEATURES_CSV = "data/modeling_universe_enriched_features.csv"
OUTPUT_DIAGNOSTICS_TXT = "data/modeling_universe_diagnostics.txt"

YF_CACHE_DIR = "data/yfinance_cache"
os.makedirs(YF_CACHE_DIR, exist_ok=True)

HORIZON_TRADING_DAYS = 60

# Approx trading-day windows
WIN_RET_1M = 21
WIN_RET_3M = 63
WIN_RET_6M = 126

WIN_VOL_20D = 20
WIN_VOL_60D = 60

WIN_LIQ_20D = 20
WIN_LIQ_60D = 60

DO_STATEMENT_BASED_FUNDAMENTALS = True
ONLY_TICKERS_WITH_EVENTS = True
MAX_TICKERS_FOR_RUN: Optional[int] = None

# Practical: yfinance download can fail; we skip tickers we can't download.
DOWNLOAD_AUTO_ADJUST = False


# -------------------------
# Utility helpers
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def normalize_change_type(x: str) -> Optional[str]:
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


def extract_ticker_parts(ticker_normalized: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (code_int_str, exchange_suffix) from yfinance-like ticker_normalized.

    Example: "1378.HK" -> ("1378","HK")
    """
    if ticker_normalized is None or (isinstance(ticker_normalized, float) and math.isnan(ticker_normalized)):
        return None, None
    s = str(ticker_normalized).strip()
    if not s:
        return None, None
    parts = s.split(".")
    if len(parts) < 2:
        return None, None
    code = parts[0]
    exch = parts[1].upper()
    try:
        code_int_str = str(int(code))
    except Exception:
        return None, exch
    return code_int_str, exch


def last_trading_day_index(prices_dates: np.ndarray, dt: np.datetime64) -> int:
    """Index of last trading day strictly less than dt.

    prices_dates must be sorted, normalized to datetime64[D].
    Returns -1 if no trading day exists before dt.
    """
    pos = np.searchsorted(prices_dates, dt, side="left") - 1
    return int(pos)


# -------------------------
# Stage 1: Load + standardize + map to eligible universe
# -------------------------
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_events = pd.read_csv(
        INPUT_EVENT_CSV,
        dtype={
            "index": str,
            "change_type": str,
            "ticker": str,
            "ticker_normalized": str,
            "company": str,
        },
    )
    df_eligible = safe_read_csv(INPUT_ELIGIBLE_CSV)

    print("Loaded event dataset:")
    print(f"  rows={len(df_events)} cols={list(df_events.columns)}")
    print("Loaded eligible dataset:")
    print(f"  rows={len(df_eligible)} cols={list(df_eligible.columns)}")

    return df_events, df_eligible


def standardize_event_tickers_and_dates(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), ["date", "index", "change_type", "ticker_normalized"]].head(10)
        raise ValueError("Failed to parse some `date` values. Example rows:\n" + bad.to_string(index=False))
    df["date"] = df["date"].dt.normalize()

    df["change_type"] = df["change_type"].map(normalize_change_type)
    if df["change_type"].isna().any():
        bad = df.loc[df["change_type"].isna(), ["change_type", "index", "ticker_normalized"]].head(10)
        raise ValueError("Encountered unknown `change_type` values. Example rows:\n" + bad.to_string(index=False))

    for col in ["index", "ticker", "ticker_normalized", "company"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def build_eligible_universe(df_eligible: pd.DataFrame) -> Dict[str, str]:
    required_cols = ["证券代码", "中文简称", "英文简称"]
    missing = [c for c in required_cols if c not in df_eligible.columns]
    if missing:
        raise KeyError(f"eligible_securities.csv missing columns: {missing}. Found: {list(df_eligible.columns)}")

    df = df_eligible.copy()
    df["证券代码"] = df["证券代码"].astype(str).str.strip()
    code_numeric = pd.to_numeric(df["证券代码"], errors="coerce")
    df = df[code_numeric.notna()].copy()
    df["code_int_str"] = code_numeric[code_numeric.notna()].astype(int).astype(str)

    # HK yfinance tickers
    df["ticker_yf"] = df["code_int_str"].apply(lambda s: f"{s}.HK")

    mapping = (
        df[["code_int_str", "ticker_yf"]]
        .drop_duplicates()
        .set_index("code_int_str")["ticker_yf"]
        .to_dict()
    )
    print(f"Eligible universe size (mapped): {len(mapping)} tickers")
    return mapping


def map_events_to_eligible(df_events: pd.DataFrame, eligible_map: Dict[str, str]) -> pd.DataFrame:
    df = df_events.copy()
    parsed = df["ticker_normalized"].apply(extract_ticker_parts)
    df[["eligible_code", "exchange"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # Only map HK events.
    df["ticker_yf"] = np.where(
        (df["exchange"] == "HK") & (df["eligible_code"].notna()),
        df["eligible_code"].map(eligible_map),
        np.nan,
    )

    before = len(df)
    df = df[df["ticker_yf"].notna()].copy()
    after = len(df)

    eligible_codes = set(eligible_map.keys())
    event_codes_hk = set(
        df_events.loc[df_events["ticker_normalized"].notna(), "ticker_normalized"]
        .map(extract_ticker_parts)
        .map(lambda x: x[0])
        .dropna()
        .tolist()
    )
    overlap = event_codes_hk.intersection(eligible_codes)

    print("\nMapping diagnostics:")
    print(f"  event rows total: {before}")
    print(f"  mapped to eligible (HK-only): {after}")
    print(f"  dropped rows: {before-after}")
    print(f"  eligible codes: {len(eligible_codes)}")
    print(f"  event HK codes (from ticker_normalized): {len(event_codes_hk)}")
    print(f"  overlap codes: {len(overlap)}")

    return df


# -------------------------
# Stage 2: yfinance history + caching
# -------------------------
def yf_cache_path(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", ticker)
    return os.path.join(YF_CACHE_DIR, f"{safe}.pkl")


def fetch_prices_yfinance(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str = YF_CACHE_DIR,
    auto_adjust: bool = DOWNLOAD_AUTO_ADJUST,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    start_str = str(pd.Timestamp(start).date())
    end_str = str(pd.Timestamp(end).date())

    for i, ticker in enumerate(tickers):
        cache_path = yf_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                cached = pd.read_pickle(cache_path)
                cached_start = pd.to_datetime(cached.get("start"))
                cached_end = pd.to_datetime(cached.get("end"))
                if cached_start <= start and cached_end >= end:
                    out[ticker] = cached["prices"]
                    continue
            except Exception:
                pass

        print(f"Downloading prices from yfinance: ({i+1}/{len(tickers)}) {ticker}")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_str, end=end_str, auto_adjust=auto_adjust, actions=False)
            if hist is None or hist.empty:
                print(f"  Warning: empty history for {ticker}")
                continue

            hist = hist.copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.sort_index()

            needed = ["Open", "High", "Low", "Close", "Volume"]
            keep_cols = [c for c in needed if c in hist.columns]
            hist = hist[keep_cols]

            payload = {"ticker": ticker, "start": start_str, "end": end_str, "prices": hist}
            pd.to_pickle(payload, cache_path)
            out[ticker] = hist
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
            continue

    return out


# -------------------------
# Stage 3-4: Price features + next-event labels
# -------------------------
def compute_price_features_for_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in prices.columns:
        raise KeyError("prices missing Close")

    df = prices.copy()
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    log_ret = np.log(close).diff()
    dollar_volume = close * volume

    feat = pd.DataFrame(index=df.index)

    feat["ret_1m"] = close.pct_change(WIN_RET_1M)
    feat["ret_3m"] = close.pct_change(WIN_RET_3M)
    feat["ret_6m"] = close.pct_change(WIN_RET_6M)

    feat["vol_20d"] = log_ret.rolling(WIN_VOL_20D).std()
    feat["vol_60d"] = log_ret.rolling(WIN_VOL_60D).std()

    feat["dollar_volume"] = dollar_volume
    feat["avg_dollar_volume_20d"] = dollar_volume.rolling(WIN_LIQ_20D).mean()
    feat["avg_dollar_volume_60d"] = dollar_volume.rolling(WIN_LIQ_60D).mean()

    feat["volume"] = volume
    feat["avg_volume_20d"] = volume.rolling(WIN_LIQ_20D).mean()
    feat["avg_volume_60d"] = volume.rolling(WIN_LIQ_60D).mean()

    # Market cap/turnover placeholders (filled only if shares are available)
    feat["market_cap_approx"] = np.nan
    feat["share_turnover"] = np.nan

    return feat


def fetch_shares_outstanding_static(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort static shares outstanding from yfinance info.

    We avoid using marketCap snapshot directly. We use shares as a stable multiplier
    to compute market cap approximations at date t using close.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            return None, None
        shares = float(shares)
        if not np.isfinite(shares) or shares <= 0:
            return None, None
        return shares, "yfinance_info_static_sharesOutstanding"
    except Exception:
        return None, None


def extract_statement_ttm_features(
    quarterly_financials: Optional[pd.DataFrame],
    quarterly_balance_sheet: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Extract ttm revenue, net income, book equity and ROE/ROA if possible."""

    def pick_metric(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for c in candidates:
            if c in df.index:
                return df.loc[c]
        return None

    if quarterly_financials is None or quarterly_financials.empty:
        return pd.DataFrame()

    fin = quarterly_financials.copy()
    fin.columns = pd.to_datetime(fin.columns, errors="coerce")
    fin = fin.loc[:, fin.columns.notna()]
    fin = fin.sort_index(axis=1)
    period_end = fin.columns

    bs = quarterly_balance_sheet.copy() if quarterly_balance_sheet is not None else None
    if bs is not None and not bs.empty:
        bs.columns = pd.to_datetime(bs.columns, errors="coerce")
        bs = bs.loc[:, bs.columns.notna()]
        bs = bs.sort_index(axis=1)

    revenue_s = pick_metric(fin, ["Total Revenue", "Total Revenues", "Revenue", "TotalRevenue"])
    net_income_s = pick_metric(
        fin,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "NetIncome",
            "Net Income Applicable To Common Shares",
        ],
    )

    equity_s = None
    total_assets_s = None
    if bs is not None and not bs.empty:
        equity_s = pick_metric(
            bs,
            [
                "Total Stockholder Equity",
                "Total equity",
                "TotalEquity",
                "Stockholders Equity",
                "Total shareholders equity",
            ],
        )
        total_assets_s = pick_metric(bs, ["Total Assets", "TotalAssets", "Total assets"])

    def to_float_aligned(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None:
            return None
        s2 = s.copy()
        s2.index = pd.to_datetime(s2.index, errors="coerce")
        s2 = s2.loc[period_end]
        return pd.to_numeric(s2, errors="coerce")

    revenue_s = to_float_aligned(revenue_s)
    net_income_s = to_float_aligned(net_income_s)
    equity_s = to_float_aligned(equity_s)
    total_assets_s = to_float_aligned(total_assets_s)

    out = pd.DataFrame(index=period_end)
    out["revenue_ttm"] = revenue_s.rolling(4, min_periods=4).sum() if revenue_s is not None else np.nan
    out["net_income_ttm"] = net_income_s.rolling(4, min_periods=4).sum() if net_income_s is not None else np.nan
    out["book_value"] = equity_s if equity_s is not None else np.nan
    out["total_assets"] = total_assets_s if total_assets_s is not None else np.nan

    out["roe"] = out["net_income_ttm"] / out["book_value"]
    out["roa"] = out["net_income_ttm"] / out["total_assets"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def fetch_statement_based_fundamentals_for_ticker(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        q_fin = getattr(t, "quarterly_financials", None)
        q_bs = getattr(t, "quarterly_balance_sheet", None)
        if q_fin is None or q_fin.empty:
            return pd.DataFrame()
        return extract_statement_ttm_features(q_fin, q_bs)
    except Exception:
        return pd.DataFrame()


def label_and_features_for_ticker(
    ticker_yf: str,
    eligible_code: str,
    df_events_mapped: pd.DataFrame,
    observation_dates: np.ndarray,
    prices: pd.DataFrame,
    horizon_trading_days: int = HORIZON_TRADING_DAYS,
    do_fundamentals: bool = DO_STATEMENT_BASED_FUNDAMENTALS,
    shares_outstanding_static: Optional[float] = None,
    shares_source: Optional[str] = None,
) -> pd.DataFrame:
    # Event stream for this ticker: Inclusion/Exclusion only
    ev = df_events_mapped[
        (df_events_mapped["ticker_yf"] == ticker_yf)
        & (df_events_mapped["change_type"].isin(["Inclusion", "Exclusion"]))
    ].copy()
    if ev.empty:
        return pd.DataFrame()

    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev = ev.sort_values("date")
    ev_dates = ev["date"].to_numpy().astype("datetime64[D]")
    ev_types = ev["change_type"].to_numpy()

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices_dates = pd.to_datetime(prices.index).normalize().to_numpy().astype("datetime64[D]")
    if len(prices_dates) < 30:
        return pd.DataFrame()

    price_feat = compute_price_features_for_prices(prices)

    market_cap_is_approx = 0
    market_cap_available = 0
    turnover_available = 0
    if shares_outstanding_static is not None:
        market_cap_is_approx = 1
        market_cap_available = 1
        turnover_available = 1
        close = prices["Close"].astype(float)
        volume = prices["Volume"].astype(float) if "Volume" in prices.columns else pd.Series(index=prices.index, data=np.nan)
        price_feat["market_cap_approx"] = close * float(shares_outstanding_static)
        price_feat["share_turnover"] = volume / float(shares_outstanding_static)

    # For each observation date t:
    #  - find the next Inclusion/Exclusion event for this ticker after t
    #  - accept only if it occurs within horizon_trading_days after t
    positions = np.searchsorted(ev_dates, observation_dates, side="right")
    obs_idx = np.array([last_trading_day_index(prices_dates, d) for d in observation_dates], dtype=int)
    valid_obs_mask = obs_idx >= 0

    labeled_rows: List[dict] = []

    for j, t_obs in enumerate(observation_dates):
        pos = int(positions[j])
        if pos >= len(ev_dates):
            continue
        if not valid_obs_mask[j]:
            continue

        next_date = ev_dates[pos]
        next_type = ev_types[pos]

        idx_t = int(obs_idx[j])
        idx_next = last_trading_day_index(prices_dates, next_date)
        if idx_next < 0:
            continue

        if (idx_next - idx_t) > horizon_trading_days:
            continue

        row = {
            "ticker_yf": ticker_yf,
            "eligible_code": eligible_code,
            "date": pd.to_datetime(t_obs),
            "next_event_type": str(next_type),
            "y_inclusion_next": 1 if next_type == "Inclusion" else 0,
            "anchor_trade_date": pd.to_datetime(prices_dates[idx_t]),
        }
        row.update(price_feat.iloc[idx_t].to_dict())

        # Provenance / availability flags
        row["shares_source"] = shares_source if shares_outstanding_static is not None else None
        row["market_cap_available"] = market_cap_available
        row["market_cap_is_approx"] = market_cap_is_approx
        row["turnover_available"] = turnover_available

        labeled_rows.append(row)

    if not labeled_rows:
        return pd.DataFrame()

    out = pd.DataFrame(labeled_rows).sort_values(["ticker_yf", "date"]).reset_index(drop=True)

    # Best-effort fundamentals: align to latest period end <= observation date
    if do_fundamentals:
        fund = fetch_statement_based_fundamentals_for_ticker(ticker_yf)
        if fund is not None and not fund.empty:
            fund_reset = fund.reset_index()
            if "index" in fund_reset.columns:
                fund_reset = fund_reset.rename(columns={"index": "fund_period_end"})
            else:
                fund_reset = fund_reset.rename(columns={fund_reset.columns[0]: "fund_period_end"})

            merged = pd.merge_asof(
                out.sort_values("date"),
                fund_reset.sort_values("fund_period_end"),
                left_on="date",
                right_on="fund_period_end",
                direction="backward",
            )
            merged["fundamentals_available"] = (merged["revenue_ttm"].notna() | merged["net_income_ttm"].notna()).astype(int)
            merged["fundamentals_source"] = "yfinance_quarterly_statements_best_effort"
            out = merged
        else:
            out["fundamentals_available"] = 0
            out["fundamentals_source"] = None
    else:
        out["fundamentals_available"] = 0
        out["fundamentals_source"] = None

    return out


def quality_diagnostics(df: pd.DataFrame, mapping_stats: Dict[str, int], start: pd.Timestamp, end: pd.Timestamp, horizon: int) -> str:
    lines: List[str] = []
    lines.append("Modeling dataset diagnostics")
    lines.append(f"  label horizon (trading days): {horizon}")
    lines.append(f"  yfinance price download range (approx): {start.date()} to {end.date()}")
    lines.append(f"  labeled rows: {len(df)}")

    if len(df) > 0:
        vc = df["next_event_type"].value_counts(dropna=False)
        lines.append("  label balance (next_event_type):")
        for k, v in vc.items():
            lines.append(f"    - {k}: {v}")

        if vc.min() < 20:
            lines.append("  WARNING: very few minority-class labeled rows; later classification may be unstable.")
        if df["y_inclusion_next"].nunique(dropna=True) < 2:
            lines.append("  WARNING: only one class appears; classification is ill-posed.")

        # Feature missingness
        exclude = {
            "ticker_yf",
            "eligible_code",
            "date",
            "next_event_type",
            "y_inclusion_next",
            "anchor_trade_date",
            "shares_source",
            "fundamentals_source",
        }
        feature_cols = [c for c in df.columns if c not in exclude]
        null_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
        lines.append("  top missingness (fraction):")
        for col, rate in null_rates.head(10).items():
            lines.append(f"    - {col}: {rate:.3f}")

        # Pre-event feature sanity: anchor_trade_date must be strictly before date
        ok = pd.to_datetime(df["anchor_trade_date"]) < pd.to_datetime(df["date"])
        if not ok.all():
            lines.append(f"  WARNING: {((~ok).sum())} rows have anchor_trade_date not strictly before date.")

    lines.append("")
    lines.append("Conclusion")
    if len(df) == 0:
        lines.append("  The dataset is empty after label filtering; adjust horizon and/or universe mapping or prioritize descriptive analysis.")
    else:
        vc = df["next_event_type"].value_counts(dropna=False)
        if len(vc) < 2:
            lines.append("  Data does not support an Inclusion-vs-Exclusion classifier right now (one class missing).")
        elif vc.min() < 50:
            lines.append("  Dataset may be too small for a serious classifier; prioritize descriptive event analysis first.")
        else:
            lines.append("  Dataset size appears adequate for a conservative baseline model later; still verify leakage and evaluate using time splits.")

    lines.append("")
    lines.append("Next dataset enhancements (recommended)")
    lines.append("  - Build a non-included candidate universe with time-varying eligibility flags.")
    lines.append("  - Add shares/float history so market cap/turnover are point-in-time, not static.")
    lines.append("  - Add sector/industry with historical as-of dates (or filing-based alignment).")
    lines.append("  - Add index eligibility criteria if you can obtain historical rules/changes.")
    lines.append("  - Add a richer price/volume panel source with better event-date alignment (announcement vs trading date).")

    lines.append("")
    lines.append("Mapping coverage (stage 1):")
    for k, v in mapping_stats.items():
        lines.append(f"  - {k}: {v}")

    return "\n".join(lines)


def main() -> None:
    df_events_raw, df_eligible_raw = load_inputs()
    df_events_clean = standardize_event_tickers_and_dates(df_events_raw)
    eligible_map = build_eligible_universe(df_eligible_raw)
    df_events_mapped = map_events_to_eligible(df_events_clean, eligible_map)

    print("\nMapped events change_type counts:")
    print(df_events_mapped["change_type"].value_counts(dropna=False))

    df_events_incl_excl = df_events_mapped[df_events_mapped["change_type"].isin(["Inclusion", "Exclusion"])].copy()
    print(f"\nInclusion/Exclusion mapped rows: {len(df_events_incl_excl)}")

    observation_dates = np.sort(df_events_mapped["date"].dt.normalize().unique()).astype("datetime64[D]")

    if ONLY_TICKERS_WITH_EVENTS:
        eligible_tickers = sorted(df_events_incl_excl["ticker_yf"].unique().tolist())
    else:
        eligible_tickers = sorted(df_events_mapped["ticker_yf"].unique().tolist())

    if MAX_TICKERS_FOR_RUN is not None:
        eligible_tickers = eligible_tickers[:MAX_TICKERS_FOR_RUN]

    print(f"\nEligible tickers for this run: {len(eligible_tickers)}")
    print(f"Number of observation dates: {len(observation_dates)}")

    min_date = pd.to_datetime(df_events_mapped["date"].min())
    max_date = pd.to_datetime(df_events_mapped["date"].max())
    yf_start = min_date - pd.Timedelta(days=500)
    yf_end = max_date + pd.Timedelta(days=250)

    mapping_stats = {
        "event_rows_total_raw": int(len(df_events_raw)),
        "event_rows_total_mapped": int(len(df_events_mapped)),
        "inclusion_exclusion_rows_mapped": int(len(df_events_incl_excl)),
        "unique_mapped_tickers": int(df_events_mapped["ticker_yf"].nunique()),
    }

    all_rows: List[pd.DataFrame] = []
    for k, ticker_yf in enumerate(eligible_tickers):
        print(f"\nProcessing ticker {k+1}/{len(eligible_tickers)}: {ticker_yf}")
        sub = df_events_mapped[df_events_mapped["ticker_yf"] == ticker_yf]
        if sub.empty:
            continue
        eligible_code = str(sub["eligible_code"].dropna().iloc[0])

        prices_map = fetch_prices_yfinance([ticker_yf], yf_start, yf_end)
        if ticker_yf not in prices_map:
            print("  Warning: no prices downloaded, skipping.")
            continue
        prices = prices_map[ticker_yf]

        shares, shares_source = (None, None)
        if prices is not None and not prices.empty:
            shares, shares_source = fetch_shares_outstanding_static(ticker_yf)

        labeled_features = label_and_features_for_ticker(
            ticker_yf=ticker_yf,
            eligible_code=eligible_code,
            df_events_mapped=df_events_mapped,
            observation_dates=observation_dates,
            prices=prices,
            horizon_trading_days=HORIZON_TRADING_DAYS,
            do_fundamentals=DO_STATEMENT_BASED_FUNDAMENTALS,
            shares_outstanding_static=shares,
            shares_source=shares_source,
        )

        if not labeled_features.empty:
            all_rows.append(labeled_features)

    df_modeling = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    print(f"\nFinal assembled dataset rows: {len(df_modeling)}")
    if not df_modeling.empty:
        print(df_modeling["next_event_type"].value_counts(dropna=False))

    os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV) or ".", exist_ok=True)
    df_modeling.to_csv(OUTPUT_FEATURES_CSV, index=False)

    diagnostics = quality_diagnostics(
        df_modeling,
        mapping_stats=mapping_stats,
        start=yf_start,
        end=yf_end,
        horizon=HORIZON_TRADING_DAYS,
    )
    with open(OUTPUT_DIAGNOSTICS_TXT, "w", encoding="utf-8") as f:
        f.write(diagnostics)

    print("\nSaved:")
    print(" -", OUTPUT_FEATURES_CSV)
    print(" -", OUTPUT_DIAGNOSTICS_TXT)
    print("\nDiagnostics preview (first 2000 chars):")
    print(diagnostics[:2000])


if __name__ == "__main__":
    main()

