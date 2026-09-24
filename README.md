# Hang Seng Index Event Research

Research pipeline built to support equity research at **RAYS Capital**, a Hong Kong-based fund.

The working hypothesis: pre-event market data (momentum, liquidity, volatility, size) and fundamentals can help identify stocks that are likely to **enter or exit** Hang Seng family indices — and, after a Stock Connect inclusion, whether to **keep or sell**.

This repo turns historical index reconstitution events (2016–2026) into cleaned datasets, features, event-study results, and simple walk-forward baselines.

## What’s in here

```
data/          event sheets, eligible universe, and research outputs
scripts/       Python pipelines and baseline models
```

Yahoo Finance price/fundamental caches and the local `.venv` are ignored by git (they can be regenerated).

## Data

| File | Role |
| --- | --- |
| `data/hang_seng_events_2016_2026_master.csv` | Master reconstitution file: date, index, change type, ticker, company |
| `data/eligible_securities.csv` | Hang Seng eligible HK universe (used as the “no change” candidate set) |

Change types in the master file: **Inclusion**, **Exclusion**, **Removal**.

Indices covered include the Hang Seng Index, Hang Seng Composite, HSTECH, Stock Connect / Greater Bay Area series, China Enterprises, and related ESG / biotech / China A benchmarks.

Outputs land in `data/research_outputs/` (normalized events, coverage audits, features, charts, and model tables).

## How it works

1. **Clean events** — parse mixed HK / A-share tickers, normalize index names, and flag the primary symbol.
2. **Pull prices** — download Yahoo Finance history for event names and the eligible universe (cached locally).
3. **Build leakage-safe features** — returns, liquidity, volatility, and a market-cap proxy as of a date *before* the announcement.
4. **Event study** — pre/post windows around reconstitutions, plus charts and summary tables.
5. **Model** — walk-forward logistic baselines for inclusion vs exclusion/removal, and a Stock Connect keep/sell follow-on using fundamentals.

Features are anchored before the event date (typically ~120 calendar days) so announcement-day information is not used as a predictor.

## Results

Pre-event, inclusions already look different from exclusions. In the ~1,800 feature-enriched events, names that were added had about **4.5%** 20-day momentum versus **1.7%** for exclusions, and roughly **2×** the trailing dollar volume. Hang Seng Index inclusions were even stronger: about **8.5%** in the 20 days before the announcement.

That signal also showed up out of sample. A walk-forward logistic baseline (inclusion vs exclusion/removal, features frozen 120 days before the event) reached **ROC-AUC ~0.78** and **~71% accuracy** on held-out months, versus a **54%** majority-class baseline. Ranking was the more useful result for screening: the top 5% of scores were inclusions **~92–94%** of the time, and the top 1% was **100%** across the test windows. The same ranking edge held in each fold from 2019 through early 2025.

A liquidity-only variant was weaker on AUC but still concentrated true inclusions at the top of the list (**precision@5% ~81%**, **precision@10% ~84%**).

These are ranking results on historical reconstitutions, not live P&L. Full tables live in `data/research_outputs/`.
