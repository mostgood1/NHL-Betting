First-10 Goal Probability Calibration
====================================

Summary
-------
We calibrate the probability of at least one goal in the first 10 minutes of P1 using PBP-derived labels. On a PBP sample (2019–2024 fetch, 2023–24 used for initial check), the best-performing scales were:

- FIRST10_SCALE (from P1 goals): 0.55
- FIRST10_TOTAL_SCALE (fallback from total goals): 0.15

These are now the defaults and can be overridden with environment variables. Predictions expose both the expected first-10 goals (lambda) and the probability:

- first_10min_proj (lambda)
- first_10min_prob = 1 - exp(-first_10min_proj)

How to Re-Run the Backtest
--------------------------
1) Ensure PBP data is fetched and ingested, then rebuild features:
   - scripts/nhl_pbp_fetch.R (requires R + fastRhockey)
   - python scripts/ingest_pbp_to_periods.py
   - python -m nhl_betting.data.game_features

2) Run the backtest:
   - python scripts/backtest_first10.py --source pbp --start 2023-10-01 --end 2024-07-01 --out data/processed/first10_backtest_pbp_grid.csv

3) Inspect results in the console and the saved grid CSV. Adjust env vars as desired and re-run predictions.

Environment Knobs
-----------------
- FIRST10_FROM_P1 (default 1): when enabled, prefer P1-based lambda. Set 0/false to disable.
- FIRST10_SCALE (default 0.55): fraction of P1 goals expected in first 10 minutes.
- FIRST10_TOTAL_SCALE (default 0.15): fallback fraction when P1 unavailable; applied to derived P1 share of total.

Artifacts
---------
- data/processed/first10_backtest_pbp_grid.csv — grid of scales with Brier/LogLoss
- data/raw/games_with_features.csv — features including integer-like PBP labels where available
- data/raw/games_with_periods.csv — merged game rows with period splits and period_source

Notes
-----
- If you expand PBP coverage or seasons, re-run the backtest; defaults can be revisited.
- A quick coverage check is available via scripts/pbp_coverage_report.py; output CSV at data/processed/pbp_coverage.csv.
