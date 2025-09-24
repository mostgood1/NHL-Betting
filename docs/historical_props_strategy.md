# Historical Player Props Sourcing Strategy (Draft)

## Objectives
Acquire sufficient historical player prop lines & prices (SOG, Goals, Saves; later Assists/Points) to:
1. Calibrate distributional models vs market expectations.
2. Evaluate edge realization (closing line vs opening line drift).
3. Backtest staking strategies.

## Minimum Viable Historical Set
- 1.5–2 prior regular seasons of: line, over/under prices, timestamp (approx), and eventual result (actual stat).
- Coverage priority order: SOG > Goals > Saves > Assists > Points.

## Potential Data Sources
1. OddsAPI (if supports historical player props via paid tier). Pros: structured JSON; Cons: cost, possible gaps.
2. Self-archived forward collection (start now, build corpus going forward). Pros: free; Cons: no retro coverage.
3. Third-party archived scrapes (manual import if accessible). Pros: breadth; Cons: one-off ingestion & cleaning.
4. Synthetic reconstruction using closing lines + modeled vig removal (only for baselines; not a substitute for real line evolution).

## Proposed Approach
Phase A (Immediate): Begin daily archiving of current Bovada props (SOG, Goals, Saves) using existing `BovadaClient.fetch_props_odds` into `data/props/player_props_lines/raw/date=YYYY-MM-DD/*.parquet`.
- Include fields: date, player (raw string), market, line, side, odds, book, collected_at (UTC timestamp), source='bovada:live'.
- Pivot to canonical line format for `player_props_lines` fact table: combine OVER & UNDER rows into one record with over_price & under_price.

Phase B (Enhancement): Add additional books (OddsAPI realtime feeds) to broaden market consensus; unify player name normalization to NHL player_id.

Phase C (Historical Backfill): Attempt bulk export if an external historical dataset becomes available; design importer script to map to schema (fuzzy name matching -> ID, dedupe by identical line & price within short window).

Phase D (Evolution): Track mid-day line movement by running collection every X minutes (configurable, default 15) and version rows (`first_seen_at`, `last_seen_at`).

## Data Normalization
- Player Name to ID Mapping: Use roster snapshots (recent + historical game logs). Matching algorithm:
  1. Exact match first (case-insensitive, strip punctuation).
  2. If multiple, prefer player active on the teams playing that date.
  3. Fuzzy fallback (Levenshtein distance <=2) flagged for manual review.
- Market Canonicalization: SOG, GOALS, SAVES initial; ASSISTS, POINTS later.
- Price Standardization: Store American odds as int, compute implied probabilities & remove vig lazily at modeling time.

## Schema Mapping (Raw -> Fact)
Raw rows (over & under):
- Combine by (date, player_id, market, line, book) selecting contemporary over/under prices.
- If only one side present, keep partial (set missing side to NULL) but mark `incomplete=1` (optional column) for QC.
- Assign `is_current=1` initially; subsequent collection runs set prior row `is_current=0` and insert new.

## Collection Frequency & Storage
- Pre-season / early dev: once daily (midday ET).
- After deployment: every 15 mins from 10:00 ET to first game start; every 5 mins final 60 mins pre-faceoff (config tiers).
- Storage layout (Parquet partitioning):
  - `data/props/raw_props/date=YYYY-MM-DD/book=bovada/*.parquet`
  - `data/props/player_props_lines/date=YYYY-MM-DD/*.parquet`

## Quality Checks
- Null rate of player_id after mapping (<2% target).
- Duplicate (market,line) entries per player per timestamp window (should be low; detect same over+under pair repeated).
- Line movement tracking: median absolute price change per market per day.

## Backfill Gap Strategy
If no historical dataset obtained:
- Bootstrap models using player game logs only (distributional assumption) and evaluate vs newly collected current-season lines.
- After 30+ days of collection, recalibrate priors with empirical market means.

## Tooling & Scripts
- New module: `nhl_betting/data/player_props.py` (collector & normalizer).
  - `collect_bovada_props(date: str) -> pd.DataFrame`
  - `normalize_props(df_raw: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame`
  - `write_props(df_norm: pd.DataFrame, date: str)`
- CLI entry (later): `props-collect --date YYYY-MM-DD --book bovada --interval 15` (interval=0 => single snapshot)

## Open Questions
- Retain intermediate snapshots vs collapse to daily open/close? (Initial: retain all for movement analysis.)
- Add implied hold calculation now or at model time? (Defer to model step.)

## Next Actions
1. Implement `player_props.py` raw collection + normalization stubs.
2. Wire daily snapshot into existing `daily_update` sequence (after odds; before predictions).
3. Start persisting new daily data immediately for forward fill.
