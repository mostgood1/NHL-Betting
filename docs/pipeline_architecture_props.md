# Props Pipeline Architecture (Draft)

## Overview
Integrates new props data ingestion, feature building, modeling, and predictions into existing daily pipeline without disrupting current odds & recommendations workflow.

## Existing Daily Flow (Simplified)
1. Fetch/update games & odds.
2. Run model predictions (team-level / recommendations).
3. Commit & push artifacts.

## Extended Flow (Target)
1. (A) Roster snapshots build (pre-games)
2. (B) Player game logs incremental update (post previous day finals)
3. (C) Usage aggregates refresh (incremental)
4. (D) Props line collection snapshot (book-specific) [repeatable intra-day]
5. (E) Feature build for target date (skater / goalie props)
6. (F) Model training or refresh (conditional: schedule-based or trigger-based)
7. (G) Inference: compute probabilities for current lines
8. (H) Persist predictions & edges
9. (I) Commit & push

## Incremental Update Contracts
- Roster snapshots: idempotent per date (overwrite allowed, version bump optional later).
- Player game logs: append-only for new completed games; detect duplicates by (game_pk, player_id).
- Usage aggregates: recompute only for players appearing in newly finalized games; reuse prior rows for others.
- Props lines: multi-snapshot per date; maintain current vs historical pricing by is_current flag.

## Modules Mapping
- `nhl_betting/data/rosters.py` -> step (A)
- `nhl_betting/data/player_game_logs.py` (to implement) -> steps (B)
- `nhl_betting/data/usage_aggregates.py` (to implement) -> step (C)
- `nhl_betting/data/player_props.py` -> step (D)
- `nhl_betting/features/props_features.py` (to implement) -> step (E)
- `nhl_betting/models/props_train.py` (to implement) -> step (F)
- `nhl_betting/models/props_infer.py` (to implement) -> step (G)

## Orchestration
Add optional flag to `daily_update.py`:
- `--with-props` executes steps A through H (single daily snapshot path).
- Future: separate `intra_day_props.py` for repeating D+G+H on schedule.

## Error Handling
- Non-critical failures (e.g., props collection returns empty) log warning and continue main pipeline.
- Model absence: fallback to rolling mean lambda from recent game logs (graceful degradation pattern).

## Configuration
- YAML or env-based config for: collection interval, backfill toggles, feature window sizes, dispersion threshold for NB switch.

## Outputs
- Data directories under `data/props/`:
  - `roster_snapshots/date=YYYY-MM-DD/*.parquet`
  - `player_props_lines/date=YYYY-MM-DD/*.parquet`
  - `predictions/date=YYYY-MM-DD/{market}.parquet`
  - `models/{market}/model_meta.json`

## Logging & Metrics
- Count of players with predictions per market
- Empty/failed collections
- Model dispersion statistic & last refit timestamp

## Security & Rate Limits
- Throttle Stats API requests (already set ~3/sec); reuse caching for back-to-back modules where possible.

## Next Steps
1. Implement stub writer for roster snapshots triggered by a new CLI command.
2. Add `--with-props` flag to daily update orchestrator calling new stubs (no-op safe if data missing).
3. Implement feature builder & SOG Poisson baseline.
4. Add predictions persistence and integrate into UI or API endpoint (later phase).
