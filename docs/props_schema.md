# Player Props Data Schema (Draft)

## Goals
Provide normalized, incremental data structures to support modeling SOG, Goals, Assists, Points, and Goalie Saves with efficient daily updates and reproducible historical backfills.

## Core Principles
- Immutable fact tables (game logs, props lines) + slowly changing snapshots (rosters/line usage) + derived aggregates.
- IDs: use NHL `gamePk` + `player_id` (Stats API ID) + `team_id`. Avoid string names as keys; keep display names separately.
- Timestamps: store game date in UTC (date component) + faceoff datetime when available.
- Nullable fields for yet-to-start games; do not overwrite finalized stats.

## Tables

### 1. `player_game_logs`
Granular per-player per-game performance.

Columns:
- game_pk (int, PK part)
- player_id (int, PK part)
- team_id (int)
- opponent_team_id (int)
- date (date)
- is_home (bool)
- position (string) e.g. 'F','D','G'
- toi (float, minutes) total
- toi_pp (float) power play minutes
- toi_sh (float) shorthanded minutes
- shifts (int)
- shots (int)
- shot_attempts (int) (Corsi For individually credited if available) 
- goals (int)
- assists (int)
- points (int) (redundant = goals + assists for convenience)
- blocks (int)
- hits (int)
- penalties (int)
- saves (int, goalies)
- shots_against (int, goalies)
- goals_against (int, goalies)
- win (int, goalie win flag)
- empty_net_goals (int, if quickly available; optional)
- created_at (timestamp)
- updated_at (timestamp)

Indexes:
- (player_id, date)
- (team_id, date)
- (game_pk, player_id) PK

### 2. `roster_snapshots`
Roster + line/unit context captured per game day (pre-game). If updated later (e.g., scratches), capture additional snapshot rows with a sequence.

Columns:
- snapshot_date (date)
- game_pk (int NULL) (if pre-game lineup tied to a specific game, else NULL for off-day)
- team_id (int)
- player_id (int)
- position (string)
- line_slot (string NULL) values like 'L1','L2','L3','L4','D1','D2','D3'
- pp_unit (int NULL) 1,2
- pk_unit (int NULL) 1,2
- projected (bool) whether player is projected vs confirmed
- status (string) e.g. 'active','scratch','IR'
- source (string) (StatsAPI, manual, other)
- version (int) increments on same-date updates
- created_at (timestamp)

PK: (snapshot_date, team_id, player_id, version)

### 3. `usage_aggregates`
Rolling / seasonal per-player rates to accelerate model features.

Columns:
- player_id (int)
- thru_date (date) (inclusive date for which stats up to previous game are aggregated)
- games_played (int)
- toi_avg (float)
- toi_pp_avg (float)
- toi_sh_avg (float)
- shots_per_60 (float)
- shot_attempts_per_60 (float)
- goals_per_60 (float)
- assists_per_60 (float)
- points_per_60 (float)
- saves_per_60 (float, if goalie)
- shots_against_per_60 (float)
- onice_cf_pct (float NULL) (team share advanced) optional
- recent_window (int) (e.g. 10)
- recent_shots_per_60 (float)
- recent_goals_per_60 (float)
- recent_saves_per_60 (float)
- created_at (timestamp)

PK: (player_id, thru_date)

### 4. `player_props_lines`
Historical and current offered prop lines & prices by sportsbook.

Columns:
- line_date (date) (date the line first seen)
- game_pk (int)
- player_id (int)
- market (string) e.g. 'SOG','GOALS','ASSISTS','POINTS','SAVES'
- line (float) e.g. 2.5
- over_price (int) American odds
- under_price (int)
- sportsbook (string) e.g. 'Bovada'
- first_seen_at (timestamp)
- last_seen_at (timestamp)
- is_current (bool)

Unique index (game_pk, player_id, market, line, sportsbook, first_seen_at)
Partial index for current active lines (is_current=1).

### 5. `player_prop_predictions`
Model outputs snapshot for auditing edges.

Columns:
- run_id (string UUID or date-time batch id)
- game_pk (int)
- player_id (int)
- market (string)
- line (float)
- proj_mean (float)
- dist_param2 (float NULL) (e.g. k for NB) 
- p_over (float)
- p_under (float)
- fair_odds_over (int)
- fair_odds_under (int)
- edge_over_pct (float)
- edge_under_pct (float)
- model_version (string)
- created_at (timestamp)

PK: (run_id, game_pk, player_id, market, line)

### 6. `game_metadata` (optional future)
- game_pk, date, venue, home_team_id, away_team_id, timezone, start_time_utc, status

## Derivation & ETL Notes
- `player_game_logs`: derive from NHL Stats API live feed or boxscore after game final; partial stats optional mid-game for live edges (separate staging table if needed).
- `roster_snapshots`: daily pre-cutoff pull of team rosters (Stats API `teams/{id}/roster`) + heuristic line assignment (TOI-based from trailing N games) if no explicit line data.
- `usage_aggregates`: built incrementally each day from prior aggregates + yesterday's games; avoid full recompute.
- `player_props_lines`: ingest bookmaker APIs periodically; de-duplicate by (market,line,price tuple). When price update occurs, close prior row (update last_seen_at, set is_current=0) and insert new current row.
- `player_prop_predictions`: generated inside daily update and optionally intra-day refresh. Keep run_id for reproducibility.

## Minimal Backfill Order
1. Backfill `player_game_logs` (two seasons)
2. Derive initial `usage_aggregates`
3. Begin capturing `roster_snapshots` going forward (historical optional phase)
4. Start ingesting live `player_props_lines`
5. Generate `player_prop_predictions` once models ready

## Index & Performance Considerations
- Frequent query patterns: current lines for today's games (filter by date, is_current=1). Add composite index (market, line_date, is_current).
- Aggregation updates: maintain materialized recent rate windows by incremental delta rather than full historical scan.

## Storage Format
Initial implementation may use Parquet files per table under `data/props/` with partitioning by date (e.g., `player_game_logs/date=YYYY-MM-DD/*.parquet`). Option to migrate to SQLite/Postgres later without schema change.

## Next Steps
- Validate column naming with existing models in `nhl_betting/models/props.py` (e.g., shots vs goals field names align).
- Implement extraction module stubs: `nhl_betting/data/player_game_logs.py`, `nhl_betting/data/rosters.py`.
- Add feature computation builder to populate `usage_aggregates`.
