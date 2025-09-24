# Feature Engineering Design for Player Props (Draft)

## Overview
Feature sets target distributional modeling for SOG, Goals, Saves first; extensible to Assists & Points. Focus on interpretable rate-based predictors with hierarchical shrinkage potential.

## Shared Base Features (All Skater Markets)
- date
- player_id
- team_id
- opponent_team_id
- is_home (binary)
- rest_days (int)
- b2b (binary) (from team schedule features)
- proj_toi (minutes)
- proj_toi_pp (minutes)
- proj_toi_sh (minutes)
- line_slot_encoded (ordinal: L1=4, L2=3, L3=2, L4=1, else 0)
- pp_unit_onehot (PP1, PP2)
- pk_unit_flag
- rolling_games_played_10

## Rate & Form Features
(Compute per player up to previous game, exclude current)
- shots_per_60_recent_5, shots_per_60_recent_10
- goals_per_60_recent_5, goals_per_60_recent_10
- assists_per_60_recent_5, points_per_60_recent_5
- pp_shots_share_recent_10 (player PP shots / team PP shots)
- pp_goals_share_recent_10
- sh_toi_share_recent_10
- offensive_zone_start_pct (future optional)

## Opponent Suppression Features
(Team-level allowed rates last N games)
- opp_shots_allowed_per_60_recent_10
- opp_goals_allowed_per_60_recent_10
- opp_shots_blocked_rate_recent_10
- opp_pk_time_per_game (discipline) -> influences PP TOI expectation
- opp_save_pct_recent_10 (for goals model context)

## Team Context Features
- team_shots_per_60_recent_10
- team_goals_per_60_recent_10
- team_pp_opportunities_per_game_recent_10
- team_pp_conversion_pct_recent_10

## SOG-Specific Features
- individual_shot_attempts_share_recent_10 (player shot attempts / team attempts)
- last_game_shots (recency spike)
- std_dev_shots_recent_10 (volatility) -> may relate to over probability distribution tail
- cumulative_shots_season_per_60 (season baseline)

## Goals-Specific Features
- shooting_pct_recent_25 (player goals / shots) with shrinkage toward league mean
- xSOG_proxy (shots_per_60 * shooting_pct_recent_25)
- pp_shooting_pct_recent_25
- rebound_goals_share (optional if data available)

## Saves (Goalies)
Base features adapt:
- goalie_proj_toi (usually 60)
- shots_against_per_60_team_def_allowed_recent_10 (opponent offense strength)
- opponent_shots_for_per_60_recent_10
- goalie_saves_per_60_recent_10
- goalie_shots_faced_per_60_recent_10
- goalie_save_pct_recent_10 (stability)
- team_def_blocks_per_60_recent_10 (negative correlation with shots reaching net)

## Assists / Points (Later Phase)
- primary_assist_rate_recent_10
- secondary_assist_rate_recent_10
- linemate_goals_rate_recent_10
- onice_shot_share (player on-ice shots / team shots when on ice)

## Categorical Encodings
- line_slot -> ordinal + one-hot
- position group (F vs D vs G)
- opponent_team_id (use target encoding or leave numeric ID for tree models)

## Interaction Ideas (later optimization)
- proj_toi * shots_per_60_recent_10 (expected shots baseline)
- pp_unit_onehot * team_pp_opportunities_per_game_recent_10 (PP usage interaction)
- goalie_save_pct_recent_10 * opponent_shots_for_per_60_recent_10 (expected saves calibration)

## Leakage Avoidance
- All rolling stats computed excluding current game (shift by 1).
- Use only finalized games before date cutoff.

## Data Dependencies
- player_game_logs (for all per-player rates)
- roster_snapshots (for line_slot, proj_toi)
- usage_aggregates (can pre-compute some rate windows)
- team game-level aggregate stats (shots for/against) - may extend existing team features module or add new.

## Computation Strategy
1. Ingest player_game_logs subset up to date D-1.
2. Compute rolling windows (5, 10, 25) with minimum games threshold; apply shrinkage: blended = w * player_rate + (1-w) * league_mean where w = games / (games + k).
3. Merge opponent suppression features from team-level allowed stats table (build if missing).
4. Merge roster snapshot for projection features (proj_toi etc.).
5. Output feature frame per (game_pk, player_id, market) needed (initially just players scheduled that day producing at least minimal expected TOI).

## Output Schema (Initial SOG Model)
- player_id
- game_pk
- date
- feature columns (prefixed e.g. f_*)
- target (shots) for training rows (historical) / None for inference rows

## Versioning & Reproducibility
- Add `feature_version` string constant in code; bump if logic changes.
- Cache intermediate rolling aggregates to avoid recomputation for each market.

## Next Steps
- Implement `nhl_betting/features/props_features.py` with builder functions:
  - `build_skater_base_features(date: str) -> pd.DataFrame`
  - `build_sog_features(date: str)` (extends base)
  - Shared helpers for rolling rates & shrinkage.
