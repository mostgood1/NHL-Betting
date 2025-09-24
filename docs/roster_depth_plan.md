# Roster & Depth Acquisition Plan (Draft)

## Objectives
1. Daily capture of active NHL rosters with reliable player IDs.
2. Infer likely even-strength lines, power play (PP) units, and penalty kill (PK) units using recent TOI and shift distribution.
3. Provide projections of time on ice components (overall, PP, SH) per player for prop models.

## Data Sources (NHL Stats API)
- Teams list: `/api/v1/teams`
- Team roster (current): `/api/v1/teams/{teamId}?expand=team.roster`
- Team schedule for upcoming games: `/api/v1/schedule?teamId={teamId}&date={YYYY-MM-DD}` (for validation of game presence)
- Live feed (optional for lines later): `/api/v1/game/{gamePk}/feed/live` (contains shift events and TOI post-game)
- Boxscore: `/api/v1/game/{gamePk}/boxscore` (provides skater and goalie TOI breakdowns, PP/SH)

## Core Steps (Daily Pre-Game Window)
1. Enumerate teams.
2. Pull current roster for each team with player IDs & primary position.
3. Retrieve last N (e.g., 10) finalized games per team to compute usage metrics:
   - Even-strength TOI / game
   - PP TOI / game
   - SH TOI / game
4. Rank forwards and defensemen separately by recent EV TOI to assign line slots heuristically:
   - Forwards: group into lines of 3 in order (L1, L2, L3, L4). If player has fewer than M games recently, mark line_slot NULL (uncertain).
   - Defense: pair into D1, D2, D3 (top 6) by EV TOI; extras become depth (line_slot NULL or 'D7').
5. PP Units: rank skaters by PP TOI; assign top 5 to PP1, next 5 to PP2; tie-break by overall TOI or points.
6. PK Units: rank by SH TOI similarly (top 4 for PK1, next 4 PK2).
7. Goalies: Determine likely starter via last 3 games pattern + rest heuristic (starter candidate has higher recent share and 1+ rest days). Mark projected=true for probable starter.
8. Produce a roster snapshot record for each player with:
   - line_slot
   - pp_unit
   - pk_unit
   - projected flag (True until confirmed)

## Heuristics & Edge Cases
- Injured/Scratch Detection: If player on roster but zero TOI last 5 games while available, mark status='scratch?/reserve'. (Future: integrate injury feed if available.)
- Small Sample Thresholds: If games_played_recent < 3, set projected flag but leave line_slot None to avoid false certainty.
- Trade / New Call-Ups: No prior data → assign provisional line_slot after sorting known players; unknowns placed on bottom line or extras until 1 game recorded.

## Time On Ice Projection Method
Projected TOI per player = blend of:
- Recent EV TOI rolling mean (weight 0.7) + season EV mean (0.3)
- Adjust for back-to-back (reduce top forwards/defense by 5–8%)
- Goalie starter: if confirmed, projected TOI = 60; otherwise 0 or partial (if tandem expectation ~50/50 preseason) set 30 as placeholder.
PP TOI share within team = player's recent PP TOI / sum of all PP TOI among top 8 PP candidates (capped) * projected team PP minutes (baseline 5.5 per game, adjustable by opponent penalty rate).

## Data Outputs (Intermediate DataFrame Columns)
- player_id
- team_id
- date
- position
- line_slot
- pp_unit
- pk_unit
- proj_toi
- proj_toi_pp
- proj_toi_sh
- starter_goalie (bool)
- projected (bool)
- status
- source='heuristic'

## Implementation Skeleton
Module: `nhl_betting/data/rosters.py`

Functions:
- `fetch_current_roster(team_id: int) -> List[Dict]`
- `recent_team_games(team_id: int, n: int) -> List[int]` (returns gamePks)
- `build_usage_frame(team_id: int, game_pks: List[int]) -> pd.DataFrame` (rows per player with aggregated TOI splits)
- `infer_lines(usage_df: pd.DataFrame) -> pd.DataFrame` (adds line_slot, pp_unit, pk_unit)
- `project_toi(usage_df: pd.DataFrame, lines_df: pd.DataFrame, context: Dict) -> pd.DataFrame`
- `build_roster_snapshot(date: str) -> pd.DataFrame` (loops teams, concatenates)

CLI/Typer integration later: `props-roster-snapshot --date YYYY-MM-DD` writing Parquet under `data/props/roster_snapshots/date=.../`.

## Validation Metrics
- Coverage: % of active roster players with non-null line_slot (target >85% regular season)
- PP Assignment Accuracy (later when confirmations exist): compare heuristic vs actual PP TOI share top 5.
- Goalie Starter Precision (vs actual starter minutes threshold > 45): target >70% early, improve iteratively.

## Future Enhancements
- Integrate confirmed line sources if available.
- Use shift-level data to refine pairings mid-season.
- Add injury/reserve feed ingestion to set status explicitly.
