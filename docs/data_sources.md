# Data Sources Inventory (NHL Betting)

This document lists primary data sources to support roster tracking, lineups/injuries, TOI/shift charts, simulation inputs, and props.

Reliable, public sources:

- NHL Web API (api-web.nhle.com/v1)
  - Schedule: `/schedule/{YYYY-MM-DD}` (games, start times, current states)
  - Boxscore: `/gamecenter/{gamePk}/boxscore` (players, stats)
  - Play-by-play: `/gamecenter/{gamePk}/play-by-play` (events)
  - Linescore: `/gamecenter/{gamePk}/linescore` (period/clock)
  - Teams: `/teams` or `/standings/now` for active teams
  - Rosters: `/roster/{TEAM}/current` (current active roster lists)
  - Pros: Canonical, stable, good coverage
  - Cons: No official injury feed; roster attributes vary; shifts not exposed here

- NHL Stats API (api.nhle.com/stats/rest)
  - Shift charts: `/en/shiftcharts?cayenneExp=gameId={gameId}` (player shifts with start/end times)
  - Other endpoints include skater/goalie summaries, but shiftcharts is key for co-TOI and on-ice detection.
  - Pros: True per-shift intervals for accurate co-TOI; open access
  - Cons: Occasional downtime; schema changes possible

- MoneyPuck
  - Skater/Team xG rates; used for team-level offensive pace and calibration.
  - Pros: Public CSVs; rich metrics
  - Cons: Update cadence; attribution differences

- OddsAPI (provider feed)
  - Book lines for ML/PL/Totals and player props; canonical lines used across the project.
  - Pros: Consistent schema; provider-specific metadata
  - Cons: Rate limits; provider coverage varies

- Team/League media (injuries)
  - No single official API. Injuries typically posted via team PR/beat writers; some aggregation sites require licensing (e.g., Rotowire).
  - Current approach: baseline manual overrides + inference from scratches and roster changes; extendable to licensed feeds if available.

Planned adapters & usage

- Roster & lineups:
  - Primary via NHL Web API `roster/{TEAM}/current` + TOI-based inference.
  - Augment via shiftcharts to estimate co-TOI and validate expected lines.

- Injuries:
  - Manual overrides (`injury-update --overrides-path`) + future adapter to licensed feeds.
  - Leverage scratches in boxscore for day-of status when available.

- Simulation:
  - Baseline period-level rates from historical goals-per-team (`config.json: base_mu`).
  - Integrate lineup-based rotations and simple score effects.
  - Future: hazard models conditioned on on-ice units, special teams, pulled goalie, and score/home/away effects.

Notes
- We avoid scraping restricted content; prefer official APIs and public CSVs.
- Where licensed or restricted sources are considered, code will be designed behind optional adapters.
