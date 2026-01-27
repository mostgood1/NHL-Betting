# Play-by-Play Hockey Simulation Engine (Design)

## Goals
- True play-by-play simulation that models lines/rotations and realistic event sequences
- Covers events: faceoffs, passes, zone entries, shots, saves, blocks, penalties, goals, stoppages
- Produces aggregated period totals, game totals, and derived props (SOG, GOALS, ASSISTS, POINTS, BLOCKS, SAVES)
- Supports game-to-props linkage and per-period projections

## Core Concepts
- Game clock: 3 periods × 20:00; optional OT
- Shifts (line rotations): forward lines (F1-F4) and defensive pairs (D1-D3)
- Goalie: per-team goalie with save ability
- Stoppages: faceoff events to restart play; shift changes happen on-the-fly and at stoppages

## State & Entities
- Team: roster (players + positions), lineup config (F lines, D pairs), team rates (pace, shot/share, penalty/PK/PP)
- Player: per-player skills (shot rate, pass rate, block rate, xG on shot, assist chance, penalty draw/take)
- GameState: time, score, possession, manpower (5v5, PP, PK, EN), current lines (F/D), goalie

## Event Model
- Faceoff: determines initial possession; weighted by center FO skill
- Pass: advances time; swap possession based on turnover probability
- Zone Entry: controlled entries increase shot quality (xG)
- Shot: can be blocked, saved, or goal; block chance depends on defenders; save chance depends on goalie
- Penalty: creates PP/PK states; PP increases shot rates and xG
- Goal: updates score, triggers stoppage and faceoff
- Stoppage: faceoff; potential shift change

## Time Advance
- Discrete-event simulation: sample next event type with intensity/weights given current state
- Draw event times using exponential clocks (Poisson for shot attempts, penalties, etc.) with context modifiers (PP/PK, lines)

## Lines & Shifts
- Initialize lines from `roster_snapshot_{date}.csv` and optional `lineups_co_toi_{date}.csv`
- Shift algorithm: target TOI distribution per player/line; rotate lines at ~45-60s shifts, with flexible adjustments based on stoppages and fatigue

## Aggregation
- Track per-event contributions to players and teams
- Aggregate period totals: goals, shots (on goal), blocks, saves, assists, points, TOI
- Provide game-level aggregates and per-period breakdown

## Outputs
- Per-period tables: team totals
- Player box scores: points, shots, blocks, saves, TOI
- Props derivation: lambda estimates for markets from event counts and distributions

## Calibration & Inputs
- Use historical rates (team pace, penalty/min, PP conversion, shot rate per player) from processed datasets
- Sync with NN projections for initial priors; use sim to generate variability and correlations

## Extensibility
- Optional OT, empty-net behavior, goalie pulls
- Injuries/scratches; live updates from line combinations

## Implementation Plan
1. Build Simulator with `GameState`, `Event`, `LineRotation`
2. Implement 5v5 baseline events with shot/pass/block/save/goal chain
3. Add penalties and PP/PK states, with modifiers
4. Add line rotation scheduler (target TOI per player) and stoppage handling
5. Validate against daily artifacts; calibrate rates
6. Integrate outputs for per-period and player props

