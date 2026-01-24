# NHL Sim Engine (Baseline)

This package scaffolds a possession→period→game simulation engine to produce period/game outcomes and player box scores for props.

Current state:
- Period-level baseline using per-60 team rates (shots/goals) and multinomial allocation to players proportional to projected TOI.
- Placeholders for possession-level modeling and line-combo conditioning.

Key modules:
- `state.py`: `GameState`, `TeamState`, `PlayerState`, `Event`.
- `models.py`: `RateModels` and simple `TeamRates` / `PlayerRates`.
- `engine.py`: `GameSimulator`, `PeriodSimulator`, `PossessionSimulator`.

Roadmap:
- Integrate TOI/shift charts to derive line combos and co-TOI features.
- Hazard-based possession events conditioned on on-ice players, score effects, and special teams.
- Calibration weekly vs actual distributions at period/game and props.
