# Props uniformity monitoring and NN cascade

This doc summarizes the end-to-end setup to prevent uniform projections, prefer neural models, and apply per-player reconciliation bias.

## Pipeline

- Projections (per date): `python -m nhl_betting.cli props-project-all --date YYYY-MM-DD`
  - Cascade: NN ➜ traditional (EWMA) ➜ conservative fallback
  - Source tagging per row: `source` in {`nn`, `trad`, `fallback`} (+`+bias` if applied)
  - Bias merge: auto-applies `data/processed/player_props_bias_YYYY-MM-DD.csv` when present
  - Summary includes counts by source and dispersion stats per market.

- Bias (optional): `python scripts/build_player_props_bias.py --date YYYY-MM-DD`
  - EWMA over recent vs long-term, shrunk and clipped [0.6, 1.4]
  - Output: `data/processed/player_props_bias_YYYY-MM-DD.csv`

- Recommendations: `python -m nhl_betting.cli props-recommendations --date YYYY-MM-DD`
  - Uses `props_projections_all_DATE.csv` preferentially for lambdas
  - Computes p_over/EV and writes `props_recommendations_DATE.csv`

## Neural models

- Train single market: `python -m nhl_betting.scripts.train_nn_props train --market SOG --epochs 40`
- Train all markets: `python -m nhl_betting.scripts.train_nn_props train_all --epochs 40`
- Models/metadata live in `data/models/nn_props/` and are loaded via ONNX/QNN when available.

Robustness in `NNPropsModel`:
- Name normalization and abbreviation recovery (e.g., "A. Lastname")
- Last-name matching fallback with optional team preference
- Role inference from `role` or `position`, and fallback if role filter over-prunes

## Uniformity checks

- Quick analyzer: `python scripts/analyze_props_uniformity.py --date YYYY-MM-DD`.
  - Prints source shares, lambda dispersion, top/bottom players, and EV spread.
  - Exits with code 2 if std or nn-share fall below thresholds (configurable).

## VS Code task

- `.vscode/tasks_recompute_recs.json` provides handy task: "Recompute recs (today & tomorrow)".
  - Runs `scripts/recompute_recs.py` for today/tomorrow with `FIRST10_BLEND=1`.

## Troubleshooting

- If counts show all `trad`, ensure the NN metadata and ONNX exist under `data/models/nn_props/` and that `player_game_stats.csv` has enough history.
- If name mismatches occur (abbreviations or accents), the new fuzzy matching and last-name fallback should recover most cases.
- If projections CSV has only a header, verify roster cache and lines ingestion for the date; projections pipeline now guards against writing empty files without data.
