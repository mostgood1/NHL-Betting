# Nolines Props Gates and Weekly Auto-Tune

This project produces player props recommendations both from EV-based simulations and from "nolines" simulations for markets where odds/lines may be missing (currently `SAVES` and `BLOCKS`).

## Current Gates
- SOG: 0.75 (probability gate)
- GOALS / ASSISTS / POINTS: 0.60
- SAVES: 0.65 (auto-tuned weekly within 0.65–0.68)
- BLOCKS: 0.92

## Weekly Auto-Tune for SAVES
- Trigger: Each Monday, the daily script runs a 7-day nolines monitor and adjusts the `SAVES` gate.
- Source: `data/processed/props_nolines_monitor.json` (computed via `props-nolines-monitor`).
- Heuristic:
  - If accuracy < 0.88 or Brier > 0.16 → `SAVES` gate = 0.68
  - Else if accuracy < 0.90 or Brier > 0.15 → gate = 0.67
  - Else if accuracy < 0.92 or Brier > 0.14 → gate = 0.66
  - Else → gate = 0.65
- Range: Constrained to 0.65–0.68 to preserve pick volume and consistency.

## Outputs
- Nolines simulations: `data/processed/props_simulations_nolines_{YYYY-MM-DD}.csv`
- Nolines recommendations: `data/processed/props_recommendations_nolines_{YYYY-MM-DD}.csv`
- Combined recommendations: `data/processed/props_recommendations_combined_{YYYY-MM-DD}.{csv,json}`
- Monitor: `data/processed/props_nolines_monitor.json`

## How to Run Manually (PowerShell)
```powershell
. .\activate_npu.ps1; . .\.venv\Scripts\Activate.ps1
# 7-day monitor with current gates
python -m nhl_betting.cli props-nolines-monitor --window-days 7 --markets 'SAVES,BLOCKS' --min-prob-per-market "SAVES=0.65,BLOCKS=0.92"
# Today’s nolines and combined recommendations
python -m nhl_betting.cli props-recommendations-nolines --date $(Get-Date -Format yyyy-MM-dd) --markets 'SAVES,BLOCKS' --top 400 --min-prob-per-market "SAVES=0.65,BLOCKS=0.92"
python -m nhl_betting.cli props-recommendations-combined --date $(Get-Date -Format yyyy-MM-dd)
```