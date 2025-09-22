# NHL Betting Predictive Engine

A pragmatic, extensible engine to generate NHL betting edges: moneyline winners, totals (over/under), puck line, and derivative props (shots on goal, goalie saves/goals).

## What this includes
- Data fetchers for NHL Stats API (public, free) to pull schedules, results, teams, and basic player stats.
- Feature engineering for rolling team strength, rest/back-to-back, and home/away effects.
- Baseline models:
  - Elo-style team ratings updated each game
  - Poisson goals model with home/away adjustments to derive totals and puck line probabilities
  - Logistic moneyline from Elo difference
- CLI to fetch, train, and predict edges for a date or slate.

This is a starting point built for iteration. It won’t beat the books out-of-the-box; you’ll improve it by adding xG, goalie adjustments, lines/injuries, travel, and market-aware calibration.

## Quickstart

1. Create a virtual environment and install deps

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run a smoke test

```
python -m nhl_betting.cli smoke
```

3. Fetch recent games and train

```
python -m nhl_betting.cli fetch --season 2023
python -m nhl_betting.cli train
python -m nhl_betting.cli predict --date 2024-10-01
```

Outputs will print to console and save to `data/`.

## Roadmap ideas
- Expected goals (xG) via shot location data; team attacking/defensive strengths via Dixon-Coles.
- Goalie model using rolling save%, quality starts, and confirmed starters.
- Player props using Poisson/negative binomial with usage and linemates.
- Market integration (close odds, vig removal) and calibration (Platt/isotonic).
- Ensemble models and Bayesian updating.

## Disclaimer
Gambling involves risk. This repository is for educational purposes only and offers no guarantees of profitability. Use responsibly.

## Deploy to Render

This repo is configured to run on Render as a Python web service.

Included files:
- `render.yaml` (Render Blueprint): installs dependencies and starts `uvicorn` on `$PORT`.
- `Procfile`: process declaration for Render/Dokku-style runtimes.
- `runtime.txt`: pins Python version.

Steps:
1. Create a new Web Service from this repo on Render.
2. Set environment variable `ODDS_API_KEY` (The Odds API key) in Render.
3. Deploy. On first startup, the app bootstraps models in the background (fetch ~two seasons; train Elo). Pages may be partially populated until this completes.
4. Visit the service URL. The `/` page will generate predictions for the selected date and fetch odds (Bovada first, then The Odds API fallback).

Notes:
- The app automatically falls back to The Odds API if Bovada odds aren’t available.
- Use the header’s “Refresh Odds” to force an odds refresh for the selected date.

## Local development

1. Python 3.11+. Create a venv and install deps:
  - `pip install -r requirements.txt`
2. Add `.env` with `ODDS_API_KEY=...` (for The Odds API fallback).
3. Run the server:
  - `uvicorn nhl_betting.web.app:app --host 127.0.0.1 --port 8010 --reload`
4. Open `http://127.0.0.1:8010`.

CLI examples:
- Train from existing data: `python -m nhl_betting.cli train`
- Predict using Bovada: `python -m nhl_betting.cli predict --date 2025-10-07 --odds-source bovada`
- Odds API snapshot: `python -m nhl_betting.cli predict --date 2025-10-07 --odds-source oddsapi --snapshot 2025-10-07T20:00:00Z --odds-best`
