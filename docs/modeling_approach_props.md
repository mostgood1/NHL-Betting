# Modeling Approach for Player Props (Draft)

## Objectives
Produce calibrated predictive distributions for SOG, Goals, Saves (phase 1) and later Assists/Points, generating probabilities over bookmaker lines and fair odds.

## Target Distributions & Rationale
- SOG: Poisson or Negative Binomial (NB) if overdispersion (var > mean) persists after feature conditioning. Start Poisson baseline; test NB via dispersion statistic.
- Goals: Zero-inflated Poisson (ZIP) or Poisson with shooting-percentage feature; typically low counts; start Poisson with player-level random effect.
- Saves: Poisson / NB depending on variance; correlation with projected shots against. Use conditional mean from opponent shot projection.
- Assists: Poisson or ZIP (lower frequency per game for some players); consider share of on-ice goals.
- Points: Compound (Goals + Assists) or direct NB; initial approach = independent Poisson goals + assists -> convolution; simplify by modeling points directly once features robust.

## Hierarchical Structure
- Random intercept per player (partial pooling) for base rate.
- Optional team-level intercept for systematic style differences.
- For NB: shared dispersion parameter (global) or hierarchical shrinkage toward league value.

## Feature Integration
- Log-link: log(lambda) = Xβ + player_re + team_re.
- For SOG NB: log(lambda) = offset(log(proj_toi/average_toi)) + coefficients * features + random effects.

## Fitting Strategy
Phase 1 (Fast Iteration):
- Use statsmodels GLM Poisson for baseline; store coefficients.
- Evaluate dispersion: (Pearson chi-square / df). If >1.25, move to NB.
Phase 2 (Regularized):
- L2 (ridge) via scikit-learn PoissonRegressor for stability when adding many features.
Phase 3 (Hierarchical):
- PyMC or pyro Bayesian model for partial pooling (optional; consider performance/time tradeoff). Cache posterior means.

## Calibration & Evaluation Metrics
- RMSE / MAE on counts.
- Log-likelihood / Deviance.
- Brier score for binary over chosen book lines.
- Probability Integral Transform (PIT) histogram for calibration.
- Coverage of central prediction intervals (e.g., 80%).

## Edge Computation
Given predicted PMF p(k), probability over line L.5 = sum_{k>L} p(k). Fair decimal odds = 1 / P(over). American odds conversion standard. Edge% = (implied_prob_book - model_prob) / model_prob for side underpriced (flip sign for under bet evaluation).

## Handling Missing Features
- Impute proj_toi with roster heuristic or league average by position if missing.
- For players with <3 games history: blend league mean via prior weight: lambda = w * player_rate + (1-w) * league_mean; w = games/(games+k0), choose k0 ~ 8.

## Outlier & Robustness Checks
- Cap extreme feature values at reasonable percentiles (e.g., 2.5 / 97.5) to reduce undue influence.
- Monitor players returning from injury (long absence) -> reset recent windows weighting.

## Model Artifacts
- Store per-market: model_version, fit_date, coefficient vector (JSON), training sample size, dispersion metric.
- Save to `data/props/models/{market}/model_meta.json` and coefficients parquet.

## Inference Pipeline
1. Build feature frame for today.
2. Load market model artifact (if absent, fallback simple rolling mean lambda).
3. Compute lambda (and k if NB) per player.
4. Produce PMF up to max_k (SOG: 15, Goals: 5, Saves: 70).
5. For each offered line: compute p_over, p_under, fair prices, edge.

## Refit Frequency
- Daily incremental refit (rolling window last 400 days) or expanding window with decay.
- Trigger full refit weekly or when dispersion shift threshold exceeded.

## Roadmap Extensions
- Incorporate bookmaker line as a feature (for assists/points) to exploit anchoring but guard vs leakage by training on historical lines only.
- Add time-of-day or travel fatigue features (back-to-back road specifics).
- Joint modeling of correlated skater linemates (multivariate hierarchical) if marginal gains justify complexity.

## Next Implementation Steps
1. Implement baseline Poisson trainer for SOG: `train_sog_model(date: str)`.
2. Add evaluation script computing dispersion & calibration plots (text summary first).
3. Integrate fallback to rolling mean if model artifact missing or insufficient data.
