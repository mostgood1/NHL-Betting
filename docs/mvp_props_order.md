# MVP Implementation Order & Acceptance Criteria (Draft)

## Phase 0 (Completed Design Artifacts)
- Scope, schema, roster plan, historical sourcing strategy, feature design, modeling approach, pipeline architecture.

## Phase 1: Data Foundations
1. Roster snapshots operational
   - Command creates `data/props/roster_snapshots/date=YYYY-MM-DD/*.parquet`.
   - Acceptance: >90% of likely skaters for day have non-null proj_toi.
2. Player props daily snapshot (Bovada) stored
   - Acceptance: File exists even if empty; schema columns validated.
3. Player game logs module (backfill + incremental) (TBD implementation)
   - Acceptance: Two recent finalized slates ingested with consistent counts vs boxscore totals.

## Phase 2: Feature & Baseline Model (SOG)
4. Feature builder for SOG
   - Acceptance: DataFrame with required base + SOG-specific features for today's players; no leakage (all features use data <= D-1).
5. Baseline Poisson SOG model training
   - Acceptance: Trains without error, saves artifact with coefficients & dispersion metric.
6. Inference producing predictions & probabilities over offered lines
   - Acceptance: Predictions parquet file with p_over, fair odds for >80% of offered SOG lines.

## Phase 3: Additional Markets
7. Saves model (reuse feature pipeline with goalie-specific features)
   - Acceptance: Predictions for starting goalies >70% coverage.
8. Goals model (Poisson baseline)
   - Acceptance: p_over for most common goal lines (0.5, 1.5) generated.

## Phase 4: Integration & Edges
9. Integrate props predictions into daily_update with `--with-props`
   - Acceptance: Running daily update with flag outputs predictions + edges without failing base workflow.
10. Edge calculation & export
    - Acceptance: Each prediction row has edge_over_pct/edge_under_pct when corresponding book prices available.

## Phase 5: Quality & Refinement
11. Overdispersion detection & optional NB switch
    - Acceptance: Automatic switch logs reason when dispersion > threshold.
12. Calibration reporting
    - Acceptance: Text summary of Brier score & PIT coverage printed during model training.

## Phase 6: Optional Enhancements
13. Intra-day line movement collection (multiple snapshots)
14. Bayesian hierarchical upgrade for SOG
15. Assists & Points models

## Overall Acceptance Criteria for MVP (Phases 1-2)
- End-to-end SOG predictions for current date with edges in <60s run time.
- Model dispersion within acceptable range or NB fallback engaged.
- At least one week of archived SOG lines accumulating automatically.

## Risk Mitigation
- Missing historical props: fallback to game log priors.
- Incomplete roster: set conservative proj_toi for unknown players (e.g., 8 min skater, 0 goalie not starter).

## Next Action Proposal
Implement CLI entry / stub functions for roster snapshot & props collection integration into `daily_update`.
