# Props Recommendations Hardening: Next Steps

This document outlines follow‑up improvements to make the props recommendations + photo/logo enrichment pipeline more robust, observable, and low‑maintenance.

## 1. Player Identity & Photo Robustness

Goal: Maintain near‑100% headshot coverage with minimal runtime work.

Planned actions:
1. Persistent player_id cache
   - File: `data/models/player_id_cache.json` (name_normalized -> { player_id, last_seen, sources }).
   - Populate during props collection & historical stats ingest.
   - On each request, consult cache first (O(1) lookup, skip ad‑hoc enrichment passes).
   - Scheduled refresh job (weekly) to prune entries not seen in N days.
2. Name mismatch report
   - Generate `data/processed/player_id_mismatches_<date>.csv` listing rows where player_id is null or multiple candidates found.
   - Columns: source, raw_name, normalized_name, candidate_ids, chosen_id (nullable), reason.
   - Surface link on debug JSON: `mismatches_csv`.
3. Manual overrides layer
   - Add optional `data/models/player_id_overrides.json` (normalized_name -> player_id).
   - Apply early (before algorithmic matching) so manual corrections win.
4. Silhouette fallback audit
   - Track silhouettes served per day; alert if proportion > threshold (e.g. >5%).
   - Simple CSV append: `date, total_players, photos_remote, silhouettes`.

## 2. Non‑Player Row Filtering

Some feeds include aggregate/stat rows (e.g., "Team Totals", "Total Shots On Goal").

Actions:
1. Centralize a `is_non_player_row(name: str) -> bool` helper (regex + keyword list) in `player_props.py`.
2. Apply during canonical line ingestion so these rows never appear downstream.
3. Add unit test covering examples (positive + negative cases).

## 3. Debug & Verification Tooling

Enhancements:
1. `/props/recommendations?debug=2` already returns JSON metrics. Add fields:
   - `player_count`, `with_photo`, `with_logo`, `silhouette_pct`.
2. Add `/props/recommendations?debug=3` to stream (first 25) players with missing photos + reason (no id, 404 headshot, etc.).
3. Script `scripts/verify_props_assets.py`:
   - Fetch debug=2 JSON; assert thresholds (photos_remote >= X, silhouette_pct <= Y).
   - Exit non‑zero on failure for CI.
4. Include this script in a lightweight GitHub Action (future) or local pre‑deploy checklist.

## 4. Performance & Startup

Because the app now only serves precomputed artifacts:
1. Defer large CSV loads until first access (lazy module‑level caches with TTL).
2. Add simple in‑process LRU for parsed recommendations (keyed by date). Invalidation: file mtime change.
3. Guard against accidental heavy recompute by ensuring all training / projection CLI code paths are NOT imported in `web.app` (only lightweight helpers).

## 5. Logging & Observability

Add structured log entries for every recommendations request:
```
{"event":"props_recommendations_render","date":"2025-10-14","player_count":500,"photos_remote":472,"silhouettes":28,"render_ms":123}
```
Implementation details:
1. Timing decorator or context manager around route body.
2. Write to stdout (picked up by platform logs) — avoid external dependencies.
3. Add a rolling daily summary file `data/processed/props_asset_summary_<date>.json` (optional future step).

## 6. Testing Additions

New tests (outline):
1. `test_player_name_normalization_variants` – covers edge cases (initials, dict‑like strings, accents removed, punctuation, Jr./III variants).
2. `test_non_player_filter` – ensures aggregation/statistic strings are excluded.
3. `test_recommendations_debug_json` – uses TestClient to call debug=2 and assert required keys and invariants (photos_remote + silhouettes sum = player_count).
4. `test_photo_url_format` – verifies URL conforms to BAM pattern for a sample of player_ids.

## 7. Manual Ops Playbook (Runbook)

Add a runbook section (README update) describing:
1. Collect lines: `python -m nhl_betting.cli props-collect --date YYYY-MM-DD`.
2. Generate recommendations: `python -m nhl_betting.cli props-recommendations --date YYYY-MM-DD`.
3. Start dev server: `powershell -File dev_server.ps1`.
4. Verify assets: `python scripts/verify_props_assets.py --date YYYY-MM-DD` (future).
5. Investigate missing photos via `/props/recommendations?date=YYYY-MM-DD&debug=3`.

## 8. Future (Optional) Enhancements

Ideas (evaluate after core hardening):
1. Cache headshot HTTP status (avoid repeated 404 fetch attempts) with a small on-disk map (player_id -> ok|missing + last_checked_date).
2. Pre-generate a static JSON mapping: `player_assets.json` (id -> {photo_url, team_logo_url, last_seen}).
3. Add lightweight CDN or image proxy (only if external rate limits or latency become an issue).
4. Partial hydration front-end: deliver JSON + client-side templating for faster perceived load (only if template rendering becomes bottleneck).

## 9. Acceptance Criteria Checklist

- [ ] Headshot coverage >= 95% for active props players.
- [ ] Silhouettes proportion < 5% on a standard slate.
- [ ] Debug JSON includes: player_count, photos_remote, silhouettes, logos, silhouette_pct.
- [ ] CI (or local) verification script enforces thresholds.
- [ ] Non-player rows absent from recommendations CSV.
- [ ] Manual override file recognized in pipeline (test included).

---
This plan can be implemented incrementally; prioritize sections 1–3 first to lock in data quality and observability, then proceed to performance and tests.
