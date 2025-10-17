# Render Periodic Updates - Odds & Recommendations Refresh

## Overview

**YES! The Render site DOES perform periodic checks** for updated odds and refreshes recommendations throughout the day.

## Cron Jobs Summary

### 1. **Hourly Props Recommendations Refresh** ⭐ MAIN UPDATER
**Schedule:** Every hour at minute 0 (`0 * * * *`)
**Endpoint:** `/api/cron/light-refresh`
**What it does:**
- ✅ Fetches fresh odds from Bovada
- ✅ Updates `predictions_{date}.csv` with new odds (skips started games)
- ✅ Recomputes team edges and recommendations
- ✅ **Regenerates props recommendations** from canonical lines + NN projections
- ✅ Writes updated `props_recommendations_{date}.csv`
- ✅ Upserts to GitHub for persistence

**Props Refresh Flow:**
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Canonical Lines (OddsAPI + Bovada)                  │
│    - data/props/player_props_lines/date={date}/*.parquet   │
│    - Lines from bookmakers with current Over/Under odds     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Load NN Projections (Precomputed)                        │
│    - data/processed/props_projections_all_{date}.csv       │
│    - Lambda values from neural network models               │
│    - NOT recomputed (uses morning's projections)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Vectorized EV Calculation (FAST!)                        │
│    - For each (player, market, line):                       │
│      * p_over = P(X > line | lambda) via Poisson            │
│      * EV_over = p_over × (decimal_odds - 1) - (1-p_over)  │
│      * EV_under = similar calculation                        │
│    - Choose better side (Over vs Under)                     │
│    - No history scans, no model inference needed            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Filter & Rank                                             │
│    - Keep only bets with EV >= min_ev (default 0.0)        │
│    - Sort by EV descending                                   │
│    - Take top N (default 200)                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Write & Persist                                           │
│    - Save to props_recommendations_{date}.csv               │
│    - Upsert to GitHub (automatic backup)                    │
│    - Available immediately on /props/recommendations        │
└─────────────────────────────────────────────────────────────┘
```

**Why This Works:**
- **Fast:** No model inference, just math on precomputed projections
- **Fresh:** Odds updated hourly as bookmakers adjust lines
- **Smart:** Skips games that already started
- **Resilient:** Falls back to GitHub if local files missing

### 2. **Games Recompute**
**Schedule:** Daily at 18:30 UTC (~2:30 PM ET) (`30 18 * * *`)
**What it does:**
- Fetches fresh Bovada odds for games
- Recomputes edges without re-running models
- Lighter refresh for team-level predictions

### 3. **Nightly Core**
**Schedule:** Daily at 08:00 UTC (~3-4 AM ET) (`0 8 * * *`)
**What it does:**
- Captures closing lines from yesterday
- Reconciles predictions vs actual results
- Reconciles props recommendations vs actual stats

### 4. **Keepalive Health**
**Schedule:** Every 5 minutes (`*/5 * * * *`)
**What it does:**
- Pings health endpoints to prevent cold starts
- Keeps server warm on free Render plan

## Key Implementation Details

### `_refresh_props_recommendations()` Function

**Location:** `nhl_betting/web/app.py` lines 5550-5690

**Core Logic:**
```python
def _refresh_props_recommendations(date: str, min_ev: float = 0.0, top: int = 200):
    # 1. Load canonical lines (parquet files)
    lines = pd.concat([
        pd.read_parquet("data/props/player_props_lines/date={date}/bovada.parquet"),
        pd.read_parquet("data/props/player_props_lines/date={date}/oddsapi.parquet"),
    ])
    
    # 2. Load NN projections (from morning's run)
    proj = pd.read_csv(f"props_projections_all_{date}.csv")
    lam_map = {(player.lower(), market.upper()): proj_lambda for ...}
    
    # 3. Merge and calculate probabilities (vectorized!)
    merged = lines.merge(lam_map, on=["player_norm", "market"], how="left")
    
    # For each market (SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS):
    p_over = poisson.sf(line, mu=proj_lambda)  # Survival function (P(X > line))
    
    # 4. Calculate EV for both sides
    ev_over = p_over × (decimal_odds_over - 1) - (1 - p_over)
    ev_under = (1 - p_over) × (decimal_odds_under - 1) - p_over
    
    # 5. Choose better side and filter
    chosen_side = "Over" if ev_over >= ev_under else "Under"
    chosen_ev = max(ev_over, ev_under)
    
    filtered = merged[merged["ev"] >= min_ev].sort_values("ev", ascending=False).head(top)
    
    # 6. Write and upsert
    filtered.to_csv(f"props_recommendations_{date}.csv", index=False)
    _gh_upsert_file_if_better_or_same(...)  # Push to GitHub
```

**Performance:**
- ⚡ **Very fast:** No model inference, pure math
- 📊 **Vectorized:** Uses NumPy/Pandas for speed
- 💾 **Cached:** Reads precomputed projections (not recomputing)
- 🔄 **Live odds:** Always using latest bookmaker lines

### `_inject_bovada_odds_into_predictions()` Function

**What it does:**
- Fetches current Bovada odds via API
- Updates `predictions_{date}.csv` with fresh moneyline, totals, spreads
- **Skips games that already started** (respects `skip_started=True`)
- Only updates odds fields, preserves model predictions

**Smart Filtering:**
```python
# Check game state from NHL API
if gameState in ["LIVE", "FINAL", "IN_PROGRESS"]:
    # Skip updating odds for this game
    continue
```

This prevents:
- ❌ Updating odds after puck drop (meaningless)
- ❌ Overwriting closing lines with stale data
- ❌ Confusing users with odds that can't be bet anymore

## What Gets Updated vs What Stays Static

### Updated Every Hour ✅
- **Odds for props** (Over/Under lines and prices)
- **Odds for games** (Moneyline, totals, spreads)
- **Expected value calculations** (based on fresh odds vs static projections)
- **Best side recommendations** (Over vs Under based on EV)
- **Ranked list of edges** (sorted by EV with latest odds)

### Stays Static (Until Next Morning) 📌
- **NN model projections** (props_projections_all_{date}.csv)
  - Generated once in the morning by daily updater
  - Not recomputed hourly (too expensive, unnecessary)
  - Projections are for the entire game, don't change mid-day
- **Team-level predictions** (predictions_{date}.csv model columns)
  - period1_home_proj, period1_away_proj, etc.
  - first_10min_proj
  - Model probabilities (p_over, p_under based on model)

## Why This Design?

### Efficient & Fast ⚡
- **Model inference is expensive** (requires PyTorch, NPU, 50+ ms per player)
- **Odds updates are cheap** (just API calls and math)
- **Hourly model runs would:**
  - Slow down the site
  - Waste compute resources
  - Not improve projections (player abilities don't change during the day)

### Projections vs Odds 🎯
- **Player projections:** Based on season-long patterns, don't change intraday
  - Connor McDavid's expected SOG is ~4.2 whether you check at 10 AM or 6 PM
- **Bookmaker odds:** Change constantly based on:
  - Betting volume (sharp money)
  - Line movements
  - Injury news
  - Public perception

**The sweet spot:** Update odds hourly, keep projections static → Fresh EV calculations without expensive recomputation!

### Real-World Example

**Morning (8 AM ET):**
```
[Daily Updater Runs]
→ NN projects Connor McDavid: SOG lambda = 4.2
→ Writes to props_projections_all_2025-10-17.csv
→ Initial odds: Over 3.5 SOG at -110

EV calculation:
P(Over 3.5) = 0.68 (from Poisson(4.2))
Decimal odds = 1.909
EV = 0.68 × (1.909 - 1) - 0.32 = 0.30 (30% edge!)
```

**Afternoon (2 PM ET):**
```
[Hourly Refresh Runs]
→ Same projection: SOG lambda = 4.2 (unchanged, from morning file)
→ Updated odds: Over 3.5 SOG at -130 (line moved!)

EV calculation:
P(Over 3.5) = 0.68 (same, from morning projection)
Decimal odds = 1.769 (odds got worse)
EV = 0.68 × (1.769 - 1) - 0.32 = 0.20 (20% edge, still good but worse)
```

**Late (6 PM ET):**
```
[Hourly Refresh Runs]
→ Same projection: SOG lambda = 4.2 (still unchanged)
→ Updated odds: Over 3.5 SOG at -150 (line moved more!)

EV calculation:
P(Over 3.5) = 0.68 (same)
Decimal odds = 1.667 (odds got even worse)
EV = 0.68 × (1.667 - 1) - 0.32 = 0.13 (13% edge, marginal now)
```

**Insight:** The projection stays at 4.2 SOG all day (correct!), but the edge shrinks as odds move against you. This is **exactly what you want** - tracking when the market catches up to your model.

## Render Cron Configuration

**File:** `render.yaml`

```yaml
# Hourly props refresh
- type: cron
  name: props-recs-hourly
  schedule: "0 * * * *"  # Top of every hour
  runtime: image
  image:
    url: docker.io/curlimages/curl:8.11.0
  dockerCommand: >
    sh -lc 'curl -fsS -H "Authorization: Bearer $REFRESH_CRON_TOKEN" 
    "https://nhl-betting.onrender.com/api/cron/light-refresh?min_ev=0&top=200" || true'
  envVars:
    - key: REFRESH_CRON_TOKEN
      sync: false
```

**What happens:**
1. Render's scheduler triggers at hour 0 (e.g., 1:00 PM, 2:00 PM, 3:00 PM, etc.)
2. Curl container starts and makes HTTP request to your web service
3. Web service receives request with auth token
4. `api_cron_light_refresh()` handler executes:
   - Fetches fresh Bovada odds
   - Updates predictions CSV (team bets)
   - **Regenerates props recommendations** with fresh odds
5. Updated files written to disk and pushed to GitHub
6. Users see updated recommendations on `/props/recommendations`

## Monitoring & Verification

### Check Last Update Time
```bash
# In web UI, look at any props recommendation card
# Shows "Last Updated: 2:05 PM ET" or similar timestamp

# Or via API
curl https://nhl-betting.onrender.com/api/last-updated?date=2025-10-17
```

### Check Cron Job Logs (Render Dashboard)
```
Navigate to: Render Dashboard → Crons → props-recs-hourly → Logs
Look for: "OK" responses with row counts
```

### Verify Odds Changes
```bash
# Compare odds at different times of day
# Morning: Check props recommendations
# Afternoon: Refresh page - odds should be different
# The proj_lambda stays the same, but EV changes as odds move
```

### Manual Trigger (for testing)
```bash
# With proper auth token
curl -H "Authorization: Bearer $TOKEN" \
  "https://nhl-betting.onrender.com/api/cron/light-refresh?min_ev=0&top=200"
```

## Limitations & Trade-offs

### ✅ Pros
- **Fast:** Hourly updates without expensive recomputation
- **Fresh odds:** Always showing latest bookmaker lines
- **Accurate EV:** Comparing static projections to moving odds (correct approach)
- **Efficient:** Low compute cost on Render

### ⚠️ Considerations
- **Projections don't update intraday:** If injury news breaks at 3 PM, projection won't reflect it until next morning
  - **Mitigation:** Most injury news happens before game day (morning updates catch it)
- **Late scratches:** Player scratched at 6:30 PM won't be removed from recommendations
  - **Mitigation:** User responsibility to check lineups before betting
- **Line movement meaning:** Odds moving might indicate sharp information you don't have
  - **Mitigation:** This is a feature! Shows when market disagrees with your model

## Best Practices for Users

### 1. **Check Recommendations Throughout the Day**
- Morning: See initial edges based on opening odds
- Afternoon: See which edges persist vs which odds moved
- Evening: Final check before game time

### 2. **Monitor Line Movement**
```
If a prop shows:
- Morning: Over 3.5 SOG at -110 (EV = 30%)
- Afternoon: Over 3.5 SOG at -140 (EV = 15%)

This means: Sharp money is hammering the Over, market catching up to your model.
Action: Either bet now before it gets worse, or pass (market might know something).
```

### 3. **Verify Lineups Before Betting**
- Check team Twitter/official sites for late scratches
- Verify starting goalies confirmed (affects Saves props)
- Look for last-minute injury updates

### 4. **Use EV Thresholds**
- **High EV (>20%):** Strong edge, bet if lineup confirmed
- **Medium EV (10-20%):** Decent edge, bet smaller amounts
- **Low EV (0-10%):** Marginal, only bet if very confident in model

## Summary

**YES - The Render site performs periodic updates:**

✅ **Hourly props recommendations refresh** (every hour at :00)
✅ **Fresh odds from Bovada** (always latest lines)
✅ **Regenerated EV calculations** (projections vs new odds)
✅ **Smart game filtering** (skips started games)
✅ **GitHub persistence** (changes are saved and backed up)
✅ **Fast & efficient** (no model recomputation needed)

**The workflow is:**
1. **Morning:** Daily updater projects ALL player stats using NN models
2. **Hourly:** Cron job fetches fresh odds and recalculates EV
3. **Display:** UI shows latest edges with current bookmaker lines
4. **Throughout day:** Odds update, projections stay constant, EV adjusts accordingly

This is the **optimal design** for a betting intelligence system:
- Heavy computation (NN inference) runs once daily
- Light updates (odds + math) run every hour
- Users always see fresh edges without site slowdown
- Cost-efficient on Render free/starter plans

🎯 **Your complete system is tracking odds movements in real-time and updating betting recommendations hourly!**
