# NHL Betting Frontend Evaluation & Deployment Plan

## Current Status: October 17, 2025

### ✅ Local Instance Components

#### 1. **Core Data Pipeline**
- **NPU-Accelerated NN Models**: All 6 player props models (SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS) running with QNNExecutionProvider
- **Game Predictions**: Elo/Poisson models for moneyline, totals, puck lines
- **Daily Update**: Automated via `daily_update.ps1` with `PROPS_PRECOMPUTE_ALL=1`
- **Data Sources**: OddsAPI (primary), Bovada (fallback), NHL Web API

#### 2. **HTML Pages** (User-Facing)
```
/                           - Dashboard with today's games and edges
/props                      - Player props overview with filters
/props/all                  - Complete props projections table
/props/recommendations      - Top EV props bets (NN-powered)
/props/reconciliation       - Historical accuracy tracking
/recommendations            - Game outcome recommendations
/reconciliation             - Game prediction accuracy
/odds-coverage              - Odds availability by book/market
```

#### 3. **API Endpoints** (Data Access)
```
/api/predictions            - Game predictions with probabilities
/api/edges                  - Top edges by EV
/api/props/projections      - NN-generated props projections
/api/props/recommendations  - Props bets with EV calculations
/api/props/all.json         - All player projections (CSV also available)
/api/last-updated           - Data freshness timestamp
/api/status                 - System health check
/api/scoreboard             - Live game scores
```

#### 4. **Cron/Automation Endpoints** (For Render)
```
/api/cron/props-projections   - Trigger NN projections generation
/api/cron/props-recommendations - Generate props bets
/api/cron/props-fast          - Fast props update (odds only)
/api/cron/light-refresh       - Quick odds refresh
/api/cron/capture-closing     - Capture closing lines
/api/cron/retune              - Retrain models
```

### 📊 Current Data Files (from Oct 17 daily update)

**Game Predictions:**
- `data/processed/predictions_2025-10-17.csv` - 4 games
- `data/processed/predictions_2025-10-18.csv` - 13 games
- `data/processed/edges_2025-10-17.csv` - Edge rankings

**Player Props (NN-Powered):**
- `data/processed/props_projections_all_2025-10-17.csv` - 2,271 projections
- `data/processed/props_recommendations_2025-10-17.csv` - 154 bets (top: Ryan Hartman ASSISTS +69.95% EV)
- `data/processed/props_recommendations_history.csv` - Rolling history

**Odds Data:**
- `data/props/player_props_lines/date=2025-10-17/oddsapi.parquet` - 198 lines
- `data/props/player_props_lines/date=2025-10-18/oddsapi.parquet` - 239 lines

### 🎯 Local Instance Evaluation Checklist

#### A. Data Freshness & Accuracy
- [ ] Verify `/api/last-updated` returns correct timestamps
- [ ] Check props projections use NN models (proj_lambda values match NPU output)
- [ ] Validate game predictions have Elo ratings and probabilities
- [ ] Confirm odds data is from today (not stale)

#### B. EV Calculations
- [ ] Props recommendations sorted by EV descending
- [ ] EV formula: `ev = (p_over * decimal_odds) - 1` for OVER side
- [ ] Negative EV bets filtered out (min_ev threshold applied)
- [ ] Multiple books compared (best odds selected per prop)

#### C. User Experience
- [ ] Dashboard loads < 2 seconds
- [ ] Props page filterable by team, market, player
- [ ] Recommendations table sortable by EV, probability, line
- [ ] Historical reconciliation shows accuracy metrics
- [ ] Mobile-responsive design

#### D. Predictability Features
- [ ] Confidence scores visible (probability percentages)
- [ ] Historical win rate by market/model type
- [ ] Calibration charts (predicted vs actual)
- [ ] Edge persistence (edges that held vs moved)

### 🚀 Render Deployment Requirements

#### 1. **Environment Variables** (Must Set)
```bash
PROPS_PRECOMPUTE_ALL=1           # Enable NN projections
PROPS_INCLUDE_BOVADA=0           # OddsAPI only (faster)
SKIP_PROPS_CALIBRATION=1         # Skip slow calibration step
PROPS_NO_COMPUTE=1               # Disable on-demand compute (use precomputed only)
ODDS_API_KEY=<your_key>          # OddsAPI key
QNN_SDK_ROOT=/opt/qnn            # Path to QNN SDK (if NPU available on server)
```

#### 2. **Build Configuration** (`render.yaml`)
```yaml
services:
  - type: web
    name: nhl-betting
    runtime: python
    region: oregon
    plan: starter
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn nhl_betting.web.app:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PROPS_PRECOMPUTE_ALL
        value: "1"
      - key: PROPS_NO_COMPUTE
        value: "1"
    
  - type: cron
    name: daily-update
    runtime: python
    schedule: "0 10 * * *"  # 10 AM UTC = 6 AM ET
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python -m nhl_betting.scripts.daily_update --days-ahead 2 --no-reconcile"
    
  - type: cron
    name: intraday-props-refresh
    runtime: python
    schedule: "*/30 9-23 * * *"  # Every 30 min during game hours
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python -m nhl_betting.cli props-fast --date $(date +%Y-%m-%d)"
```

#### 3. **Intraday Odds Updates Strategy**

**Problem**: Odds change throughout the day; need to update recommendations without full retraining.

**Solution**: Fast Props Workflow
```python
# props-fast command (already exists in CLI)
# 1. Fetch fresh odds from OddsAPI
# 2. Load precomputed NN projections (props_projections_all_{date}.csv)
# 3. Recalculate EV with new odds
# 4. Update props_recommendations_{date}.csv
# 5. Update props_recommendations_history.csv
```

**Implementation**:
- Render cron job every 30 minutes: `props-fast --date today`
- Web app serves from updated CSV files
- No model retraining needed (projections stay static)
- Only odds/EV recalculated (fast: < 10 seconds)

#### 4. **Data Persistence** (Critical)

**Challenge**: Render ephemeral filesystem; data lost on restart.

**Solutions**:
A. **Git-based persistence** (current approach):
   - Daily update auto-commits CSVs to GitHub
   - Web app reads from local files
   - ✅ Pros: Simple, version-controlled
   - ❌ Cons: Slow commits, merge conflicts

B. **External storage** (recommended):
   - Use Render persistent disk ($1/GB/month)
   - Mount at `/data` for CSV files
   - Models stored separately (S3/R2)
   - ✅ Pros: Fast, reliable
   - ❌ Cons: Additional cost

C. **Database** (future):
   - PostgreSQL for historical data
   - Redis for caching
   - ✅ Pros: Query optimization, scalability
   - ❌ Cons: More complex, migration needed

### 🔄 Synchronized Workflow (Local ↔ Render)

#### Local Development:
1. Run `daily_update.ps1` nightly (or on demand)
2. Generates all CSVs with NN projections
3. Commit to GitHub automatically
4. Launch `.\launch_local.ps1` for testing

#### Render Production:
1. Daily cron at 6 AM ET runs `daily_update`
2. Generates fresh predictions and props
3. Commits to GitHub (or saves to persistent disk)
4. Intraday cron every 30 min runs `props-fast`
5. Updates only odds/EV calculations
6. Web app serves latest data

#### Consistency Guarantees:
- ✅ Same CSV file formats
- ✅ Same NN models (sync model files)
- ✅ Same calculation logic
- ✅ Same API endpoints
- ✅ Timestamps on all data for cache busting

### 📋 Action Items

#### Immediate (Local Testing):
1. [ ] Open http://127.0.0.1:8000 and verify dashboard loads
2. [ ] Check /props/recommendations shows NN-powered bets
3. [ ] Verify Ryan Hartman ASSISTS Over 0.5 appears with ~70% EV
4. [ ] Test /api/props/projections.json returns 2,271 rows
5. [ ] Confirm /api/last-updated shows today's timestamp

#### Pre-Deployment (Render Prep):
6. [ ] Create `render.yaml` with cron jobs
7. [ ] Set environment variables in Render dashboard
8. [ ] Test `props-fast` command locally
9. [ ] Upload NN model files (.onnx, .pt, metadata.npz) to Render
10. [ ] Configure persistent disk or S3 for data storage

#### Post-Deployment (Verification):
11. [ ] Render URL matches local http://127.0.0.1:8000 exactly
12. [ ] Daily cron runs successfully at 6 AM ET
13. [ ] Intraday props-fast updates EV every 30 min
14. [ ] API responses < 500ms for all endpoints
15. [ ] Historical data persists across deploys

### 🎯 Success Criteria

**Local Instance:**
- ✅ NPU models running (confirmed)
- ✅ Daily update generates all files (confirmed: 462s runtime)
- ✅ Props recommendations with EV > 0 (confirmed: 154 bets)
- ⏳ Frontend loads and displays correctly
- ⏳ All API endpoints return expected data

**Render Instance:**
- ⏳ Identical output to local instance
- ⏳ Intraday odds updates every 30 min
- ⏳ Props recommendations adjust dynamically
- ⏳ No data loss across deploys
- ⏳ < 2 second page load times

### 📊 Monitoring & Alerts

**Health Checks:**
- `/health` - Overall system status
- `/health/props` - Props data freshness
- `/health/render` - Render-specific diagnostics
- `/api/status` - API availability

**Metrics to Track:**
- Data update lag (minutes since last refresh)
- API response times (p50, p95, p99)
- EV distribution (histogram of recommended bets)
- Accuracy over time (win rate by market)
- NPU utilization (if available on Render)

---

## Next Steps

1. **Test local frontend comprehensively**
2. **Document any issues or missing features**
3. **Create Render configuration**
4. **Deploy to Render staging environment**
5. **Verify parity between local and Render**
6. **Enable intraday cron jobs**
7. **Monitor for 24 hours**
8. **Go live** 🚀
