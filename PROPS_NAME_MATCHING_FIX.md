# Props NN Predictions Not Being Used - Name Matching Issue

## Problem Identified

Props recommendations showing all players with identical `proj_lambda` values (fallbacks) despite NN models producing variable predictions.

**Root Cause**: Player name mismatch between projections and betting lines.

- **Projections file** (`props_projections_all_{date}.csv`): Abbreviated names like "A. Levshunov"
- **Betting lines**: Full names like "Artyom Levshunov"
- **Join**: Uses exact match on normalized names → always fails
- **Result**: All players fall back to hardcoded defaults (SOG: 2.4, GOALS: 0.35, etc.)

## Evidence

```bash
# Projections have variable predictions
props_projections_all_2025-10-17.csv:
  - A. Levshunov SOG: 2.1069
  - Mean: 1.24, Std: 2.98, Range: 0.0-27.0 ✅ GOOD VARIABILITY

# But recommendations all have same values
props_recommendations_2025-10-17.csv:
  - Artyom Levshunov SOG: 2.4 ← FALLBACK
  - All SOG: 2.4, All GOALS: 0.35 ← ALL FALLBACKS
  - Std: 0.0 ❌ NO VARIABILITY
```

## Solution

Modify `props_recommendations` function in `nhl_betting/cli.py` to create name variants for matching:

1. When building `lam_map` from projections file (line ~2150)
2. Create both full and abbreviated variants of each name  
3. Use "initial + lastname" format matching

### Implementation

Add name variant function before lam_map building (after line 2140):

```python
def _create_name_variants(name: str) -> list[str]:
    """Create name variants for fuzzy matching (full name and abbreviated)."""
    name = (name or "").strip()
    if not name:
        return []
    
    # Normalize
    norm = " ".join(name.split())
    variants = [norm.lower()]
    
    # If abbreviated (e.g., "A. Levshunov"), try to keep as-is
    if "." in norm:
        variants.append(norm.replace(".", "").lower())
    
    # If full name (e.g., "Artyom Levshunov"), create abbreviated
    parts = [p for p in norm.split() if p and not p.endswith(".")]
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        # Add "A. Levshunov" format
        abbreviated = f"{first[0].upper()}. {last}"
        variants.append(abbreviated.lower())
        variants.append(abbreviated.replace(".", "").lower())
    
    return list(set(variants))
```

Then modify lam_map building (~line 2150-2158):

```python
lam_map: dict[tuple[str, str], float] = {}
try:
    proj_all_path = PROC_DIR / f"props_projections_all_{date}.csv"
    if proj_all_path.exists():
        _proj = pd.read_csv(proj_all_path)
        if _proj is not None and not _proj.empty and {"player","market","proj_lambda"}.issubset(_proj.columns):
            for _, rr in _proj.iterrows():
                try:
                    player_name = _norm_name(rr.get("player"))
                    market = str(rr.get("market")).upper()
                    val = float(rr.get("proj_lambda")) if pd.notna(rr.get("proj_lambda")) else None
                    
                    if player_name and market and val is not None:
                        # Create variants for better matching
                        for variant in _create_name_variants(player_name):
                            key = (variant.lower(), market)
                            lam_map[key] = val
                except Exception:
                    pass
except Exception:
    lam_map = lam_map
```

And update work normalization (~line 2249):

```python
# Create normalized name variants for join
def _norm_for_join(name: str) -> str:
    """Normalize name for joining - prefer abbreviated format."""
    variants = _create_name_variants(name)
    # Try to find the most compact variant (abbreviated if available)
    return min(variants, key=len) if variants else ""

work["player_norm"] = work["player_display"].astype(str).map(_norm_for_join)
```

## Expected Result

After fix:
- "Artyom Levshunov" in lines → variants include "a. levshunov"
- "A. Levshunov" in projections → variants include "a levshunov", "a. levshunov"
- Join succeeds on "a. levshunov" or "a levshunov"
- Recommendations use actual NN predictions instead of fallbacks
- Variability restored (std: 0.62, range: 1.29-3.60 like NN models produce)

## Files to Modify

1. `nhl_betting/cli.py` - props_recommendations function (~line 2140-2260)

## Testing

After fix:
```bash
# Regenerate recommendations
python -m nhl_betting.cli props-recommendations --date 2025-10-17 --min-ev 0 --top 400

# Verify variability
python check_props_recommendations.py

# Expected:
# SOG: proj_lambda std > 0.5 (not 0.0)
# GOALS: proj_lambda std > 0.1 (not 0.0)
# All markets showing variable predictions matching projections file
```
