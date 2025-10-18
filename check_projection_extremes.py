import pandas as pd
import numpy as np

print("="*80)
print("GAME PROJECTION SANITY CHECK")
print("="*80)

# Load predictions
pred = pd.read_csv('data/processed/predictions_2025-10-17.csv')

# Load historical games to compare against
hist = pd.read_csv('data/raw/games_with_features.csv')
hist['total_goals'] = hist['home_goals'] + hist['away_goals']

print(f"\n📊 HISTORICAL BASELINE (3,136 games)")
print(f"   Average total goals: {hist['total_goals'].mean():.2f}")
print(f"   Std dev: {hist['total_goals'].std():.2f}")
print(f"   Min: {hist['total_goals'].min():.0f}, Max: {hist['total_goals'].max():.0f}")
print(f"   90th percentile: {hist['total_goals'].quantile(0.90):.2f}")

print("\n" + "="*80)
print("TODAY'S PROJECTIONS")
print("="*80)

for _, r in pred.iterrows():
    print(f"\n{r['home']} vs {r['away']}")
    print(f"  Projected: {r['proj_home_goals']:.2f} - {r['proj_away_goals']:.2f} (Total: {r['model_total']:.2f})")
    print(f"  Moneyline: {r['p_home_ml']:.1%} home / {r['p_away_ml']:.1%} away")
    print(f"  Total line used: {r['total_line_used']:.1f}")
    
    # Flag extreme projections
    warnings = []
    if r['proj_home_goals'] > 5.5:
        warnings.append(f"⚠️ Home projection {r['proj_home_goals']:.2f} very high (>5.5)")
    if r['proj_away_goals'] < 1.5:
        warnings.append(f"⚠️ Away projection {r['proj_away_goals']:.2f} very low (<1.5)")
    if r['model_total'] > hist['total_goals'].quantile(0.90):
        warnings.append(f"⚠️ Total {r['model_total']:.2f} in top 10% of historical games")
    if r['model_total'] < hist['total_goals'].quantile(0.10):
        warnings.append(f"⚠️ Total {r['model_total']:.2f} in bottom 10% of historical games")
    
    # Check team balance
    imbalance = abs(r['proj_home_goals'] - r['proj_away_goals'])
    if imbalance > 3.0:
        warnings.append(f"⚠️ Large goal differential: {imbalance:.2f} (>3.0)")
    
    if warnings:
        for w in warnings:
            print(f"  {w}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check Poisson lambda allocation
print("\nAssuming Poisson allocation formula:")
print("  lam_h = 0.7 * (total * p_home) + 0.3 * base_home")
print("  lam_a = 0.7 * (total * p_away) + 0.3 * base_away")
print("\nWhere base_home=3.2 (3.05*1.05), base_away=2.9 (3.05*0.95)")

for _, r in pred.iterrows():
    base_h = 3.05 * 1.05
    base_a = 3.05 * 0.95
    
    # Current formula (70/30)
    lam_h_70 = 0.7 * (r['total_line_used'] * r['p_home_ml']) + 0.3 * base_h
    lam_a_70 = 0.7 * (r['total_line_used'] * r['p_away_ml']) + 0.3 * base_a
    
    # Proposed 50/50 formula
    lam_h_50 = 0.5 * (r['total_line_used'] * r['p_home_ml']) + 0.5 * base_h
    lam_a_50 = 0.5 * (r['total_line_used'] * r['p_away_ml']) + 0.5 * base_a
    
    print(f"\n{r['home'][:15]} vs {r['away'][:15]}")
    print(f"  Current (70/30): {lam_h_70:.2f} - {lam_a_70:.2f} = {lam_h_70+lam_a_70:.2f}")
    print(f"  Proposed (50/50): {lam_h_50:.2f} - {lam_a_50:.2f} = {lam_h_50+lam_a_50:.2f}")
    print(f"  Actual projection: {r['proj_home_goals']:.2f} - {r['proj_away_goals']:.2f} = {r['model_total']:.2f}")
    
    # Check if proposed is more reasonable
    if abs((lam_h_50 + lam_a_50) - hist['total_goals'].mean()) < abs(r['model_total'] - hist['total_goals'].mean()):
        print(f"  ✓ 50/50 split closer to historical average")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nThe 70/30 split causes extreme projections when ML probability is lopsided.")
print("Consider:")
print("1. Change to 50/50 split (more conservative)")
print("2. Use recent form data (goals_last10) instead of pure allocation")
print("3. Add dampening for extreme probabilities (cap at 25%-75%)")
