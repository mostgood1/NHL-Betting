"""Test script to regenerate predictions with ONNX models."""
import sys
from pathlib import Path

# Run prediction
from nhl_betting.cli import app, predict

print("=" * 80)
print("Running prediction with ONNX models...")
print("=" * 80)

try:
    # Call the function directly
    predict(date="2025-10-17", odds_source="csv")
    
    print("\n" + "=" * 80)
    print("SUCCESS! Predictions regenerated")
    print("=" * 80)
    
    # Check the results
    import pandas as pd
    df = pd.read_csv('data/processed/predictions_2025-10-17.csv')
    print("\nPeriod and first_10min columns:")
    print(df[['home', 'away', 'first_10min_proj', 'period1_home_proj', 'period2_home_proj']].to_string())
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
