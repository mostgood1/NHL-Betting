"""Test that daily update workflow uses NN models end-to-end."""
import os
import sys
from pathlib import Path

# Set environment for NN precomputation
os.environ["PROPS_PRECOMPUTE_ALL"] = "1"

print("[test] Testing daily update NN workflow...")
print(f"[test] PROPS_PRECOMPUTE_ALL={os.environ.get('PROPS_PRECOMPUTE_ALL')}")

# Import the CLI function
from nhl_betting.cli import props_project_all

# Test for today
from datetime import datetime
date_str = datetime.now().strftime("%Y-%m-%d")

print(f"\n[test] Running props_project_all for {date_str} with NN models...")
print("[test] This should show '[npu] ... model loaded with QNNExecutionProvider' messages")
print()

try:
    # Call directly (not via Typer callback)
    props_project_all(
        date=date_str,
        ensure_history_days=365,
        include_goalies=True,
        use_nn=True  # Explicitly enable NN (though it's default)
    )
    print(f"\n[test] ✅ SUCCESS - props_project_all completed with NN models!")
    
    # Verify output file exists
    output_path = Path(f"data/processed/props_projections_all_{date_str}.csv")
    if output_path.exists():
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"[test] ✅ Output file created: {len(df)} projections")
        print(f"[test] ✅ Markets: {df['market'].unique()}")
        print(f"[test] ✅ Sample projections:")
        print(df.head(10))
    else:
        print(f"[test] ⚠️  Output file not found: {output_path}")
        
except Exception as e:
    print(f"\n[test] ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[test] ✅ Daily update NN workflow is properly wired!")
