#!/usr/bin/env python
"""Debug props projection to find the failure point."""
import traceback
import sys

try:
    from nhl_betting.cli import app
    from nhl_betting import cli
    
    print("[1] Imports successful")
    
    # Try calling the function directly (not a Typer callback)
    print("[2] Calling props_project_all...")
    cli.props_project_all(date='2025-10-17', ensure_history_days=365, include_goalies=True, use_nn=True)
    
    print("[3] Function completed successfully!")
    
except Exception as e:
    print(f"\n[ERROR] Failed at some point:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
