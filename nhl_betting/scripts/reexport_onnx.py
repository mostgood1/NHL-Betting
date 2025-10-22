"""Re-export ONNX models with current exporter settings.

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1; python -m nhl_betting.scripts.reexport_onnx
"""
from __future__ import annotations

from nhl_betting.models.nn_games import NNGameModel
from nhl_betting.utils.io import MODEL_DIR


def main():
    md = MODEL_DIR / "nn_games"
    tasks = ["FIRST_10MIN", "PERIOD_GOALS", "TOTAL_GOALS", "MONEYLINE", "GOAL_DIFF"]
    for mt in tasks:
        try:
            m = NNGameModel(model_type=mt, model_dir=md)
            p = m.export_onnx(opset_version=18)
            print(f"[reexport] {mt}: {p}")
        except Exception as e:
            print(f"[reexport] {mt} failed: {e}")


if __name__ == "__main__":
    main()
