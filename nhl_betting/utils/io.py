from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

for p in [DATA_DIR, RAW_DIR, PROC_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


essential_csv_kwargs = dict(index=False)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **essential_csv_kwargs)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
