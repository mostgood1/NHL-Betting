from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
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
    # Atomic write: write to temp file then replace
    # Use same directory to ensure os.replace works across filesystems on Windows
    dirpath = str(path.parent)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=dirpath, suffix=".tmp") as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp, **essential_csv_kwargs)
    try:
        os.replace(tmp_path, path)
    finally:
        # Best-effort cleanup if replace failed
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
