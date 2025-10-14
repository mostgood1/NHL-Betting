from __future__ import annotations

from pathlib import Path
from typing import Dict, Set
import json
import pandas as pd


def _unwrap_dictish_name(val: str) -> str:
    try:
        s = str(val or "").strip()
        if s.startswith("{") and s.endswith("}"):
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                v = d.get("default") or d.get("name") or d.get("fullName")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return str(val or "")


def _load_cache(cache_path: Path) -> Dict[str, str]:
    try:
        if cache_path.exists() and cache_path.stat().st_size > 2:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Only keep string-to-string entries
                    return {str(k): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except Exception:
        return {}
    return {}


def _save_cache(cache_path: Path, mapping: Dict[str, str]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
    except Exception:
        # best-effort
        pass


def get_positions_map(target_names: Set[str], stats_csv: Path, cache_path: Path) -> Dict[str, str]:
    """Return a map of player full name -> last known primary position.

    Uses a persistent JSON cache to avoid repeatedly scanning the entire stats CSV.
    Will scan the CSV in chunks only for names missing from the cache, then update cache.
    """
    if not target_names:
        return {}
    cache = _load_cache(cache_path)
    missing = {str(n) for n in target_names if str(n) not in cache}
    if not missing:
        return {k: cache[k] for k in target_names if k in cache}
    # Build positions for missing names by scanning stats CSV in chunks
    if stats_csv and Path(stats_csv).exists():
        seen = set()
        try:
            for chunk in pd.read_csv(stats_csv, usecols=["player", "primary_position"], chunksize=100_000):
                chunk["player"] = chunk["player"].astype(str).map(_unwrap_dictish_name)
                sub = chunk[chunk["player"].isin(missing)].dropna(subset=["primary_position"]).copy()
                for _, rr in sub.iterrows():
                    nm = str(rr.get("player") or "")
                    if nm and nm not in seen:
                        cache[nm] = str(rr.get("primary_position"))
                        seen.add(nm)
                # Early stop if we've found all
                if len(seen) >= len(missing):
                    break
        except Exception:
            pass
        # Persist updated cache (best-effort)
        _save_cache(cache_path, cache)
    # Return subset for targets that are now in cache
    return {k: cache[k] for k in target_names if k in cache}
