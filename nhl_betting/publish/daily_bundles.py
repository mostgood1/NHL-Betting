from __future__ import annotations

import json
import os
import re
import tempfile
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..utils.io import PROC_DIR


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dirpath = str(path.parent)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=dirpath, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(text)
    try:
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def bundles_root(proc_dir: Path = PROC_DIR) -> Path:
    return proc_dir / "bundles"


def bundle_path(date: str, proc_dir: Path = PROC_DIR) -> Path:
    return bundles_root(proc_dir) / f"date={date}" / "bundle.json"


def manifest_path(proc_dir: Path = PROC_DIR) -> Path:
    return bundles_root(proc_dir) / "manifest.json"


def _validate_ymd(date: str) -> str:
    s = str(date or "").strip()
    if not _DATE_RE.fullmatch(s):
        raise ValueError(f"Invalid date '{date}'. Expected YYYY-MM-DD")
    return s


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _df_to_rows(df: pd.DataFrame, keep: Optional[list[str]] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    d2 = df
    if keep:
        cols = [c for c in keep if c in d2.columns]
        if cols:
            d2 = d2[cols]
    if limit is not None and limit > 0:
        d2 = d2.head(int(limit))

    # JSONResponse does not automatically coerce numpy/pandas scalars.
    # Convert to plain Python types to avoid TypeError: Object of type int64 is not JSON serializable.
    try:
        import numpy as _np
        import datetime as _dt

        def _cell(v: Any) -> Any:
            try:
                if v is None:
                    return None
                # Pandas NA / NaN
                try:
                    if pd.isna(v):
                        return None
                except Exception:
                    pass
                # Numpy scalars
                try:
                    if isinstance(v, _np.generic):
                        return v.item()
                except Exception:
                    pass
                # Timestamps / datetimes
                if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
                    try:
                        return v.isoformat()
                    except Exception:
                        return str(v)
                return v
            except Exception:
                return None

        try:
            d2 = d2.applymap(_cell)
        except Exception:
            for c in d2.columns:
                try:
                    d2[c] = d2[c].map(_cell)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        return d2.to_dict(orient="records")
    except Exception:
        return []


def _pick_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _json_sanitize(obj: Any) -> Any:
    """Convert an object graph to strict-JSON-safe Python primitives.

    Starlette's JSONResponse serializes with allow_nan=False, so any NaN/Inf will
    raise and result in a 500. This sanitizer converts NaN/Inf -> None and
    numpy/pandas scalars -> plain Python types.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import datetime as _dt
    except Exception:
        _dt = None

    def _san(x: Any) -> Any:
        if x is None:
            return None

        # dict
        if isinstance(x, dict):
            out: dict[str, Any] = {}
            for k, v in x.items():
                try:
                    ks = str(k)
                except Exception:
                    ks = "key"
                out[ks] = _san(v)
            return out

        # list/tuple
        if isinstance(x, (list, tuple)):
            return [_san(v) for v in x]

        # pandas NA / NaN
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        # numpy scalar
        try:
            if _np is not None and isinstance(x, _np.generic):
                return _san(x.item())
        except Exception:
            pass

        # datetime-like
        try:
            if isinstance(x, pd.Timestamp):
                return x.isoformat()
        except Exception:
            pass
        try:
            if _dt is not None and isinstance(x, (_dt.datetime, _dt.date)):
                return x.isoformat()
        except Exception:
            pass

        # floats: map NaN/Inf -> None
        if isinstance(x, float):
            try:
                if not math.isfinite(x):
                    return None
            except Exception:
                return None

        return x

    try:
        return _san(obj)
    except Exception:
        return obj


def discover_dates(proc_dir: Path = PROC_DIR) -> list[str]:
    """Discover available dates from processed artifacts.

    We keep this conservative and fast by scanning a few canonical patterns.
    """
    dates: set[str] = set()
    for pat in [
        "predictions_????-??-??.csv",
        "predictions_sim_????-??-??.csv",
        "recommendations_????-??-??.csv",
        "props_recommendations_????-??-??.csv",
        "props_recommendations_combined_????-??-??.csv",
        "props_recommendations_sim_????-??-??.csv",
    ]:
        try:
            for p in proc_dir.glob(pat):
                m = _DATE_RE.search(p.name)
                if m:
                    dates.add(m.group(0))
        except Exception:
            pass

    # Also include any already-built bundles
    try:
        for p in bundles_root(proc_dir).glob("date=*/bundle.json"):
            m = _DATE_RE.search(str(p))
            if m:
                dates.add(m.group(0))
    except Exception:
        pass

    return sorted(dates)


def build_daily_bundle(date: str, proc_dir: Path = PROC_DIR) -> dict[str, Any]:
    """Compile a stable JSON bundle for a single date from existing artifacts.

    This function is *pure* (no writes). Use `write_daily_bundle` to persist.
    """
    d = _validate_ymd(date)
    # Select best-available sources
    predictions_p = _pick_existing(proc_dir / f"predictions_sim_{d}.csv", proc_dir / f"predictions_{d}.csv")
    edges_p = _pick_existing(proc_dir / f"edges_sim_{d}.csv", proc_dir / f"edges_{d}.csv")
    game_recs_p = _pick_existing(proc_dir / f"recommendations_sim_{d}.csv", proc_dir / f"recommendations_{d}.csv")
    props_recs_p = _pick_existing(
        proc_dir / f"props_recommendations_sim_{d}.csv",
        proc_dir / f"props_recommendations_combined_{d}.csv",
        proc_dir / f"props_recommendations_{d}.csv",
    )
    rec_game_p = proc_dir / f"reconciliation_{d}.json"
    rec_props_p = proc_dir / f"reconciliation_props_{d}.json"

    predictions_df = _safe_read_csv(predictions_p) if predictions_p else pd.DataFrame()
    edges_df = _safe_read_csv(edges_p) if edges_p else pd.DataFrame()
    game_recs_df = _safe_read_csv(game_recs_p) if game_recs_p else pd.DataFrame()
    props_recs_df = _safe_read_csv(props_recs_p) if props_recs_p else pd.DataFrame()

    # Normalize key columns across different artifact variants.
    # - sim predictions use `totals_line_used` while legacy uses `total_line_used`
    # - puck line is fixed at +/-1.5 for our sim outputs, so provide a display line
    try:
        if predictions_df is not None and not predictions_df.empty:
            if "total_line_used" not in predictions_df.columns and "totals_line_used" in predictions_df.columns:
                predictions_df["total_line_used"] = predictions_df["totals_line_used"]
            if "pl_line_used" not in predictions_df.columns:
                predictions_df["pl_line_used"] = 1.5
    except Exception:
        pass

    # Keep bundles reasonably small; UI can fetch large tables via existing /api/* endpoints.
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _now_utc_iso(),
        "date": d,
        "files": {
            "predictions": f"data/processed/{predictions_p.name}" if predictions_p and predictions_p.exists() else None,
            "edges": f"data/processed/{edges_p.name}" if edges_p and edges_p.exists() else None,
            "game_recommendations": f"data/processed/{game_recs_p.name}" if game_recs_p and game_recs_p.exists() else None,
            "props_recommendations": f"data/processed/{props_recs_p.name}" if props_recs_p and props_recs_p.exists() else None,
            "reconciliation_games": f"data/processed/{rec_game_p.name}" if rec_game_p.exists() else None,
            "reconciliation_props": f"data/processed/{rec_props_p.name}" if rec_props_p.exists() else None,
        },
        "data": {
            "games": {
                "predictions": {
                    "count": int(len(predictions_df)) if predictions_df is not None and not predictions_df.empty else 0,
                    "rows": _df_to_rows(
                        predictions_df,
                        keep=[
                            "date",
                            "home",
                            "away",
                            "venue",
                            "game_state",
                            "p_home_ml",
                            "p_away_ml",
                            "total_line_used",
                            "p_over",
                            "p_under",
                            "pl_line_used",
                            "p_home_pl_-1.5",
                            "p_away_pl_+1.5",
                            "home_ml_odds",
                            "away_ml_odds",
                            "over_odds",
                            "under_odds",
                            "home_pl_-1.5_odds",
                            "away_pl_+1.5_odds",
                            "proj_home_goals",
                            "proj_away_goals",
                            "model_total",
                            "model_spread",
                        ],
                        limit=250,
                    ),
                },
                "edges": {
                    "count": int(len(edges_df)) if edges_df is not None and not edges_df.empty else 0,
                    "rows": _df_to_rows(edges_df, keep=["market", "side", "price", "ev", "home", "away", "totals_line"], limit=500),
                    "source": edges_p.name if edges_p else None,
                },
                "recommendations": {
                    "count": int(len(game_recs_df)) if game_recs_df is not None and not game_recs_df.empty else 0,
                    "rows": _df_to_rows(game_recs_df, keep=["market", "side", "price", "ev", "home", "away", "totals_line"], limit=200),
                },
                "reconciliation": _safe_read_json(rec_game_p),
            },
            "props": {
                "recommendations": {
                    "count": int(len(props_recs_df)) if props_recs_df is not None and not props_recs_df.empty else 0,
                    "rows": _df_to_rows(
                        props_recs_df,
                        keep=["team", "player", "market", "line", "side", "price", "ev", "book", "prob"],
                        limit=500,
                    ),
                    "source": props_recs_p.name if props_recs_p else None,
                },
                "reconciliation": _safe_read_json(rec_props_p),
            },
        },
    }
    return _json_sanitize(bundle)


def write_daily_bundle(date: str, proc_dir: Path = PROC_DIR) -> Path:
    d = _validate_ymd(date)
    out = bundle_path(d, proc_dir)
    obj = build_daily_bundle(d, proc_dir=proc_dir)
    _atomic_write_json(out, obj)
    return out


def build_manifest(proc_dir: Path = PROC_DIR) -> dict[str, Any]:
    dates = discover_dates(proc_dir)
    bundles: dict[str, Any] = {}
    for d in dates:
        p = bundle_path(d, proc_dir)
        bundles[d] = {
            "path": f"data/processed/bundles/date={d}/bundle.json" if p.exists() else None,
            "exists": bool(p.exists()),
        }
    return {
        "schema_version": 1,
        "generated_at_utc": _now_utc_iso(),
        "latest": dates[-1] if dates else None,
        "dates": dates,
        "bundles": bundles,
    }


def write_manifest(proc_dir: Path = PROC_DIR) -> Path:
    out = manifest_path(proc_dir)
    obj = build_manifest(proc_dir)
    _atomic_write_json(out, obj)
    return out
