from __future__ import annotations

from typing import Optional
import asyncio

import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..utils.io import PROC_DIR

# Import shared recompute/settlement/reconciliation
try:
    from ..core.recs_shared import (
        recompute_edges_and_recommendations as _recs_recompute_shared,
        backfill_settlement_for_date as _bf_settlement,
        reconcile_extended as _reconcile,
    )
except Exception:
    _recs_recompute_shared = None
    _bf_settlement = None
    _reconcile = None

router = APIRouter()


@router.get("/api/game-recs/recompute")
async def api_game_recs_recompute(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = Query(0.0, description="Minimum EV filter when generating recommendations"),
):
    try:
        if _recs_recompute_shared is None:
            return JSONResponse({"ok": False, "error": "shared_module_unavailable"}, status_code=500)
        if not date:
            return JSONResponse({"ok": False, "error": "date_required"}, status_code=400)
        # Run compute off the event loop
        recs = await asyncio.to_thread(_recs_recompute_shared, date, float(min_ev))
        rsp = {
            "ok": True,
            "date": date,
            "count": len(recs) if isinstance(recs, list) else None,
            "recommendations_csv": str((PROC_DIR / f"recommendations_{date}.csv").resolve()),
            "edges_csv": str((PROC_DIR / f"edges_{date}.csv").resolve()),
        }
        return JSONResponse(rsp)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/api/game-recs")
async def api_game_recs(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    min_ev: Optional[float] = Query(None, description="Optional EV filter applied to returned rows"),
    top: Optional[int] = Query(None, description="Optional maximum rows returned, sorted by EV desc"),
    compute: int = Query(0, description="If 1 and file missing, recompute first using shared module"),
):
    try:
        if not date:
            return JSONResponse({"ok": False, "error": "date_required"}, status_code=400)
        p = PROC_DIR / f"recommendations_{date}.csv"
        if (not p.exists()) and compute == 1 and _recs_recompute_shared is not None:
            await asyncio.to_thread(_recs_recompute_shared, date, 0.0)
        if not p.exists():
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        # Read recommendations CSV; handle empty file gracefully
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        except Exception:
            # Any unexpected read error -> treat as no rows rather than 500
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        if min_ev is not None:
            try:
                df = df[pd.to_numeric(df["ev"], errors="coerce") >= float(min_ev)]
            except Exception:
                pass
        try:
            df = df.sort_values("ev", ascending=False)
        except Exception:
            pass
        if top is not None and top > 0:
            df = df.head(int(top))
        return JSONResponse({"ok": True, "date": date, "count": int(len(df)), "rows": df.to_dict(orient="records")})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/api/game-edges")
async def api_game_edges(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    top: Optional[int] = Query(None, description="Optional maximum rows returned, sorted by EV desc"),
):
    try:
        if not date:
            return JSONResponse({"ok": False, "error": "date_required"}, status_code=400)
        p = PROC_DIR / f"edges_{date}.csv"
        if not p.exists():
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        except Exception:
            return JSONResponse({"ok": True, "date": date, "rows": [], "count": 0})
        try:
            df = df.sort_values("ev", ascending=False)
        except Exception:
            pass
        if top is not None and top > 0:
            df = df.head(int(top))
        return JSONResponse({"ok": True, "date": date, "count": int(len(df)), "rows": df.to_dict(orient="records")})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/api/reconciliation")
async def api_reconciliation(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    compute: int = Query(0, description="If 1, attempt to compute reconciliation if missing"),
):
    try:
        if not date:
            return JSONResponse({"ok": False, "error": "date_required"}, status_code=400)
        p = PROC_DIR / f"reconciliation_{date}.json"
        if (not p.exists()) and compute == 1:
            try:
                if _bf_settlement is not None:
                    await asyncio.to_thread(_bf_settlement, date)
                if _reconcile is not None:
                    await asyncio.to_thread(_reconcile, date, 100.0)
            except Exception:
                pass
        if not p.exists():
            return JSONResponse({"ok": True, "date": date, "summary": None, "rows": []})
        import json as _json
        obj = _json.loads(p.read_text(encoding="utf-8"))
        return JSONResponse({"ok": True, **obj})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
