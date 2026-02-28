from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _finite(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s[np.isfinite(s)]


def _summ(x: pd.Series) -> dict:
    x = _finite(x)
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "mae": float(np.mean(np.abs(x))),
        "rmse": float(math.sqrt(float(np.mean(x * x)))),
    }


def _parse_toi_to_seconds(v: object) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        vv = float(v)
        return vv if np.isfinite(vv) else None
    s = str(v).strip()
    if not s:
        return None
    # Expected formats: "MM:SS" or "HH:MM:SS".
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return None
    if len(nums) == 2:
        mm, ss = nums
        return float(60 * mm + ss)
    if len(nums) == 3:
        hh, mm, ss = nums
        return float(3600 * hh + 60 * mm + ss)
    return None


def load_sim_boxscores(date: str, *, proc_dir: Path) -> pd.DataFrame:
    p = proc_dir / f"props_boxscores_sim_{date}.csv"
    if not p.exists() or getattr(p.stat(), "st_size", 0) == 0:
        raise SystemExit(f"Missing sim boxscores: {p}")
    df = pd.read_csv(p)
    if df is None or df.empty:
        raise SystemExit(f"Empty sim boxscores: {p}")

    # Aggregate per player per game.
    # IMPORTANT: sim files include BOTH per-period rows (period=1..3) and a game-total row (period=0).
    # Using a naive groupby-sum across all periods will double-count stats and TOI.
    group_cols = ["date", "team", "player_id", "player", "game_home", "game_away"]
    for c in group_cols:
        if c not in df.columns:
            raise SystemExit(f"Sim boxscores missing required column: {c}")

    if "period" not in df.columns:
        raise SystemExit(f"Sim boxscores missing required column: period")

    sum_cols = [
        "shots",
        "goals",
        "assists",
        "points",
        "blocks",
        "saves",
        "toi_sec",
    ]
    for c in sum_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    is_dressed = pd.to_numeric(df.get("is_dressed"), errors="coerce") if "is_dressed" in df.columns else None

    # Prefer period=0 totals when present; fallback to summing periods if not.
    df0 = df[pd.to_numeric(df["period"], errors="coerce").fillna(-1).astype(int) == 0].copy()
    if df0 is not None and not df0.empty:
        agg = df0.groupby(group_cols, as_index=False)[sum_cols].sum()
        if is_dressed is not None:
            dressed = (
                df0.assign(_d=is_dressed.fillna(0).astype(int))
                .groupby(group_cols, as_index=False)["_d"]
                .max()
            )
            agg = agg.merge(dressed, on=group_cols, how="left").rename(columns={"_d": "is_dressed"})
        else:
            agg["is_dressed"] = 1
    else:
        dfp = df[pd.to_numeric(df["period"], errors="coerce").fillna(-1).astype(int) > 0].copy()
        agg = dfp.groupby(group_cols, as_index=False)[sum_cols].sum()
        if is_dressed is not None:
            dressed = (
                dfp.assign(_d=is_dressed.fillna(0).astype(int))
                .groupby(group_cols, as_index=False)["_d"]
                .max()
            )
            agg = agg.merge(dressed, on=group_cols, how="left").rename(columns={"_d": "is_dressed"})
        else:
            agg["is_dressed"] = 1

    agg = agg.rename(
        columns={
            "shots": "sim_shots",
            "goals": "sim_goals",
            "assists": "sim_assists",
            "points": "sim_points",
            "blocks": "sim_blocks",
            "saves": "sim_saves",
            "toi_sec": "sim_toi_sec",
        }
    )
    agg["sim_toi_min"] = pd.to_numeric(agg["sim_toi_sec"], errors="coerce").fillna(0.0) / 60.0
    return agg


def _iter_actual_chunks(raw_path: Path, usecols: Iterable[str], chunksize: int = 200_000):
    yield from pd.read_csv(raw_path, usecols=list(usecols), chunksize=int(chunksize))


def load_actual_boxscores(date_et: str, *, raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists() or getattr(raw_path.stat(), "st_size", 0) == 0:
        raise SystemExit(f"Missing raw player game stats: {raw_path}")

    usecols = [
        "date",
        "team",
        "player_id",
        "primary_position",
        "role",
        "shots",
        "goals",
        "assists",
        "blocked",
        "saves",
        "timeOnIce",
    ]

    parts: list[pd.DataFrame] = []
    for chunk in _iter_actual_chunks(raw_path, usecols=usecols):
        dt = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
        et = dt.dt.tz_convert("America/New_York")
        mask = et.dt.strftime("%Y-%m-%d") == str(date_et)
        if not bool(mask.any()):
            continue
        sub = chunk.loc[mask].copy()
        sub["date_et"] = str(date_et)
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    for c in ["shots", "goals", "assists", "blocked", "saves"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df["toi_sec"] = df["timeOnIce"].map(_parse_toi_to_seconds)
    df["toi_sec"] = pd.to_numeric(df["toi_sec"], errors="coerce").fillna(0.0)
    df["points"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0.0) + pd.to_numeric(
        df["assists"], errors="coerce"
    ).fillna(0.0)

    keep = [
        "date_et",
        "team",
        "player_id",
        "primary_position",
        "role",
        "shots",
        "goals",
        "assists",
        "points",
        "blocked",
        "saves",
        "toi_sec",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(
        columns={
            "shots": "act_shots",
            "goals": "act_goals",
            "assists": "act_assists",
            "points": "act_points",
            "blocked": "act_blocks",
            "saves": "act_saves",
            "toi_sec": "act_toi_sec",
        }
    )
    df["act_toi_min"] = pd.to_numeric(df["act_toi_sec"], errors="coerce").fillna(0.0) / 60.0
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile simulated props boxscores vs actual player boxscores.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (ET)")
    ap.add_argument("--proc-dir", default="data/processed", help="Processed data directory")
    ap.add_argument("--raw", default="data/raw/player_game_stats.csv", help="Raw player game stats CSV")
    ap.add_argument(
        "--out",
        default="",
        help="Optional output CSV path (default: data/processed/boxscore_reconciliation_{date}.csv)",
    )
    args = ap.parse_args()

    date = str(args.date)
    proc_dir = Path(args.proc_dir)
    raw_path = Path(args.raw)

    sim = load_sim_boxscores(date, proc_dir=proc_dir)
    act = load_actual_boxscores(date, raw_path=raw_path)

    print(f"date={date}")
    print(f"sim_rows={len(sim)} act_rows={len(act)}")
    if act.empty:
        print("No actual rows for this ET date (raw feed may not be updated yet).")
        return 0

    # Join by player_id + team (safer than id alone for traded players).
    sim["player_id"] = pd.to_numeric(sim["player_id"], errors="coerce")
    act["player_id"] = pd.to_numeric(act["player_id"], errors="coerce")
    joined = sim.merge(act, on=["player_id", "team"], how="left")
    n_match = int(joined["act_shots"].notna().sum()) if "act_shots" in joined.columns else 0
    print(f"matched_rows={n_match}/{len(joined)}")

    # Compute deltas where actual exists.
    metrics = [
        ("shots", "sim_shots", "act_shots"),
        ("goals", "sim_goals", "act_goals"),
        ("assists", "sim_assists", "act_assists"),
        ("points", "sim_points", "act_points"),
        ("blocks", "sim_blocks", "act_blocks"),
        ("saves", "sim_saves", "act_saves"),
        ("toi_min", "sim_toi_min", "act_toi_min"),
    ]

    out = joined.copy()
    for name, a, b in metrics:
        if a in out.columns and b in out.columns:
            out[f"delta_{name}"] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")

    def _subset(df: pd.DataFrame, label: str) -> None:
        print(f"\n==== {label} ====\n")
        for name, _, _ in metrics:
            col = f"delta_{name}"
            if col not in df.columns:
                continue
            s = _summ(df[col])
            if not s.get("n"):
                continue
            print(
                f"{name:8s} n={s['n']:4d} mean={s['mean']:+.3f} med={s['median']:+.3f} "
                f"p10={s['p10']:+.3f} p90={s['p90']:+.3f} mae={s['mae']:.3f} rmse={s['rmse']:.3f}"
            )

    have_actual = out["act_shots"].notna() if "act_shots" in out.columns else pd.Series([False] * len(out))
    act_toi = pd.to_numeric(out.get("act_toi_min"), errors="coerce")
    active = have_actual & (act_toi.notna()) & (act_toi > 0)
    print(f"active_rows={int(active.sum())}/{int(have_actual.sum())} (actual TOI>0)")

    _subset(out.loc[have_actual].copy(), "ALL (matched)")
    _subset(out.loc[active].copy(), "ACTIVE ONLY")

    if "role" in out.columns:
        role = out["role"].astype(str).str.lower()
        _subset(out.loc[have_actual & (role == "skater")].copy(), "SKATERS (matched)")
        _subset(out.loc[active & (role == "skater")].copy(), "SKATERS (active)")
        _subset(out.loc[have_actual & (role != "skater")].copy(), "GOALIES (matched)")
        _subset(out.loc[active & (role != "skater")].copy(), "GOALIES (active)")

    if "primary_position" in out.columns:
        for pos in ["C", "L", "R", "D", "G"]:
            mask = have_actual & (out["primary_position"].astype(str).str.upper() == pos)
            if int(mask.sum()) >= 30:
                _subset(out.loc[mask].copy(), f"POS {pos}")

    out_path = Path(args.out) if str(args.out).strip() else (proc_dir / f"boxscore_reconciliation_{date}.csv")
    try:
        out.to_csv(out_path, index=False)
        print(f"\nwrote={out_path}")
    except Exception as e:
        print(f"\nfailed_write={out_path} err={e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
