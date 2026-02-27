from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def _norm_name(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = _NON_WORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _coerce_player_id(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.round().astype("Int64")


def _summ(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit prop lines coverage vs sim boxscores")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--lines-dir",
        default=None,
        help="Override lines directory (default: data/props/player_props_lines/date=YYYY-MM-DD)",
    )
    ap.add_argument(
        "--sim-path",
        default=None,
        help="Override sim CSV path (default: data/processed/props_boxscores_sim_YYYY-MM-DD.csv)",
    )
    ap.add_argument(
        "--only-current",
        action="store_true",
        help="Only consider rows where is_current is True when available",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path (writes per-lined-player coverage table)",
    )
    ap.add_argument("--top", type=int, default=30, help="How many problematic players to print")
    args = ap.parse_args()

    date = args.date
    lines_dir = Path(args.lines_dir) if args.lines_dir else Path("data/props/player_props_lines") / f"date={date}"
    sim_path = Path(args.sim_path) if args.sim_path else Path("data/processed") / f"props_boxscores_sim_{date}.csv"

    if not lines_dir.exists():
        raise SystemExit(f"Missing lines dir: {lines_dir}")
    if not sim_path.exists() or sim_path.stat().st_size == 0:
        raise SystemExit(f"Missing sim file: {sim_path}")

    line_files = sorted([p for p in lines_dir.glob("*") if p.suffix.lower() in {".csv", ".parquet", ".pq"}])
    if not line_files:
        raise SystemExit(f"No line files found in: {lines_dir}")

    frames: list[pd.DataFrame] = []
    for p in line_files:
        try:
            dfp = _read_table(p)
        except Exception as e:
            print(f"warn: failed reading {p.name}: {e}")
            continue
        dfp["_src_file"] = p.name
        frames.append(dfp)

    if not frames:
        raise SystemExit(f"No readable line files in: {lines_dir}")
    lines = pd.concat(frames, ignore_index=True)

    if args.only_current and "is_current" in lines.columns:
        lines = lines[lines["is_current"].astype(bool)].copy()

    if "player_id" not in lines.columns:
        lines["player_id"] = np.nan
    if "player_name" not in lines.columns and "player" in lines.columns:
        lines["player_name"] = lines["player"]
    if "player_name" not in lines.columns:
        lines["player_name"] = ""
    if "market" not in lines.columns and "prop" in lines.columns:
        lines["market"] = lines["prop"]
    if "market" not in lines.columns:
        lines["market"] = ""

    lines["player_id_int"] = _coerce_player_id(lines["player_id"])
    lines["player_name_norm"] = lines["player_name"].map(_norm_name)
    lines["market"] = lines["market"].astype(str)

    key_id = lines["player_id_int"].notna()
    lines["join_key"] = np.where(key_id, lines["player_id_int"].astype(str), "name::" + lines["player_name_norm"])

    agg_dict: dict[str, object] = {
        "n_lines": ("market", "size"),
        "n_markets": ("market", "nunique"),
        "player_id_int": ("player_id_int", "first"),
        "player_name": ("player_name", "first"),
        "player_name_norm": ("player_name_norm", "first"),
        "markets": ("market", lambda x: ",".join(sorted({str(v) for v in x.dropna().unique()}))[:200]),
    }
    if "book" in lines.columns:
        agg_dict["books"] = ("book", lambda x: ",".join(sorted({str(v) for v in x.dropna().unique()})))
    else:
        agg_dict["books"] = ("market", lambda _: "")

    agg = lines.groupby("join_key", dropna=False).agg(**agg_dict).reset_index()

    sim = pd.read_csv(sim_path)
    if "player_id" not in sim.columns:
        raise SystemExit(f"sim file missing player_id: {sim_path}")
    sim["player_id_int"] = _coerce_player_id(sim["player_id"])
    if "player" in sim.columns:
        sim["player_name"] = sim["player"]
    elif "player_name" not in sim.columns:
        sim["player_name"] = ""
    sim["player_name_norm"] = sim["player_name"].map(_norm_name)

    if "period" in sim.columns and (sim["period"] == 0).any():
        sim_tot = sim[sim["period"] == 0].copy()
    else:
        sum_cols = [c for c in ["shots", "goals", "assists", "points", "blocks", "saves", "toi_sec"] if c in sim.columns]
        first_cols = [c for c in ["team", "game_home", "game_away", "date", "is_dressed", "player_name", "player_name_norm"] if c in sim.columns]
        sim_tot = sim.groupby("player_id_int", dropna=False)[sum_cols].sum().reset_index()
        if first_cols:
            first = sim.sort_values(["player_id_int"]).groupby("player_id_int", dropna=False)[first_cols].first().reset_index()
            sim_tot = sim_tot.merge(first, on="player_id_int", how="left")

    sim_tot["sim_toi_min"] = pd.to_numeric(sim_tot.get("toi_sec"), errors="coerce") / 60.0
    if "is_dressed" not in sim_tot.columns:
        sim_tot["is_dressed"] = np.nan

    sim_by_id = sim_tot.dropna(subset=["player_id_int"]).drop_duplicates(subset=["player_id_int"], keep="first")
    sim_by_name = (
        sim_tot.assign(player_name_norm=sim_tot["player_name_norm"].fillna(""))
        .sort_values(["player_name_norm"])
        .drop_duplicates(subset=["player_name_norm"], keep="first")
    )

    out = agg.merge(
        sim_by_id[["player_id_int", "player_name", "team", "is_dressed", "sim_toi_min"]].rename(
            columns={
                "player_name": "sim_player_name",
                "team": "sim_team",
                "is_dressed": "sim_is_dressed",
            }
        ),
        on="player_id_int",
        how="left",
    )

    needs_name = out["player_id_int"].isna() | out["sim_player_name"].isna()
    name_join = out.loc[needs_name, ["join_key", "player_name_norm"]].merge(
        sim_by_name[["player_name_norm", "player_id_int", "player_name", "team", "is_dressed", "sim_toi_min"]].rename(
            columns={
                "player_id_int": "sim_player_id_int_by_name",
                "player_name": "sim_player_name_by_name",
                "team": "sim_team_by_name",
                "is_dressed": "sim_is_dressed_by_name",
                "sim_toi_min": "sim_toi_min_by_name",
            }
        ),
        on="player_name_norm",
        how="left",
    )
    out = out.merge(name_join, on=["join_key", "player_name_norm"], how="left")

    out["match_type"] = np.where(
        out["sim_player_name"].notna(),
        "id",
        np.where(out["sim_player_name_by_name"].notna(), "name", "none"),
    )
    out["matched"] = out["match_type"] != "none"
    out["match_player_id_int"] = out["player_id_int"]
    out.loc[out["match_type"] == "name", "match_player_id_int"] = out.loc[out["match_type"] == "name", "sim_player_id_int_by_name"]
    out["match_player_name"] = out["sim_player_name"].fillna(out["sim_player_name_by_name"])
    out["match_team"] = out["sim_team"].fillna(out["sim_team_by_name"])
    out["match_is_dressed"] = out["sim_is_dressed"].fillna(out["sim_is_dressed_by_name"])
    out["match_sim_toi_min"] = out["sim_toi_min"].fillna(out["sim_toi_min_by_name"])

    n_total = int(len(out))
    n_with_id = int(out["player_id_int"].notna().sum())
    n_missing_id = int(out["player_id_int"].isna().sum())
    n_matched = int(out["matched"].sum())
    n_unmatched = int((~out["matched"]).sum())
    print(f"date={date} lines_dir={lines_dir}")
    print(f"line_files={[p.name for p in line_files]}")
    print(f"lined_players={n_total} with_player_id={n_with_id} missing_player_id={n_missing_id}")
    print(f"matched_to_sim={n_matched} unmatched_to_sim={n_unmatched}")

    matched = out[out["matched"]].copy()
    matched["match_is_dressed"] = pd.to_numeric(matched["match_is_dressed"], errors="coerce")
    n_d0 = int((matched["match_is_dressed"] == 0).sum())
    n_d1 = int((matched["match_is_dressed"] == 1).sum())
    n_du = int(matched["match_is_dressed"].isna().sum())
    print(f"matched_dressed: dressed={n_d1} undressed={n_d0} unknown={n_du}")

    toi = pd.to_numeric(matched["match_sim_toi_min"], errors="coerce")
    ts = _summ(toi)
    if ts.get("n"):
        print(f"matched_sim_toi_min: n={ts['n']} mean={ts['mean']:.2f} p10={ts['p10']:.2f} p50={ts['p50']:.2f} p90={ts['p90']:.2f}")

    for thr in (0.1, 1.0, 5.0, 10.0):
        frac = float((toi < thr).mean())
        print(f"matched_sim_toi_min<{thr:g}: {frac:.1%}")

    prob = out.copy()
    prob["problem_score"] = 0
    prob.loc[prob["match_type"] == "none", "problem_score"] += 100
    prob.loc[(prob["match_type"] != "none") & (pd.to_numeric(prob["match_is_dressed"], errors="coerce") == 0), "problem_score"] += 50
    prob.loc[(prob["match_type"] != "none") & (pd.to_numeric(prob["match_sim_toi_min"], errors="coerce") < 5), "problem_score"] += 20
    prob.loc[(prob["match_type"] != "none") & (pd.to_numeric(prob["match_sim_toi_min"], errors="coerce") < 1), "problem_score"] += 10
    prob = prob.sort_values(["problem_score", "n_markets", "n_lines"], ascending=[False, False, False])

    print(f"\nTop {args.top} problematic lined players:")
    cols = [
        "problem_score",
        "match_type",
        "player_id_int",
        "player_name",
        "match_player_id_int",
        "match_player_name",
        "match_team",
        "match_is_dressed",
        "match_sim_toi_min",
        "n_markets",
        "markets",
    ]
    cols = [c for c in cols if c in prob.columns]
    show = prob.head(args.top)[cols].copy()
    if "match_sim_toi_min" in show.columns:
        show["match_sim_toi_min"] = pd.to_numeric(show["match_sim_toi_min"], errors="coerce").round(2)
    print(show.to_string(index=False))

    out_path = Path(args.out) if args.out else Path("data/processed") / f"props_lines_sim_coverage_{date}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
