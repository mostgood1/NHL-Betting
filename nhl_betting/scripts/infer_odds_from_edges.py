import argparse
import math
from pathlib import Path
import pandas as pd

from ..utils.io import PROC_DIR


def american_from_r(r: float) -> float | None:
    """Convert net decimal payout r (profit per 1 stake) to American odds.

    r >= 1 -> +100*r; r < 1 -> -100/r
    """
    try:
        if r is None or not math.isfinite(r) or r <= 0:
            return None
        if r >= 1:
            return round(100.0 * r, 0)
        return round(-100.0 / r, 0)
    except Exception:
        return None


def infer_r_from_ev_p(ev: float, p: float) -> float | None:
    """Given EV per 1 stake and win probability p, infer net decimal payout r.

    Assumes EV = p*r - (1-p).
    """
    try:
        if p is None or ev is None:
            return None
        if p <= 0 or p >= 1:
            return None
        return (ev + (1.0 - p)) / p
    except Exception:
        return None


def run(date: str):
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    edges_path = PROC_DIR / f"edges_{date}.csv"
    if not pred_path.exists() or not edges_path.exists():
        raise SystemExit(f"Missing files for {date}. Need {pred_path.name} and {edges_path.name}")
    dfp = pd.read_csv(pred_path)
    dfe = pd.read_csv(edges_path)

    # Map market -> probability column in predictions
    prob_map = {
        "home_ml": "p_home_ml",
        "away_ml": "p_away_ml",
        "over": "p_over",
        "under": "p_under",
        "home_pl_-1.5": "p_home_pl_-1.5",
        "away_pl_+1.5": "p_away_pl_+1.5",
    }

    # Normalize edges market labels: ev_X -> X
    dfe = dfe.copy()
    dfe["market"] = dfe["market"].astype(str).str.replace("^ev_", "", regex=True)

    # Join on date/home/away
    on_cols = ["date", "home", "away"]
    merged = pd.merge(dfe, dfp[on_cols + list(set(prob_map.values()))], on=on_cols, how="left")

    rows = []
    for _, r in merged.iterrows():
        mkt = r.get("market")
        ev = r.get("ev")
        prob_col = prob_map.get(str(mkt))
        if not prob_col or prob_col not in r:
            continue
        p = r.get(prob_col)
        rnet = infer_r_from_ev_p(ev, p)
        amer = american_from_r(rnet) if rnet is not None else None
        rows.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "market": mkt,
            "p": p,
            "ev": ev,
            "net_decimal": rnet,
            "american_inferred": amer,
        })

    out = pd.DataFrame(rows)
    out_path = PROC_DIR / f"inferred_odds_{date}.csv"
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Infer book odds from EV and model probabilities")
    ap.add_argument("--date", required=True, help="Slate date YYYY-MM-DD")
    args = ap.parse_args()
    path = run(args.date)
    print(f"Wrote {path}")
