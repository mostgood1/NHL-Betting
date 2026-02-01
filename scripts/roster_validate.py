import argparse
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"

def validate_roster(file: Path, min_team_size: int, max_team_size: int):
    if not file.exists():
        raise FileNotFoundError(f"Roster master not found at {file}")
    try:
        df = pd.read_csv(file)
    except Exception:
        df = pd.read_parquet(str(file).replace(".csv", ".parquet"))
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            c = cols.get(n.lower())
            if c:
                return c
        return None
    col_team = get_col("team_abbr","team","abbr")
    col_pid = get_col("player_id","id")
    col_name = get_col("full_name","player","name")
    col_pos = get_col("position","pos")
    if any(c is None for c in [col_team, col_pid, col_name]):
        raise ValueError("Missing required columns: team_abbr/player_id/full_name")
    df[col_team] = df[col_team].astype(str).str.upper().str.strip()
    df[col_name] = df[col_name].astype(str).str.strip()
    if col_pos:
        df[col_pos] = df[col_pos].astype(str).str.upper().str.strip()
    issues = {"teams": {}, "duplicates": {}, "missing": {}, "unknown": []}
    warnings = []
    severe = []
    for abbr, grp in df.groupby(col_team):
        n = int(len(grp))
        issues["teams"][abbr] = {"count": n}
        if n < min_team_size or n > max_team_size:
            msg = f"Team {abbr} size {n} outside [{min_team_size},{max_team_size}]"
            warnings.append(msg)
            if n < (min_team_size // 2) or n > (max_team_size + 10):
                severe.append(msg)
    missing_names = df[df[col_name].eq("") | df[col_name].isna()]
    if not missing_names.empty:
        issues["missing"]["names_count"] = int(len(missing_names))
        warnings.append(f"Missing names: {len(missing_names)}")
    if pd.api.types.is_numeric_dtype(df[col_pid]):
        df[col_pid] = df[col_pid].astype(int)
    pid_team = df.groupby(col_pid)[col_team].nunique()
    dup_pid = pid_team[pid_team > 1]
    if not dup_pid.empty:
        issues["duplicates"]["player_id_multi_teams"] = dup_pid.index.tolist()
        severe.append(f"Player IDs across multiple teams: {len(dup_pid)}")
    name_team = df.groupby(col_name)[col_team].nunique()
    dup_names = name_team[name_team > 1]
    if not dup_names.empty:
        issues["duplicates"]["names_multi_teams"] = dup_names.index.tolist()[:50]
        warnings.append(f"Player names across multiple teams: {len(dup_names)}")
    summary = {
        "file": str(file),
        "min_team_size": min_team_size,
        "max_team_size": max_team_size,
        "issues": issues,
        "warnings": warnings,
        "severe": severe,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate roster_master plausibility")
    parser.add_argument("--file", default=str(PROC_DIR / "roster_master.csv"), help="Path to roster_master.csv")
    parser.add_argument("--min-team-size", type=int, default=8)
    parser.add_argument("--max-team-size", type=int, default=35)
    parser.add_argument("--out", default=str(PROC_DIR / "roster_master_validation.json"))
    args = parser.parse_args()
    file = Path(args.file)
    out = Path(args.out)
    summary = validate_roster(file, args.min_team_size, args.max_team_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print({"teams_checked": len(summary["issues"].get("teams", {})), "warnings": len(summary["warnings"]), "severe": len(summary["severe"])})
    if summary["severe"]:
        print("[SEVERE] Issues detected. Consider rebuilding roster_master.")
    elif summary["warnings"]:
        print("[WARN] Warnings detected. Consider rebuilding roster_master.")
    else:
        print("[OK] Roster master looks plausible.")


if __name__ == "__main__":
    main()
