import pandas as pd, json, os
cmp_path = r"C:\Users\mostg\OneDrive\Coding\NHL-Betting\data\processed\props_boxscores_compare_2026-01-27.csv"
df = pd.read_csv(cmp_path)
markets = {
    'SOG': ('sog_sim','sog_act'),
    'GOALS': ('goals_sim','goals_act'),
    'ASSISTS': ('assists_sim','assists_act'),
    'POINTS': ('points_sim','points_act'),
    'SAVES': ('saves_sim','saves_act'),
    'BLOCKS': ('blocks_sim','blocks_act'),
}
summary = {}
report = {}
for m, (sim_col, act_col) in markets.items():
    if sim_col not in df.columns or act_col not in df.columns:
        continue
    tmp = df[[sim_col, act_col, 'player', 'team']].copy()
    tmp['abs_err'] = (tmp[sim_col].fillna(0) - tmp[act_col].fillna(0)).abs()
    # overall MAE
    try:
        summary[f'mae_{m.lower()}'] = float(tmp['abs_err'].mean())
    except Exception:
        summary[f'mae_{m.lower()}'] = None
    # top players by absolute error
    top = tmp.sort_values('abs_err', ascending=False).head(15)
    report[m] = [
        {
            'player': str(r['player']),
            'team': str(r['team']),
            'sim': float(r[sim_col]) if pd.notna(r[sim_col]) else None,
            'actual': float(r[act_col]) if pd.notna(r[act_col]) else None,
            'abs_err': float(r['abs_err']) if pd.notna(r['abs_err']) else None,
        }
        for _, r in top.iterrows()
    ]
out = {
    'date': '2026-01-27',
    'rows': int(len(df)),
    'summary_mae': summary,
    'top_abs_errors': report,
}
# write to processed
proc_dir = r"C:\Users\mostg\OneDrive\Coding\NHL-Betting\data\processed"
out_path = os.path.join(proc_dir, 'props_boxscores_compare_top_2026-01-27.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(out_path)
