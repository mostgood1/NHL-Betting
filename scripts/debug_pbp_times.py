import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nhl_betting.data.nhl_api_web import NHLWebClient

def main(date: str):
    c = NHLWebClient()
    sb = c.scoreboard_day(date)
    print(f"[games] {len(sb)}")
    for g in sb:
        pk = g.get('gamePk')
        home = g.get('home'); away = g.get('away')
        print(f"\n[game] {home} vs {away} pk={pk}")
        if not pk: continue
        pbp = c.play_by_play(int(pk)) or {}
        plays = pbp.get('plays') if isinstance(pbp, dict) else None
        if not isinstance(plays, list):
            print('  [no plays list]')
            continue
        for p in plays:
            tkey = str(p.get('typeDescKey') or p.get('type') or '').lower()
            if 'goal' not in tkey: continue
            per = None
            try:
                pdsc = p.get('periodDescriptor') or {}
                per = int(pdsc.get('number') or pdsc.get('period') or p.get('period') or 0)
            except Exception:
                per = int(p.get('period') or 0)
            if per != 1: continue
            tr = p.get('timeRemaining'); tip = p.get('timeInPeriod'); tt = p.get('time')
            print('  goal: timeRemaining=', tr, ' timeInPeriod=', tip, ' time=', tt)
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: debug_pbp_times.py YYYY-MM-DD')
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
