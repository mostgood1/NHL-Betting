import sys, os, json
from datetime import datetime
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.data.nhl_api_web import NHLWebClient

def _abbr(team: str) -> str:
    try:
        from nhl_betting.web.teams import get_team_assets
        return (get_team_assets(str(team)).get('abbr') or '').upper()
    except Exception:
        return ''

def inspect(date: str, team_filter: str | None = None):
    c = NHLWebClient()
    rows = c.scoreboard_day(date)
    print('[games]', [ (r.get('gamePk'), r.get('home'), r.get('away')) for r in rows ])
    for g in rows:
        gid = g.get('gamePk')
        home, away = g.get('home'), g.get('away')
        if team_filter and (team_filter.lower() not in str(home).lower() and team_filter.lower() not in str(away).lower()):
            continue
        print('\n==', gid, f"{away} @ {home}")
        try:
            pbp = c.play_by_play(int(gid))
        except Exception as e:
            print('[web] pbp error', e); pbp = {}
        plays = pbp.get('plays') if isinstance(pbp, dict) else []
        seen = 0
        for p in plays:
            t = (str(p.get('typeDescKey') or p.get('type')) or '').lower()
            if t != 'goal':
                continue
            seen += 1
            if seen <= 3:
                sample = {
                    'timeRemaining': p.get('timeRemaining'),
                    'timeInPeriod': p.get('timeInPeriod'),
                    'period': p.get('period') or (p.get('periodDescriptor') or {}).get('number') or None,
                    'team.triCode': ((p.get('team') or {}).get('triCode') or (p.get('team') or {}).get('abbrev') or (p.get('team') or {}).get('abbreviation')),
                    'teamAbbrev': p.get('teamAbbrev') or p.get('teamTriCode'),
                    'details.eventOwnerAbbrev': (p.get('details') or {}).get('eventOwnerAbbrev'),
                    'details.eventOwner': (p.get('details') or {}).get('eventOwner'),
                    'club': p.get('club'),
                }
                print('[web goal]', sample)
        # Stats REST fallback (modern)
        try:
            import requests
            r = requests.get(f"https://api.nhle.com/stats/rest/en/game/{int(gid)}/playbyplay", timeout=20)
            if r.ok:
                js = r.json() or {}
                plays = js.get('plays') or js.get('data') or []
                count = 0
                for ev in plays:
                    try:
                        # Common fields in REST: periodDescriptor.number, timeInPeriod, teamAbbrev
                        per = (ev.get('periodDescriptor') or {}).get('number') or ev.get('period')
                        clk = ev.get('timeInPeriod') or ev.get('periodTime')
                        tkey = (ev.get('teamAbbrev') or (ev.get('team') or {}).get('triCode') or (ev.get('team') or {}).get('abbreviation'))
                        t = (str(ev.get('typeDescKey') or ev.get('type') or '').lower())
                        if t == 'goal' and int(per or 0) == 1:
                            count += 1
                            print('[stats-rest goal]', {'period': per, 'timeInPeriod': clk, 'team': tkey})
                            if count >= 3:
                                break
                    except Exception:
                        continue
        except Exception as e:
            print('[stats-rest] error', e)


def main(argv):
    date = argv[1] if len(argv) > 1 else datetime.today().strftime('%Y-%m-%d')
    team_filter = argv[2] if len(argv) > 2 else None
    inspect(date, team_filter)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
