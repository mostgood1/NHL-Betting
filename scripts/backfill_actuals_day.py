import sys
from nhl_betting.data.collect import collect_player_game_stats

def main(date: str):
    try:
        collect_player_game_stats(date, date, source='stats')
        print('[backfill] stats done')
    except Exception as e:
        print('[backfill] stats error:', e)
    try:
        collect_player_game_stats(date, date, source='web')
        print('[backfill] web done')
    except Exception as e:
        print('[backfill] web error:', e)

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else None
    if not d:
        print('Usage: backfill_actuals_day.py YYYY-MM-DD')
        sys.exit(2)
    main(d)
