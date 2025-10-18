from nhl_betting.data.odds_api import OddsAPIClient

client = OddsAPIClient()
from_dt = '2025-10-17T00:00:00Z'
to_dt = '2025-10-17T23:59:59Z'

print("Fetching NHL events for 2025-10-17...")
evs, _ = client.list_events('icehockey_nhl', commence_from_iso=from_dt, commence_to_iso=to_dt)

print(f'\nTotal events found: {len(evs)}')
print('\nEvents:')
for e in evs:
    print(f"  {e.get('away_team')} @ {e.get('home_team')}")
    print(f"    Event ID: {e.get('id')}")
    print(f"    Commence: {e.get('commence_time')}")
    print()

# Now check if we can get props for each event
print("\nChecking props availability for each event:")
markets = "player_shots_on_goal,player_goals,player_assists,player_points"
bookmakers = "fanduel,draftkings,pinnacle"

for e in evs:
    ev_id = e.get('id')
    print(f"\n{e.get('away_team')} @ {e.get('home_team')} (ID: {ev_id}):")
    try:
        eo, _ = client.event_odds('icehockey_nhl', ev_id, markets=markets, regions='us', bookmakers=bookmakers)
        if eo and eo.get('bookmakers'):
            total_outcomes = sum(len(m.get('outcomes', [])) for bk in eo.get('bookmakers', []) for m in bk.get('markets', []))
            print(f"  ✓ Has props: {len(eo.get('bookmakers', []))} bookmakers, {total_outcomes} total outcomes")
        else:
            print(f"  ✗ No props available")
    except Exception as ex:
        print(f"  ✗ Error: {ex}")
