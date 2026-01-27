from nhl_betting.sim.pbp_engine import simulate_game

if __name__ == "__main__":
    totals = simulate_game("BOS", "NSH")
    for period in [1,2,3]:
        t = totals[period]
        print(f"Period {period}")
        print(f"  HOME: goals={t['HOME'].goals:.1f}, shots={t['HOME'].shots:.1f}, blocked={t['HOME'].blocked:.1f}, saves={t['HOME'].saves:.1f}")
        print(f"  AWAY: goals={t['AWAY'].goals:.1f}, shots={t['AWAY'].shots:.1f}, blocked={t['AWAY'].blocked:.1f}, saves={t['AWAY'].saves:.1f}")
