from scripts.backfill_live_lens_signal_timing import enrich_signal_snapshots


def test_enrich_signal_snapshots_backfills_goal_timing_from_score_changes():
    rows = [
        {
            "asof_utc": "2026-04-19T20:00:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 0, "away": 0},
                    "guidance": {"elapsed_min": 10.0},
                    "signals": [{"market": "TOTAL", "driver_meta": {}}],
                }
            ],
        },
        {
            "asof_utc": "2026-04-19T20:01:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 1, "away": 0},
                    "guidance": {"elapsed_min": 11.0},
                    "signals": [{"market": "TOTAL", "driver_meta": {}}],
                }
            ],
        },
        {
            "asof_utc": "2026-04-19T20:03:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 1, "away": 0},
                    "guidance": {"elapsed_min": 13.0},
                    "signals": [{"market": "TOTAL", "driver_meta": {}}],
                }
            ],
        },
    ]

    out = enrich_signal_snapshots(rows)

    first_meta = out[0]["games"][0]["signals"][0]["driver_meta"]
    second_meta = out[1]["games"][0]["signals"][0]["driver_meta"]
    third_meta = out[2]["games"][0]["signals"][0]["driver_meta"]

    assert first_meta["score_state_age_sec"] == 600
    assert "time_since_last_goal_sec" not in first_meta
    assert second_meta["last_goal_team"] == "home"
    assert second_meta["time_since_last_goal_sec"] == 0
    assert second_meta["score_state_age_sec"] == 0
    assert third_meta["last_goal_team"] == "home"
    assert third_meta["time_since_last_goal_sec"] == 120
    assert third_meta["score_state_age_sec"] == 120


def test_enrich_signal_snapshots_uses_signal_elapsed_when_guidance_missing():
    rows = [
        {
            "asof_utc": "2026-04-19T20:00:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 0, "away": 0},
                    "signals": [{"market": "TOTAL", "elapsed_min": 12.0, "driver_meta": {}}],
                }
            ],
        }
    ]

    out = enrich_signal_snapshots(rows)
    meta = out[0]["games"][0]["signals"][0]["driver_meta"]

    assert meta["score_state_age_sec"] == 720


def test_enrich_signal_snapshots_marks_double_score_jump_as_both():
    rows = [
        {
            "asof_utc": "2026-04-19T20:00:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 1, "away": 0},
                    "guidance": {"elapsed_min": 15.0},
                    "signals": [{"market": "ML", "driver_meta": {}}],
                }
            ],
        },
        {
            "asof_utc": "2026-04-19T20:02:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 2, "away": 1},
                    "guidance": {"elapsed_min": 17.0},
                    "signals": [{"market": "ML", "driver_meta": {}}],
                }
            ],
        },
    ]

    out = enrich_signal_snapshots(rows)
    meta = out[1]["games"][0]["signals"][0]["driver_meta"]

    assert meta["last_goal_team"] == "both"
    assert meta["time_since_last_goal_sec"] == 0
    assert meta["score_state_age_sec"] == 0


def test_enrich_signal_snapshots_backfills_pp_state_age_from_saved_pp_context():
    rows = [
        {
            "asof_utc": "2026-04-19T20:00:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 0, "away": 0},
                    "signals": [{"market": "TOTAL", "driver_meta": {"pp_team": "home", "pp_sec_remaining_est": 45}}],
                }
            ],
        }
    ]

    out = enrich_signal_snapshots(rows)
    meta = out[0]["games"][0]["signals"][0]["driver_meta"]

    assert meta["pp_state_age_sec"] == 75


def test_enrich_signal_snapshots_can_fill_missing_flow_meta_from_matching_state_guidance():
    rows = [
        {
            "asof_utc": "2026-04-19T20:00:00+00:00",
            "games": [
                {
                    "gamePk": 2025030151,
                    "score": {"home": 0, "away": 0},
                    "signals": [{"market": "PERIOD_TOTAL", "driver_meta": {}}],
                }
            ],
        }
    ]

    out = enrich_signal_snapshots(
        rows,
        state_guidance_by_key={
            (2025030151, "2026-04-19T20:00:00+00:00"): {
                "pp_team": "away",
                "pp_sec_remaining_est": 44,
                "home_empty_net": False,
                "away_empty_net": True,
                "late_state_mode": "multi_goal",
                "projection_driver_tags": ["pace:down", "pressure:away", "late:multi_goal"],
            }
        },
    )
    meta = out[0]["games"][0]["signals"][0]["driver_meta"]

    assert meta["pp_team"] == "away"
    assert meta["pp_sec_remaining_est"] == 44
    assert meta["pp_state_age_sec"] == 76
    assert meta["home_empty_net"] is False
    assert meta["away_empty_net"] is True
    assert meta["late_state_mode"] == "multi_goal"
    assert meta["trigger_tags"] == ["pace:down", "pressure:away", "late:multi_goal"]