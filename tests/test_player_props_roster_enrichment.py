import pandas as pd

import nhl_betting.data.player_props as player_props


def test_load_cached_roster_enrichment_merges_dated_team_with_master_player_id(tmp_path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    (proc_dir / "roster_2026-03-16.csv").write_text(
        "player,position,team\n"
        "Brent Burns,F,COL\n"
        "David Pastrnak,F,BOS\n",
        encoding="utf-8",
    )
    (proc_dir / "roster_master.csv").write_text(
        "player_id,full_name,team_abbr\n"
        "8470613,Brent Burns,COL\n"
        "8477956,David Pastrnak,BOS\n",
        encoding="utf-8",
    )

    roster = player_props._load_cached_roster_enrichment("2026-03-16", proc_dir=proc_dir)

    assert roster is not None
    roster = roster.set_index("full_name")
    assert roster.loc["Brent Burns", "player_id"] == 8470613
    assert roster.loc["Brent Burns", "team"] == "COL"
    assert roster.loc["David Pastrnak", "player_id"] == 8477956
    assert roster.loc["David Pastrnak", "team"] == "BOS"


def test_normalize_player_names_fills_team_from_player_id_variant_match():
    raw = pd.DataFrame(
        [
            {
                "player": "JG Pageau",
                "market": "SOG",
                "line": 1.5,
                "odds": -120,
                "side": "OVER",
                "book": "draftkings",
                "date": "2026-03-16",
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {"full_name": "Jean-Gabriel Pageau", "player_id": 8476476, "team": "NYI"},
        ]
    )

    norm = player_props.normalize_player_names(raw, roster)

    assert norm.loc[0, "player_id"] == 8476476
    assert norm.loc[0, "team"] == "NYI"


def test_build_roster_enrichment_preserves_existing_team_when_team_id_missing(monkeypatch):
    monkeypatch.setattr(
        player_props._rosters,
        "build_all_team_roster_snapshots",
        lambda: pd.DataFrame(
            [
                {"full_name": "Brent Burns", "player_id": 8470613, "team": "COL", "position": "F", "team_id": None},
            ]
        ),
    )
    monkeypatch.setattr(player_props._rosters, "list_teams", lambda: [{"id": None, "abbreviation": "COL", "name": "COL"}])

    roster = player_props._build_roster_enrichment()

    assert roster.loc[0, "team"] == "COL"