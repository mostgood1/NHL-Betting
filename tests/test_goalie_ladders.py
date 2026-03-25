from fastapi.testclient import TestClient

from nhl_betting.web.app import app


client = TestClient(app)


def test_goalie_ladders_api_returns_rows_for_real_histogram_date():
    response = client.get("/api/goalie-ladders", params={"date": "2026-03-24"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "2026-03-24"
    assert payload["prop"] == "saves"
    assert payload["found"] is True
    assert isinstance(payload["rows"], list)
    assert payload["rows"]

    first_row = payload["rows"][0]
    assert "goalieName" in first_row
    assert "ladder" in first_row
    assert isinstance(first_row["ladder"], list)


def test_goalie_ladders_page_renders_shell():
    response = client.get("/goalie-ladders", params={"date": "2026-03-24"})

    assert response.status_code == 200
    assert "Goalie Ladders" in response.text
    assert "/static/goalie_ladders.js" in response.text