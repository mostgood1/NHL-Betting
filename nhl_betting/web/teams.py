from __future__ import annotations

# Minimal mapping of full team names to abbreviations and logo URLs (light theme)

_NAME_TO_ABBR = {
    "anaheim ducks": "ANA",
    "arizona coyotes": "ARI",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "carolina hurricanes": "CAR",
    "columbus blue jackets": "CBJ",
    "calgary flames": "CGY",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "montréal canadiens": "MTL",
    "new jersey devils": "NJD",
    "nashville predators": "NSH",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "san jose sharks": "SJS",
    "seattle kraken": "SEA",
    "st. louis blues": "STL",
    "st louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "winnipeg jets": "WPG",
    "washington capitals": "WSH",
}


def _norm(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_team_assets(name: str) -> dict:
    """Return dict with name, abbr, and logo URLs for the given team name.

    Uses NHL assets CDN with abbreviation-based SVGs.
    """
    abbr = _NAME_TO_ABBR.get(_norm(name))
    if not abbr:
        return {
            "name": name,
            "abbr": None,
            "logo_light": None,
            "logo_dark": None,
        }
    base = "https://assets.nhle.com/logos/nhl/svg"
    return {
        "name": name,
        "abbr": abbr,
        "logo_light": f"{base}/{abbr}_light.svg",
        "logo_dark": f"{base}/{abbr}_dark.svg",
    }
