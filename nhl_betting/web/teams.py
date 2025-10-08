from __future__ import annotations

# Minimal mapping of full team names to abbreviations and logo URLs (light theme)

_NAME_TO_ABBR = {
    "anaheim ducks": "ANA",
    "utah mammoth": "UTA",
    "utah hockey club": "UTA",
    "utah hc": "UTA",
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
    """Return dict with name, abbr, and logo URLs for the given team name or abbreviation.

    Accepts either a full team name (e.g., "Florida Panthers") or a 2-3 letter
    abbreviation (e.g., "FLA"). Uses NHL assets CDN with abbreviation-based SVGs.
    """
    if name is None:
        return {
            "name": None,
            "abbr": None,
            "logo_light": None,
            "logo_dark": None,
            "division": None,
            "conference": None,
        }
    # First try full-name mapping
    abbr = _NAME_TO_ABBR.get(_norm(str(name)))
    # If not found, try direct abbreviation pass-through
    if not abbr:
        raw = str(name).strip().upper()
        # Some feeds provide 2- or 3-letter abbreviations; accept if known
        _ALL_ABBRS = set(_NAME_TO_ABBR.values())
        if raw in _ALL_ABBRS or (len(raw) in (2, 3) and raw.isalpha()):
            abbr = raw
    if not abbr:
        return {
            "name": name,
            "abbr": None,
            "logo_light": None,
            "logo_dark": None,
            "division": None,
            "conference": None,
        }
    base = "https://assets.nhle.com/logos/nhl/svg"
    # Minimal static mapping for division/conference (2024-25 structure)
    _DIV_CONF = {
        # Eastern Conference
        "BOS": ("Atlantic", "East"),
        "BUF": ("Atlantic", "East"),
        "DET": ("Atlantic", "East"),
        "FLA": ("Atlantic", "East"),
        "MTL": ("Atlantic", "East"),
        "OTT": ("Atlantic", "East"),
        "TBL": ("Atlantic", "East"),
        "TOR": ("Atlantic", "East"),
        "CAR": ("Metropolitan", "East"),
        "CBJ": ("Metropolitan", "East"),
        "NJD": ("Metropolitan", "East"),
        "NYI": ("Metropolitan", "East"),
        "NYR": ("Metropolitan", "East"),
        "PHI": ("Metropolitan", "East"),
        "PIT": ("Metropolitan", "East"),
        "WSH": ("Metropolitan", "East"),
        # Western Conference
        "ANA": ("Pacific", "West"),
        "CGY": ("Pacific", "West"),
        "EDM": ("Pacific", "West"),
        "LAK": ("Pacific", "West"),
        "SJS": ("Pacific", "West"),
        "SEA": ("Pacific", "West"),
        "VAN": ("Pacific", "West"),
        "VGK": ("Pacific", "West"),
        "CHI": ("Central", "West"),
        "COL": ("Central", "West"),
        "DAL": ("Central", "West"),
        "MIN": ("Central", "West"),
        "NSH": ("Central", "West"),
        "STL": ("Central", "West"),
        "WPG": ("Central", "West"),
        # Arizona historical; Utah current in Central/West
        "ARI": ("Central", "West"),
        "UTA": ("Central", "West"),
        # Additional aliases can be added as needed
    }
    div, conf = _DIV_CONF.get(abbr, (None, None))
    return {
        "name": name,
        "abbr": abbr,
        "logo_light": f"{base}/{abbr}_light.svg",
        "logo_dark": f"{base}/{abbr}_dark.svg",
        "division": div,
        "conference": conf,
    }
