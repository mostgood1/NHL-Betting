from nhl_betting.web.app import app
for r in app.routes:
    try:
        if 'projections' in r.path:
            print(r.path)
    except Exception:
        pass
