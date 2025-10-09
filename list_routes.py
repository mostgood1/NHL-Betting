from nhl_betting.web.app import app; print('\n'.join([r.path for r in app.routes if 'projections' in r.path and 'history' in r.path]))
