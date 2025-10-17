"""List all routes in the FastAPI app."""
from nhl_betting.web.app import app

print("=" * 80)
print("NHL BETTING WEB APP - ROUTES")
print("=" * 80)

# Group routes by category
html_routes = []
api_routes = []
static_routes = []

for route in app.routes:
    try:
        path = route.path
        methods = getattr(route, 'methods', set())
        
        if '/static' in path:
            static_routes.append((path, methods))
        elif '/api/' in path:
            api_routes.append((path, methods))
        else:
            html_routes.append((path, methods))
    except Exception:
        pass

print("\n📄 HTML PAGES:")
print("-" * 80)
for path, methods in sorted(html_routes):
    print(f"  {path:50} {', '.join(sorted(methods))}")

print("\n🔌 API ENDPOINTS:")
print("-" * 80)
for path, methods in sorted(api_routes):
    print(f"  {path:50} {', '.join(sorted(methods))}")

print("\n📦 STATIC FILES:")
print("-" * 80)
for path, methods in sorted(static_routes):
    print(f"  {path:50} {', '.join(sorted(methods))}")

print(f"\n✅ Total routes: {len(html_routes) + len(api_routes) + len(static_routes)}")
print("=" * 80)
