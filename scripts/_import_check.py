import sys
try:
    import nhl_betting.web.app as app  # noqa: F401
    sys.stdout.write("IMPORT_OK\n")
except Exception as e:
    sys.stdout.write(f"IMPORT_ERR: {e}\n")
    raise
