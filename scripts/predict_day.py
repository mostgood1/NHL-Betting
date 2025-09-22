from nhl_betting.cli import predict_core

if __name__ == "__main__":
    # Example: generate predictions for a given date using the web source
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else "2024-03-01"
    source = sys.argv[2] if len(sys.argv) > 2 else "web"
    predict_core(date=date, source=source)
