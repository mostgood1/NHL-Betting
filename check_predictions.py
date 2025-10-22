import pandas as pd

df = pd.read_csv('data/processed/predictions_2025-10-17.csv')

print('Period predictions check:\n')
for idx, row in df.iterrows():
    print(f"{row['away']} @ {row['home']}:")
    print(f"  Period 1: Away {row['period1_away_proj']:.2f} | Home {row['period1_home_proj']:.2f}")
    print(f"  Period 2: Away {row['period2_away_proj']:.2f} | Home {row['period2_home_proj']:.2f}")
    print(f"  Period 3: Away {row['period3_away_proj']:.2f} | Home {row['period3_home_proj']:.2f}")
    # Show both lambda and probability if available
    lam = row.get('first_10min_proj')
    prob = row.get('first_10min_prob')
    if pd.notna(prob):
        print(f"  First 10min: lambda={lam:.2f} | prob={prob:.1%}")
    else:
        # fallback for older files: approximate prob from lambda
        if pd.notna(lam):
            import math
            print(f"  First 10min: lambda={lam:.2f} | prob~={(1-math.exp(-lam)):.1%}")
        else:
            print("  First 10min: n/a")
    print()
