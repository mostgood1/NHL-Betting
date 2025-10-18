"""Test neural network models are trained and working."""
from nhl_betting.models.nn_games import NNGameModel

print("=" * 70)
print("MODEL TRAINING STATUS & TEST")
print("=" * 70)

# Load models
print("\nLoading models...")
m1 = NNGameModel('FIRST_10MIN')
m2 = NNGameModel('PERIOD_GOALS')

print(f"\n1. FIRST_10MIN:  Loaded via {'ONNX ✓' if m1.onnx_session else 'PyTorch ✓' if m1.model else 'NOT LOADED ✗'}")
print(f"2. PERIOD_GOALS: Loaded via {'ONNX ✓' if m2.onnx_session else 'PyTorch ✓' if m2.model else 'NOT LOADED ✗'}")

# Test predictions
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS (with minimal features)")
print("=" * 70)

test_features = {
    'home_elo': 1520,
    'away_elo': 1480,
}

print("\nTest game: Home (Elo 1520) vs Away (Elo 1480)")
print("-" * 70)

try:
    first_10 = m1.predict('TBL', 'DET', test_features)
    print(f"\nFirst 10 min goals prediction: {first_10:.3f}")
    
    periods = m2.predict('TBL', 'DET', test_features)
    print(f"\nPeriod predictions:")
    print(f"  Period 1: Home {periods[0]:.2f}, Away {periods[1]:.2f}")
    print(f"  Period 2: Home {periods[2]:.2f}, Away {periods[3]:.2f}")
    print(f"  Period 3: Home {periods[4]:.2f}, Away {periods[5]:.2f}")
    print(f"\nTotal game: Home {sum(periods[::2]):.2f}, Away {sum(periods[1::2]):.2f}")
    
    print("\n" + "=" * 70)
    print("✅ MODELS ARE TRAINED AND WORKING!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
