# NPU-Accelerated Modeling with Qualcomm Hardware

This document explains how to use your Qualcomm NPU (Neural Processing Unit) for accelerated NHL betting model inference.

## System Requirements

- **Processor**: Qualcomm Snapdragon X Elite/Plus (ARMv8 64-bit)
- **OS**: Windows 11 ARM64
- **Python**: 3.10+
- **ONNX Runtime**: 1.20+ with QNN execution provider

## Current Status

✅ **Detected**: Your system has a Qualcomm ARMv8 processor  
✅ **Installed**: PyTorch 2.5.1, ONNX 1.19.1, ONNX Runtime 1.23.1  
⚠️ **Missing**: QNNExecutionProvider (Qualcomm NPU driver)

**Available providers**: `['AzureExecutionProvider', 'CPUExecutionProvider']`

To enable NPU acceleration, you need to install the Qualcomm AI Engine Direct SDK.

## Installation Steps

### 1. Install Qualcomm AI Engine Direct SDK

The QNN execution provider for ONNX Runtime requires Qualcomm's NPU drivers:

1. **Download SDK**:
   - Visit: https://www.qualcomm.com/developer/software/neural-processing-sdk
   - Download "Qualcomm AI Engine Direct SDK for Windows on Snapdragon"
   - Requires registration (free)

2. **Install SDK**:
   ```powershell
   # Run the SDK installer
   # Follow prompts to install QNN libraries
   ```

3. **Install onnxruntime-qnn** (if available):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install onnxruntime-qnn
   ```
   
   If not available on PyPI, the SDK includes the necessary libraries.

4. **Set environment variables** (SDK installation should do this automatically):
   ```powershell
   $env:QNN_SDK_ROOT = "C:\Qualcomm\AIStack\QNN\<version>"
   $env:PATH += ";$env:QNN_SDK_ROOT\lib\aarch64-windows-msvc"
   ```

### 2. Verify NPU is Available

```powershell
.\.venv\Scripts\Activate.ps1
python -m nhl_betting.models.nn_inference
```

Expected output after SDK installation:
```
NPU Availability Check:
  available: True
  all_providers: ['QNNExecutionProvider', 'CPUExecutionProvider', ...]
  onnxruntime_version: 1.23.1
  recommendation: NPU is ready!
```

## Architecture Overview

### Current Models (Lightweight - CPU Optimized)
- **Elo ratings**: Pure arithmetic, no ML
- **Poisson stats**: scipy calculations
- **Props models**: Rolling window averages

### New Neural Network Models (NPU-Ready)
We've added PyTorch-based neural networks that can leverage your NPU:

1. **Player Props Prediction**:
   - Input: Recent game stats, team context, opponent strength
   - Architecture: Feedforward network (64→32→1)
   - Output: Poisson lambda parameter
   - Markets: SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS

2. **Inference Pipeline**:
   - Train in PyTorch on CPU
   - Export to ONNX format
   - Run inference on NPU via QNN provider
   - Automatic fallback to CPU if NPU unavailable

## Usage

### Training Models

Train a single market:
```powershell
.\.venv\Scripts\Activate.ps1

# Train shots on goal model
python -m nhl_betting.scripts.train_nn_props train --market SOG --epochs 50

# Train goals model
python -m nhl_betting.scripts.train_nn_props train --market GOALS --epochs 50
```

Train all markets:
```powershell
python -m nhl_betting.scripts.train_nn_props train-all --epochs 50
```

### Using NPU for Inference

Set environment variable to enable NPU:
```powershell
$env:USE_NPU = "1"

# Run props recommendations with NPU acceleration
python -m nhl_betting.cli props-recommendations --date 2025-10-16 --min-ev 0 --top 400
```

Without NPU (CPU fallback):
```powershell
$env:USE_NPU = "0"
python -m nhl_betting.cli props-recommendations --date 2025-10-16
```

### Benchmarking Performance

Compare CPU vs NPU inference speed:
```powershell
python -m nhl_betting.scripts.train_nn_props benchmark --market SOG --num-runs 1000
```

Expected output:
```
Benchmark Results:
==================================================

CPU:
  Avg time: 0.523 ms
  Throughput: 1912.0 inferences/sec
  Providers: ['CPUExecutionProvider']

NPU:
  Avg time: 0.089 ms
  Throughput: 11235.0 inferences/sec
  Providers: ['QNNExecutionProvider', 'CPUExecutionProvider']

Speedup (NPU vs CPU): 5.88x
✓ NPU is significantly faster!
```

## Integration with Daily Pipeline

Update your daily automation to use NPU models:

### Option 1: Environment Variable (Recommended)
```powershell
# In daily_update.ps1 or your scheduled task
$env:USE_NPU = "1"
.\daily_update.ps1 -DaysAhead 2
```

### Option 2: Modify daily_update.py
Add NPU-aware model selection in `nhl_betting/scripts/daily_update.py`:

```python
# At top of file
use_npu = os.getenv("USE_NPU", "0") == "1"

# In props collection/projection functions
if use_npu:
    # Use NN models with NPU acceleration
    from nhl_betting.models.nn_props import NNPropsModel
    shots_model = NNPropsModel("SOG")
else:
    # Use existing Poisson models
    from nhl_betting.models.props import SkaterShotsModel
    shots_model = SkaterShotsModel()
```

## Model Files

After training, models are saved in `data/models/nn_props/`:

```
data/models/nn_props/
├── sog_model.pt              # PyTorch weights
├── sog_model.onnx            # ONNX export (NPU-ready)
├── sog_metadata.npz          # Feature columns, normalization params
├── goals_model.pt
├── goals_model.onnx
├── goals_metadata.npz
└── ...
```

## Performance Expectations

### Small Models (Props Prediction)
- **CPU**: ~0.5ms per player
- **NPU**: ~0.1ms per player (5-10x speedup)
- **Batch inference** (multiple players): Even better NPU gains

### Benefits
1. **Faster daily updates**: Complete props projections in seconds instead of minutes
2. **Real-time recommendations**: Instant recalculation when lines change
3. **Richer features**: Can afford more complex models without timeout concerns
4. **Scalability**: Handle 1000+ player projections with ease

## Roadmap

### Phase 1 (Complete)
- ✅ PyTorch neural network models for props
- ✅ ONNX export pipeline
- ✅ NPU inference wrapper with QNN support
- ✅ Training and benchmarking scripts

### Phase 2 (Next Steps)
- [ ] Install Qualcomm AI Engine Direct SDK
- [ ] Verify QNN provider available
- [ ] Train models on full 2-year history
- [ ] Benchmark NPU vs CPU performance
- [ ] Update daily pipeline to use NPU models

### Phase 3 (Future Enhancements)
- [ ] Advanced features: linemate effects, TOI projections, opponent adjustments
- [ ] Ensemble models combining NN + Poisson
- [ ] Win probability prediction with situational awareness
- [ ] xG (expected goals) model using shot/game context
- [ ] Market-aware calibration networks

## Troubleshooting

### "QNNExecutionProvider not available"
- Install Qualcomm AI Engine Direct SDK (see Installation Steps)
- Verify PATH includes QNN library directory
- Check SDK documentation for your specific Windows ARM version

### "Import torch could not be resolved"
```powershell
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "ONNX model not found"
Train models first:
```powershell
python -m nhl_betting.scripts.train_nn_props train-all
```

### Models not improving over Poisson baselines
- Collect more historical data (need 2+ seasons)
- Add more features (opponent strength, TOI, linemates)
- Tune hyperparameters (learning rate, hidden dims, window size)

## References

- [Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/software/neural-processing-sdk)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [PyTorch to ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## Support

For issues with:
- **SDK installation**: Qualcomm developer forums
- **Model training**: Check `data/raw/player_game_stats.csv` exists and has sufficient history
- **NPU integration**: Run `python -m nhl_betting.models.nn_inference` to diagnose

---

**Next Action**: Install the Qualcomm AI Engine Direct SDK to enable NPU acceleration. Once installed, train your models and benchmark the performance gains!
