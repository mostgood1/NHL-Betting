import torch
import torch.onnx
import numpy as np
from pathlib import Path

# Load the period_goals model
model_dir = Path('data/models/nn_games')
model_path = model_dir / 'period_goals_model.pt'
onnx_path = model_dir / 'period_goals_model.onnx'

print(f"Loading model from {model_path}...")
# Load state dict and reconstruct model
from nhl_betting.models.nn_games import NNGameModel
model = NNGameModel(input_dim=95, hidden_dims=[128, 64, 32], dropout=0.3, output_type='poisson')
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# Create dummy input (95 features based on the model architecture)
dummy_input = torch.randn(1, 95)

print(f"Exporting to ONNX: {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"✓ Exported to {onnx_path}")
print(f"Size: {onnx_path.stat().st_size} bytes")
