import os
from pathlib import Path
import platform

print("platform:", platform.system(), platform.release(), platform.machine())
print("QNN_SDK_ROOT:", os.environ.get("QNN_SDK_ROOT"))

try:
    import onnxruntime as ort
except Exception as e:
    print("[error] onnxruntime import failed:", e)
    raise

print("onnxruntime:", ort.__version__)
print("available_providers:", ort.get_available_providers())

onnx_path = Path('data/models/nn_games/first_10min_model.onnx')
if not onnx_path.exists():
    print("[error] ONNX model not found:", onnx_path)
    raise SystemExit(1)

providers = []
avail = set(ort.get_available_providers())
qnn_root = os.environ.get("QNN_SDK_ROOT", "")
backend_path = None
if qnn_root and "QNNExecutionProvider" in avail:
    # Let ORT resolve QNN via environment; no explicit backend_path first
    providers.append("QNNExecutionProvider")

# Optional: DirectML if available
if "DmlExecutionProvider" in avail:
    providers.append("DmlExecutionProvider")

providers.append("CPUExecutionProvider")

print("trying providers:", providers)

sess = ort.InferenceSession(str(onnx_path), providers=providers)
print("session providers:", sess.get_providers())
print("inputs:", [i.name for i in sess.get_inputs()])
print("outputs:", [o.name for o in sess.get_outputs()])
