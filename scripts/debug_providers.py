import os, platform, json
try:
    import onnxruntime as ort
    ort_avail = True
except Exception as e:
    ort = None
    ort_avail = False
    print("[error] onnxruntime import failed:", e)

print("platform:", platform.system(), platform.release(), platform.machine())
print("python:", platform.python_version())
print("QNN_SDK_ROOT:", os.environ.get("QNN_SDK_ROOT"))

if ort_avail:
    try:
        print("onnxruntime:", ort.__version__)
        print("get_all_providers:", getattr(ort, 'get_all_providers', lambda: ['n/a'])())
        print("get_available_providers:", ort.get_available_providers())
    except Exception as e:
        print("[error] provider introspection failed:", e)

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from nhl_betting.models.nn_games import NNGameModel

print("-- constructing FIRST_10MIN model (will attempt ONNX load) --")
m = NNGameModel("FIRST_10MIN")
if m.onnx_session:
    try:
        print("session providers:", m.onnx_session.get_providers())
        print("inputs:", [i.name for i in m.onnx_session.get_inputs()])
        print("outputs:", [o.name for o in m.onnx_session.get_outputs()])
    except Exception as e:
        print("[warn] could not query session:", e)
else:
    print("onnx_session not created (PyTorch fallback? or no model files)")
