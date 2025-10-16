import os
os.environ['ORT_LOGGING_LEVEL'] = '0'  # Verbose logging

import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
print()

# Check if provider DLL exists
import sys
capi_path = os.path.join(os.path.dirname(ort.__file__), 'capi')
qnn_dll = os.path.join(capi_path, 'onnxruntime_providers_qnn.dll')
print(f"QNN Provider DLL exists: {os.path.exists(qnn_dll)}")
if os.path.exists(qnn_dll):
    print(f"  Size: {os.path.getsize(qnn_dll):,} bytes")
print()

# List all DLLs in capi
import glob
print("All DLLs in capi folder:")
for dll in sorted(glob.glob(os.path.join(capi_path, '*.dll'))):
    print(f"  {os.path.basename(dll)}")
print()

# Try to explicitly load QNN provider
print("Attempting to register QNN provider...")
try:
    # This is what happens internally when you request QNNExecutionProvider
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0
    
    print("  Creating session with QNN provider...")
    # Dummy minimal model
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    
    # Create a minimal ONNX model
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    identity_node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([identity_node], 'test', [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    
    model_bytes = model.SerializeToString()
    
    # Try to create session with QNN
    sess = ort.InferenceSession(
        model_bytes,
        sess_options=session_options,
        providers=['QNNExecutionProvider', 'CPUExecutionProvider']
    )
    
    print(f"  ✓ Session created successfully!")
    print(f"  Active providers: {sess.get_providers()}")
    
except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
