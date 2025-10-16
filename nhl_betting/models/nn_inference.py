"""NPU-accelerated inference using ONNX Runtime with Qualcomm QNN execution provider.

This module provides a wrapper for running ONNX models on Qualcomm NPU hardware
via the QNN (Qualcomm Neural Network) execution provider, with automatic fallback
to CPU if NPU is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

# ONNX Runtime will be installed separately
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class NPUInferenceSession:
    """Wrapper for ONNX Runtime inference with NPU acceleration.
    
    Automatically selects Qualcomm QNN execution provider if available,
    otherwise falls back to CPU.
    
    Usage:
        session = NPUInferenceSession("model.onnx", use_npu=True)
        outputs = session.run({"features": input_array})
    """
    
    def __init__(
        self,
        onnx_path: str | Path,
        use_npu: bool | None = None,
        verbose: bool = False,
    ):
        """Initialize inference session.
        
        Args:
            onnx_path: Path to ONNX model file
            use_npu: If True, try to use NPU; if False, use CPU; if None, auto-detect from env
            verbose: Print provider info
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        # Determine whether to use NPU
        if use_npu is None:
            use_npu = os.getenv("USE_NPU", "0") == "1"
        
        self.use_npu = use_npu
        self.verbose = verbose
        
        # Create session with appropriate providers
        self.session = self._create_session()
        
        # Cache input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        if self.verbose:
            print(f"[NPU] Session created with providers: {self.session.get_providers()}")
            print(f"[NPU] Inputs: {self.input_names}")
            print(f"[NPU] Outputs: {self.output_names}")
    
    def _create_session(self) -> Any:
        """Create ONNX Runtime session with optimal execution providers."""
        providers = self._get_providers()
        
        # Session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Attempt to create session with preferred providers
        for provider_list in providers:
            try:
                session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=sess_options,
                    providers=provider_list,
                )
                if self.verbose:
                    actual = session.get_providers()
                    print(f"[NPU] Created session with: {actual}")
                return session
            except Exception as e:
                if self.verbose:
                    print(f"[NPU] Failed to create session with {provider_list}: {e}")
                continue
        
        # Final fallback: CPU only
        return ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    
    def _get_providers(self) -> List[List[str]]:
        """Get ordered list of execution provider preferences.
        
        Returns list of provider lists to try in order:
        1. QNN (Qualcomm NPU) if requested
        2. CPU as fallback
        """
        available = ort.get_available_providers()
        
        if self.verbose:
            print(f"[NPU] Available providers: {available}")
        
        providers_to_try = []
        
        if self.use_npu:
            # Try QNN provider first (Qualcomm NPU)
            if "QNNExecutionProvider" in available:
                providers_to_try.append(["QNNExecutionProvider", "CPUExecutionProvider"])
            else:
                if self.verbose:
                    print(
                        "[NPU] QNNExecutionProvider not available. "
                        "Install onnxruntime-qnn or Qualcomm AI Engine Direct SDK."
                    )
        
        # Always include CPU fallback
        providers_to_try.append(["CPUExecutionProvider"])
        
        return providers_to_try
    
    def run(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference on the model.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            output_names: Optional list of output names to return (default: all)
        
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if output_names is None:
            output_names = self.output_names
        
        # Run inference
        outputs = self.session.run(output_names, inputs)
        
        # Return as dictionary
        return {name: arr for name, arr in zip(output_names, outputs)}
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the inference device."""
        providers = self.session.get_providers()
        is_npu = "QNNExecutionProvider" in providers
        
        return {
            "providers": providers,
            "using_npu": is_npu,
            "onnx_path": str(self.onnx_path),
            "input_names": self.input_names,
            "output_names": self.output_names,
        }


def benchmark_inference(
    onnx_path: str | Path,
    input_shape: tuple,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """Benchmark NPU vs CPU inference performance.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Shape of input tensor (e.g., (1, 10) for batch=1, features=10)
        num_runs: Number of inference runs to average
        warmup_runs: Number of warmup runs before timing
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {}
    
    # Generate random input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    for use_npu, label in [(False, "CPU"), (True, "NPU")]:
        try:
            session = NPUInferenceSession(onnx_path, use_npu=use_npu, verbose=False)
            input_name = session.input_names[0]
            
            # Warmup
            for _ in range(warmup_runs):
                session.run({input_name: dummy_input})
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_runs):
                session.run({input_name: dummy_input})
            elapsed = time.perf_counter() - start
            
            avg_time_ms = (elapsed / num_runs) * 1000
            throughput = num_runs / elapsed
            
            results[label] = {
                "avg_time_ms": avg_time_ms,
                "throughput_per_sec": throughput,
                "providers": session.session.get_providers(),
            }
            
        except Exception as e:
            results[label] = {"error": str(e)}
    
    # Compute speedup
    if "CPU" in results and "NPU" in results:
        if "avg_time_ms" in results["CPU"] and "avg_time_ms" in results["NPU"]:
            results["speedup"] = results["CPU"]["avg_time_ms"] / results["NPU"]["avg_time_ms"]
    
    return results


def check_npu_availability() -> Dict[str, Any]:
    """Check if Qualcomm NPU is available for inference.
    
    Returns:
        Dictionary with NPU availability info
    """
    if not ONNX_AVAILABLE:
        return {
            "available": False,
            "reason": "onnxruntime not installed",
        }
    
    providers = ort.get_available_providers()
    has_qnn = "QNNExecutionProvider" in providers
    
    return {
        "available": has_qnn,
        "all_providers": providers,
        "onnxruntime_version": ort.__version__,
        "recommendation": (
            "NPU is ready!" if has_qnn else
            "Install onnxruntime-qnn or Qualcomm AI Engine Direct SDK for NPU support"
        ),
    }


if __name__ == "__main__":
    # Quick check
    info = check_npu_availability()
    print("NPU Availability Check:")
    for k, v in info.items():
        print(f"  {k}: {v}")
