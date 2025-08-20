from __future__ import annotations

import time
import statistics
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch


TensorOrNested = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]


def count_params(model: torch.nn.Module) -> int:
    """
    Count trainable parameters only.

    Args:
        model: A torch.nn.Module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Approximate model size in megabytes based on parameter tensors.

    Args:
        model: A torch.nn.Module.

    Returns:
        Size in megabytes (MB), computed as total parameter bytes / 1e6.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return float(total_bytes) / 1e6


def _as_args_kwargs(sample_inputs: TensorOrNested) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Normalize sample inputs to (args, kwargs)."""
    if isinstance(sample_inputs, torch.Tensor):
        return (sample_inputs,), {}
    if isinstance(sample_inputs, Mapping):
        return tuple(), dict(sample_inputs)
    if isinstance(sample_inputs, (list, tuple)):
        return tuple(sample_inputs), {}
    # Fallback: pass as single positional argument
    return (sample_inputs,), {}


def _first_tensor(sample_inputs: TensorOrNested) -> Optional[torch.Tensor]:
    if isinstance(sample_inputs, torch.Tensor):
        return sample_inputs
    if isinstance(sample_inputs, Mapping):
        for v in sample_inputs.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(sample_inputs, (list, tuple)):
        for v in sample_inputs:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    return None


def _slice_batch(sample_inputs: TensorOrNested, batch_start: int, batch_end: int) -> TensorOrNested:
    """Slice along the first (batch) dimension for all tensors within nested structures."""
    if isinstance(sample_inputs, torch.Tensor):
        return sample_inputs[batch_start:batch_end]
    if isinstance(sample_inputs, Mapping):
        # Preserve mapping type if mutable, otherwise fall back to dict
        out: MutableMapping[str, Any] = type(sample_inputs)() if isinstance(sample_inputs, MutableMapping) else {}
        for k, v in sample_inputs.items():
            out[k] = _slice_batch(v, batch_start, batch_end)
        return out
    if isinstance(sample_inputs, list):
        return [
            _slice_batch(v, batch_start, batch_end)
            for v in sample_inputs
        ]
    if isinstance(sample_inputs, tuple):
        return tuple(
            _slice_batch(v, batch_start, batch_end)
            for v in sample_inputs
        )
    # Unknown type, return as-is
    return sample_inputs


def _batch_size(sample_inputs: TensorOrNested) -> Optional[int]:
    t = _first_tensor(sample_inputs)
    return int(t.size(0)) if t is not None and t.dim() >= 1 else None


def _any_tensor_is_cuda(sample_inputs: TensorOrNested) -> bool:
    t = _first_tensor(sample_inputs)
    return bool(t is not None and t.is_cuda)


def try_flops(model: torch.nn.Module, sample_inputs: TensorOrNested) -> Optional[Dict[str, Any]]:
    """
    Attempt to compute FLOPs/MACs using available libraries.

    Tries THOP first, then fvcore. All imports are local and optional.

    Args:
        model: The model to analyze.
        sample_inputs: Example inputs to the model. Can be Tensor, tuple/list of Tensors, or dict of Tensors.

    Returns:
        A dict with FLOP statistics, or None if no supported library is available.
    """
    args, kwargs = _as_args_kwargs(sample_inputs)

    # Prefer eval/no-grad for profiling safety
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # 1) Try THOP
            try:
                from importlib import import_module

                thop = import_module("thop")
                profile = getattr(thop, "profile")

                # THOP accepts positional inputs only; if only kwargs are provided, try to pass as a single dict
                thop_inputs: Tuple[Any, ...]
                if args:
                    thop_inputs = args
                elif kwargs:
                    thop_inputs = (kwargs,)
                else:
                    thop_inputs = tuple()

                macs, params = profile(model, inputs=thop_inputs, verbose=False)
                flops = 2 * macs  # Common convention: FLOPs ≈ 2 * MACs for multiply-add
                return {
                    "library": "thop",
                    "flops": int(flops),
                    "macs": int(macs),
                    "params": int(params),
                }
            except Exception:
                pass

            # 2) Try fvcore
            try:
                from importlib import import_module

                fvcore_nn = import_module("fvcore.nn")
                FlopCountAnalysis = getattr(fvcore_nn, "FlopCountAnalysis")

                # fvcore supports args and kwargs
                if kwargs:
                    fca = FlopCountAnalysis(model, (args, kwargs))
                else:
                    fca = FlopCountAnalysis(model, args)
                total_flops = int(fca.total())
                return {
                    "library": "fvcore",
                    "flops": total_flops,
                    "macs": None,
                    "params": count_params(model),
                }
            except Exception:
                pass

    finally:
        if was_training:
            model.train()

    return None


def _invoke_forward(forward_fn: Any, sample_inputs: TensorOrNested) -> Any:
    args, kwargs = _as_args_kwargs(sample_inputs)
    return forward_fn(*args, **kwargs)


def _time_cuda(forward_fn: Any, sample_inputs: TensorOrNested, iters: int) -> Tuple[List[float], float, int]:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    per_iter_ms: List[float] = []
    total_samples = 0
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            starter.record()
            _ = _invoke_forward(forward_fn, sample_inputs)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
            per_iter_ms.append(float(elapsed_ms))
            bs = _batch_size(sample_inputs) or 1
            total_samples += bs
    total_time_sec = sum(per_iter_ms) / 1000.0
    return per_iter_ms, total_time_sec, total_samples


def _time_cpu(forward_fn: Any, sample_inputs: TensorOrNested, iters: int) -> Tuple[List[float], float, int]:
    per_iter_ms: List[float] = []
    total_samples = 0
    any_cuda = _any_tensor_is_cuda(sample_inputs)
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = _invoke_forward(forward_fn, sample_inputs)
            if any_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_iter_ms.append(float((t1 - t0) * 1000.0))
            bs = _batch_size(sample_inputs) or 1
            total_samples += bs
    total_time_sec = sum(per_iter_ms) / 1000.0
    return per_iter_ms, total_time_sec, total_samples


def _summarize_timings(per_iter_ms: List[float], batch_size: int) -> Dict[str, float]:
    if not per_iter_ms:
        return {"avg_ms": 0.0, "median_ms": 0.0, "std_ms": 0.0}
    per_sample_ms = [t / max(1, batch_size) for t in per_iter_ms]
    avg_ms = float(statistics.mean(per_sample_ms))
    median_ms = float(statistics.median(per_sample_ms))
    std_ms = float(statistics.pstdev(per_sample_ms)) if len(per_sample_ms) > 1 else 0.0
    return {"avg_ms": avg_ms, "median_ms": median_ms, "std_ms": std_ms}


def measure_latency(
    forward_fn: Any,
    sample_inputs: TensorOrNested,
    warmup: int = 20,
    iters: int = 100,
    use_cuda: bool = True,
) -> Dict[str, Any]:
    """
    Measure forward latency for batch size 1 and batch size N (if N > 1),
    returning per-sample latency stats and throughput.

    Args:
        forward_fn: Callable that computes forward pass (e.g., model.__call__ or model.forward).
        sample_inputs: Example inputs. Supports Tensor, tuple/list, or dict.
        warmup: Number of warm-up iterations.
        iters: Number of timed iterations.
        use_cuda: If True and CUDA is available, use CUDA events; otherwise use CPU timer.

    Returns:
        Dict with latency stats for batch size 1 and N, plus throughput.
    """
    bs = _batch_size(sample_inputs)
    if bs is None:
        raise ValueError("Could not infer batch size from sample_inputs; ensure inputs include at least one Tensor with a batch dimension.")

    device_mode = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"

    # Prepare inputs for B=1 and B=N
    inputs_b1 = _slice_batch(sample_inputs, 0, 1)
    inputs_bn = sample_inputs if bs > 1 else None

    # Warmup
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = _invoke_forward(forward_fn, inputs_b1)
            if device_mode == "cuda":
                torch.cuda.synchronize()
        if inputs_bn is not None:
            for _ in range(max(0, warmup)):
                _ = _invoke_forward(forward_fn, inputs_bn)
                if device_mode == "cuda":
                    torch.cuda.synchronize()

    # Timed runs
    if device_mode == "cuda":
        b1_times, b1_total_sec, b1_total_samples = _time_cuda(forward_fn, inputs_b1, iters)
        if inputs_bn is not None:
            bn_times, bn_total_sec, bn_total_samples = _time_cuda(forward_fn, inputs_bn, iters)
        else:
            bn_times, bn_total_sec, bn_total_samples = [], 0.0, 0
    else:
        b1_times, b1_total_sec, b1_total_samples = _time_cpu(forward_fn, inputs_b1, iters)
        if inputs_bn is not None:
            bn_times, bn_total_sec, bn_total_samples = _time_cpu(forward_fn, inputs_bn, iters)
        else:
            bn_times, bn_total_sec, bn_total_samples = [], 0.0, 0

    b1_stats = _summarize_timings(b1_times, batch_size=1)
    if bn_times:
        bn_stats = _summarize_timings(bn_times, batch_size=bs)
    else:
        bn_stats = None

    result: Dict[str, Any] = {
        "device": device_mode,
        "warmup": int(warmup),
        "iters": int(iters),
        "batch_size_1": {
            **b1_stats,
            "throughput_samples_per_s": float(b1_total_samples / b1_total_sec) if b1_total_sec > 0 else 0.0,
        },
    }

    if bn_stats is not None:
        result["batch_size_N"] = {
            **bn_stats,
            "batch_size": int(bs),
            "throughput_samples_per_s": float(bn_total_samples / bn_total_sec) if bn_total_sec > 0 else 0.0,
        }
    else:
        result["batch_size_N"] = None

    return result


def env_info() -> Dict[str, Any]:
    """
    Report environment information.

    Returns:
        Dict with GPU name (or "CPU"), torch/cuda/cuDNN versions, and precision used.
    """
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    torch_version = torch.__version__
    cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available() else None

    return {
        "device": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "precision": "fp32",
        "default_dtype": str(torch.get_default_dtype()),
    }



