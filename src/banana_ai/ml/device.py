"""Device selection and CPU configuration for BananaAI."""

import os
import platform
from typing import Any

import torch


def _xpu_backend():
    return getattr(torch, "xpu", None)


def _xpu_available() -> bool:
    backend = _xpu_backend()
    if backend is None or not callable(getattr(backend, "is_available", None)):
        return False
    try:
        return bool(backend.is_available())
    except (AttributeError, RuntimeError):
        return False


def get_device() -> torch.device:
    """Select CUDA, supported XPU, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _xpu_available():
        return torch.device("xpu")
    return torch.device("cpu")


def get_device_name(device: torch.device | None = None) -> str:
    selected = device or get_device()
    if selected.type == "cuda":
        return torch.cuda.get_device_name(selected)
    if selected.type == "xpu":
        backend = _xpu_backend()
        return str(backend.get_device_name(selected)) if backend else "Intel XPU"
    return platform.processor() or "CPU"


def configure_cpu_threads() -> int:
    """Set CPU threads from the environment or a conservative hardware default."""
    configured = os.getenv("BANANA_TORCH_THREADS")
    if configured:
        try:
            thread_count = int(configured)
        except ValueError as exc:
            raise ValueError("BANANA_TORCH_THREADS must be a positive integer") from exc
        if thread_count < 1:
            raise ValueError("BANANA_TORCH_THREADS must be a positive integer")
    else:
        logical_processors = os.cpu_count() or 1
        thread_count = min(4, logical_processors)
    torch.set_num_threads(thread_count)
    return thread_count


def amp_supported(device: torch.device | None = None) -> bool:
    """Report whether AMP is available on the selected accelerator.

    V1 deliberately does not enable AMP; this is only capability reporting.
    """
    selected = device or get_device()
    return selected.type in {"cuda", "xpu"} and hasattr(torch, "amp")


def get_device_report() -> dict[str, Any]:
    device = get_device()
    xpu_available = _xpu_available()
    xpu_name = "Unavailable"
    if xpu_available:
        xpu_name = get_device_name(torch.device("xpu"))
    return {
        "pytorch_version": torch.__version__,
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable",
        "xpu_available": xpu_available,
        "xpu_device": xpu_name,
        "cpu": platform.processor() or "CPU",
        "physical_cores": None,
        "logical_processors": os.cpu_count() or 1,
        "torch_cpu_threads": torch.get_num_threads(),
        "amp_supported": amp_supported(device),
    }


def main() -> None:
    configure_cpu_threads()
    report = get_device_report()
    print("# BANANA AI DEVICE REPORT")
    labels = {
        "pytorch_version": "PyTorch version",
        "selected_device": "Selected device",
        "cuda_available": "CUDA available",
        "cuda_version": "CUDA version",
        "cuda_gpu": "CUDA GPU",
        "xpu_available": "XPU available",
        "xpu_device": "XPU device",
        "cpu": "CPU",
        "physical_cores": "Physical cores",
        "logical_processors": "Logical processors",
        "torch_cpu_threads": "Torch CPU threads",
        "amp_supported": "AMP supported",
    }
    for key, label in labels.items():
        print(f"{label}: {report[key]}")


if __name__ == "__main__":
    main()
