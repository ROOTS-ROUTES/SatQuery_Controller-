"""
Hardware auto-detection.

This is what runs the moment the pendrive is plugged into a new machine:
it inspects the host's RAM and GPU, and decides which of the two
supported inference paths to use — CPU or GPU. Nothing here loads any
model weights; it only produces a small profile dict that model_loader
uses to decide how to load things.
"""

import subprocess
from dataclasses import dataclass, asdict

import psutil

from config import MIN_VRAM_GB_FOR_GPU_PATH


@dataclass
class HardwareProfile:
    accelerator: str          # "gpu" or "cpu" — the decision this module exists to make
    total_ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    compute_capability: tuple | None  # e.g. (8, 6) for an RTX 3050
    reason: str                # human-readable justification, goes into the execution trace

    def to_dict(self):
        return asdict(self)


def _detect_gpu():
    """
    Returns (available, name, vram_gb, compute_capability) using torch if
    it's importable with CUDA support; falls back to nvidia-smi parsing
    if torch isn't available yet at detection time.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            capability = torch.cuda.get_device_capability(0)
            return True, name, vram_gb, capability
    except ImportError:
        pass

    # Fallback: query nvidia-smi directly (works even before torch is installed).
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if output:
            name, mem_mb = output.split(",")[:2]
            return True, name.strip(), float(mem_mb) / 1024, None
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return False, None, None, None


def detect_hardware() -> HardwareProfile:
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_available, gpu_name, gpu_vram_gb, compute_capability = _detect_gpu()

    if gpu_available and gpu_vram_gb is not None and gpu_vram_gb >= MIN_VRAM_GB_FOR_GPU_PATH:
        return HardwareProfile(
            accelerator="gpu",
            total_ram_gb=round(total_ram_gb, 1),
            gpu_available=True,
            gpu_name=gpu_name,
            gpu_vram_gb=round(gpu_vram_gb, 1),
            compute_capability=compute_capability,
            reason=(
                f"NVIDIA GPU '{gpu_name}' detected with {gpu_vram_gb:.1f} GB VRAM "
                f"(>= {MIN_VRAM_GB_FOR_GPU_PATH} GB threshold) — routing to GPU path."
            ),
        )

    if gpu_available:
        reason = (
            f"NVIDIA GPU '{gpu_name}' detected but only {gpu_vram_gb:.1f} GB VRAM "
            f"(< {MIN_VRAM_GB_FOR_GPU_PATH} GB threshold) — routing to CPU path for safety."
        )
    else:
        reason = "No NVIDIA GPU detected — routing to CPU path."

    return HardwareProfile(
        accelerator="cpu",
        total_ram_gb=round(total_ram_gb, 1),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram_gb, 1) if gpu_vram_gb else None,
        compute_capability=compute_capability,
        reason=reason,
    )


if __name__ == "__main__":
    profile = detect_hardware()
    print("Detected hardware profile:")
    for k, v in profile.to_dict().items():
        print(f"  {k}: {v}")
