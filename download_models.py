"""
Run this ONCE, on a machine with internet access, before treating the
pendrive as offline. Downloads:
  1. The vision tool's base model (Qwen2-VL-2B-Instruct) — your fine-tuned
     adapters get merged onto this at runtime.
  2. Both brain LLM GGUF tiers (CPU and GPU) — hardware_detect.py picks
     between them at runtime; downloading both now means the pendrive works
     correctly regardless of which machine it ends up plugged into later.

After this has been run once, model_loader.py automatically detects these
local files and loads everything with local_files_only=True / from local
GGUF paths — no network access is attempted at runtime, ever.

Usage:
    pip install -r requirements.txt
    python download_models.py
"""

import os

from huggingface_hub import snapshot_download, hf_hub_download

from config import (
    VISION_TOOL_BASE_MODEL_ID,
    VISION_TOOL_BASE_MODEL_LOCAL_DIR,
    BRAIN_MODELS_DIR,
    BRAIN_MODEL_CPU_PATH,
    BRAIN_MODEL_CPU_REPO,
    BRAIN_MODEL_CPU_FILENAME,
    BRAIN_MODEL_GPU_PATH,
    BRAIN_MODEL_GPU_REPO,
    BRAIN_MODEL_GPU_FILENAME,
)


def download_vision_base_model():
    print(f"[1/3] Downloading vision tool base model: {VISION_TOOL_BASE_MODEL_ID}")
    print(f"       -> {VISION_TOOL_BASE_MODEL_LOCAL_DIR}")
    snapshot_download(
        repo_id=VISION_TOOL_BASE_MODEL_ID,
        local_dir=VISION_TOOL_BASE_MODEL_LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print("       Done.\n")


def download_brain_tier(step_label, repo_id, filename, dest_path):
    print(f"{step_label} Downloading brain LLM tier: {repo_id} ({filename})")
    print(f"       -> {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # hf_hub_download caches into the HF cache dir; copy the real file to
    # our expected local path so it's a self-contained, portable file.
    import shutil
    shutil.copy(downloaded_path, dest_path)
    print("       Done.\n")


def main():
    print("Downloading all models needed for fully offline operation.")
    print("This requires internet access and may take a while (a few GB total).")
    print("This is the ONLY step in the whole project that needs internet —")
    print("everything after this runs fully offline.\n")

    download_vision_base_model()
    download_brain_tier("[2/3]", BRAIN_MODEL_CPU_REPO, BRAIN_MODEL_CPU_FILENAME, BRAIN_MODEL_CPU_PATH)
    download_brain_tier("[3/3]", BRAIN_MODEL_GPU_REPO, BRAIN_MODEL_GPU_FILENAME, BRAIN_MODEL_GPU_PATH)

    print("All models downloaded. This entire folder is now self-contained —")
    print("copy it to the pendrive and it will run fully offline on any machine.")


if __name__ == "__main__":
    main()
