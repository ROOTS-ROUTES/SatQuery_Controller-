"""
Diagnostic: compares the parameter names saved inside a LoRA adapter's
safetensors file against the parameter names the currently-installed
transformers/peft actually expect, to pinpoint exactly what changed if
pinning transformers==4.46.0 didn't resolve the "missing adapter keys"
warning.

Usage:
    venv\\Scripts\\python.exe diagnose_adapter_keys.py
"""

import os
from safetensors import safe_open

from config import BACKBONE_ADAPTER_DIR
from model_loader import find_adapter_dir

adapter_dir = find_adapter_dir(BACKBONE_ADAPTER_DIR)
weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")

print(f"Reading key names from: {weights_path}\n")
with safe_open(weights_path, framework="pt") as f:
    saved_keys = list(f.keys())

print(f"Adapter file contains {len(saved_keys)} tensors. First 10 keys:")
for k in saved_keys[:10]:
    print(f"  {k}")

print("\nNow loading the base model to compare against its real module names...")
import torch
from transformers import Qwen2VLForConditionalGeneration
from config import VISION_TOOL_BASE_MODEL_ID, VISION_TOOL_BASE_MODEL_LOCAL_DIR

model_source = VISION_TOOL_BASE_MODEL_LOCAL_DIR if os.path.isdir(VISION_TOOL_BASE_MODEL_LOCAL_DIR) else VISION_TOOL_BASE_MODEL_ID
model = Qwen2VLForConditionalGeneration.from_pretrained(model_source, torch_dtype=torch.float32, local_files_only=os.path.isdir(VISION_TOOL_BASE_MODEL_LOCAL_DIR))

live_module_names = {name for name, _ in model.named_modules()}

print(f"\nLive model has {len(live_module_names)} named modules. Checking overlap...")

# Strip the LoRA-specific suffixes/prefixes to get to the base module path
# for a rough comparison.
def normalize(key):
    key = key.replace("base_model.model.", "")
    key = key.replace(".lora_A.default.weight", "")
    key = key.replace(".lora_B.default.weight", "")
    return key

saved_module_paths = {normalize(k) for k in saved_keys}
matches = saved_module_paths & live_module_names
missing = saved_module_paths - live_module_names

print(f"\n{len(matches)} / {len(saved_module_paths)} adapter module paths matched a live module.")
if missing:
    print(f"\nFirst 10 UNMATCHED paths from the adapter file (these don't exist in the live model):")
    for m in list(missing)[:10]:
        print(f"  {m}")
    print("\nFirst 10 live module names containing 'visual' or 'lora'-target-like names, for comparison:")
    candidates = [n for n in live_module_names if "visual" in n or "q_proj" in n][:10]
    for c in candidates:
        print(f"  {c}")
else:
    print("\nAll adapter paths matched — the mismatch (if the warning persists) is elsewhere.")
