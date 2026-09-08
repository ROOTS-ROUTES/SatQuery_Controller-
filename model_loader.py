"""
Loads two separate things:

1. The BRAIN — a small general-purpose instruction-following LLM (GGUF via
   llama.cpp), picked by tier (CPU/GPU) from hardware_detect's decision.
   This is what reasons about the query and decides whether to call the
   vision tool. It never sees images directly.

2. The VISION TOOL — your fine-tuned Qwen2-VL-2B, with the backbone and
   task LoRA adapters merged onto it. This is what the brain calls when it
   needs to actually look at an image. Loading logic here (4-bit GPU vs
   fp32+int8 CPU) is the same as before this architecture change — only its
   role changed, from "the whole system" to "a tool the brain calls."
"""

import os
import glob

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from llama_cpp import Llama

from config import (
    VISION_TOOL_BASE_MODEL_ID,
    VISION_TOOL_BASE_MODEL_LOCAL_DIR,
    BACKBONE_ADAPTER_DIR,
    ADAPTER_REGISTRY,
    MIN_COMPUTE_CAPABILITY_FOR_BF16,
    BRAIN_MODEL_CPU_PATH,
    BRAIN_MODEL_GPU_PATH,
    BRAIN_CONTEXT_WINDOW,
    BRAIN_N_GPU_LAYERS_WHEN_GPU,
)
from hardware_detect import HardwareProfile


# ============================== Brain LLM ==============================

def load_brain_llm(hw: HardwareProfile):
    """
    Returns (llama_cpp.Llama instance, trace_dict). Tries the GPU-tier
    model with GPU offload first if hw says GPU; falls back to the CPU-tier
    model (still with GPU offload if some VRAM is free, else pure CPU) if
    the GPU-tier model fails to fit — a 4GB card running the GPU-tier brain
    *and* the vision tool simultaneously is a genuinely tight fit, so this
    fallback matters in practice, not just in theory.
    """
    if hw.accelerator == "gpu" and os.path.exists(BRAIN_MODEL_GPU_PATH):
        try:
            print(f"  Loading GPU-tier brain: {os.path.basename(BRAIN_MODEL_GPU_PATH)}")
            llm = Llama(
                model_path=BRAIN_MODEL_GPU_PATH,
                n_ctx=BRAIN_CONTEXT_WINDOW,
                n_gpu_layers=BRAIN_N_GPU_LAYERS_WHEN_GPU,
                chat_format="chatml-function-calling",
                verbose=False,
            )
            return llm, {"brain_model": os.path.basename(BRAIN_MODEL_GPU_PATH), "brain_tier": "gpu", "brain_gpu_offload": True}
        except Exception as e:
            print(f"  GPU-tier brain failed to load ({e}) — falling back to CPU-tier brain.")

    # CPU-tier model, either because hw.accelerator == "cpu", or as a
    # fallback after the GPU-tier brain failed to fit.
    print(f"  Loading CPU-tier brain: {os.path.basename(BRAIN_MODEL_CPU_PATH)}")
    n_gpu_layers = BRAIN_N_GPU_LAYERS_WHEN_GPU if hw.accelerator == "gpu" else 0
    llm = Llama(
        model_path=BRAIN_MODEL_CPU_PATH,
        n_ctx=BRAIN_CONTEXT_WINDOW,
        n_gpu_layers=n_gpu_layers,
        chat_format="chatml-function-calling",
        verbose=False,
    )
    return llm, {
        "brain_model": os.path.basename(BRAIN_MODEL_CPU_PATH),
        "brain_tier": "cpu",
        "brain_gpu_offload": n_gpu_layers != 0,
    }


# ============================== Vision tool ==============================

def find_adapter_dir(root: str) -> str:
    """Locates adapter_config.json, searching one level of subdirectories if needed."""
    if os.path.exists(os.path.join(root, "adapter_config.json")):
        return root
    matches = glob.glob(os.path.join(root, "*", "adapter_config.json"))
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError(
        f"Could not find adapter_config.json in '{root}' or any direct subfolder."
    )


def find_processor_dir(root: str) -> str:
    """Locates preprocessor_config.json, which may sit above a nested adapter subfolder."""
    if os.path.exists(os.path.join(root, "preprocessor_config.json")):
        return root
    matches = glob.glob(os.path.join(root, "**", "preprocessor_config.json"), recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError(f"Could not find preprocessor_config.json under '{root}'.")


def pick_gpu_dtype(compute_capability):
    """Ampere+ (compute capability >= 8.0) gets bf16; older GPUs get fp16."""
    if compute_capability and compute_capability[0] >= MIN_COMPUTE_CAPABILITY_FOR_BF16:
        return torch.bfloat16
    return torch.float16


def _resolve_vision_base_source():
    if os.path.isdir(VISION_TOOL_BASE_MODEL_LOCAL_DIR) and os.path.exists(
        os.path.join(VISION_TOOL_BASE_MODEL_LOCAL_DIR, "config.json")
    ):
        return VISION_TOOL_BASE_MODEL_LOCAL_DIR, True
    print(
        f"  NOTE: no local vision base model found at {VISION_TOOL_BASE_MODEL_LOCAL_DIR}. "
        f"Falling back to downloading '{VISION_TOOL_BASE_MODEL_ID}' from Hugging Face — "
        f"requires internet, will fail offline. Run download_models.py once to fix this."
    )
    return VISION_TOOL_BASE_MODEL_ID, False


def _load_vision_base_gpu(hw: HardwareProfile):
    model_source, is_local = _resolve_vision_base_source()
    dtype = pick_gpu_dtype(hw.compute_capability)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=is_local,
    )
    return model, dtype, "4-bit NF4 (bitsandbytes)"


def _load_vision_base_cpu():
    model_source, is_local = _resolve_vision_base_source()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=is_local,
    )
    return model, torch.float32, "fp32 (CPU) + dynamic int8 quantization"


def _merge_adapter(model, adapter_root: str, label: str):
    adapter_dir = find_adapter_dir(adapter_root)
    print(f"  Merging {label} adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    return model


def load_vision_tool(hw: HardwareProfile, task: str):
    """
    Returns (model, processor, trace_dict) for the vision tool — this is
    your fine-tuned model, called by the brain via one of the specialist
    tools defined in config.TOOLS, not talked to directly by the user.
    """
    adapter_dir = ADAPTER_REGISTRY.get(task)
    fallback_used = False
    if adapter_dir is None or not os.path.isdir(adapter_dir):
        print(f"  Adapter for task '{task}' not found — falling back to 'vqa'.")
        task = "vqa"
        adapter_dir = ADAPTER_REGISTRY["vqa"]
        fallback_used = True

    print(f"Loading vision tool base model on {hw.accelerator.upper()} path...")
    if hw.accelerator == "gpu":
        model, dtype, precision_desc = _load_vision_base_gpu(hw)
    else:
        model, dtype, precision_desc = _load_vision_base_cpu()

    processor_dir = find_processor_dir(adapter_dir)
    processor = AutoProcessor.from_pretrained(processor_dir)

    model = _merge_adapter(model, BACKBONE_ADAPTER_DIR, "backbone (RS domain adaptation)")
    model = _merge_adapter(model, adapter_dir, f"task ({task})")

    if hw.accelerator == "cpu":
        print("  Applying dynamic int8 quantization to linear layers (CPU speed-up)...")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    model.eval()

    trace = {
        "vision_tool_base_model": VISION_TOOL_BASE_MODEL_ID,
        "vision_tool_accelerator": hw.accelerator,
        "vision_tool_precision": precision_desc,
        "backbone_adapter": BACKBONE_ADAPTER_DIR,
        "task_adapter": adapter_dir,
        "task_adapter_fallback_used": fallback_used,
    }
    return model, processor, trace
