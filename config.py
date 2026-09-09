"""
Central configuration for the SatQuery AI agentic controller.

Edit the paths below to point at wherever you've placed the extracted
adapter folders on this machine / pendrive. Everything else in the
codebase reads from here rather than hardcoding paths, so this is the
only file you should need to touch when adding a new adapter or moving
folders around.
"""

import os

# --- Vision tool (your fine-tuned model — NOT the brain) ---
# The brain LLM below is text-only and cannot see images; when it needs to
# analyze one, it calls this as a tool. This is the model you fine-tuned.
VISION_TOOL_BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# If a local copy exists here (produced by download_models.py, run once
# on a machine with internet before going offline), it's used instead of
# VISION_TOOL_BASE_MODEL_ID, with local_files_only=True so no network call
# is ever made at runtime — this is what makes "plug into any offline
# device" possible.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
VISION_TOOL_BASE_MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "vision_tool_base_model")

BACKBONE_ADAPTER_DIR = os.path.join(MODELS_DIR, "backbone_adapter")  # RS domain adaptation LoRA (always applied)
VQA_ADAPTER_DIR = os.path.join(MODELS_DIR, "vqa_adapter")            # VQA task LoRA

# --- Brain LLM (the actual "brain" — text-only, does the reasoning + tool-calling) ---
# GGUF format via llama.cpp: one file per tier, runs on CPU or GPU, any OS,
# no CUDA toolkit required on the target machine. Auto-selected by
# hardware_detect.py's accelerator decision.
BRAIN_MODELS_DIR = os.path.join(MODELS_DIR, "brain_llm")
BRAIN_MODEL_CPU_PATH = os.path.join(BRAIN_MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
BRAIN_MODEL_GPU_PATH = os.path.join(BRAIN_MODELS_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")

# Hugging Face source repos/filenames for download_models.py (one-time, online step).
BRAIN_MODEL_CPU_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
BRAIN_MODEL_CPU_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
BRAIN_MODEL_GPU_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
BRAIN_MODEL_GPU_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

BRAIN_CONTEXT_WINDOW = 8192
BRAIN_N_GPU_LAYERS_WHEN_GPU = -1  # -1 = offload all layers to GPU (llama.cpp convention)

# --- Task -> adapter registry ---
# This is the extension point: when the grounding / change-VQA adapters
# are ready, add them here with their own key. Every tool below maps to
# one of these keys; if the mapped adapter isn't present yet, the vision
# tool loader falls back to "vqa" and says so in the execution trace.
ADAPTER_REGISTRY = {
    "vqa": VQA_ADAPTER_DIR,
    # "grounding": os.path.join(MODELS_DIR, "grounding_adapter"),
    # "change_vqa": os.path.join(MODELS_DIR, "change_vqa_adapter"),
}

# --- Specialist tools exposed to the brain ---
# Each tool is a genuinely distinct function in the brain's eyes — it picks
# which one to call, the controller only validates the choice (right number
# of images for this tool) and executes it. This replaces keyword-based task
# classification: the brain itself is the task classifier now.
#
# Each entry: the OpenAI-style function spec (what the brain sees), which
# adapter it maps to, how many images it requires, and how to turn the
# brain's arguments into the actual prompt sent to the vision model.

TOOLS = [
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "answer_visual_question",
                "description": (
                    "Answers a specific factual question about ONE satellite image "
                    "(optical or SAR) — e.g. what land-cover or objects are present, "
                    "whether something specific is visible, counts, comparisons within "
                    "the single image. Use this for general single-image questions that "
                    "aren't specifically about describing the whole scene or locating one "
                    "particular object."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The precise, self-contained question to ask about the image."},
                    },
                    "required": ["question"],
                },
            },
        },
        "adapter_key": "vqa",
        "min_images": 1,
        "max_images": 1,
        "build_prompt": lambda args: args["question"],
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "describe_scene",
                "description": (
                    "Produces a free-form caption / scene description of ONE satellite "
                    "image — land-cover types and major objects visible overall. Use this "
                    "when the user wants a general description or summary of the image "
                    "rather than an answer to a specific question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Optional: a particular aspect to focus the description on (e.g. 'water bodies', 'urban areas'). Omit for a general description.",
                        }
                    },
                    "required": [],
                },
            },
        },
        "adapter_key": "caption",  # falls back to "vqa" until a dedicated captioning adapter exists
        "min_images": 1,
        "max_images": 1,
        "build_prompt": lambda args: (
            f"Describe the land-cover and major objects visible in this image, focusing on {args['focus']}."
            if args.get("focus")
            else "Describe the land-cover and major objects visible in this image."
        ),
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "locate_object",
                "description": (
                    "Locates/highlights a SPECIFIC object or feature the user described "
                    "within ONE satellite image (e.g. 'the water body', 'the largest "
                    "building'). Use this when the user asks to find, highlight, or point "
                    "out something specific, rather than asking a general question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_description": {"type": "string", "description": "What to locate, as described by the user."},
                    },
                    "required": ["object_description"],
                },
            },
        },
        "adapter_key": "grounding",  # falls back to "vqa" until the grounding adapter exists
        "min_images": 1,
        "max_images": 1,
        "build_prompt": lambda args: f"Locate and describe the position of {args['object_description']} in this image.",
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "analyze_change",
                "description": (
                    "Compares TWO satellite images of the SAME location taken at "
                    "different times and answers a question about what changed between "
                    "them. Requires exactly two images. Use this for any before/after, "
                    "'what changed', or change-over-time question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The specific question about what changed between the two images."},
                    },
                    "required": ["question"],
                },
            },
        },
        "adapter_key": "change_vqa",  # falls back to "vqa" until the change-VQA adapter exists
        "min_images": 2,
        "max_images": 2,
        "build_prompt": lambda args: args["question"],
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "analyze_optical_sar_fusion",
                "description": (
                    "Jointly analyzes a co-registered OPTICAL and SAR image pair of the "
                    "SAME location to extract information more reliably than either "
                    "image alone (e.g. distinguishing built-up vs water regions using "
                    "both modalities together). Requires exactly two images: one optical, "
                    "one SAR. Use this specifically when the user asks to combine or use "
                    "both optical and SAR imagery together."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The specific fusion question to answer using both images together."},
                    },
                    "required": ["question"],
                },
            },
        },
        "adapter_key": "fusion",  # falls back to "vqa" until the fusion adapter exists
        "min_images": 2,
        "max_images": 2,
        "build_prompt": lambda args: args["question"],
    },
]

TOOLS_BY_NAME = {t["spec"]["function"]["name"]: t for t in TOOLS}

# --- Hardware routing thresholds ---
# Minimum VRAM (GB) required to take the GPU path for BOTH the brain LLM
# and the vision tool together. On a tight 4GB card (e.g. RTX 3050 laptop),
# this is genuinely close: ~2GB for the GPU-tier brain (Q4_K_M) + ~1.5-2GB
# for the 4-bit vision tool + overhead. model_loader.py catches CUDA OOM
# on the brain specifically and falls back to running the brain on CPU
# while keeping the vision tool on GPU, rather than crashing — so this
# threshold doesn't need to be set defensively high; the fallback handles
# the tight-fit case gracefully.
MIN_VRAM_GB_FOR_GPU_PATH = 3.0

# Ampere or newer (compute capability >= 8.0) supports native bf16 tensor
# cores; older GPUs (Turing/Pascal, e.g. T4, P100, most laptop GPUs like
# the RTX 30-series *do* qualify, but GTX/older RTX 20-series do not) fall
# back to fp16 automatically — see model_loader.pick_gpu_dtype().
MIN_COMPUTE_CAPABILITY_FOR_BF16 = 8

# --- Generation defaults ---
# One pass now carries both the answer and the structured findings block, so
# this needs to be a little larger than the original 128 — but not much: on
# the modest hardware this prototype targets, KV-cache size is the memory
# ceiling, so keep it tight.
MAX_NEW_TOKENS = 192

# --- Structured findings prompt ---------------------------------------------
# The vision tool is asked to return its analysis in a fixed, machine-parseable
# shape (land-cover fractions, key findings, a preliminary conclusion) rather
# than free prose. The brain then synthesises those findings into the answer the
# user actually sees — which is what lets it draw a conclusion instead of just
# repeating the tool's raw output.
STRUCTURED_PROMPT = """You are analysing a satellite image for a remote-sensing question.
Respond with EXACTLY this structure and nothing else:

LAND COVER
water: <fraction between 0 and 1>
vegetation: <fraction between 0 and 1>
bare soil: <fraction between 0 and 1>
built-up: <fraction between 0 and 1>

KEY FINDINGS
- <one concise, specific finding>
- <one concise, specific finding>
- <one concise, specific finding>

CONCLUSION
<one or two sentences that directly answer the question, based on the findings>
"""
