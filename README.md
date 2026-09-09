# SatQuery AI — Agentic Controller (Brain + Vision Tool)

Plug this drive into any machine, run one command, and get the full SatQuery AI
interface in a browser — fully offline after a one-time setup step.

## Architecture: brain + tool, not one model doing everything

```
User question (+ optional image(s))
        │
        ▼
  BRAIN LLM  (Qwen2.5-Instruct, GGUF via llama.cpp — text only, cannot see images)
        │
        ├─ no image / doesn't need one → answers directly
        │
        └─ needs to look at the image → picks and calls the right specialist tool
                    │             (answer_visual_question / describe_scene /
                    │              locate_object / analyze_change /
                    │              analyze_optical_sar_fusion — see below)
                    │
                    ▼
            VISION TOOL  (your fine-tuned Qwen2-VL-2B: backbone + task LoRA adapters)
                    │
                    ▼
            tool result → fed back to the BRAIN
                    │
                    ▼
            BRAIN synthesizes the final answer for the user
```

The brain is a small general-purpose instruction-following LLM whose job is
reasoning and deciding *when* to call the vision tool — it never receives
image data itself. Your fine-tuned model (backbone + adapters) is the tool
that actually looks at satellite imagery, called only when needed. This
matches the "agentic controller selects and executes specialist tools"
language directly rather than having one vision model do everything with
scripted keyword routing.

**Two tiers of brain LLM ship on the drive**, auto-selected by
`hardware_detect.py`:

| Tier | Model | When used |
|---|---|---|
| CPU | Qwen2.5-1.5B-Instruct (GGUF, Q4_K_M, ~1GB) | No GPU, or GPU-tier brain doesn't fit |
| GPU | Qwen2.5-3B-Instruct (GGUF, Q4_K_M, ~2GB) | NVIDIA GPU with enough free VRAM |

Both are the same model family (Qwen2.5-Instruct) because it has reliable
native tool-calling support, which is the actual mechanism that makes "brain
calls fine-tuned model as a tool" work — not every small LLM supports this
well.

**Note on VRAM**: on a tight 4GB card, running the GPU-tier brain (~2GB)
*and* the vision tool (~1.5–2GB in 4-bit) at the same time is genuinely
close to the limit. `model_loader.py` catches a CUDA out-of-memory error on
the brain specifically and automatically falls back to the CPU-tier brain
(vision tool stays on GPU) rather than crashing — so this is handled, not
just a known risk.

## Honesty note on "fully offline"

Everything runs with **zero network access at runtime** after setup. The one
unavoidable step: the base model weights and both brain GGUF files (~5-6 GB
total) have to be downloaded from somewhere once — no software can bundle
Hugging Face models without ever touching the internet at all. So:

> **One machine, once, with internet** → prepares this folder → **copy the
> whole folder to the pendrive** → runs on any number of offline machines
> after that, forever, with no further internet access.

## One-time setup (needs internet, do this once)

The launchers (`run_windows.bat` / `run_mac_linux.sh`) handle creating a
virtual environment and installing Python packages automatically on first
run — see "Running it" below. Run the launcher once first (it's fine that
`models/` is still empty — the UI will just show a loading error, which is
expected at this point). Then close it and download the actual model
weights using that same virtual environment's Python directly, so there's
no need to fuss with activating it manually:

**Windows:**
```bat
venv\Scripts\python.exe download_models.py
```
**Mac/Linux:**
```bash
venv/bin/python download_models.py
```

This downloads:
1. `Qwen/Qwen2-VL-2B-Instruct` → `models/vision_tool_base_model/`
2. Both brain GGUF tiers → `models/brain_llm/`

**Do this before you disconnect from the internet / copy the folder to the
pendrive.** After it finishes, run the launcher again — it'll load the real
models this time instead of erroring. Once `models/` is fully populated,
**copy the entire folder** onto the pendrive.

### A note on `llama-cpp-python` and GPU support

PyPI has no prebuilt wheel for `llama-cpp-python` at all — a plain `pip
install` always builds it from source, which fails on Windows with a
long-path error while extracting llama.cpp's bundled web UI assets (you may
have hit this already). Both launchers now install it from the
maintainer's **prebuilt CPU wheel index** automatically:
```bash
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
This gets you working CPU inference immediately, no compiler needed. For
**GPU offload** (the brain's GPU tier actually using the GPU), install a
CUDA-specific prebuilt wheel instead, matching your installed CUDA version
(11.8, 12.1–12.5, 13.0, or 13.2):
```bash
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```
(swap `cu121` for whichever CUDA version `nvidia-smi` reports). If you skip
this, the brain still works — it just always runs on CPU regardless of
`hardware_detect.py`'s decision, which is a performance loss, not a crash.

The frontend is already built (`frontend_dist/`) and included — no Node.js
needed on the target machine.

## Moving to another device (or cloning this repo from GitHub)

The code in this repo is everything *except* the model weights — `models/`
(~7.4 GB), `venv/`, and the prebuilt `frontend_dist/` bundle are
intentionally not committed. After cloning (or copying the folder), do this:

**Re-downloadable — run once with internet** (`download_models.py` fetches
them from Hugging Face into `models/`):
1. `models/vision_tool_base_model/` — Qwen/Qwen2-VL-2B-Instruct
2. `models/brain_llm/` — both Qwen2.5 GGUF tiers (CPU + GPU)

**NOT re-downloadable — must be copied from a machine that has them**
(pendrive, Drive, etc.):
- `models/backbone_adapter/` — the remote-sensing domain-adaptation LoRA
- `models/vqa_adapter/` — the fine-tuned VQA task LoRA

These two folders are *your fine-tuned weights*; `download_models.py` can
only fetch public base models. Without them the vision tool silently runs
on the base Qwen2-VL model instead (worse answers — no fine-tuning).

**The frontend bundle:** `frontend_dist/` is served by `server.py` but not
committed. Either copy it from an existing machine, or rebuild it from the
[SatQuery-AI frontend repo](https://github.com/ROOTS-ROUTES/SatQuery-AI):
`npm install && npm run build`, then copy `dist/*` into `frontend_dist/`.

Then run the launcher once (`run_windows.bat` / `run_mac_linux.sh`) — it
creates the `venv/` and installs packages automatically on first run.

## Running it (fully offline, any machine, from here on)

- **Windows**: double-click `run_windows.bat`
- **Mac/Linux**: `./run_mac_linux.sh`

**First time on a new machine**: the launcher creates a virtual environment
and runs `pip install -r requirements.txt` automatically — this is the only
slow run, and can take several minutes (longer if `llama-cpp-python` has to
compile). **Every run after that is instant** — a marker file
(`venv/.setup_complete`) tells the launcher to skip setup and go straight to
launching.

**The browser UI appears within a second or two of clicking the file** —
not after the models finish loading. `server.py` starts the web server and
opens the browser immediately, while the brain LLM and vision tool load in
a background thread. If you try to ask a question before loading finishes,
you'll get a friendly "still starting up, try again in a few seconds"
message through the same chat interface — check `GET /api/status` any time
if you want to confirm loading state directly (`{"ready": true/false}`).

You still need Python 3.10+ installed on the target machine — the launcher
detects if it's missing and tells you where to get it, rather than failing
silently. For a truly zero-setup pendrive (no Python install required
either), the next step would be bundling a portable Python distribution —
ask if you want that built out.

## What the execution trace now contains

Since the brain does real tool-calling instead of scripted routing, the
trace reflects an actual decision:
```json
{
  "brain": {"brain_model": "qwen2.5-3b-instruct-q4_k_m.gguf", "brain_tier": "gpu", ...},
  "tool_called": "answer_visual_question",
  "tool_call": {"tool": "answer_visual_question", "arguments": {"question": "..."}, "result": "...", "validation_error": null},
  "tools_offered": ["answer_visual_question", "describe_scene", "locate_object"],
  "vision_tool_load_info": {...},
  "num_images": 1,
  "latency_seconds": 4.2
}
```
If the user asks something that doesn't need image analysis, `tool_called`
is `null` and the brain answered directly — genuinely conditional, not
hardcoded.

## Frontend changes (for offline use — unrelated to the brain/tool change)

- **`ImagePicker.tsx`** (new) — local file upload, the offline-capable input
  path. The original frontend's only image input was a live map picker
  requiring internet (Esri tiles); that's now gated behind
  `VITE_ENABLE_MAP_MODE=false` by default.
- **Google Fonts CDN links removed** from `index.html` — falls back to
  system fonts.
- **`ChatMessage.images` field** added end-to-end so uploaded images travel
  with each message.
- Production `.env` set explicitly (`VITE_USE_MOCK=false`) before building,
  since the repo's default would otherwise ship in mock mode.

## Specialist tool selection is now genuinely brain-driven

Earlier this used one generic `analyze_satellite_image` tool with keyword-based
task classification happening *inside* the tool. That's been replaced: the
brain is now offered several distinct specialist tools and picks one itself.

| Tool | Maps to adapter | Images required |
|---|---|---|
| `answer_visual_question` | `vqa` | 1 |
| `describe_scene` | `caption` (falls back to `vqa` until trained) | 1 |
| `locate_object` | `grounding` (falls back to `vqa` until trained) | 1 |
| `analyze_change` | `change_vqa` (falls back to `vqa` until trained) | 2 |
| `analyze_optical_sar_fusion` | `fusion` (falls back to `vqa` until trained) | 2 |

**Input compatibility is checked before the brain even sees the tool list**:
`controller.get_available_tools(num_images)` filters to only tools whose
image-count requirement matches what's actually attached — a 2-image tool is
never even offered when only one image was uploaded, rather than being
offered and then failing. The controller re-validates the brain's choice
afterward too (defense in depth), and if a mismatch somehow occurs anyway,
it's reported back to the brain as a tool-error message rather than crashing.

This means `controller.py` no longer decides *which* task to run — it only
validates and executes whatever the brain decides. As real grounding /
change-VQA / fusion adapters get trained, just point their `adapter_key` in
`config.py`'s `ADAPTER_REGISTRY` at the real folder — no other code changes,
and the fallback-to-vqa note disappears from the trace automatically.

## Known limitations (current prototype)

- Only the `vqa` adapter is actually trained today — the other four tools
  are real, distinct choices the brain can make, but currently execute via
  the VQA adapter as a fallback (visible in the trace) until their own
  adapters are trained.
- No multi-turn conversation history is replayed into the brain today (each
  query is handled independently).
- CPU-tier brain and CPU-path vision tool are both functional but slow —
  fine for a demo, not fast.
- `locate_object`'s output is currently a text description of where
  something is, not real pixel coordinates/bounding boxes — that requires
  the grounding adapter to actually be trained for structured output.
