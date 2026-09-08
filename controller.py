"""
The agentic controller.

The BRAIN chooses which specialist tool to call — VQA, captioning,
grounding, change analysis, or optical-SAR fusion — based on the tools it's
offered and the user's question. This is genuine tool selection, not
keyword-based task classification: the brain sees several distinct tools
with distinct names and descriptions, and picks one.

The controller's job is now exactly what the PRD's agentic-controller
requirements describe: check input compatibility (filter which tools are
even valid for the number of images attached, BEFORE offering them to the
brain), execute whatever the brain decides, and produce an auditable trace
of the decision. It does not decide the task itself.
"""

import json
import time

from PIL import Image
from qwen_vl_utils import process_vision_info

from config import MAX_NEW_TOKENS, TOOLS, TOOLS_BY_NAME
from hardware_detect import detect_hardware
from model_loader import load_brain_llm, load_vision_tool

MODALITY_TAG_SINGLE = "This is a satellite image (optical or SAR)."
MODALITY_TAG_PAIR = "These are two co-registered satellite images of the same location."

SYSTEM_PROMPT = (
    "You are SatQuery AI, an assistant for analyzing remote-sensing satellite imagery. "
    "You cannot see images yourself — you have specialist tools available for that. "
    "You will only be offered tools that are actually valid for the number of images "
    "currently attached, so choose whichever offered tool best matches what the user "
    "is asking; if none of the offered tools fit and no images are relevant to the "
    "question, answer directly instead. After a tool returns a result, synthesize a "
    "clear, direct answer for the user — do not just repeat the tool output verbatim."
)


def check_compatibility(image_paths):
    if len(image_paths) > 2:
        raise ValueError("At most two images (a pair) are supported in this prototype.")
    for p in image_paths:
        try:
            Image.open(p).verify()
        except Exception as e:
            raise ValueError(f"Could not open image '{p}': {e}")
    return True


def get_available_tools(num_images: int):
    """
    Input-compatibility check, done BEFORE the brain ever sees the tool
    list: only tools whose min/max image requirement matches what's
    actually attached are offered. This is what stops the brain from
    picking a 2-image tool when only one image was uploaded, rather than
    catching that mistake after the fact.
    """
    return [t for t in TOOLS if t["min_images"] <= num_images <= t["max_images"]]


class SatQueryController:
    def __init__(self):
        self.hw = detect_hardware()
        print("=== Hardware detection ===")
        print(f"  {self.hw.reason}")

        print("\n=== Loading brain LLM ===")
        self.brain, self.brain_trace = load_brain_llm(self.hw)

        # Vision tool is loaded lazily and cached per adapter — reloaded
        # only when the brain picks a specialist tool mapping to a
        # different adapter than what's currently loaded.
        self._vision_loaded_adapter_key = None
        self.vision_model = None
        self.vision_processor = None
        self.vision_trace = None

    def _ensure_vision_tool_loaded(self, adapter_key: str):
        if self._vision_loaded_adapter_key == adapter_key:
            return
        print(f"\n=== Loading vision tool for adapter: {adapter_key} ===")
        self.vision_model, self.vision_processor, self.vision_trace = load_vision_tool(self.hw, adapter_key)
        self._vision_loaded_adapter_key = adapter_key

    def _execute_tool(self, tool_name: str, args: dict, image_paths):
        tool = TOOLS_BY_NAME[tool_name]
        self._ensure_vision_tool_loaded(tool["adapter_key"])

        question = tool["build_prompt"](args)
        modality_tag = MODALITY_TAG_PAIR if len(image_paths) == 2 else MODALITY_TAG_SINGLE
        prompt = f"{modality_tag} {question}"

        content = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.vision_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        if self.hw.accelerator == "gpu":
            inputs = inputs.to(self.vision_model.device)

        output_ids = self.vision_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        answer = self.vision_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return answer, prompt

    def run_query(self, image_paths, query: str):
        start_time = time.time()
        check_compatibility(image_paths)

        available_tools = get_available_tools(len(image_paths))
        tool_specs = [t["spec"] for t in available_tools]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        if tool_specs:
            response = self.brain.create_chat_completion(messages=messages, tools=tool_specs, tool_choice="auto")
        else:
            # No images attached (or an image count nothing can use) — no
            # tools to offer, the brain just answers directly.
            response = self.brain.create_chat_completion(messages=messages)

        choice = response["choices"][0]
        msg = choice["message"]

        tool_call_info = None
        chosen_tool_name = None
        vision_prompt_sent = None
        validation_error = None

        if msg.get("tool_calls"):
            tool_call = msg["tool_calls"][0]
            chosen_tool_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            # Defensive re-validation: the brain was only offered valid
            # tools, but re-check anyway rather than trusting it blindly —
            # this is the "controller validates" half of the design.
            tool = TOOLS_BY_NAME.get(chosen_tool_name)
            if tool is None:
                validation_error = f"Unknown tool '{chosen_tool_name}' requested."
            elif not (tool["min_images"] <= len(image_paths) <= tool["max_images"]):
                validation_error = (
                    f"Tool '{chosen_tool_name}' requires between {tool['min_images']} and "
                    f"{tool['max_images']} image(s), but {len(image_paths)} were provided."
                )

            if validation_error:
                tool_result_content = f"Error: {validation_error}"
            else:
                vision_answer, vision_prompt_sent = self._execute_tool(chosen_tool_name, args, image_paths)
                tool_result_content = vision_answer

            tool_call_info = {
                "tool": chosen_tool_name,
                "arguments": args,
                "result": tool_result_content,
                "validation_error": validation_error,
            }

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result_content,
            })
            final_response = self.brain.create_chat_completion(messages=messages)
            final_answer = final_response["choices"][0]["message"]["content"]
        else:
            final_answer = msg["content"]

        elapsed = time.time() - start_time

        execution_trace = {
            "brain": self.brain_trace,
            "tools_offered": [t["spec"]["function"]["name"] for t in available_tools],
            "tool_called": chosen_tool_name,
            "tool_call": tool_call_info,
            "vision_prompt_sent": vision_prompt_sent,
            "vision_tool_load_info": self.vision_trace,
            "num_images": len(image_paths),
            "latency_seconds": round(elapsed, 2),
        }
        return final_answer, execution_trace
