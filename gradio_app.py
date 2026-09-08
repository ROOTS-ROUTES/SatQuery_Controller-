"""
ChatGPT-style local web UI for SatQuery AI, built with Gradio.

This is what should launch automatically when the pendrive is plugged
in and its autorun/launch script is triggered: it starts a local web
server and opens the browser to it — no installation required beyond
the Python environment already bundled on the drive.

Usage:
    python gradio_app.py
Then open the printed http://127.0.0.1:xxxx URL (Gradio also opens it
automatically in most environments).
"""

import json

import gradio as gr

from controller import SatQueryController

print("Starting SatQuery AI — loading controller...")
controller = SatQueryController()


def handle_query(image1, image2, question):
    image_paths = [p for p in [image1, image2] if p is not None]
    if not image_paths:
        return "Please upload at least one image.", ""
    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        answer, trace = controller.run_query(image_paths, question.strip())
    except Exception as e:
        return f"Error: {e}", ""

    trace_str = json.dumps(trace, indent=2, default=str)
    return answer, trace_str


with gr.Blocks(title="SatQuery AI") as demo:
    gr.Markdown("# SatQuery AI\nUpload one image (single-image query) or two (pair/change/fusion query), then ask a question.")

    with gr.Row():
        image1 = gr.Image(type="filepath", label="Image 1")
        image2 = gr.Image(type="filepath", label="Image 2 (optional — for pairs)")

    question = gr.Textbox(label="Your question", placeholder="e.g. Describe the land-cover and major objects visible in this image.")
    submit_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=4)
    with gr.Accordion("Show me why (execution trace)", open=False):
        trace_box = gr.Textbox(label="Execution trace", lines=12)

    submit_btn.click(
        fn=handle_query,
        inputs=[image1, image2, question],
        outputs=[answer_box, trace_box],
    )

if __name__ == "__main__":
    demo.launch()
