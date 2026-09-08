"""
FastAPI backend for SatQuery AI.

Implements exactly the contract the frontend (frontend-main) expects:
    POST /api/chat
        request:  { model, messages: [{ role, content, images?: string[] }], stream, ... }
        response: text/event-stream of lines "data: {\"token\": \"...\"}", ending
                  with "data: [DONE]"

Startup sequence (this is the important part): the web server — and
therefore the UI — comes up immediately. The brain + vision tool load in a
BACKGROUND THREAD started right as the server starts, not before it. If a
chat request arrives while loading is still in progress, it gets a friendly
"still starting up" message through the exact same streaming contract
(no special-case handling needed on the frontend). This is what makes
"click the file, see the UI right away, everything else loads behind it"
actually true rather than the UI being blocked behind model loading.

Image handling: the frontend sends locally-uploaded images as base64 data
URLs on the last user message's `images` field (see ImagePicker.tsx on the
frontend side). This file decodes those into temp files before handing them
to the controller, which expects file paths.
"""

import base64
import json
import os
import re
import tempfile
import threading
import webbrowser

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="SatQuery AI")

# --- Background controller loading ---
# Populated by _load_controller_in_background() shortly after the server
# starts. Left as None until then; /api/chat checks this and responds
# gracefully rather than blocking or erroring if a request arrives early.
controller = None
controller_load_error = None
controller_ready = threading.Event()


def _load_controller_in_background():
    global controller, controller_load_error
    try:
        print("Loading controller (hardware detection + brain LLM)...")
        from controller import SatQueryController  # imported here, not at module top,
        # so a slow/heavy import doesn't delay the server starting to accept connections.
        controller = SatQueryController()
        print("Controller ready — brain LLM loaded, vision tool will load on first image query.\n")
    except Exception as e:
        controller_load_error = str(e)
        print(f"Controller failed to load: {e}")
    finally:
        controller_ready.set()


threading.Thread(target=_load_controller_in_background, daemon=True).start()


DATA_URL_RE = re.compile(r"^data:image/(?P<ext>\w+);base64,(?P<data>.+)$", re.DOTALL)


def _save_data_url_to_temp(data_url: str) -> str:
    match = DATA_URL_RE.match(data_url)
    if not match:
        raise ValueError("Expected a base64 image data URL (data:image/...;base64,...)")
    ext = match.group("ext")
    raw = base64.b64decode(match.group("data"))
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _sse(token: str) -> str:
    return f"data: {json.dumps({'token': token})}\n\n"


@app.get("/api/status")
async def status():
    """Polled by nothing on the frontend today (no frontend change needed),
    but handy to check manually (e.g. curl) while troubleshooting a slow
    first load."""
    return {
        "ready": controller_ready.is_set() and controller is not None,
        "loading": not controller_ready.is_set(),
        "error": controller_load_error,
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user_msg is None:
        return StreamingResponse(iter([_sse("No user message received."), "data: [DONE]\n\n"]), media_type="text/event-stream")

    if not controller_ready.is_set():
        return StreamingResponse(
            iter([_sse("SatQuery AI is still starting up (loading the model) — please wait a few seconds and try again."), "data: [DONE]\n\n"]),
            media_type="text/event-stream",
        )
    if controller is None:
        return StreamingResponse(
            iter([_sse(f"SatQuery AI failed to start: {controller_load_error}"), "data: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    query_text = last_user_msg.get("content", "")
    image_data_urls = last_user_msg.get("images", []) or []
    temp_paths = [_save_data_url_to_temp(url) for url in image_data_urls]

    def event_stream():
        try:
            answer, trace = controller.run_query(temp_paths, query_text)
        except Exception as e:
            yield _sse(f"Error: {e}")
            yield "data: [DONE]\n\n"
            return

        # The controller currently returns the full answer at once (no
        # token-level streaming from generate() yet) — stream it back in
        # word-sized chunks so the UI's streaming typing effect still
        # works. Swapping in real token streaming later needs no frontend change.
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield _sse(word + (" " if i < len(words) - 1 else ""))

        trace_md = "\n\n---\n**Execution trace:**\n```json\n" + json.dumps(trace, indent=2, default=str) + "\n```"
        yield _sse(trace_md)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Serve the built frontend (dist/) as static files ---
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend_dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    print(f"WARNING: frontend build not found at {FRONTEND_DIST} — API will run without a UI.")


def _open_browser_when_ready(url: str, delay_seconds: float = 1.2):
    threading.Timer(delay_seconds, lambda: webbrowser.open(url)).start()


if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 8000
    # Opens shortly after uvicorn starts accepting connections — NOT after
    # the controller finishes loading, since that now happens in the
    # background thread above instead of blocking here.
    _open_browser_when_ready(f"http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
