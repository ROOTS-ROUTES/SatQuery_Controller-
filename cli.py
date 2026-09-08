"""
Minimal command-line interface for testing the agentic controller
without needing a browser UI. Good for a quick sanity check right after
plugging the pendrive into a new machine, before demoing the full UI.

Usage:
    python cli.py
"""

import json

from controller import SatQueryController


def main():
    print("SatQuery AI — CLI prototype")
    print("Loading controller (this will download the base model on first run "
          "if it isn't cached locally)...\n")
    controller = SatQueryController()

    while True:
        print("\n" + "-" * 60)
        image_input = input("Image path(s), comma-separated for a pair (or 'quit'): ").strip()
        if image_input.lower() in ("quit", "exit"):
            break
        image_paths = [p.strip() for p in image_input.split(",") if p.strip()]

        query = input("Question: ").strip()
        if not query:
            continue

        try:
            answer, trace = controller.run_query(image_paths, query)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print("\nAnswer:")
        print(" ", answer)
        print("\nExecution trace:")
        print(json.dumps(trace, indent=2, default=str))


if __name__ == "__main__":
    main()
