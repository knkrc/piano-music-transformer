"""Entry point for Hugging Face Spaces, which expects app.py at the repository root.

The interface itself lives in ``pmt.demo`` so that it is importable and testable.

    uv run --extra demo python app.py
"""

from pmt.demo import Demo, build_interface

demo = build_interface(Demo())

if __name__ == "__main__":
    demo.launch()
