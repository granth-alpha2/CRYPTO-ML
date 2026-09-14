"""
main.py
=======
Simple entry point for the CRYPTO ML project.

This is a thin wrapper around run_pipeline.py.
Run this file to execute the full pipeline.

Usage:
    python main.py                       # Full pipeline
    python main.py --skip-encrypt        # Skip slow HE step
    python main.py --step train          # One step only
    python main.py --help                # Show all options
"""

import sys
import os


def main():
    # Ensure project root is on the path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Delegate to run_pipeline.py
    from run_pipeline import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
