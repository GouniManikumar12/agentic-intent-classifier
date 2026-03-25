#!/usr/bin/env python3
"""End-to-end production pipeline entry point (Colab-friendly).

Runs ``training/run_full_training_pipeline.py`` and forwards CLI args.

Typical:
  python complete_pipeline.py --complete
  python complete_pipeline.py --skip-full-eval --complete   # Colab: skip heavy eval suites
  python training/pipeline_verify.py                        # only check artifacts
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    script = BASE_DIR / "training" / "run_full_training_pipeline.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


if __name__ == "__main__":
    main()
