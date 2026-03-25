from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    HEAD_CONFIGS,
    IAB_CLASSIFIER_MODEL_DIR,
    MULTITASK_INTENT_MODEL_DIR,
    _looks_like_local_hf_model_dir,
)


def verify_production_artifacts() -> tuple[bool, list[tuple[str, bool, str]]]:
    """Return (all_ok, rows of (label, ok, path_str))."""
    rows: list[tuple[str, bool, str]] = []
    ok = True

    for label, path in (
        ("multitask weights", MULTITASK_INTENT_MODEL_DIR / "multitask_model.pt"),
        ("multitask metadata", MULTITASK_INTENT_MODEL_DIR / "metadata.json"),
        ("multitask tokenizer config", MULTITASK_INTENT_MODEL_DIR / "config.json"),
    ):
        exists = path.exists()
        rows.append((label, exists, str(path)))
        ok = ok and exists

    iab_dir = IAB_CLASSIFIER_MODEL_DIR
    iab_ok = _looks_like_local_hf_model_dir(iab_dir)
    rows.append(("IAB classifier (HF layout)", iab_ok, str(iab_dir)))
    ok = ok and iab_ok

    for slug, cfg in HEAD_CONFIGS.items():
        path = cfg.calibration_path
        exists = path.exists()
        rows.append((f"calibration {slug}", exists, str(path)))
        ok = ok and exists

    return ok, rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify multitask, IAB, and calibration artifacts exist for production inference."
    )
    args = parser.parse_args()
    all_ok, rows = verify_production_artifacts()
    for label, row_ok, path in rows:
        status = "OK " if row_ok else "MISS"
        print(f"[{status}] {label}: {path}")
    if not all_ok:
        print(
            "\nFix: run training/run_full_training_pipeline.py (or train_multitask_intent, train_iab, "
            "calibrate_confidence for each head).",
            flush=True,
        )
        return 1
    print("\nAll production artifacts present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
