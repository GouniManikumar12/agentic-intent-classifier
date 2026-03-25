#!/usr/bin/env python3
"""
Upload trained artifacts to Hugging Face Hub.

This repo uses local-path inference. The upload is intended so you can later
download these directories into the same folder layout and run inference.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload trained intent/IAB artifacts to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF repo id, e.g. 'yourname/admesh-intent-iab-v1'.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token. If omitted, uses env HF_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private.",
    )
    parser.add_argument(
        "--include-multitask",
        action="store_true",
        help="Upload multitask intent model output directory.",
    )
    parser.add_argument(
        "--include-iab",
        action="store_true",
        help="Upload IAB classifier model output directory.",
    )
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Upload artifacts/calibration directory.",
    )
    parser.add_argument(
        "--include-hf-readme",
        action="store_true",
        help="Upload a Hugging Face model card file as README.md in the Hub repo root.",
    )
    parser.add_argument(
        "--hf-readme-path",
        default="HF_MODEL_CARD.md",
        help="Local path to the HF model card markdown to upload as README.md (relative to repo root).",
    )
    parser.add_argument(
        "--multitask-dir",
        default="multitask_intent_model_output",
        help="Path to multitask intent output directory (relative to this script's base).",
    )
    parser.add_argument(
        "--iab-dir",
        default="iab_classifier_model_output",
        help="Path to IAB classifier model output directory (relative to this script's base).",
    )
    parser.add_argument(
        "--calibration-dir",
        default="artifacts/calibration",
        help="Path to calibration artifacts directory (relative to this script's base).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.token:
        print("Missing HF token. Provide --token or set env HF_TOKEN.", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent

    multitask_dir = (repo_root / args.multitask_dir).resolve()
    iab_dir = (repo_root / args.iab_dir).resolve()
    calibration_dir = (repo_root / args.calibration_dir).resolve()
    hf_readme_path = (repo_root / args.hf_readme_path).resolve()

    to_upload: list[tuple[str, Path]] = []
    if args.include_multitask:
        to_upload.append(("multitask_intent_model_output", multitask_dir))
    if args.include_iab:
        to_upload.append(("iab_classifier_model_output", iab_dir))
    if args.include_calibration:
        to_upload.append(("artifacts/calibration", calibration_dir))
    if args.include_hf_readme:
        to_upload.append(("README.md", hf_readme_path))

    if not to_upload:
        print("Nothing to upload. Pass --include-multitask, --include-iab, and/or --include-calibration.", file=sys.stderr)
        return 2

    # Import lazily so `--dry-run` works without extra deps.
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        print("Missing dependency: huggingface_hub. Install with: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for repo_path, local_dir in to_upload:
        if not local_dir.exists():
            print(f"[SKIP] {repo_path}: local path does not exist: {local_dir}", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"[DRY] Would upload {local_dir} -> {args.repo_id}:{repo_path}")
            continue
        # Upload single README.md file (Hub model card) vs directories
        if repo_path == "README.md":
            print(f"[UPLOAD] {local_dir} -> {args.repo_id}:README.md")
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local_dir),
                path_in_repo="README.md",
            )
            continue

        print(f"[UPLOAD] {local_dir} -> {args.repo_id}:{repo_path}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            path_in_repo=repo_path,
        )

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

