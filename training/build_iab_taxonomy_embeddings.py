from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from iab_retrieval import build_iab_taxonomy_embedding_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local embedding index for IAB taxonomy nodes.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    args = parser.parse_args()

    summary = build_iab_taxonomy_embedding_index(batch_size=args.batch_size)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
