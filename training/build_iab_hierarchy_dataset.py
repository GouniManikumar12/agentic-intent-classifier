from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_DIFFICULTY_DATA_DIR, IAB_HEAD_CONFIG, IAB_HIERARCHY_HEAD_CONFIGS, IAB_HIERARCHY_SUMMARY_PATH
from iab_hierarchy import format_iab_hierarchy_text
from iab_taxonomy import parse_path_label, path_to_label, write_training_graph


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def dedupe_rows(rows: list[dict], label_field: str) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row["text"], row[label_field])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_level_rows(rows: list[dict], level: int, label_field: str) -> list[dict]:
    level_rows = []
    for row in rows:
        path = parse_path_label(row["iab_path"])
        if len(path) < level:
            continue
        parent_path = path[: level - 1]
        payload = {
            "text": format_iab_hierarchy_text(row["text"], parent_path),
            label_field: path_to_label(path[:level]),
            "source_iab_path": row["iab_path"],
            "source_depth": len(path),
            "parent_path": path_to_label(parent_path) if parent_path else "",
        }
        for field in ("prompt_family", "evidence_strength", "hard_negative", "negative_iab_path"):
            if field in row:
                payload[field] = row[field]
        level_rows.append(payload)
    return dedupe_rows(level_rows, label_field)


def load_source_rows(split_name: str) -> list[dict]:
    rows = load_jsonl(IAB_HEAD_CONFIG.split_paths[split_name])
    difficulty_path = IAB_DIFFICULTY_DATA_DIR / f"{split_name}.jsonl"
    rows.extend(load_jsonl(difficulty_path))
    return rows


def summarize_level_rows(rows: list[dict], label_field: str) -> dict:
    label_counts = Counter(row[label_field] for row in rows)
    by_strength = Counter(row.get("evidence_strength", "unspecified") for row in rows)
    by_family = Counter(row.get("prompt_family", "unspecified") for row in rows)
    return {
        "count": len(rows),
        "label_count": len(label_counts),
        "min_rows_per_label": min(label_counts.values()) if label_counts else 0,
        "max_rows_per_label": max(label_counts.values()) if label_counts else 0,
        "hard_negative_count": sum(1 for row in rows if row.get("hard_negative")),
        "parent_safe_count": sum(1 for row in rows if row.get("evidence_strength") == "weak"),
        "rows_by_evidence_strength": dict(sorted(by_strength.items())),
        "rows_by_prompt_family": dict(sorted(by_family.items())),
    }


def main() -> None:
    write_training_graph()
    summary = {}
    for level, config in IAB_HIERARCHY_HEAD_CONFIGS.items():
        label_field = config.label_field
        level_summary = {}
        for split_name in ("train", "val", "test"):
            source_rows = load_source_rows(split_name)
            level_rows = build_level_rows(source_rows, level, label_field)
            write_jsonl(config.split_paths[split_name], level_rows)
            level_summary[split_name] = summarize_level_rows(level_rows, label_field)
        summary[config.slug] = level_summary
    IAB_HIERARCHY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IAB_HIERARCHY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
