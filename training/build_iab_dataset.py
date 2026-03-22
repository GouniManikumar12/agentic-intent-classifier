from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_HEAD_CONFIG
from iab_taxonomy import get_iab_taxonomy, path_to_label

OUTPUT_DIR = IAB_HEAD_CONFIG.data_dir

GENERAL_TEMPLATES = (
    "what is {term}",
    "how does {term} work",
    "{term} basics",
    "{term} guide",
    "help me understand {term}",
    "{term} examples",
)

DISCOVERY_TEMPLATES = (
    "best {term}",
    "top {term}",
    "compare {term} options",
    "{term} for beginners",
)

BUYING_TEMPLATES = (
    "which {term} should i buy",
    "best {term} to buy this year",
    "compare {term} before buying",
    "{term} buying guide",
)

ACTION_TEMPLATES = (
    "book {term}",
    "find {term}",
    "get {term}",
)

TYPO_VARIANTS = {
    "laptop": "labtop",
    "smartphone": "smart phone",
    "e-readers": "ereaders",
}

SUPPLEMENTAL_LABEL_PROMPTS = {
    "Automotive > Auto Buying and Selling": [
        "Which car to buy in 2026",
        "best used suv to buy this year",
        "how should i compare cars before buying",
    ],
    "Business and Finance > Business > Sales": [
        "What is CRM software?",
        "HubSpot vs Zoho for a small team",
        "Which CRM should I buy for a 3-person startup?",
    ],
    "Business and Finance > Business > Marketing and Advertising": [
        "Best AI SEO tools for content teams",
    ],
    "Business and Finance > Business > Business I.T.": [
        "How do I reset my password?",
    ],
    "Food & Drink > Dining Out": [
        "Book a table for 2 tonight",
    ],
    "Food & Drink > Alcoholic Beverages": [
        "what is best vodka drink should i try",
        "best whiskey cocktail for beginners",
    ],
    "Technology & Computing > Artificial Intelligence": [
        "What is intent classification in NLP?",
    ],
    "Technology & Computing > Computing > Computer Software and Applications": [
        "Start my free trial",
    ],
    "Technology & Computing > Computing > Laptops": [
        "Which laptop to buy in 2026",
        "Which labtop to buy in 2026",
        "best laptop for work and study",
    ],
    "Technology & Computing > Computing > Desktops": [
        "Best desktop for gaming",
    ],
    "Technology & Computing > Consumer Electronics > Smartphones": [
        "Best smartphone to buy this year",
    ],
    "Technology & Computing > Computing > Computer Software and Applications > Communication": [
        "best communication software for remote teams",
    ],
}

HARD_CASES = [
    {"text": "Which car to buy in 2026", "iab_path": "Automotive > Auto Buying and Selling"},
    {"text": "What is CRM software?", "iab_path": "Business and Finance > Business > Sales"},
    {"text": "How do I reset my password?", "iab_path": "Business and Finance > Business > Business I.T."},
    {"text": "Book a table for 2 tonight", "iab_path": "Food & Drink > Dining Out"},
    {"text": "what is best vodka drink should i try", "iab_path": "Food & Drink > Alcoholic Beverages"},
    {"text": "What is intent classification in NLP?", "iab_path": "Technology & Computing > Artificial Intelligence"},
    {
        "text": "best communication software for remote teams",
        "iab_path": "Technology & Computing > Computing > Computer Software and Applications > Communication",
    },
    {"text": "Which laptop to buy in 2026", "iab_path": "Technology & Computing > Computing > Laptops"},
    {"text": "Which labtop to buy in 2026", "iab_path": "Technology & Computing > Computing > Laptops"},
]

EXTENDED_CASES = [
    {"text": "best used suv to buy this year", "iab_path": "Automotive > Auto Buying and Selling"},
    {"text": "HubSpot vs Zoho for a small team", "iab_path": "Business and Finance > Business > Sales"},
    {"text": "Best AI SEO tools for content teams", "iab_path": "Business and Finance > Business > Marketing and Advertising"},
    {"text": "founder playbook for seed stage growth", "iab_path": "Business and Finance > Business > Startups"},
    {
        "text": "small business operations software for a local agency",
        "iab_path": "Business and Finance > Business > Small and Medium-sized Business",
    },
    {"text": "best whiskey cocktail for beginners", "iab_path": "Food & Drink > Alcoholic Beverages"},
    {"text": "how does web hosting work", "iab_path": "Technology & Computing > Computing > Internet > Web Hosting"},
    {"text": "Best desktop for gaming", "iab_path": "Technology & Computing > Computing > Desktops"},
    {"text": "Best smartphone to buy this year", "iab_path": "Technology & Computing > Consumer Electronics > Smartphones"},
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_key(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def singularize(term: str) -> str:
    cleaned = normalize_text(term)
    if cleaned.endswith("ies") and len(cleaned) > 4:
        return cleaned[:-3] + "y"
    if cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
        return cleaned[:-1]
    return cleaned


def tokenize(term: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", normalize_text(term)) if token]


def looks_buyable(path: tuple[str, ...]) -> bool:
    joined = normalize_text(" ".join(path))
    tokens = set(tokenize(joined))
    buyable_tokens = {
        "buying",
        "selling",
        "loan",
        "loans",
        "software",
        "application",
        "applications",
        "laptops",
        "laptop",
        "desktops",
        "desktop",
        "smartphones",
        "smartphone",
        "tablets",
        "camera",
        "cameras",
        "camcorders",
        "wearable",
        "hosting",
        "flights",
        "flight",
        "hotel",
        "hotels",
        "rentals",
        "cars",
        "car",
        "property",
        "real",
        "estate",
    }
    return bool(tokens & buyable_tokens)


def looks_actionable(path: tuple[str, ...]) -> bool:
    joined = normalize_text(" ".join(path))
    return any(token in joined for token in ("booking", "reservations", "hosting", "communication"))


def base_terms_for_path(path: tuple[str, ...]) -> set[str]:
    terms = set()
    leaf = normalize_text(path[-1])
    full_label = normalize_text(path_to_label(path))
    terms.add(leaf)
    terms.add(full_label)

    if len(path) >= 2:
        terms.add(normalize_text(f"{path[-2]} {path[-1]}"))
        terms.add(normalize_text(f"{path[-1]} {path[-2]}"))
    if len(path) >= 3:
        terms.add(normalize_text(f"{path[-2]} {path[-1]} {path[0]}"))

    leaf_singular = singularize(leaf)
    terms.add(leaf_singular)
    if leaf_singular != leaf and len(path) >= 2:
        terms.add(normalize_text(f"{path[-2]} {leaf_singular}"))

    for source, replacement in TYPO_VARIANTS.items():
        if source in leaf:
            terms.add(leaf.replace(source, replacement))
            terms.add(full_label.replace(source, replacement))

    return {term for term in terms if term}


def apply_templates(term: str, templates: tuple[str, ...]) -> set[str]:
    return {template.format(term=term).strip() for template in templates}


def build_category_rows(path: tuple[str, ...], target_rows_per_label: int | None = None) -> list[dict]:
    label = path_to_label(path)
    prompt_set: set[str] = set()
    terms = base_terms_for_path(path)

    for term in terms:
        prompt_set.update(apply_templates(term, GENERAL_TEMPLATES))
        prompt_set.update(apply_templates(term, DISCOVERY_TEMPLATES))

    leaf = normalize_text(path[-1])
    prompt_set.add(leaf)
    prompt_set.add(f"{leaf} overview")
    prompt_set.add(f"learn about {leaf}")

    if looks_buyable(path):
        for term in sorted(terms):
            shopping_term = singularize(term)
            prompt_set.update(apply_templates(shopping_term, BUYING_TEMPLATES))
        prompt_set.add(f"best {singularize(leaf)}")
        prompt_set.add(f"top {singularize(leaf)} to buy")

    if looks_actionable(path):
        prompt_set.update(apply_templates(singularize(leaf), ACTION_TEMPLATES))

    supplemental_rows = [
        {"text": normalize_text(text), "iab_path": label}
        for text in SUPPLEMENTAL_LABEL_PROMPTS.get(label, [])
        if text.strip()
    ]
    prompt_set.update(row["text"] for row in supplemental_rows)

    generated_rows = [{"text": normalize_text(text), "iab_path": label} for text in prompt_set if text.strip()]
    generated_rows = sorted(generated_rows, key=lambda row: stable_key(f'{row["iab_path"]}::{row["text"]}'))

    if target_rows_per_label is None:
        return generated_rows

    chosen_by_text = {row["text"]: row for row in supplemental_rows}
    remaining = [row for row in generated_rows if row["text"] not in chosen_by_text]
    room = max(target_rows_per_label - len(chosen_by_text), 0)
    for row in remaining[:room]:
        chosen_by_text[row["text"]] = row

    rows = list(chosen_by_text.values())
    if len(rows) < 6:
        rows.extend(remaining[room : room + (6 - len(rows))])
    return sorted(rows, key=lambda row: stable_key(f'{row["iab_path"]}::{row["text"]}'))


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(rows)
    if total < 6:
        raise ValueError(f"Each IAB label needs at least 6 prompts for splitting, got {total}")
    val_count = max(1, total // 6)
    test_count = max(1, total // 6)
    test_rows = rows[:test_count]
    val_rows = rows[test_count : test_count + val_count]
    train_rows = rows[test_count + val_count :]
    return train_rows, val_rows, test_rows


def validate_label_space() -> list[str]:
    taxonomy = get_iab_taxonomy()
    taxonomy_labels = [node.path_label for node in taxonomy.nodes]
    if tuple(taxonomy_labels) != IAB_HEAD_CONFIG.labels:
        raise ValueError("IAB head labels do not match taxonomy-derived labels")
    return taxonomy_labels


def assert_full_coverage(rows: list[dict], labels: list[str]) -> None:
    covered = {row["iab_path"] for row in rows}
    missing = [label for label in labels if label not in covered]
    if missing:
        raise ValueError(f"Missing generated rows for {len(missing)} labels; first 10: {missing[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full-taxonomy IAB dataset.")
    parser.add_argument(
        "--target-rows-per-label",
        type=int,
        default=0,
        help="Optional cap per IAB label. Use 0 for uncapped full dataset.",
    )
    args = parser.parse_args()

    labels = validate_label_space()
    target_rows_per_label = args.target_rows_per_label if args.target_rows_per_label > 0 else None

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    per_label_counts: dict[str, dict[str, int]] = {}

    for label in labels:
        path = tuple(label.split(" > "))
        category_rows = build_category_rows(path, target_rows_per_label=target_rows_per_label)
        train_split, val_split, test_split = split_rows(category_rows)
        train_rows.extend(train_split)
        val_rows.extend(val_split)
        test_rows.extend(test_split)
        per_label_counts[label] = {
            "total": len(category_rows),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
        }

    assert_full_coverage(train_rows + val_rows + test_rows, labels)

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUTPUT_DIR / "val.jsonl", val_rows)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_rows)
    write_jsonl(OUTPUT_DIR / "hard_cases.jsonl", HARD_CASES)
    write_jsonl(OUTPUT_DIR / "extended_cases.jsonl", EXTENDED_CASES)

    summary = {
        "head": IAB_HEAD_CONFIG.slug,
        "label_count": len(labels),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "hard_cases_count": len(HARD_CASES),
        "extended_cases_count": len(EXTENDED_CASES),
        "target_rows_per_label": target_rows_per_label,
        "sample_label_counts": dict(list(per_label_counts.items())[:10]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
