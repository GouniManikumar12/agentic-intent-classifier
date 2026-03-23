from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import IAB_DATASET_SUMMARY_PATH, IAB_HEAD_CONFIG
from iab_taxonomy import IabNode, get_iab_taxonomy, path_to_label, write_training_graph

OUTPUT_DIR = IAB_HEAD_CONFIG.data_dir

TRAIN_ROWS_BY_DEPTH = {
    1: 80,
    2: 24,
    3: 12,
    4: 8,
}

DATA_MIX = {
    "alias": 0.4,
    "discovery": 0.25,
    "action": 0.15,
    "vague": 0.1,
    "contrastive": 0.1,
}

TYPO_VARIANTS = {
    "laptop": "labtop",
    "smartphone": "smart phone",
    "e-reader": "ereader",
    "e-readers": "ereaders",
    "university": "universitiy",
}

ALIAS_TEMPLATES = (
    "{term}",
    "what is {term}",
    "how does {term} work",
    "{term} guide",
    "learn about {term}",
    "{term} basics",
    "{term} overview",
)

DISCOVERY_TEMPLATES = (
    "best {term}",
    "top {term}",
    "compare {term}",
    "{term} options",
    "which {term} is best",
)

ACTION_TEMPLATES = (
    "find {term}",
    "get {term}",
    "book {term}",
    "reserve {term}",
    "buy {term}",
)

VAGUE_TEMPLATES = (
    "help me choose {term}",
    "looking into {term}",
    "need advice about {term}",
    "exploring {term}",
)

CONTRASTIVE_TEMPLATES = (
    "need {positive}, not {negative}",
    "looking for {positive} instead of {negative}",
    "{positive} rather than {negative}",
    "researching {positive}, not {negative}",
)

TIER1_ALIAS_BANK = {
    "Automotive": (
        "car buying",
        "used cars",
        "vehicle shopping",
        "auto advice",
    ),
    "Business and Finance": (
        "business software",
        "sales and marketing",
        "company operations",
        "financial planning",
    ),
    "Books and Literature": (
        "books to read",
        "fiction books",
        "reading ideas",
        "book recommendations",
    ),
    "Education": (
        "study options",
        "education programs",
        "school guidance",
        "student advice",
    ),
    "Entertainment": (
        "movies to watch",
        "film picks",
        "watch tonight",
        "entertainment ideas",
    ),
    "Family and Relationships": (
        "family advice",
        "parenting help",
        "relationship guidance",
        "raising kids",
    ),
    "Food & Drink": (
        "places to eat",
        "drinks to try",
        "restaurant ideas",
        "food recommendations",
    ),
    "Healthy Living": (
        "fitness ideas",
        "exercise guidance",
        "running advice",
        "healthy habits",
    ),
    "Home & Garden": (
        "home projects",
        "garden advice",
        "house updates",
        "backyard ideas",
    ),
    "Medical Health": (
        "medical advice",
        "doctor guidance",
        "symptom help",
        "health questions",
    ),
    "Personal Finance": (
        "saving money",
        "budget planning",
        "retirement planning",
        "financial guidance",
    ),
    "Real Estate": (
        "renting advice",
        "apartment search",
        "home buying",
        "housing options",
    ),
    "Sports": (
        "sports rules",
        "game tactics",
        "soccer advice",
        "match guidance",
    ),
    "Style & Fashion": (
        "fashion ideas",
        "shoe shopping",
        "outfit help",
        "style picks",
    ),
    "Technology & Computing": (
        "software options",
        "tech buying",
        "device comparison",
        "technology guide",
    ),
    "Travel": (
        "trip planning",
        "places to stay",
        "travel options",
        "hotel booking",
    ),
}

PREFIX_ALIAS_BANK = {
    ("Business and Finance", "Business", "Sales"): (
        "crm software",
        "lead management tools",
        "sales pipeline software",
        "customer relationship management",
    ),
    ("Business and Finance", "Business", "Marketing and Advertising"): (
        "marketing tools",
        "seo software",
        "campaign analytics",
        "advertising platforms",
    ),
    ("Business and Finance", "Business", "Business I.T."): (
        "identity management",
        "password reset help",
        "access management",
        "work account security",
    ),
    ("Education", "College Education", "Postgraduate Education"): (
        "masters degree",
        "graduate schools",
        "postgraduate study",
        "universities for masters",
    ),
    ("Travel", "Travel Type", "Hotels and Motels"): (
        "hotel stay",
        "place to stay",
        "hotel booking",
        "motel booking",
    ),
    ("Real Estate", "Real Estate Renting and Leasing"): (
        "apartments for rent",
        "rental listings",
        "lease options",
        "apartment search",
    ),
    ("Technology & Computing", "Computing", "Internet", "Web Hosting"): (
        "web hosting",
        "website hosting",
        "managed hosting",
        "hosting provider",
    ),
    ("Technology & Computing", "Computing", "Computer Software and Applications", "Communication"): (
        "communication software",
        "team chat software",
        "workplace messaging",
        "internal communication tools",
    ),
    ("Technology & Computing", "Computing", "Laptops"): (
        "student laptops",
        "work laptops",
        "gaming laptops",
        "portable computers",
    ),
    ("Technology & Computing", "Computing", "Desktops"): (
        "desktop computers",
        "desktop pcs",
        "gaming desktops",
        "home office desktops",
    ),
    ("Technology & Computing", "Consumer Electronics", "Smartphones"): (
        "smartphones",
        "android phones",
        "budget phones",
        "camera phones",
    ),
    ("Food & Drink", "Dining Out"): (
        "restaurant booking",
        "book a table",
        "places to eat",
        "dining out",
    ),
    ("Food & Drink", "Alcoholic Beverages"): (
        "cocktails",
        "vodka drinks",
        "whiskey cocktails",
        "spirits to try",
    ),
    ("Entertainment", "Movies"): (
        "movies to watch",
        "film recommendations",
        "cinema picks",
        "movie night",
    ),
    ("Family and Relationships", "Parenting"): (
        "parenting advice",
        "toddler parenting",
        "teen parenting",
        "preschool parenting",
    ),
    ("Home & Garden", "Gardening"): (
        "gardening tips",
        "plant care",
        "backyard garden",
        "balcony garden",
    ),
    ("Healthy Living", "Fitness and Exercise", "Running and Jogging"): (
        "running plan",
        "5k training",
        "10k training",
        "half marathon advice",
    ),
    ("Sports", "Soccer"): (
        "soccer rules",
        "premier league",
        "offside rule",
        "soccer tactics",
    ),
}

CANONICAL_EXAMPLE_QUERIES = {
    "Automotive > Auto Buying and Selling": (
        "Which car to buy in 2026",
        "best used suv to buy this year",
    ),
    "Business and Finance > Business > Sales": (
        "What is CRM software?",
        "HubSpot vs Zoho for a small team",
    ),
    "Business and Finance > Business > Business I.T.": (
        "How do I reset my password?",
        "need help with my work account access",
    ),
    "Technology & Computing > Computing > Laptops": (
        "Which laptop to buy in 2026",
        "Which labtop to buy in 2026",
    ),
    "Technology & Computing > Consumer Electronics > Smartphones": (
        "Best smartphone to buy this year",
    ),
    "Education > College Education > Postgraduate Education": (
        "best universities to study masters",
        "need postgraduate options for a master's degree",
    ),
    "Travel > Travel Type > Hotels and Motels": (
        "Need a hotel in Chicago for two nights",
    ),
    "Real Estate > Real Estate Renting and Leasing": (
        "looking for apartments for rent near downtown",
    ),
    "Food & Drink > Dining Out": (
        "Book a table for 2 tonight",
    ),
    "Entertainment > Movies": (
        "Looking for film recommendations, not TV shows or music",
    ),
}

HARD_CASES = [
    {"text": "Which car to buy in 2026", "iab_path": "Automotive > Auto Buying and Selling"},
    {"text": "What is CRM software?", "iab_path": "Business and Finance > Business > Sales"},
    {"text": "How do I reset my password?", "iab_path": "Business and Finance > Business > Business I.T."},
    {"text": "Book a table for 2 tonight", "iab_path": "Food & Drink > Dining Out"},
    {"text": "What is intent classification in NLP?", "iab_path": "Technology & Computing > Artificial Intelligence"},
    {
        "text": "best communication software for remote teams",
        "iab_path": "Technology & Computing > Computing > Computer Software and Applications > Communication",
    },
    {"text": "best universities to study masters", "iab_path": "Education > College Education > Postgraduate Education"},
    {"text": "Need a hotel in Chicago for two nights", "iab_path": "Travel > Travel Type > Hotels and Motels"},
]

EXTENDED_CASES = [
    {"text": "best remote jobs for data analysts", "iab_path": "Careers > Job Search"},
    {"text": "how much should i save each month", "iab_path": "Personal Finance > Financial Planning"},
    {"text": "tips for parenting a toddler", "iab_path": "Family and Relationships > Parenting"},
    {"text": "best plants for a small balcony garden", "iab_path": "Home & Garden > Gardening"},
    {"text": "How do offside rules work in soccer?", "iab_path": "Sports > Soccer"},
    {"text": "Looking for film recommendations, not TV shows or music", "iab_path": "Entertainment > Movies"},
    {"text": "best shoes under 100 dollars", "iab_path": "Style & Fashion"},
    {"text": "best software for small teams", "iab_path": "Technology & Computing > Computing > Computer Software and Applications"},
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


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", normalize_text(text)) if token]


def singularize(term: str) -> str:
    cleaned = normalize_text(term)
    if cleaned in {"sales", "news"}:
        return cleaned
    if cleaned.endswith("ies") and len(cleaned) > 4:
        return cleaned[:-3] + "y"
    if cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
        return cleaned[:-1]
    return cleaned


def humanize_term(term: str) -> str:
    cleaned = normalize_text(
        term.replace("&", " and ")
        .replace("I.T.", "IT")
        .replace("I.T", "IT")
        .replace(">", " ")
    )
    cleaned = cleaned.replace("3-d", "3d")
    return re.sub(r"\s+", " ", cleaned).strip()


def expand_typos(term: str) -> set[str]:
    values = set()
    cleaned = humanize_term(term)
    for source, replacement in TYPO_VARIANTS.items():
        if source in cleaned:
            values.add(cleaned.replace(source, replacement))
    return values


def target_total_rows(depth: int, override: int | None = None) -> int:
    if override is not None and override > 0:
        return max(override, 6)
    train_count = TRAIN_ROWS_BY_DEPTH[depth]
    val_count = max(2, train_count // 4)
    test_count = max(2, train_count // 4)
    return train_count + val_count + test_count


def looks_buyable(path: tuple[str, ...]) -> bool:
    joined = " ".join(tokenize(path_to_label(path)))
    return any(
        token in joined
        for token in (
            "buy",
            "loan",
            "software",
            "application",
            "laptop",
            "desktop",
            "smartphone",
            "hosting",
            "hotel",
            "rental",
            "car",
            "real estate",
        )
    )


def looks_actionable(path: tuple[str, ...]) -> bool:
    joined = " ".join(tokenize(path_to_label(path)))
    return any(token in joined for token in ("booking", "restaurant", "dining", "hosting", "hotel", "rental"))


def matching_prefix_aliases(path: tuple[str, ...]) -> set[str]:
    aliases = set()
    for prefix, values in PREFIX_ALIAS_BANK.items():
        if path[: len(prefix)] == prefix:
            aliases.update(humanize_term(value) for value in values)
    return aliases


def exact_terms_for_node(node: IabNode) -> set[str]:
    path = node.path
    terms = {
        humanize_term(node.label),
        humanize_term(node.path_label),
    }
    if len(path) >= 2:
        parent = humanize_term(path[-2])
        leaf = humanize_term(path[-1])
        terms.add(f"{parent} {leaf}")
        terms.add(f"{leaf} {parent}")
    singular_leaf = singularize(node.label)
    if singular_leaf != humanize_term(node.label):
        terms.add(singular_leaf)
    terms.update(expand_typos(node.label))
    terms.update(matching_prefix_aliases(path))
    return {term for term in terms if term}


def moderate_terms_for_node(node: IabNode) -> set[str]:
    path = node.path
    leaf = humanize_term(path[-1])
    terms = set()
    if len(path) >= 2:
        parent = humanize_term(path[-2])
        terms.add(f"{leaf} in {parent}")
        terms.add(f"{parent} {leaf} options")
    if len(path) >= 3:
        terms.add(f"{humanize_term(path[0])} {leaf}")
    terms.update(humanize_term(value) for value in TIER1_ALIAS_BANK.get(path[0], ()))
    return {term for term in terms if term}


def weak_terms_for_node(node: IabNode) -> set[str]:
    if node.level > 2:
        return set()
    terms = {humanize_term(value) for value in TIER1_ALIAS_BANK.get(node.path[0], ())}
    if node.level == 2:
        terms.add(f"{humanize_term(node.path[-2])} options")
        terms.add(f"{humanize_term(node.path[-2])} guide")
    return {term for term in terms if term}


def candidate_negative_nodes(node: IabNode) -> list[IabNode]:
    taxonomy = get_iab_taxonomy()
    negatives: list[IabNode] = []
    negatives.extend(taxonomy.siblings(node.path)[:2])
    for candidate in taxonomy.level_nodes(node.level):
        if candidate.path == node.path:
            continue
        if candidate.path[:1] == node.path[:1]:
            negatives.append(candidate)
        if len(negatives) >= 4:
            break
    deduped = []
    seen = set()
    for candidate in negatives:
        if candidate.path in seen:
            continue
        seen.add(candidate.path)
        deduped.append(candidate)
    return deduped[:4]


def make_row(
    text: str,
    label: str,
    *,
    prompt_family: str,
    evidence_strength: str,
    hard_negative: bool = False,
    negative_iab_path: str | None = None,
) -> dict:
    payload = {
        "text": normalize_text(text),
        "iab_path": label,
        "prompt_family": prompt_family,
        "evidence_strength": evidence_strength,
        "hard_negative": hard_negative,
    }
    if negative_iab_path is not None:
        payload["negative_iab_path"] = negative_iab_path
    return payload


def apply_templates(terms: set[str], templates: tuple[str, ...], label: str, prompt_family: str, evidence_strength: str):
    rows = []
    for term in sorted(terms):
        for template in templates:
            rows.append(
                make_row(
                    template.format(term=term),
                    label,
                    prompt_family=prompt_family,
                    evidence_strength=evidence_strength,
                )
            )
    return rows


def build_contrastive_rows(node: IabNode) -> list[dict]:
    rows = []
    positive_terms = sorted(exact_terms_for_node(node))
    if not positive_terms:
        return rows
    label = node.path_label
    for negative in candidate_negative_nodes(node):
        negative_term = humanize_term(negative.label)
        for positive in positive_terms[:3]:
            for template in CONTRASTIVE_TEMPLATES:
                rows.append(
                    make_row(
                        template.format(positive=positive, negative=negative_term),
                        label,
                        prompt_family="contrastive",
                        evidence_strength="moderate",
                        hard_negative=True,
                        negative_iab_path=negative.path_label,
                    )
                )
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped = {}
    for row in rows:
        key = (row["text"], row["iab_path"])
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def select_mixed_rows(rows: list[dict], desired_total: int) -> list[dict]:
    by_family: dict[str, list[dict]] = {family: [] for family in DATA_MIX}
    for row in rows:
        by_family.setdefault(row["prompt_family"], []).append(row)
    for family_rows in by_family.values():
        family_rows.sort(key=lambda row: stable_key(f'{row["iab_path"]}::{row["text"]}'))

    selected: list[dict] = []
    selected_keys = set()
    for family, ratio in DATA_MIX.items():
        family_rows = by_family.get(family, [])
        family_target = int(round(desired_total * ratio))
        for row in family_rows[:family_target]:
            key = (row["text"], row["iab_path"])
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)

    fallback_rows = sorted(rows, key=lambda row: stable_key(f'{row["iab_path"]}::{row["text"]}'))
    for row in fallback_rows:
        if len(selected) >= desired_total:
            break
        key = (row["text"], row["iab_path"])
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
    return selected


def build_category_rows(node: IabNode, target_rows_override: int | None = None) -> list[dict]:
    label = node.path_label
    exact_terms = exact_terms_for_node(node)
    moderate_terms = moderate_terms_for_node(node)
    weak_terms = weak_terms_for_node(node)
    rows: list[dict] = []

    rows.extend(apply_templates(exact_terms, ALIAS_TEMPLATES, label, "alias", "strong"))
    rows.extend(apply_templates(exact_terms, DISCOVERY_TEMPLATES, label, "discovery", "strong"))

    if looks_buyable(node.path) or looks_actionable(node.path):
        rows.extend(apply_templates(exact_terms, ACTION_TEMPLATES, label, "action", "strong"))

    if moderate_terms:
        rows.extend(apply_templates(moderate_terms, ALIAS_TEMPLATES[:4], label, "alias", "moderate"))
        rows.extend(apply_templates(moderate_terms, DISCOVERY_TEMPLATES[:3], label, "discovery", "moderate"))
        rows.extend(apply_templates(moderate_terms, VAGUE_TEMPLATES, label, "vague", "moderate"))

    if weak_terms:
        rows.extend(apply_templates(weak_terms, VAGUE_TEMPLATES, label, "vague", "weak"))
        rows.extend(apply_templates(weak_terms, ALIAS_TEMPLATES[:2], label, "alias", "weak"))

    rows.extend(build_contrastive_rows(node))
    rows.extend(
        make_row(
            text,
            label,
            prompt_family="alias",
            evidence_strength="strong",
        )
        for text in CANONICAL_EXAMPLE_QUERIES.get(label, ())
    )

    deduped = dedupe_rows(rows)
    desired_total = target_total_rows(node.level, target_rows_override)
    selected = select_mixed_rows(deduped, desired_total)
    if len(selected) < 6:
        raise ValueError(f"Generated fewer than 6 rows for {label}")
    return sorted(selected, key=lambda row: stable_key(f'{row["iab_path"]}::{row["text"]}'))


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


def assert_full_coverage(rows: list[dict], labels: list[str]) -> None:
    covered = {row["iab_path"] for row in rows}
    missing = [label for label in labels if label not in covered]
    if missing:
        raise ValueError(f"Missing generated rows for {len(missing)} labels; first 10: {missing[:10]}")


def dataset_summary(rows: list[dict]) -> dict:
    by_label = Counter(row["iab_path"] for row in rows)
    by_depth = Counter(len(tuple(row["iab_path"].split(" > "))) for row in rows)
    by_family = Counter(row["prompt_family"] for row in rows)
    by_strength = Counter(row["evidence_strength"] for row in rows)
    return {
        "count": len(rows),
        "labels": len(by_label),
        "min_rows_per_label": min(by_label.values()) if by_label else 0,
        "max_rows_per_label": max(by_label.values()) if by_label else 0,
        "rows_by_depth": dict(sorted(by_depth.items())),
        "rows_by_prompt_family": dict(sorted(by_family.items())),
        "rows_by_evidence_strength": dict(sorted(by_strength.items())),
        "hard_negative_count": sum(1 for row in rows if row.get("hard_negative")),
        "parent_safe_count": sum(1 for row in rows if row.get("evidence_strength") == "weak"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the taxonomy-aligned IAB dataset.")
    parser.add_argument(
        "--target-rows-per-label",
        type=int,
        default=0,
        help="Optional total row cap per label. Use 0 for tier-based defaults.",
    )
    args = parser.parse_args()

    taxonomy = get_iab_taxonomy()
    write_training_graph()

    override = args.target_rows_per_label if args.target_rows_per_label > 0 else None
    labels = [node.path_label for node in taxonomy.nodes]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    per_label_counts: dict[str, dict[str, int]] = {}

    for node in taxonomy.nodes:
        category_rows = build_category_rows(node, target_rows_override=override)
        train_split, val_split, test_split = split_rows(category_rows)
        train_rows.extend(train_split)
        val_rows.extend(val_split)
        test_rows.extend(test_split)
        per_label_counts[node.path_label] = {
            "depth": node.level,
            "total": len(category_rows),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
            "hard_negative_count": sum(1 for row in category_rows if row.get("hard_negative")),
            "parent_safe_count": sum(1 for row in category_rows if row.get("evidence_strength") == "weak"),
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
        "graph_path": str(write_training_graph()),
        "train": dataset_summary(train_rows),
        "val": dataset_summary(val_rows),
        "test": dataset_summary(test_rows),
        "hard_cases_count": len(HARD_CASES),
        "extended_cases_count": len(EXTENDED_CASES),
        "target_rows_per_label_override": override,
        "sample_label_counts": dict(list(per_label_counts.items())[:10]),
    }
    IAB_DATASET_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IAB_DATASET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
