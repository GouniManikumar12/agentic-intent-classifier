from __future__ import annotations

import re

from iab_taxonomy import IabNode, get_iab_taxonomy

NORMALIZATION_REPLACEMENTS = {
    "labtop": "laptop",
    "smart phone": "smartphone",
    "e reader": "e-reader",
    "e readers": "e-readers",
}

EXACT_PATH_HINTS = {
    ("crm",): ("Business and Finance", "Business", "Sales"),
    ("hubspot",): ("Business and Finance", "Business", "Sales"),
    ("zoho",): ("Business and Finance", "Business", "Sales"),
    ("salesforce",): ("Business and Finance", "Business", "Sales"),
    ("pipedrive",): ("Business and Finance", "Business", "Sales"),
    ("password",): ("Business and Finance", "Business", "Business I.T."),
    ("login",): ("Business and Finance", "Business", "Business I.T."),
    ("log into",): ("Business and Finance", "Business", "Business I.T."),
    ("account",): ("Business and Finance", "Business", "Business I.T."),
    ("billing",): ("Business and Finance", "Business", "Business I.T."),
    ("subscription",): ("Business and Finance", "Business", "Business I.T."),
    ("restaurant",): ("Food & Drink", "Dining Out"),
    ("book a table",): ("Food & Drink", "Dining Out"),
    ("vodka",): ("Food & Drink", "Alcoholic Beverages"),
    ("whiskey",): ("Food & Drink", "Alcoholic Beverages"),
    ("whisky",): ("Food & Drink", "Alcoholic Beverages"),
    ("tequila",): ("Food & Drink", "Alcoholic Beverages"),
    ("cocktail",): ("Food & Drink", "Alcoholic Beverages"),
    ("ai seo",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("seo",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("intent classification",): ("Technology & Computing", "Artificial Intelligence"),
    ("nlp",): ("Technology & Computing", "Artificial Intelligence"),
    ("llm",): ("Technology & Computing", "Artificial Intelligence"),
    ("laptop",): ("Technology & Computing", "Computing", "Laptops"),
    ("desktop",): ("Technology & Computing", "Computing", "Desktops"),
    ("smartphone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("iphone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("android phone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("web hosting",): ("Technology & Computing", "Computing", "Internet", "Web Hosting"),
    ("free trial",): ("Technology & Computing", "Computing", "Computer Software and Applications"),
}

BUYING_TERMS = {
    "buy",
    "best",
    "top",
    "which",
    "compare",
    "comparison",
    "vs",
    "purchase",
    "shop",
    "shopping",
}

COMMERCIAL_SUBTYPES = {
    "product_discovery",
    "comparison",
    "evaluation",
    "deal_seeking",
    "provider_selection",
    "signup",
    "purchase",
    "booking",
    "contact_sales",
}


def round_score(value: float) -> float:
    return round(float(value), 4)


def normalize(text: str) -> str:
    lowered = text.strip().lower()
    for source, target in NORMALIZATION_REPLACEMENTS.items():
        lowered = lowered.replace(source, target)
    return re.sub(r"\s+", " ", lowered)


def singularize(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        return term[:-1]
    return term


def tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize(text)) if token}


def path_terms(node: IabNode) -> set[str]:
    terms = {normalize(part) for part in node.path}
    terms.add(normalize(" ".join(node.path)))
    leaf = normalize(node.label)
    terms.add(leaf)
    terms.add(singularize(leaf))
    return {term for term in terms if term}


def path_token_overlap(node: IabNode, text_tokens: set[str]) -> float:
    score = 0.0
    for part in node.path:
        tokens = tokenize(part)
        overlap = len(tokens & text_tokens)
        if overlap:
            weight = 1.2 if part == node.label else 0.6
            score += overlap * weight
    return score


def buying_bias(node: IabNode, text_tokens: set[str]) -> float:
    if not (text_tokens & BUYING_TERMS):
        return 0.0
    node_tokens = tokenize(" ".join(node.path))
    buyable_tokens = {
        "buying",
        "selling",
        "software",
        "applications",
        "application",
        "laptops",
        "desktops",
        "smartphones",
        "tablets",
        "cameras",
        "camcorders",
        "wearable",
        "hosting",
        "rentals",
        "loans",
        "estate",
        "travel",
        "hotel",
        "flight",
    }
    return 1.5 if node_tokens & buyable_tokens else 0.0


def score_node(
    node: IabNode,
    lowered: str,
    text_tokens: set[str],
    intent_type: str,
    subtype: str,
    decision_phase: str,
) -> tuple[float, str]:
    score = 0.0
    mapping_mode = "nearest_equivalent"

    for term in path_terms(node):
        if not term:
            continue
        if term in lowered:
            score += 4.0 if term == normalize(node.label) else 2.0
            if term == normalize(node.label):
                mapping_mode = "exact"

    score += path_token_overlap(node, text_tokens)
    score += buying_bias(node, text_tokens)

    if subtype in COMMERCIAL_SUBTYPES and decision_phase in {"consideration", "decision", "action"}:
        score += buying_bias(node, BUYING_TERMS) * 0.6

    if intent_type == "commercial" and "business" in tokenize(" ".join(node.path)):
        score += 0.2
    if decision_phase == "support" and normalize(node.label) in {"business i.t.", "it and internet support"}:
        score += 2.0
    if subtype in {"comparison", "evaluation", "provider_selection"} and normalize(node.label) == "sales":
        score += 1.0

    return score, mapping_mode


def fallback_path(intent_type: str, subtype: str, decision_phase: str) -> tuple[str, ...]:
    if decision_phase == "support":
        return ("Business and Finance", "Business", "Business I.T.")
    if subtype in {"signup", "purchase", "download"}:
        return ("Technology & Computing", "Computing", "Computer Software and Applications")
    if intent_type == "informational":
        return ("Education",)
    return ("Business and Finance", "Business")


def score_targets(text: str, intent_type: str, subtype: str, decision_phase: str) -> tuple[tuple[str, ...], str, float]:
    taxonomy = get_iab_taxonomy()
    lowered = normalize(text)
    text_tokens = tokenize(lowered)
    path_scores = {node.path: [0.0, "nearest_equivalent"] for node in taxonomy.nodes}

    for hints, path in EXACT_PATH_HINTS.items():
        if any(hint in lowered for hint in hints):
            path_scores[path][0] += 8.0
            path_scores[path][1] = "exact"

    for node in taxonomy.nodes:
        score, mapping_mode = score_node(node, lowered, text_tokens, intent_type, subtype, decision_phase)
        path_scores[node.path][0] += score
        if mapping_mode == "exact":
            path_scores[node.path][1] = "exact"

    best_path, (best_score, best_mode) = max(
        path_scores.items(),
        key=lambda item: (item[1][0], len(item[0]), item[0]),
    )
    sorted_scores = sorted((value[0] for value in path_scores.values()), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    if best_score <= 0:
        best_path = fallback_path(intent_type, subtype, decision_phase)
        best_mode = "nearest_equivalent"
        best_score = 1.0
        second_score = 0.0

    margin = best_score - second_score
    confidence = 0.46 + min(best_score, 8.0) * 0.05 + min(margin, 4.0) * 0.04
    confidence = max(0.48, min(confidence, 0.97))
    if best_score < 3.0:
        best_mode = "nearest_equivalent"

    return best_path, best_mode, round_score(confidence)


def map_iab_content(text: str, intent_type: str, subtype: str, decision_phase: str) -> dict:
    taxonomy = get_iab_taxonomy()
    target_path, mapping_mode, mapping_confidence = score_targets(text, intent_type, subtype, decision_phase)
    return taxonomy.build_content_object(
        path=target_path,
        mapping_mode=mapping_mode,
        mapping_confidence=mapping_confidence,
    )
