from __future__ import annotations

import re
from functools import lru_cache

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
    ("password reset",): ("Business and Finance", "Business", "Business I.T."),
    ("reset my password",): ("Business and Finance", "Business", "Business I.T."),
    ("work password",): ("Business and Finance", "Business", "Business I.T."),
    ("locked out of their accounts",): ("Business and Finance", "Business", "Business I.T."),
    ("locked out of their account",): ("Business and Finance", "Business", "Business I.T."),
    ("locked out of my account",): ("Business and Finance", "Business", "Business I.T."),
    ("locked out of our accounts",): ("Business and Finance", "Business", "Business I.T."),
    ("identity and access",): ("Business and Finance", "Business", "Business I.T."),
    ("identity management",): ("Business and Finance", "Business", "Business I.T."),
    ("access management",): ("Business and Finance", "Business", "Business I.T."),
    ("account security",): ("Business and Finance", "Business", "Business I.T."),
    ("single sign-on",): ("Business and Finance", "Business", "Business I.T."),
    ("sso",): ("Business and Finance", "Business", "Business I.T."),
    ("seo tools",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("ad attribution",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("campaign measurement",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("channel performance",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("paid and organic campaigns",): ("Business and Finance", "Business", "Marketing and Advertising"),
    ("book a table",): ("Food & Drink", "Dining Out"),
    ("restaurant reservation",): ("Food & Drink", "Dining Out"),
    ("client dinner",): ("Food & Drink", "Dining Out"),
    ("place to eat",): ("Food & Drink", "Dining Out"),
    ("reservation online",): ("Food & Drink", "Dining Out"),
    ("vodka",): ("Food & Drink", "Alcoholic Beverages"),
    ("whiskey",): ("Food & Drink", "Alcoholic Beverages"),
    ("whisky",): ("Food & Drink", "Alcoholic Beverages"),
    ("tequila",): ("Food & Drink", "Alcoholic Beverages"),
    ("cocktail",): ("Food & Drink", "Alcoholic Beverages"),
    ("intent classification",): ("Technology & Computing", "Artificial Intelligence"),
    ("nlp",): ("Technology & Computing", "Artificial Intelligence"),
    ("llm",): ("Technology & Computing", "Artificial Intelligence"),
    ("machine learning",): ("Technology & Computing", "Artificial Intelligence"),
    ("large language models",): ("Technology & Computing", "Artificial Intelligence"),
    ("text classification",): ("Technology & Computing", "Artificial Intelligence"),
    ("laptop",): ("Technology & Computing", "Computing", "Laptops"),
    ("portable computer",): ("Technology & Computing", "Computing", "Laptops"),
    ("desktop",): ("Technology & Computing", "Computing", "Desktops"),
    ("smartphone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("iphone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("android phone",): ("Technology & Computing", "Consumer Electronics", "Smartphones"),
    ("project management software",): ("Technology & Computing", "Computing", "Computer Software and Applications"),
    ("workflow software",): ("Technology & Computing", "Computing", "Computer Software and Applications"),
    ("internal workflows",): ("Technology & Computing", "Computing", "Computer Software and Applications"),
    ("internal messaging",): ("Technology & Computing", "Computing", "Computer Software and Applications", "Communication"),
    ("workplace chat",): ("Technology & Computing", "Computing", "Computer Software and Applications", "Communication"),
    ("cross-functional collaboration",): ("Technology & Computing", "Computing", "Computer Software and Applications", "Communication"),
    ("web hosting",): ("Technology & Computing", "Computing", "Internet", "Web Hosting"),
    ("website hosting",): ("Technology & Computing", "Computing", "Internet", "Web Hosting"),
    ("managed hosting provider",): ("Technology & Computing", "Computing", "Internet", "Web Hosting"),
    ("hotel",): ("Travel", "Travel Type", "Hotels and Motels"),
    ("hotels",): ("Travel", "Travel Type", "Hotels and Motels"),
    ("motel",): ("Travel", "Travel Type", "Hotels and Motels"),
    ("place to stay",): ("Travel", "Travel Type", "Hotels and Motels"),
    ("work trip",): ("Travel", "Travel Type", "Hotels and Motels"),
    ("apartments for rent",): ("Real Estate", "Real Estate Renting and Leasing"),
    ("for rent",): ("Real Estate", "Real Estate Renting and Leasing"),
    ("rental listings",): ("Real Estate", "Real Estate Renting and Leasing"),
    ("half marathon",): ("Healthy Living", "Fitness and Exercise", "Running and Jogging"),
    ("10k",): ("Healthy Living", "Fitness and Exercise", "Running and Jogging"),
    ("5k",): ("Healthy Living", "Fitness and Exercise", "Running and Jogging"),
    ("jogging",): ("Healthy Living", "Fitness and Exercise", "Running and Jogging"),
    ("soccer",): ("Sports", "Soccer"),
    ("premier league",): ("Sports", "Soccer"),
    ("offside",): ("Sports", "Soccer"),
    ("fantasy novel",): ("Books and Literature", "Fiction"),
    ("character-driven novel",): ("Books and Literature", "Fiction"),
    ("kitchen remodel",): ("Home & Garden", "Home Improvement"),
    ("bathroom renovation",): ("Home & Garden", "Home Improvement"),
    ("home improvement",): ("Home & Garden", "Home Improvement"),
    ("upgrading an older house",): ("Home & Garden", "Home Improvement"),
    ("online courses",): ("Education", "Online Education"),
    ("remote classes",): ("Education", "Online Education"),
    ("internet-based training",): ("Education", "Online Education"),
    ("professional classes",): ("Education", "Online Education"),
    ("universities",): ("Education", "College Education"),
    ("university",): ("Education", "College Education"),
    ("masters",): ("Education", "College Education", "Postgraduate Education"),
    ("master's degree",): ("Education", "College Education", "Postgraduate Education"),
    ("universities to study masters",): ("Education", "College Education", "Postgraduate Education"),
    ("graduate schools",): ("Education", "College Education", "Postgraduate Education"),
    ("postgraduate",): ("Education", "College Education", "Postgraduate Education"),
    ("medical advice",): ("Medical Health",),
    ("see a doctor",): ("Medical Health",),
    ("allergy symptoms",): ("Medical Health",),
    ("job openings",): ("Careers", "Job Search"),
    ("remote jobs",): ("Careers", "Job Search"),
    ("new role",): ("Careers", "Job Search"),
    ("product manager openings",): ("Careers", "Job Search"),
    ("retirement",): ("Personal Finance", "Financial Planning"),
    ("budgeting",): ("Personal Finance", "Financial Planning"),
    ("save each month",): ("Personal Finance", "Financial Planning"),
    ("parenting",): ("Family and Relationships", "Parenting"),
    ("preschool",): ("Family and Relationships", "Parenting"),
    ("toddler",): ("Family and Relationships", "Parenting"),
    ("teenager",): ("Family and Relationships", "Parenting"),
    ("garden",): ("Home & Garden", "Gardening"),
    ("gardening",): ("Home & Garden", "Gardening"),
    ("tomato plants",): ("Home & Garden", "Gardening"),
    ("movie",): ("Entertainment", "Movies"),
    ("movies",): ("Entertainment", "Movies"),
    ("film recommendations",): ("Entertainment", "Movies"),
    ("spirit-forward drink",): ("Food & Drink", "Alcoholic Beverages"),
    ("women's shoes",): ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"),
    ("women's sneakers",): ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"),
    ("women's footwear",): ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"),
    ("running shoes",): ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"),
    ("women sneakers",): ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"),
    ("men's shoes",): ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"),
    ("men's sneakers",): ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"),
    ("dress shoes",): ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"),
    ("men's footwear",): ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"),
    ("mens shoes",): ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"),
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
    "under",
    "cheap",
    "affordable",
    "budget",
    "price",
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

TIER1_HINTS = {
    ("Automotive",): {
        "car",
        "cars",
        "auto",
        "automotive",
        "vehicle",
        "vehicles",
        "suv",
        "suvs",
        "truck",
        "trucks",
        "sedan",
        "ev",
        "motorcycle",
    },
    ("Business and Finance",): {
        "crm",
        "sales",
        "marketing",
        "advertising",
        "business",
        "startup",
        "founder",
        "revenue",
        "lead",
        "pipeline",
        "finance",
        "accounting",
    },
    ("Food & Drink",): {
        "restaurant",
        "restaurants",
        "dinner",
        "lunch",
        "brunch",
        "eat",
        "table",
        "reservation",
        "vodka",
        "whiskey",
        "cocktail",
        "beer",
        "wine",
        "drink",
        "drinks",
    },
    ("Style & Fashion",): {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "heel",
        "heels",
        "sandals",
        "footwear",
        "fashion",
        "clothing",
        "apparel",
        "dress",
        "outfit",
        "bag",
        "bags",
        "jewelry",
        "wristwatch",
        "wristwatches",
    },
    ("Books and Literature",): {
        "book",
        "books",
        "novel",
        "novels",
        "fiction",
        "poetry",
        "comics",
        "graphic novel",
        "read",
        "reading",
    },
    ("Education",): {
        "education",
        "university",
        "universities",
        "college",
        "campus",
        "degree",
        "degrees",
        "masters",
        "master's",
        "postgraduate",
        "graduate school",
        "graduate schools",
        "admissions",
        "student",
        "students",
        "course",
        "courses",
        "class",
        "classes",
        "training",
        "learn",
        "learning",
        "online course",
        "certification",
    },
    ("Entertainment",): {
        "movie",
        "movies",
        "film",
        "films",
        "cinema",
        "soundtrack",
        "music",
        "tv",
        "show",
    },
    ("Healthy Living",): {
        "fitness",
        "exercise",
        "running",
        "jogging",
        "workout",
        "marathon",
        "10k",
        "5k",
        "nutrition",
        "wellness",
    },
    ("Medical Health",): {
        "medical",
        "doctor",
        "doctors",
        "clinic",
        "symptom",
        "symptoms",
        "diagnosis",
        "treatment",
        "allergy",
        "allergies",
        "pain",
        "injury",
        "mental health",
    },
    ("Home & Garden",): {
        "home",
        "house",
        "garden",
        "gardening",
        "plants",
        "plant",
        "backyard",
        "balcony garden",
        "tomato",
        "renovation",
        "remodel",
        "diy",
        "kitchen",
        "bathroom",
        "home improvement",
        "repair",
    },
    ("Real Estate",): {
        "apartment",
        "apartments",
        "rent",
        "rental",
        "lease",
        "leasing",
        "home buying",
        "house",
        "houses",
        "property",
    },
    ("Careers",): {
        "career",
        "careers",
        "job",
        "jobs",
        "hiring",
        "interview",
        "resume",
        "role",
        "openings",
        "salary",
        "remote work",
    },
    ("Family and Relationships",): {
        "parenting",
        "parents",
        "parent",
        "toddler",
        "teen",
        "teenager",
        "preschool",
        "daycare",
        "child",
        "children",
        "kids",
    },
    ("Personal Finance",): {
        "budget",
        "budgeting",
        "retirement",
        "saving",
        "savings",
        "debt",
        "credit",
        "loan",
        "loans",
        "investing",
        "financial aid",
        "taxes",
    },
    ("Sports",): {
        "sports",
        "soccer",
        "football",
        "premier league",
        "goal",
        "offside",
        "training drills",
        "athlete",
        "match",
    },
    ("Technology & Computing",): {
        "software",
        "app",
        "apps",
        "computer",
        "computers",
        "tech",
        "technology",
        "laptop",
        "laptops",
        "desktop",
        "desktops",
        "smartphone",
        "smartphones",
        "phone",
        "phones",
        "mobile",
        "hosting",
        "website",
        "deploy",
        "deployment",
        "ai",
        "ml",
        "nlp",
        "communication",
        "chat",
        "messaging",
    },
    ("Travel",): {
        "travel",
        "trip",
        "hotel",
        "hotels",
        "motel",
        "stay",
        "staying",
        "accommodation",
        "flight",
        "vacation",
    },
}

PATH_HINTS = {
    ("Business and Finance", "Business"): {
        "crm",
        "sales",
        "marketing",
        "startup",
        "business software",
        "lead management",
        "outbound",
    },
    ("Business and Finance", "Business", "Sales"): {
        "crm",
        "customer relationship management",
        "lead management",
        "pipeline",
        "sales software",
        "sales platform",
        "hubspot",
        "zoho",
        "salesforce",
        "pipedrive",
        "outreach",
        "apollo",
    },
    ("Business and Finance", "Business", "Marketing and Advertising"): {
        "seo",
        "ads",
        "advertising",
        "marketing software",
        "marketing tools",
        "campaign measurement",
        "attribution",
        "content marketing",
    },
    ("Business and Finance", "Business", "Business I.T."): {
        "password",
        "login",
        "account",
        "access",
        "identity",
        "permissions",
        "reset",
        "sso",
        "mfa",
        "billing",
        "subscription",
    },
    ("Food & Drink", "Dining Out"): {
        "restaurant",
        "restaurants",
        "dinner",
        "lunch",
        "brunch",
        "eat",
        "table",
        "reservation",
        "book a table",
        "book dinner",
    },
    ("Food & Drink", "Alcoholic Beverages"): {
        "vodka",
        "whiskey",
        "whisky",
        "tequila",
        "bourbon",
        "beer",
        "wine",
        "cocktail",
        "cocktails",
        "gin",
    },
    ("Style & Fashion",): {
        "shoes",
        "shoe",
        "sneakers",
        "boots",
        "footwear",
        "fashion",
        "apparel",
    },
    ("Books and Literature", "Fiction"): {
        "fiction",
        "novel",
        "novels",
        "fantasy novel",
        "thriller novel",
        "character-driven",
        "read",
    },
    ("Education", "Online Education"): {
        "online course",
        "online courses",
        "remote classes",
        "online learning",
        "internet-based training",
        "elearning",
        "e-learning",
        "learning python",
    },
    ("Education", "College Education"): {
        "college",
        "university",
        "universities",
        "campus",
        "college program",
        "college planning",
        "degree program",
    },
    ("Education", "College Education", "Postgraduate Education"): {
        "masters",
        "master's degree",
        "graduate school",
        "graduate schools",
        "postgraduate",
        "mba",
        "ms degree",
        "phd",
    },
    ("Entertainment", "Movies"): {
        "movie",
        "movies",
        "film",
        "films",
        "cinema",
        "thriller movies",
        "watch tonight",
    },
    ("Healthy Living", "Fitness and Exercise", "Running and Jogging"): {
        "running",
        "jogging",
        "marathon",
        "half marathon",
        "10k",
        "5k",
        "running plan",
        "pace",
    },
    ("Home & Garden", "Home Improvement"): {
        "home improvement",
        "remodel",
        "renovation",
        "kitchen remodel",
        "bathroom renovation",
        "diy",
        "upgrade house",
    },
    ("Home & Garden", "Gardening"): {
        "garden",
        "gardening",
        "plants",
        "plant care",
        "tomato plants",
        "balcony garden",
        "backyard garden",
    },
    ("Medical Health",): {
        "medical advice",
        "doctor",
        "symptoms",
        "diagnosis",
        "clinic",
        "allergy symptoms",
        "knee pain",
    },
    ("Careers", "Job Search"): {
        "jobs",
        "job search",
        "job openings",
        "remote jobs",
        "new role",
        "interview",
        "resume",
    },
    ("Personal Finance", "Financial Planning"): {
        "retirement",
        "budgeting",
        "save each month",
        "financial planning",
        "savings plan",
        "budgeting approach",
    },
    ("Family and Relationships", "Parenting"): {
        "parenting",
        "toddler",
        "teenager",
        "preschool",
        "child",
        "children",
        "kids",
    },
    ("Real Estate", "Real Estate Renting and Leasing"): {
        "apartment",
        "apartments",
        "rent",
        "rental",
        "lease",
        "leasing",
        "for rent",
        "rental listings",
    },
    ("Sports", "Soccer"): {
        "soccer",
        "offside",
        "premier league",
        "football tactics",
        "soccer drills",
        "midfielder",
        "striker",
    },
    ("Travel", "Travel Type", "Hotels and Motels"): {
        "hotel",
        "hotels",
        "motel",
        "motels",
        "place to stay",
        "accommodation",
        "lodging",
        "weekend trip",
    },
    ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear"): {
        "men's shoes",
        "mens shoes",
        "men sneakers",
        "men boots",
    },
    ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear"): {
        "women's shoes",
        "womens shoes",
        "women sneakers",
        "women boots",
        "heels",
    },
    ("Technology & Computing", "Artificial Intelligence"): {
        "artificial intelligence",
        "machine learning",
        "nlp",
        "llm",
        "language model",
        "intent classification",
    },
    ("Technology & Computing", "Computing"): {
        "software",
        "computer",
        "laptop",
        "desktop",
        "hosting",
        "web hosting",
        "communication software",
        "workflow software",
    },
    ("Technology & Computing", "Computing", "Computer Software and Applications"): {
        "software",
        "software applications",
        "business software",
        "productivity software",
        "workflow software",
        "project management",
    },
    ("Technology & Computing", "Computing", "Computer Software and Applications", "Communication"): {
        "communication software",
        "team chat",
        "messaging",
        "workplace communication",
        "slack",
        "microsoft teams",
        "discord",
    },
    ("Technology & Computing", "Computing", "Internet", "Web Hosting"): {
        "hosting",
        "web hosting",
        "deploy",
        "deployment",
        "website hosting",
        "managed hosting",
        "vercel",
        "netlify",
        "render",
    },
    ("Technology & Computing", "Computing", "Laptops"): {
        "laptop",
        "laptops",
        "notebook",
        "notebooks",
        "ultrabook",
        "macbook",
    },
    ("Technology & Computing", "Computing", "Desktops"): {
        "desktop",
        "desktops",
        "desktop pc",
        "desktop computer",
        "workstation",
        "gaming desktop",
        "imac",
    },
    ("Technology & Computing", "Consumer Electronics"): {
        "smartphone",
        "smartphones",
        "phone",
        "phones",
        "mobile",
        "iphone",
        "android",
    },
    ("Technology & Computing", "Consumer Electronics", "Smartphones"): {
        "smartphone",
        "smartphones",
        "phone",
        "phones",
        "iphone",
        "android",
        "mobile phone",
        "cell phone",
    },
    ("Automotive", "Auto Buying and Selling"): {
        "car",
        "cars",
        "auto buying",
        "used car",
        "vehicle shopping",
        "suv",
        "truck",
        "which car to buy",
    },
}

GENDER_MALE_HINTS = {"men", "mens", "man's", "male", "guy", "guys"}
GENDER_FEMALE_HINTS = {"women", "womens", "woman", "female", "ladies", "heels"}


def round_score(value: float) -> float:
    return round(float(value), 4)


def normalize(text: str) -> str:
    lowered = text.strip().lower()
    for source, target in NORMALIZATION_REPLACEMENTS.items():
        lowered = lowered.replace(source, target)
    return re.sub(r"\s+", " ", lowered)


def singularize(term: str) -> str:
    if term in {"sales", "news"}:
        return term
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        return term[:-1]
    return term


def tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize(text)) if token}


def phrase_in_text(lowered: str, phrase: str) -> bool:
    normalized_text = re.sub(r"[^a-z0-9]+", " ", normalize(lowered)).strip()
    normalized_phrase = re.sub(r"[^a-z0-9]+", " ", normalize(phrase)).strip()
    if not normalized_phrase:
        return False
    if f" {normalized_phrase} " not in f" {normalized_text} ":
        return False
    negation_pattern = rf"\bnot(?:\s+\w+){{0,2}}\s+{re.escape(normalized_phrase)}\b"
    return re.search(negation_pattern, normalized_text) is None


def path_terms(node: IabNode) -> set[str]:
    terms = {normalize(part) for part in node.path}
    terms.add(normalize(" ".join(node.path)))
    leaf = normalize(node.label)
    terms.add(leaf)
    terms.add(singularize(leaf))
    return {term for term in terms if term}


@lru_cache(maxsize=1)
def taxonomy_children() -> dict[tuple[str, ...], list[IabNode]]:
    taxonomy = get_iab_taxonomy()
    grouped: dict[tuple[str, ...], list[IabNode]] = {}
    for node in taxonomy.nodes:
        prefix = node.path[:-1]
        grouped.setdefault(prefix, []).append(node)
    for children in grouped.values():
        children.sort(key=lambda node: node.path)
    return grouped


def immediate_children(prefix: tuple[str, ...]) -> list[IabNode]:
    return taxonomy_children().get(prefix, [])


def phrase_score(lowered: str, phrases: set[str]) -> tuple[float, bool]:
    score = 0.0
    exact = False
    for phrase in phrases:
        normalized_phrase = normalize(phrase)
        if not normalized_phrase:
            continue
        if phrase_in_text(lowered, normalized_phrase):
            token_count = len(tokenize(normalized_phrase))
            score += 1.8 + (0.55 * min(token_count, 4))
            exact = True
    return score, exact


def token_overlap_score(tokens: set[str], candidate_tokens: set[str]) -> float:
    if not tokens or not candidate_tokens:
        return 0.0
    overlap = len(tokens & candidate_tokens)
    return overlap * 0.75


def buying_bias(path: tuple[str, ...], text_tokens: set[str]) -> float:
    if not (text_tokens & BUYING_TERMS):
        return 0.0
    path_tokens = tokenize(" ".join(path))
    buyable_tokens = {
        "buying",
        "selling",
        "software",
        "applications",
        "application",
        "laptops",
        "desktops",
        "smartphones",
        "phones",
        "hosting",
        "dining",
        "restaurants",
        "alcoholic",
        "beverages",
        "fashion",
        "footwear",
        "shopping",
    }
    return 1.25 if path_tokens & buyable_tokens else 0.0


def candidate_hint_set(path: tuple[str, ...]) -> set[str]:
    hints = set(PATH_HINTS.get(path, set()))
    if len(path) == 1:
        hints.update(TIER1_HINTS.get(path, set()))
    return hints


def candidate_score(
    node: IabNode,
    lowered: str,
    text_tokens: set[str],
    intent_type: str,
    subtype: str,
    decision_phase: str,
) -> tuple[float, bool]:
    score = 0.0
    exact = False

    term_hits, term_exact = phrase_score(lowered, path_terms(node))
    score += term_hits
    exact = exact or term_exact

    hint_hits, hint_exact = phrase_score(lowered, candidate_hint_set(node.path))
    score += hint_hits
    exact = exact or hint_exact

    score += token_overlap_score(text_tokens, tokenize(" ".join(node.path)))
    score += buying_bias(node.path, text_tokens)

    if decision_phase == "support" and node.path == ("Business and Finance", "Business", "Business I.T."):
        score += 2.0
    if subtype in {"comparison", "evaluation", "provider_selection"} and node.path == (
        "Business and Finance",
        "Business",
        "Sales",
    ):
        score += 1.4
    if subtype in COMMERCIAL_SUBTYPES and node.path == (
        "Technology & Computing",
        "Computing",
        "Computer Software and Applications",
        "Communication",
    ):
        score += 0.8
    if intent_type == "commercial" and node.path[:1] in {
        ("Automotive",),
        ("Business and Finance",),
        ("Technology & Computing",),
        ("Style & Fashion",),
        ("Food & Drink",),
    }:
        score += 0.3

    if node.path == ("Style & Fashion", "Men's Fashion", "Men's Shoes and Footwear") and not (
        text_tokens & GENDER_MALE_HINTS
    ):
        score -= 0.8
    if node.path == ("Style & Fashion", "Women's Fashion", "Women's Shoes and Footwear") and not (
        text_tokens & GENDER_FEMALE_HINTS
    ):
        score -= 0.8

    return score, exact


def fallback_path(intent_type: str, subtype: str, decision_phase: str) -> tuple[str, ...]:
    if decision_phase == "support":
        return ("Business and Finance", "Business", "Business I.T.")
    if subtype in {"signup", "purchase", "download"}:
        return ("Technology & Computing", "Computing", "Computer Software and Applications")
    if intent_type == "informational":
        return ("Education",)
    return ("Business and Finance", "Business")


def best_child(
    prefix: tuple[str, ...],
    lowered: str,
    text_tokens: set[str],
    intent_type: str,
    subtype: str,
    decision_phase: str,
) -> tuple[IabNode | None, float, float, bool]:
    candidates = immediate_children(prefix)
    if not candidates:
        return None, 0.0, 0.0, False

    scored = []
    for node in candidates:
        score, exact = candidate_score(node, lowered, text_tokens, intent_type, subtype, decision_phase)
        scored.append((node, score, exact))

    scored.sort(key=lambda item: (item[1], len(item[0].path), item[0].path), reverse=True)
    best_node, best_score, best_exact = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return best_node, best_score, second_score, best_exact


def should_descend(
    prefix: tuple[str, ...],
    best_score: float,
    second_score: float,
    best_exact: bool,
) -> bool:
    depth = len(prefix) + 1
    min_score = {1: 1.6, 2: 2.2, 3: 2.6, 4: 2.8}[depth]
    min_margin = {1: 0.2, 2: 0.35, 3: 0.55, 4: 0.7}[depth]
    if best_score < min_score:
        return False
    if best_exact:
        return True
    return (best_score - second_score) >= min_margin


def exact_path_override(lowered: str) -> tuple[tuple[str, ...], str, float] | None:
    best_match: tuple[int, int, tuple[str, ...]] | None = None
    for hints, path in EXACT_PATH_HINTS.items():
        matched_token_counts = [len(tokenize(hint)) for hint in hints if phrase_in_text(lowered, hint)]
        if not matched_token_counts:
            continue
        match = (max(matched_token_counts), len(path), path)
        if best_match is None or match > best_match:
            best_match = match
    if best_match is None:
        return None
    return best_match[2], "exact", 0.92


def score_targets(text: str, intent_type: str, subtype: str, decision_phase: str) -> tuple[tuple[str, ...], str, float]:
    lowered = normalize(text)
    override = exact_path_override(lowered)
    if override is not None:
        return override

    text_tokens = tokenize(lowered)
    selected_path: tuple[str, ...] | None = None
    mapping_mode = "nearest_equivalent"
    total_score = 0.0
    total_margin = 0.0
    prefix: tuple[str, ...] = ()

    for _depth in range(4):
        best_node, best_score, second_score, best_exact = best_child(
            prefix,
            lowered,
            text_tokens,
            intent_type,
            subtype,
            decision_phase,
        )
        if best_node is None or not should_descend(prefix, best_score, second_score, best_exact):
            break
        prefix = best_node.path
        selected_path = prefix
        total_score += best_score
        total_margin += max(best_score - second_score, 0.0)
        if best_exact and len(prefix) >= 3:
            mapping_mode = "exact"

    if selected_path is None:
        selected_path = fallback_path(intent_type, subtype, decision_phase)
        mapping_mode = "nearest_equivalent"
        total_score = 1.0
        total_margin = 0.0

    depth = len(selected_path)
    confidence = 0.47 + (depth * 0.07) + min(total_score, 10.0) * 0.025 + min(total_margin, 4.0) * 0.03
    confidence = max(0.5, min(confidence, 0.97))

    if depth <= 2:
        mapping_mode = "nearest_equivalent"

    return selected_path, mapping_mode, round_score(confidence)


def map_iab_content(text: str, intent_type: str, subtype: str, decision_phase: str) -> dict:
    taxonomy = get_iab_taxonomy()
    target_path, mapping_mode, mapping_confidence = score_targets(text, intent_type, subtype, decision_phase)
    return taxonomy.build_content_object(
        path=target_path,
        mapping_mode=mapping_mode,
        mapping_confidence=mapping_confidence,
    )
