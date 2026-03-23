from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from iab_taxonomy import get_iab_taxonomy

BENCHMARK_PATH = BASE_DIR / "data" / "iab_cross_vertical_benchmark.jsonl"
CASE_PATH = BASE_DIR / "examples" / "iab_cross_vertical_mapping_cases.json"


SCENARIOS = [
    {
        "slug": "auto-buying",
        "label": "Automotive > Auto Buying and Selling",
        "mapping_mode": "nearest_equivalent",
        "prompts": {
            "easy": "Which car should I buy for commuting?",
            "medium": "Best used SUV for a family of four",
            "hard": "I need a shortlist of practical cars before making a purchase this month",
        },
    },
    {
        "slug": "sales-crm",
        "label": "Business and Finance > Business > Sales",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "What is CRM software?",
            "medium": "HubSpot vs Zoho for a small team",
            "hard": "Need software to manage leads and pipeline for a startup sales team",
        },
    },
    {
        "slug": "marketing-tools",
        "label": "Business and Finance > Business > Marketing and Advertising",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best SEO tools for content teams",
            "medium": "How should I compare ad attribution platforms?",
            "hard": "Need software to measure channel performance across paid and organic campaigns",
        },
    },
    {
        "slug": "business-it",
        "label": "Business and Finance > Business > Business I.T.",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "How do I reset my work password?",
            "medium": "My employees keep getting locked out of their accounts",
            "hard": "Need identity and access software for login, permissions, and account security",
        },
    },
    {
        "slug": "dining-out",
        "label": "Food & Drink > Dining Out",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Book a table for six tonight",
            "medium": "Good restaurants for a client dinner downtown",
            "hard": "Need a place to eat tonight where I can make a reservation online",
        },
    },
    {
        "slug": "alcoholic-beverages",
        "label": "Food & Drink > Alcoholic Beverages",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Which whiskey cocktail should I order?",
            "medium": "Best vodka drinks for beginners",
            "hard": "Want a spirit-forward drink recommendation, not a restaurant suggestion",
        },
    },
    {
        "slug": "artificial-intelligence",
        "label": "Technology & Computing > Artificial Intelligence",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "What is intent classification in NLP?",
            "medium": "How do large language models handle text classification?",
            "hard": "Need the machine learning concept behind language understanding, not software to buy",
        },
    },
    {
        "slug": "software-apps",
        "label": "Technology & Computing > Computing > Computer Software and Applications",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best workflow software for a small operations team",
            "medium": "Need project management software for a distributed team",
            "hard": "Looking for a business software platform to organize internal workflows",
        },
    },
    {
        "slug": "communication-software",
        "label": "Technology & Computing > Computing > Computer Software and Applications > Communication",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best communication software for remote teams",
            "medium": "Slack vs Teams for internal messaging",
            "hard": "Need a workplace chat tool for cross-functional collaboration",
        },
    },
    {
        "slug": "web-hosting",
        "label": "Technology & Computing > Computing > Internet > Web Hosting",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Vercel vs Netlify for website hosting",
            "medium": "Best hosting platform for a startup website",
            "hard": "Need a managed hosting provider to deploy and run our marketing site",
        },
    },
    {
        "slug": "laptops",
        "label": "Technology & Computing > Computing > Laptops",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Which laptop should I buy for college?",
            "medium": "Best laptop for work and study under 1200",
            "hard": "Need a portable computer with good battery life for everyday work",
        },
    },
    {
        "slug": "desktops",
        "label": "Technology & Computing > Computing > Desktops",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best desktop for video editing",
            "medium": "Which desktop computer should I buy for a home office?",
            "hard": "Need a desktop PC with strong performance for creative work",
        },
    },
    {
        "slug": "smartphones",
        "label": "Technology & Computing > Consumer Electronics > Smartphones",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best phone with a good camera under 700",
            "medium": "Should I buy an iPhone or Pixel this year?",
            "hard": "Need a new smartphone with strong battery life and a clean software experience",
        },
    },
    {
        "slug": "style-fashion-parent",
        "label": "Style & Fashion",
        "mapping_mode": "nearest_equivalent",
        "prompts": {
            "easy": "Best shoes under 100 dollars",
            "medium": "Affordable fashion accessories for everyday wear",
            "hard": "Need style recommendations for clothing and footwear without a specific brand in mind",
        },
    },
    {
        "slug": "womens-shoes",
        "label": "Style & Fashion > Women's Fashion > Women's Shoes and Footwear",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best women's running shoes under 100 dollars",
            "medium": "Comfortable women's sneakers for walking all day",
            "hard": "Need women's footwear for commuting that looks polished but feels comfortable",
        },
    },
    {
        "slug": "mens-shoes",
        "label": "Style & Fashion > Men's Fashion > Men's Shoes and Footwear",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best men's sneakers for daily wear",
            "medium": "Good men's dress shoes for office use",
            "hard": "Need men's footwear that works for workdays and weekend walking",
        },
    },
    {
        "slug": "hotels",
        "label": "Travel > Travel Type > Hotels and Motels",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Need a hotel in Chicago for two nights",
            "medium": "Best hotels near Times Square for a weekend trip",
            "hard": "Looking for a place to stay during a work trip, not general travel advice",
        },
    },
    {
        "slug": "real-estate-rentals",
        "label": "Real Estate > Real Estate Renting and Leasing",
        "mapping_mode": "nearest_equivalent",
        "prompts": {
            "easy": {
                "text": "Apartments for rent near downtown Austin",
                "mapping_mode": "exact",
            },
            "medium": "Best neighborhoods to lease a two-bedroom apartment in Seattle",
            "hard": {
                "text": "Need rental listings for a short move, not home-buying advice",
                "mapping_mode": "exact",
            },
        },
    },
    {
        "slug": "running-and-jogging",
        "label": "Healthy Living > Fitness and Exercise > Running and Jogging",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best running plan for a first 10k",
            "medium": "How should I train for a half marathon as a beginner?",
            "hard": "Need guidance on building a weekly jogging routine without getting injured",
        },
    },
    {
        "slug": "soccer",
        "label": "Sports > Soccer",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "How do offside rules work in soccer?",
            "medium": "Best soccer drills for beginner players",
            "hard": "Need help understanding football tactics for the Premier League, not fantasy sports",
        },
    },
    {
        "slug": "fiction",
        "label": "Books and Literature > Fiction",
        "mapping_mode": "nearest_equivalent",
        "prompts": {
            "easy": {
                "text": "Recommend a good fantasy novel to read",
                "mapping_mode": "exact",
            },
            "medium": "Best fiction books for a long flight",
            "hard": {
                "text": "Looking for a character-driven novel, not comics or poetry",
                "mapping_mode": "exact",
            },
        },
    },
    {
        "slug": "home-improvement",
        "label": "Home & Garden > Home Improvement",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "How much does a kitchen remodel usually cost?",
            "medium": "Best tools for a DIY bathroom renovation",
            "hard": "Need practical advice for upgrading an older house, not interior decor inspiration",
        },
    },
    {
        "slug": "online-education",
        "label": "Education > Online Education",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "Best online courses for learning Python",
            "medium": "What are good platforms for remote professional classes?",
            "hard": "Need internet-based training options I can finish after work hours",
        },
    },
    {
        "slug": "postgraduate-education",
        "label": "Education > College Education > Postgraduate Education",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "best universities to study masters",
            "medium": "which graduate schools have strong data science programs",
            "hard": "need postgraduate options for a master's degree, not short online courses",
        },
    },
    {
        "slug": "medical-health",
        "label": "Medical Health",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "what do these allergy symptoms mean",
            "medium": "when should i see a doctor for persistent knee pain",
            "hard": "need medical advice about symptoms, not wellness or fitness tips",
        },
    },
    {
        "slug": "careers-job-search",
        "label": "Careers > Job Search",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "best remote jobs for data analysts",
            "medium": "where should i look for product manager openings",
            "hard": "need help finding a new role and preparing for interviews",
        },
    },
    {
        "slug": "personal-finance",
        "label": "Personal Finance > Financial Planning",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "how much should i save each month",
            "medium": "best budgeting approach for a growing family",
            "hard": "need help planning savings and retirement, not business finance advice",
        },
    },
    {
        "slug": "parenting",
        "label": "Family and Relationships > Parenting",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "tips for parenting a toddler",
            "medium": "how do i help my teenager spend less time online",
            "hard": "need parenting advice for a child starting preschool",
        },
    },
    {
        "slug": "gardening",
        "label": "Home & Garden > Gardening",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "best plants for a small balcony garden",
            "medium": "how often should i water tomato plants",
            "hard": "need gardening advice for a shady backyard, not interior decor ideas",
        },
    },
    {
        "slug": "movies",
        "label": "Entertainment > Movies",
        "mapping_mode": "exact",
        "prompts": {
            "easy": "What movie should we watch tonight?",
            "medium": "Best thriller movies from the last few years",
            "hard": "Looking for film recommendations, not TV shows or music",
        },
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_expected(label: str, mapping_mode: str) -> dict:
    taxonomy = get_iab_taxonomy()
    content = taxonomy.build_content_object_from_label(label, mapping_mode=mapping_mode, mapping_confidence=0.9)
    expected = {
        "model_output.classification.iab_content.tier1.label": content["tier1"]["label"],
        "model_output.classification.iab_content.mapping_mode": mapping_mode,
    }
    if "tier2" in content:
        expected["model_output.classification.iab_content.tier2.label"] = content["tier2"]["label"]
    if "tier3" in content:
        expected["model_output.classification.iab_content.tier3.label"] = content["tier3"]["label"]
    if "tier4" in content:
        expected["model_output.classification.iab_content.tier4.label"] = content["tier4"]["label"]
    return expected


def build_rows() -> tuple[list[dict], list[dict]]:
    benchmark_rows: list[dict] = []
    cases: list[dict] = []
    for scenario in SCENARIOS:
        for difficulty, prompt_config in scenario["prompts"].items():
            if isinstance(prompt_config, dict):
                text = prompt_config["text"]
                mapping_mode = prompt_config.get("mapping_mode", scenario["mapping_mode"])
            else:
                text = prompt_config
                mapping_mode = scenario["mapping_mode"]
            benchmark_rows.append(
                {
                    "difficulty": difficulty,
                    "iab_path": scenario["label"],
                    "source": "iab_cross_vertical_benchmark",
                    "text": text,
                }
            )
            cases.append(
                {
                    "id": f"{scenario['slug']}-{difficulty}",
                    "status": "must_fix",
                    "text": text,
                    "notes": f"Cross-vertical {difficulty} IAB mapping case for {scenario['label']}.",
                    "expected": build_expected(scenario["label"], mapping_mode),
                }
            )
    return benchmark_rows, cases


def main() -> None:
    benchmark_rows, cases = build_rows()
    write_jsonl(BENCHMARK_PATH, benchmark_rows)
    CASE_PATH.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"benchmark: {len(benchmark_rows)} rows")
    print(f"cases: {len(cases)} rows")


if __name__ == "__main__":
    main()
