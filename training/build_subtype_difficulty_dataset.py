from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "subtype_difficulty"
BENCHMARK_PATH = BASE_DIR / "data" / "subtype_benchmark.jsonl"

TRAIN_THEMES = (
    {
        "product": "CRM",
        "products": "CRM tools",
        "provider_a": "HubSpot",
        "provider_b": "Zoho",
        "team": "a 3-person startup",
        "goal": "manage leads better",
        "asset": "CRM buyer guide",
        "task": "import leads from a spreadsheet",
        "setup_task": "set up pipelines and invite teammates",
        "billing_object": "invoice",
        "account_object": "workspace",
        "booking_target": "demo",
        "budget": "$100 per month",
    },
    {
        "product": "analytics platform",
        "products": "analytics tools",
        "provider_a": "Mixpanel",
        "provider_b": "Amplitude",
        "team": "a product team",
        "goal": "measure activation",
        "asset": "analytics implementation checklist",
        "task": "generate a retention report",
        "setup_task": "connect events and configure dashboards",
        "billing_object": "renewal charge",
        "account_object": "dashboard login",
        "booking_target": "strategy call",
        "budget": "$300 per month",
    },
    {
        "product": "SEO platform",
        "products": "SEO tools",
        "provider_a": "Ahrefs",
        "provider_b": "Semrush",
        "team": "a content team",
        "goal": "grow organic traffic",
        "asset": "SEO tool comparison worksheet",
        "task": "export keyword rankings",
        "setup_task": "connect Search Console and create projects",
        "billing_object": "subscription charge",
        "account_object": "account login",
        "booking_target": "sales consultation",
        "budget": "$200 per month",
    },
)

BENCHMARK_THEMES = {
    "easy": {
        "product": "help desk platform",
        "products": "help desk tools",
        "provider_a": "Zendesk",
        "provider_b": "Freshdesk",
        "team": "a support team",
        "goal": "handle tickets faster",
        "asset": "help desk setup guide",
        "task": "export open tickets",
        "setup_task": "configure queues and invite agents",
        "billing_object": "invoice",
        "account_object": "agent account",
        "booking_target": "product demo",
        "budget": "$150 per month",
    },
    "medium": {
        "product": "project management tool",
        "products": "project management platforms",
        "provider_a": "Asana",
        "provider_b": "ClickUp",
        "team": "a remote operations team",
        "goal": "track work across projects",
        "asset": "project tool buying checklist",
        "task": "create a project status report",
        "setup_task": "set up spaces and invite the team",
        "billing_object": "plan renewal",
        "account_object": "workspace login",
        "booking_target": "implementation call",
        "budget": "$250 per month",
    },
    "hard": {
        "product": "hosting platform",
        "products": "hosting providers",
        "provider_a": "Vercel",
        "provider_b": "Netlify",
        "team": "a startup launch team",
        "goal": "ship a website reliably",
        "asset": "hosting migration guide",
        "task": "deploy a new site build",
        "setup_task": "connect the domain and configure redirects",
        "billing_object": "hosting invoice",
        "account_object": "deployment account",
        "booking_target": "technical demo",
        "budget": "$80 per month",
    },
}

SUBTYPE_TEMPLATES = {
    "education": {
        "easy": (
            "What is {product}?",
            "Explain {product}.",
            "How does {provider_a} work?",
            "What does {provider_a} do?",
            "Teach me the basics of {product}.",
        ),
        "medium": (
            "Help me understand what problem {product} solves for {team}.",
            "What should a beginner know about {products}?",
            "Before I compare anything, what is a {product}?",
            "What is {product} used for in practice?",
            "Give me an overview of how {products} help with {goal}.",
        ),
        "hard": (
            "I am not shopping yet; I just need to understand what {product} is.",
            "Before I evaluate vendors, explain the role of {product} for {team}.",
            "I keep hearing about {provider_a}; what is it actually for?",
            "I only want the fundamentals of {products} right now.",
            "Clarify what people mean by {product} without recommending a tool.",
        ),
    },
    "product_discovery": {
        "easy": (
            "Best {product} for {team}.",
            "What are the top {products} for {goal}?",
            "Show me the best {products}.",
            "What {products} are worth considering?",
            "Give me options for a {product}.",
        ),
        "medium": (
            "I need a shortlist of {products} for {team}.",
            "What are strong options if I need a {product} for {goal}?",
            "Which {products} should I look at first?",
            "What tools are worth exploring before I compare finalists?",
            "Point me to promising {products} for {team}.",
        ),
        "hard": (
            "I want to discover strong options before I narrow down to a few vendors.",
            "Help me build a shortlist of {products} without picking one yet.",
            "What products belong on the first pass for {goal}?",
            "I am early in vendor discovery and need good starting options.",
            "Map the space for {products} before we decide what to compare.",
        ),
    },
    "comparison": {
        "easy": (
            "{provider_a} vs {provider_b}.",
            "Compare {provider_a} and {provider_b}.",
            "Which is better, {provider_a} or {provider_b}?",
            "Give me a comparison of {provider_a} vs {provider_b}.",
            "How does {provider_a} compare with {provider_b}?",
        ),
        "medium": (
            "Compare {provider_a} and {provider_b} for {goal}.",
            "What are the tradeoffs between {provider_a} and {provider_b} for {team}?",
            "I need a side-by-side of {provider_a} and {provider_b}.",
            "How do {provider_a} and {provider_b} stack up on price and features?",
            "Walk me through a fair comparison of {provider_a} vs {provider_b}.",
        ),
        "hard": (
            "I have already researched the market; now compare the two finalists.",
            "Help me weigh {provider_a} against {provider_b} without making the final pick for me.",
            "I want the comparison layer, not just a list of tools.",
            "Break down the tradeoffs between {provider_a} and {provider_b} for our use case.",
            "I am comparing finalists and need a structured side-by-side view.",
        ),
    },
    "evaluation": {
        "easy": (
            "Is {provider_a} good for {team}?",
            "Would {provider_a} be a good fit for {goal}?",
            "How good is {provider_a}?",
            "Is {provider_a} worth considering?",
            "Can {provider_a} handle {goal}?",
        ),
        "medium": (
            "Evaluate whether {provider_a} fits {team}.",
            "I need to assess if {provider_a} is strong enough for {goal}.",
            "What are the pros and cons of {provider_a} for our use case?",
            "Does {provider_a} look like a fit before I move to a final choice?",
            "How well would {provider_a} perform for {team}?",
        ),
        "hard": (
            "I am not asking for a shortlist or a direct comparison; I want an evaluation of {provider_a}.",
            "Pressure-test whether {provider_a} actually fits our workflow.",
            "Assess {provider_a} on its own merits before I commit to a final pick.",
            "I want a fit evaluation for {provider_a}, not a vendor comparison.",
            "Tell me whether {provider_a} seems viable for {goal} before I decide.",
        ),
    },
    "deal_seeking": {
        "easy": (
            "Best {product} under {budget}.",
            "Cheap {products} for {team}.",
            "What is the most affordable {product}?",
            "Show me lower-cost {products}.",
            "What {products} fit a budget of {budget}?",
        ),
        "medium": (
            "I need a {product} for {team} but want to stay under {budget}.",
            "What are the cheapest solid options for {goal}?",
            "Compare affordable {products} that still work well.",
            "Help me find a budget-friendly alternative to {provider_a}.",
            "Which options cost less without being terrible?",
        ),
        "hard": (
            "Price is the main filter now; I need good options that stay under {budget}.",
            "I want to optimize for value and keep costs down while choosing a {product}.",
            "I am narrowing the shortlist based on affordability first.",
            "Find the best-value route here rather than the premium option.",
            "Budget is the deciding factor, so surface the strongest low-cost options.",
        ),
    },
    "provider_selection": {
        "easy": (
            "Which should I choose, {provider_a} or {provider_b}?",
            "What is the best option for me right now?",
            "Which {product} should I go with?",
            "Help me choose the right {product}.",
            "What do you recommend between {provider_a} and {provider_b}?",
        ),
        "medium": (
            "I need to make the call between {provider_a} and {provider_b}.",
            "Which option should {team} commit to this week?",
            "Help me pick the final vendor for {goal}.",
            "We have a shortlist; which one should we choose?",
            "If you had to choose one for {team}, what would it be?",
            "{provider_a} or {provider_b} for {team} right now?",
        ),
        "hard": (
            "I am done comparing and need the final recommendation now.",
            "We have enough information; tell us which provider to commit to.",
            "I need the final pick, not another comparison round.",
            "Choose the best fit for {team} between the finalists.",
            "We are at decision time; which vendor should win?",
        ),
    },
    "signup": {
        "easy": (
            "Start my free trial.",
            "Create my account for {provider_a}.",
            "Sign me up for {provider_a}.",
            "Open a new account for {provider_a}.",
            "Register me for a trial of {provider_a}.",
        ),
        "medium": (
            "Help me create a workspace for {provider_a}.",
            "I want to activate a trial for {provider_a} today.",
            "Set up a new account so our team can try {provider_a}.",
            "Get me signed up for {provider_a} now.",
            "Open a starter account for {team}.",
            "Subscribe me to product updates from {provider_a}.",
        ),
        "hard": (
            "I am ready to get access now and need the account creation step.",
            "We decided to try it, so point me to the signup flow.",
            "I want to start using {provider_a} immediately and need the trial path.",
            "This is an access request, not a support issue; help me register.",
            "Move me into the trial/signup step for {provider_a}.",
        ),
    },
    "purchase": {
        "easy": (
            "Buy {provider_a} now.",
            "Purchase the annual plan for {provider_a}.",
            "Take me to checkout for {provider_a}.",
            "I want to pay for {provider_a}.",
            "Subscribe to the pro plan today.",
        ),
        "medium": (
            "Help me complete the purchase for {provider_a}.",
            "I am ready to place the order for {provider_a}.",
            "Charge me for the annual plan now.",
            "I want to move from trial to paid today.",
            "Where do I finish payment for {provider_a}?",
        ),
        "hard": (
            "The decision is made; I need the paid conversion step now.",
            "I am not asking for a trial anymore, I want to purchase immediately.",
            "Point me straight to the paid checkout flow.",
            "We are ready to buy and need to complete billing today.",
            "This is the purchase step, not a vendor-evaluation question.",
        ),
    },
    "booking": {
        "easy": (
            "Book a {booking_target} with {provider_a}.",
            "Schedule a {booking_target}.",
            "Reserve a meeting with {provider_a}.",
            "Set up a call for next week.",
            "Book time with the team.",
        ),
        "medium": (
            "I want to schedule a live walkthrough with {provider_a}.",
            "Find time for a {booking_target} with sales next week.",
            "Reserve a slot for a product demo.",
            "Can you help me book a meeting about {provider_a}?",
            "Put a calendar hold in place for the {booking_target}.",
        ),
        "hard": (
            "We want the live session itself, not a generic sales contact form.",
            "Help me lock in a scheduled demo rather than just talking to sales.",
            "This needs a booked slot on the calendar, not just a quote request.",
            "I want to reserve meeting time for the next step.",
            "Schedule the actual session so we can move forward.",
        ),
    },
    "download": {
        "easy": (
            "Download the {asset}.",
            "Send me the {asset}.",
            "Export the {asset}.",
            "Give me the download link for the {asset}.",
            "Open the file download for the {asset}.",
        ),
        "medium": (
            "I need the downloadable {asset} right now.",
            "Where can I get the PDF version of the {asset}?",
            "Help me export the {asset} as a file.",
            "I want the guide itself, not a summary of it.",
            "Let me download the asset for offline review.",
        ),
        "hard": (
            "I am asking for the file delivery step, not more information about the content.",
            "Please route me to the actual downloadable asset.",
            "I need the guide as a file I can keep, not a recommendation.",
            "This is a download request, not a buying-decision question.",
            "Give me the direct asset retrieval path.",
        ),
    },
    "contact_sales": {
        "easy": (
            "Talk to sales about {provider_a}.",
            "Contact sales for pricing.",
            "Put me in touch with an account executive.",
            "I need a sales conversation for {provider_a}.",
            "Connect me with the sales team.",
        ),
        "medium": (
            "I need to talk to sales about enterprise pricing for {provider_a}.",
            "Have a rep contact me about a custom quote.",
            "Route me to the sales team for a package discussion.",
            "I want a pricing conversation with sales, not a product demo.",
            "Connect me with someone on the sales side for {provider_a}.",
        ),
        "hard": (
            "I want a rep to reach out, but I am not trying to schedule the call myself right now.",
            "This is a sales-contact request rather than a booking flow.",
            "Have the account team get in touch about our requirements.",
            "We need a commercial conversation with sales, not self-serve checkout.",
            "Point me to the sales contact path for a custom deal.",
        ),
    },
    "task_execution": {
        "easy": (
            "Help me {task}.",
            "Run the next step to {task}.",
            "Create the report I need in {provider_a}.",
            "Show me how to complete the task in {provider_a}.",
            "I want to execute a task right now.",
            "Open the annotation dashboard.",
        ),
        "medium": (
            "Walk me through the exact steps to {task}.",
            "I am already in the product and need help completing a task.",
            "Tell me how to do the action itself inside {provider_a}.",
            "I need execution help for {task}, not onboarding advice.",
            "Help me finish the workflow to {task}.",
            "Send me the benchmark report for {provider_a}.",
        ),
        "hard": (
            "This is a discrete in-product action request, not a setup flow.",
            "I know the tool already; I just need help executing {task}.",
            "Guide me through the one task I am trying to complete now.",
            "I am not setting up the whole system, just doing a specific action.",
            "Focus on the immediate workflow step rather than implementation planning.",
            "Schedule the retraining pipeline rather than booking a meeting.",
        ),
    },
    "onboarding_setup": {
        "easy": (
            "How do I {setup_task} in {provider_a}?",
            "Help me set up {provider_a} for {team}.",
            "What should I configure first after signup?",
            "Show me how to onboard our team in {provider_a}.",
            "How do I get started after we bought {provider_a}?",
            "What should we do first after buying {provider_a}?",
        ),
        "medium": (
            "We already subscribed; how should we set up {provider_a} for {team}?",
            "I need rollout guidance after purchase for {provider_a}.",
            "What is the best onboarding sequence after we start using {provider_a}?",
            "Help me implement {provider_a}, not just do one task.",
            "We are past signup and need setup guidance for adoption.",
            "How do I set goals for our first month using {provider_a}?",
        ),
        "hard": (
            "This is a post-purchase setup question rather than a one-off task request.",
            "Help us implement and configure {provider_a} across the team.",
            "We are already customers; now we need onboarding and rollout help.",
            "I want setup guidance for the system as a whole, not a single workflow.",
            "What should happen first, second, and third after we activate {provider_a}?",
        ),
    },
    "troubleshooting": {
        "easy": (
            "{provider_a} keeps failing.",
            "Why is {provider_a} broken?",
            "The integration is not working.",
            "I am getting an error in {provider_a}.",
            "Help me fix a broken workflow.",
        ),
        "medium": (
            "Why does {provider_a} keep crashing when I try to work?",
            "I need help fixing a problem in the product right now.",
            "The workflow breaks every time I try to {task}.",
            "Can you troubleshoot why {provider_a} is failing?",
            "I am stuck because the system is not working correctly.",
        ),
        "hard": (
            "This is an issue-resolution request, not a billing or account-access question.",
            "Something in the product is malfunctioning and I need troubleshooting help.",
            "I need the cause of the failure diagnosed, not onboarding advice.",
            "Help me debug the broken behavior inside {provider_a}.",
            "The system is failing during use and I need it fixed.",
        ),
    },
    "account_help": {
        "easy": (
            "I cannot log into my {account_object}.",
            "How do I reset my password?",
            "My sign-in code is not working.",
            "I am locked out of my account.",
            "The password reset email never arrives.",
        ),
        "medium": (
            "Help me regain access to my {account_object}.",
            "I need support with login and access, not signup.",
            "My credentials keep getting rejected.",
            "I cannot get back into the account I already have.",
            "Can you help me unlock my existing account?",
        ),
        "hard": (
            "This is an access-recovery issue for an existing account, not a new-account request.",
            "I already have an account and need help getting back in.",
            "Treat this as login support rather than a registration flow.",
            "I need account-access help for a live customer account.",
            "This is a credentials problem, not a trial-signup question.",
        ),
    },
    "billing_help": {
        "easy": (
            "My {billing_object} is wrong.",
            "Why was I charged twice?",
            "My payment failed.",
            "How do I fix a billing issue?",
            "The invoice total looks incorrect.",
        ),
        "medium": (
            "I need help with a billing problem on my existing plan.",
            "Why did the renewal charge fail for {provider_a}?",
            "Help me resolve an incorrect payment issue.",
            "Our invoice and subscription charges do not match.",
            "I need billing support for the account we already pay for.",
        ),
        "hard": (
            "This is a billing-support issue rather than a purchasing decision.",
            "I already bought the product and need help with charges.",
            "Treat this as an account billing problem, not a checkout flow.",
            "I need a live-customer invoice issue resolved.",
            "This is post-purchase billing support, not a new-sale question.",
        ),
    },
    "follow_up": {
        "easy": (
            "Tell me more.",
            "Can you explain that?",
            "Go on.",
            "What do you mean?",
            "Show me more.",
        ),
        "medium": (
            "Can you expand on that a little?",
            "Why do you say that?",
            "Which one do you mean?",
            "Say more about that.",
            "Okay and then what?",
        ),
        "hard": (
            "I am following up on what you just said rather than starting a fresh request.",
            "This depends on prior context, so explain the previous answer a bit more.",
            "Continue the last thread rather than giving me a new recommendation.",
            "I am only asking for clarification on the prior response.",
            "Give me a little more detail on the last point.",
        ),
    },
    "emotional_reflection": {
        "easy": (
            "I feel overwhelmed by this decision.",
            "I am frustrated with the whole process.",
            "I am anxious about picking the wrong tool.",
            "This is stressing me out.",
            "I feel stuck and discouraged.",
        ),
        "medium": (
            "I keep second-guessing myself and it is making me uneasy.",
            "I am worried we will choose badly and regret it.",
            "This whole thing makes me feel burned out.",
            "I feel nervous about making the wrong call here.",
            "I am emotionally drained by trying to figure this out.",
        ),
        "hard": (
            "I do not need another recommendation right now; I need help processing how stuck I feel.",
            "This is more about how I feel than about the product options.",
            "I want support with the frustration this decision is causing me.",
            "I am reflecting on my stress around this choice, not asking for a commercial answer.",
            "Help me with the feeling of being overwhelmed rather than the task itself.",
        ),
    },
}


def split_for_index(index: int) -> str:
    bucket = index % 5
    if bucket < 3:
        return "train"
    if bucket == 3:
        return "val"
    return "test"


def build_row(text: str, subtype: str, difficulty: str, source: str) -> dict[str, str]:
    return {
        "text": text,
        "intent_subtype": subtype,
        "difficulty": difficulty,
        "source": source,
    }


def main() -> None:
    splits = {"train": [], "val": [], "test": []}
    benchmark_rows: list[dict[str, str]] = []

    for subtype, difficulty_templates in SUBTYPE_TEMPLATES.items():
        generated_index = 0
        for difficulty in ("easy", "medium", "hard"):
            for theme in TRAIN_THEMES:
                for template in difficulty_templates[difficulty]:
                    row = build_row(
                        text=template.format(**theme),
                        subtype=subtype,
                        difficulty=difficulty,
                        source="subtype_difficulty",
                    )
                    splits[split_for_index(generated_index)].append(row)
                    generated_index += 1
            benchmark_theme = BENCHMARK_THEMES[difficulty]
            for template in difficulty_templates[difficulty]:
                benchmark_rows.append(
                    build_row(
                        text=template.format(**benchmark_theme),
                        subtype=subtype,
                        difficulty=difficulty,
                        source="subtype_difficulty_benchmark",
                    )
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{split_name}: {len(rows)} rows")

    BENCHMARK_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in benchmark_rows),
        encoding="utf-8",
    )
    print(f"benchmark: {len(benchmark_rows)} rows")


if __name__ == "__main__":
    main()
