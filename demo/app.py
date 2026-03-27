"""
AdMesh Agentic Intent Classifier — Interactive Demo

Deploy as a Hugging Face Space (Gradio SDK):
    https://huggingface.co/spaces/manikumargouni/admesh_intent_classifier_demo
"""

from __future__ import annotations

import json
import time
from textwrap import dedent

import gradio as gr
from transformers import pipeline

# ── Semantic badge colours (only these are fixed) ────────────────────────────

SUCCESS = "#0D9668"
DANGER = "#DC2626"
WARNING = "#D97706"

# CSS var shortcuts — adapt to light/dark automatically
V_BG = "var(--block-background-fill)"
V_TEXT = "var(--body-text-color)"
V_MUTED = "var(--body-text-color-subdued, #71717a)"
V_BORDER = "var(--block-border-color)"
V_TRACK = "var(--block-border-color)"

# ── Model loading ────────────────────────────────────────────────────────────

clf = pipeline(
    "admesh-intent",
    model="admesh/agentic-intent-classifier",
    trust_remote_code=True,
)

# ── Colour maps ──────────────────────────────────────────────────────────────

INTENT_COLORS = {
    "commercial": "#18181b",
    "transactional": "#18181b",
    "informational": SUCCESS,
    "navigational": WARNING,
    "prohibited": DANGER,
    "ambiguous": "#71717a",
}

ELIGIBILITY_COLORS = {
    "allowed": SUCCESS,
    "allowed_with_caution": WARNING,
    "restricted": "#EA580C",
    "not_allowed": DANGER,
}

PHASE_ICONS = {
    "awareness": "🔍",
    "consideration": "⚖️",
    "decision": "🎯",
    "action": "🛒",
    "support": "🛟",
    "retention": "🔄",
}


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:20px;font-size:0.82em;font-weight:600;'
        f'font-family:Poppins,sans-serif;letter-spacing:0.02em">{text}</span>'
    )


def _confidence_bar(value: float, label: str, meets: bool) -> str:
    pct = int(value * 100)
    color = V_TEXT if meets else DANGER
    return (
        f'<div style="margin:6px 0">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82em;'
        f'font-family:Poppins,sans-serif;color:{V_TEXT}">'
        f"<span>{label}</span><span style='font-weight:600'>{pct}%</span></div>"
        f'<div style="background:{V_TRACK};border-radius:6px;height:8px;overflow:hidden;margin-top:3px">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:6px;'
        f'transition:width 0.4s ease"></div>'
        f"</div></div>"
    )


def _card(content: str) -> str:
    return (
        f'<div style="padding:20px;border-radius:14px;border:1px solid {V_BORDER};'
        f'background:{V_BG};font-family:Poppins,sans-serif">{content}</div>'
    )


# ── Classification logic ────────────────────────────────────────────────────

def classify(query: str):
    if not query or not query.strip():
        empty = (
            f'<p style="color:{V_MUTED};font-family:Poppins,sans-serif;'
            f'text-align:center;padding:24px">Enter a query above to classify.</p>'
        )
        return empty, empty, empty, empty, "{}"

    t0 = time.perf_counter()
    result = clf(query.strip())
    latency_ms = (time.perf_counter() - t0) * 1000

    intent = result["model_output"]["classification"]["intent"]
    iab = result["model_output"]["classification"]["iab_content"]
    policy = result["system_decision"]["policy"]
    opportunity = result["system_decision"]["opportunity"]
    meta = result["meta"]
    fallback = result["model_output"].get("fallback")
    cc = intent["component_confidence"]

    # ── Panel 1: Intent summary ──────────────────────────────────────────
    intent_color = INTENT_COLORS.get(intent["type"], "#71717a")
    phase_icon = PHASE_ICONS.get(intent["decision_phase"], "")

    summary_inner = dedent(f"""\
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap">
        {_badge(intent["type"], intent_color)}
        {_badge(intent["subtype"], "#3f3f46")}
        {_badge(f'{phase_icon} {intent["decision_phase"]}', "#52525b")}
      </div>
      <p style="margin:8px 0 16px 0;color:{V_TEXT};font-size:0.92em;line-height:1.5">{intent["summary"]}</p>
      <div style="display:flex;gap:28px;flex-wrap:wrap">
        <div>
          <div style="font-size:0.72em;color:{V_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">Confidence</div>
          <div style="font-size:1.6em;font-weight:700;color:{V_TEXT}">{int(intent["confidence"]*100)}%</div>
        </div>
        <div>
          <div style="font-size:0.72em;color:{V_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">Commercial Score</div>
          <div style="font-size:1.6em;font-weight:700;color:{V_TEXT}">{int(intent["commercial_score"]*100)}%</div>
        </div>
        <div>
          <div style="font-size:0.72em;color:{V_MUTED};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">Latency</div>
          <div style="font-size:1.6em;font-weight:700;color:{V_TEXT}">{latency_ms:.0f}ms</div>
        </div>
      </div>
    """)
    summary_html = _card(summary_inner)

    # ── Panel 2: Confidence breakdown ────────────────────────────────────
    bars = ""
    for head_key, display_name in [
        ("intent_type", "Intent Type"),
        ("intent_subtype", "Intent Subtype"),
        ("decision_phase", "Decision Phase"),
    ]:
        h = cc[head_key]
        bars += _confidence_bar(h["confidence"], f"{display_name}: {h['label']}", h["meets_threshold"])

    confidence_inner = dedent(f"""\
      <h4 style="margin:0 0 10px 0;font-size:0.88em;color:{V_TEXT};font-weight:600">Component Confidence</h4>
      {bars}
      <p style="margin:10px 0 0 0;font-size:0.72em;color:{V_MUTED}">
        Calibrated: {meta["calibration_enabled"]} &nbsp;|&nbsp; Strategy: {cc.get("overall_strategy", "n/a")}
      </p>
    """)
    confidence_html = _card(confidence_inner)

    # ── Panel 3: IAB + Monetization ──────────────────────────────────────
    iab_path_parts = []
    for tier in ["tier1", "tier2", "tier3", "tier4"]:
        t = iab.get(tier)
        if t and t.get("label"):
            iab_path_parts.append(t["label"])
    iab_path = " › ".join(iab_path_parts) if iab_path_parts else "—"
    is_placeholder = meta.get("iab_mapping_is_placeholder", False)

    elig = policy["monetization_eligibility"]
    elig_color = ELIGIBILITY_COLORS.get(elig, "#71717a")
    opp_type = opportunity["type"]
    opp_strength = opportunity["strength"]

    fallback_note = ""
    if fallback and fallback.get("applied"):
        fallback_note = (
            f'<div style="margin-top:10px;padding:10px 14px;background:rgba(253,230,138,0.15);'
            f'border-radius:10px;font-size:0.82em;border:1px solid rgba(253,230,138,0.4);color:{V_TEXT}">'
            f'⚠️ Fallback applied — reason: {fallback.get("reason", "unknown")}'
            f"</div>"
        )

    policy_inner = dedent(f"""\
      <h4 style="margin:0 0 10px 0;font-size:0.88em;color:{V_TEXT};font-weight:600">IAB Content Taxonomy</h4>
      <p style="margin:0;font-size:0.95em;font-weight:600;color:{V_TEXT}">{iab_path}</p>
      <p style="margin:3px 0 16px 0;font-size:0.72em;color:{V_MUTED}">
        Mode: {iab.get("mapping_mode", "—")} &nbsp;|&nbsp;
        Confidence: {int(iab.get("mapping_confidence", 0)*100)}%
        {"&nbsp;|&nbsp; ⚠️ placeholder" if is_placeholder else ""}
      </p>
      <h4 style="margin:0 0 10px 0;font-size:0.88em;color:{V_TEXT};font-weight:600">Monetization Decision</h4>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        {_badge(elig, elig_color)}
        <span style="font-size:0.82em;color:{V_MUTED}">
          Opportunity: {opp_type} ({opp_strength})
        </span>
      </div>
      <p style="margin:8px 0 0 0;font-size:0.78em;color:{V_MUTED}">{policy.get("eligibility_reason", "")}</p>
      {fallback_note}
    """)
    policy_html = _card(policy_inner)

    # ── Panel 4: Full JSON ───────────────────────────────────────────────
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    return summary_html, confidence_html, policy_html, json_str, json_str


# ── Example queries ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["Which laptop should I buy for college?"],
    ["Best running shoes under $100"],
    ["How does photosynthesis work?"],
    ["Navigate to Gmail inbox"],
    ["Compare AWS vs Azure vs GCP for startups"],
    ["Buy iPhone 16 Pro Max 256GB"],
    ["What is the meaning of life?"],
    ["Best CRM software for a 5-person startup"],
    ["How to fix a leaking faucet"],
    ["Tesla Model 3 vs Model Y review"],
]

# ── Gradio theme (monochrome) ────────────────────────────────────────────────

admesh_theme = gr.themes.Soft(
    primary_hue="neutral",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

# ── Gradio UI ────────────────────────────────────────────────────────────────

DESCRIPTION = dedent(f"""\
<div style="text-align:center;margin-bottom:8px">
  <h1 style="font-family:Poppins,sans-serif;color:{V_TEXT};font-weight:700;margin-bottom:4px">
    AdMesh Agentic Intent Classifier
  </h1>
  <p style="font-family:Poppins,sans-serif;color:{V_MUTED};font-size:1em;margin:0">
    Classify any user query into <b style="color:{V_TEXT}">intent type</b>, <b style="color:{V_TEXT}">subtype</b>,
    <b style="color:{V_TEXT}">decision phase</b>, <b style="color:{V_TEXT}">IAB content category</b>, and
    <b style="color:{V_TEXT}">monetization eligibility</b> — all in a single forward pass.
  </p>
  <p style="font-family:Poppins,sans-serif;font-size:0.85em;margin-top:10px;color:{V_MUTED}">
    <a href="https://huggingface.co/admesh/agentic-intent-classifier" style="color:{V_TEXT}">Model Card</a>
    &nbsp;&middot;&nbsp;
    <a href="https://github.com/GouniManikumar12/agentic-intent-classifier" style="color:{V_TEXT}">GitHub</a>
    &nbsp;&middot;&nbsp;
    <a href="https://useadmesh.com" style="color:{V_TEXT}">AdMesh</a>
    &nbsp;&middot;&nbsp;
    <a href="https://agenticintentprotocol.com" style="color:{V_TEXT}">Agentic Intent Protocol</a>
  </p>
</div>
""")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
.gradio-container {
    max-width: 960px !important;
    font-family: Poppins, sans-serif !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="AdMesh Intent Classifier", theme=admesh_theme) as demo:
    gr.HTML(DESCRIPTION)

    with gr.Row():
        query_input = gr.Textbox(
            label="Query",
            placeholder="Type any user query here...",
            lines=1,
            scale=4,
        )
        classify_btn = gr.Button("Classify", variant="primary", scale=1)

    gr.Examples(
        examples=EXAMPLES,
        inputs=query_input,
        label="Try these examples",
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            summary_out = gr.HTML(label="Intent Classification")
        with gr.Column():
            confidence_out = gr.HTML(label="Confidence Breakdown")

    with gr.Row():
        policy_out = gr.HTML(label="IAB & Monetization")

    with gr.Accordion("Full JSON Response", open=False):
        json_display = gr.Code(language="json", label="Raw Response")

    json_hidden = gr.Textbox(visible=False)

    LOADING_HTML = (
        f'<div style="display:flex;align-items:center;justify-content:center;padding:28px;'
        f'font-family:Poppins,sans-serif;color:{V_TEXT}">'
        f'<svg width="24" height="24" viewBox="0 0 24 24" style="animation:spin 1s linear infinite;margin-right:10px">'
        f'<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none" stroke-dasharray="31.4 31.4" />'
        f'</svg>Classifying...</div>'
        f'<style>@keyframes spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}</style>'
    )

    def _show_loading():
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            LOADING_HTML,
            LOADING_HTML,
            LOADING_HTML,
        )

    def _classify_and_unlock(query):
        results = classify(query)
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            *results,
        )

    classify_btn.click(
        fn=_show_loading,
        inputs=[],
        outputs=[query_input, classify_btn, summary_out, confidence_out, policy_out],
    ).then(
        fn=_classify_and_unlock,
        inputs=[query_input],
        outputs=[query_input, classify_btn, summary_out, confidence_out, policy_out, json_display, json_hidden],
    )
    query_input.submit(
        fn=_show_loading,
        inputs=[],
        outputs=[query_input, classify_btn, summary_out, confidence_out, policy_out],
    ).then(
        fn=_classify_and_unlock,
        inputs=[query_input],
        outputs=[query_input, classify_btn, summary_out, confidence_out, policy_out, json_display, json_hidden],
    )

if __name__ == "__main__":
    demo.launch()
