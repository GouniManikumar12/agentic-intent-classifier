---
title: AdMesh Intent Classifier
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: true
license: apache-2.0
models:
- admesh/agentic-intent-classifier
short_description: Classify queries into intent, IAB category & more
---

# AdMesh Agentic Intent Classifier — Demo

Interactive demo for [admesh/agentic-intent-classifier](https://huggingface.co/admesh/agentic-intent-classifier).

Type any user query to see:
- **Intent type** (commercial, informational, navigational, transactional, ...)
- **Intent subtype** (product_discovery, comparison, how_to, ...)
- **Decision phase** (awareness, consideration, decision, action, ...)
- **IAB Content Taxonomy 3.0** tier classification
- **Monetization eligibility** and opportunity scoring
- **Calibrated confidence** per classification head

All predictions run in a single forward pass (~5 ms GPU / ~25 ms CPU).
