# Roadmap Issue Drafts

These are the next three roadmap issues to open in GitHub once authenticated issue creation is available.

## 1. Build External Demo UI For Decision Envelope

Suggested title:

`Build external demo UI for query -> model_output -> system_decision`

Suggested body:

```md
## Goal

Add a simple external-facing demo interface on top of `/classify` so a user can paste a query and see the full decision envelope in a clean, understandable format.

## Scope

- add a lightweight UI for entering a raw query
- render `model_output.classification.intent`
- render fallback state when present
- render `system_decision.policy`
- render `system_decision.opportunity`
- include a few preloaded demo prompts

## Why

The current JSON API is enough for engineering validation, but not enough for partner demos or taxonomy walkthroughs.

## Done When

- someone can run the demo locally and inspect the full output without using curl
- the UI clearly shows query -> classification -> system decision
```

## 2. Add Better Support Handling To Intent-Type Layer

Suggested title:

`Add dedicated support handling to reduce personal_reflection fallback on account-help prompts`

Suggested body:

```md
## Goal

Reduce the current failure mode where support-like prompts such as login and billing issues collapse into `personal_reflection` or low-confidence fallback behavior.

## Scope

- review support-like prompts in the current benchmark
- decide whether to add a dedicated `support` intent-type head or a rule-based override layer
- add a fixed support-oriented evaluation set
- document the chosen approach in `known_limitations.md`

## Why

The `decision_phase` head can already separate `support` reasonably well, but the `intent_type` layer still underperforms on these cases.

## Done When

- support prompts are no longer commonly labeled as `personal_reflection`
- the combined envelope fails safe for support queries with clearer semantics
```

## 3. Add Evaluation Harness And Canonical Benchmark Runner

Suggested title:

`Add canonical benchmark runner for demo prompts and regression checks`

Suggested body:

```md
## Goal

Turn the current prompt suite and canonical examples into a repeatable regression harness.

## Scope

- add a script that runs the fixed demo prompts through `combined_inference.py`
- save outputs to a machine-readable artifact
- compare current outputs against expected behavior notes
- flag meaningful regressions in fallback behavior and phase classification

## Why

The repo now has frozen `v0.1` baselines. A benchmark runner is the clean way to protect demo quality without returning to ad hoc tuning.

## Done When

- one command runs the prompt suite end to end
- current outputs are easy to inspect and compare over time
- demo regressions become visible before external sharing
```
