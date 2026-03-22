# Agentic Intent Classifier v0.2 Phase 1

This repo now contains the v0.1 modeling baseline plus the first systemization pass needed to make it usable as a real inference service.

It classifies conversational queries into:

- `intent.type`
- `intent.decision_phase`
- `confidence`
- `commercial_score`
- fallback / policy / opportunity decisions

Phase 1 adds:

- shared config for labels, paths, thresholds, and versions
- confidence calibration artifacts with threshold-based fallback gating
- typed request/response schemas for the API
- repeatable evaluation artifacts with confusion matrices and per-class metrics

Generated model weights are intentionally not committed. Train the models locally before running inference or the demo API.

## What It Does

- predicts `intent.type` with the frozen `intent_type v0.1` classifier
- predicts `decision_phase` with the frozen `decision_phase v0.1` classifier
- calibrates confidence if calibration artifacts exist
- composes both signals into a schema-aligned decision envelope
- applies a simple rule-based system layer for fallback, monetization eligibility, and opportunity strength

## What It Does Not Do

- it does not perform robust multi-turn context tracking
- it does not yet predict the full taxonomy, such as subtype or IAB content
- it still uses heuristic combined scoring and rule-based opportunity logic
- it does not replace policy review for sensitive or regulated categories

## Project Layout

- [config.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/config.py): central labels, model paths, thresholds, and artifact locations
- [schemas.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/schemas.py): request/response schemas and validation rules
- [model_runtime.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/model_runtime.py): reusable model loading and calibrated inference
- [combined_inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/combined_inference.py): combines both model heads into one envelope
- [demo_api.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/demo_api.py): local HTTP endpoint with validated requests and responses
- [inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference.py): `intent_type` inference wrapper
- [inference_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_decision_phase.py): `decision_phase` inference wrapper
- [training/train.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train.py): `intent_type` training
- [training/train_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train_decision_phase.py): `decision_phase` training
- [training/calibrate_confidence.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/calibrate_confidence.py): fits temperature scaling and confidence thresholds
- [evaluation/run_evaluation.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_evaluation.py): writes repeatable metrics and confusion matrices
- [examples/demo_prompt_suite.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/demo_prompt_suite.json): fixed benchmark/demo prompts with current outputs
- [examples/canonical_demo_examples.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/canonical_demo_examples.json): concise showcase examples for demos and partner walkthroughs
- [examples/composed_output_examples.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/composed_output_examples.md): short walkthrough examples
- [known_limitations.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/known_limitations.md): current gaps and caveats
- [roadmap_issue_drafts.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/roadmap_issue_drafts.md): issue drafts for the next productization stage

## How To Run

Create and activate the local environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r agentic-intent-classifier/requirements.txt
```

Run the combined classifier on a single query:

```bash
cd agentic-intent-classifier
../.venv/bin/python combined_inference.py "Which CRM should I buy for a 3-person startup?"
```

If you have not trained the models yet, run the training commands below first.

Run the local demo API:

```bash
cd agentic-intent-classifier
../.venv/bin/python demo_api.py
```

Then call it:

```bash
curl -sS -X POST http://127.0.0.1:8008/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"I cannot log into my account"}'
```

Infra endpoints:

```bash
curl -sS http://127.0.0.1:8008/health
curl -sS http://127.0.0.1:8008/version
```

## Training

Retrain `intent_type`:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/train.py
```

Retrain `decision_phase`:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/train_decision_phase.py
```

Calibrate confidence after training:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/calibrate_confidence.py --head all
```

Run the repeatable evaluation suite:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/run_evaluation.py
```

Sweep `intent_type` fallback thresholds:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/sweep_intent_threshold.py
```

Artifacts are written into:

- `artifacts/calibration/*.json`
- `artifacts/evaluation/latest/`
- `artifacts/evaluation/intent_threshold_sweep.json`

## Current Status

Modeling is still intentionally frozen at:

- `intent_type v0.1`
- `decision_phase v0.1`

The current focus is calibration, validation, and evaluation quality rather than adding more heads prematurely.
# agentic-intent-classifier
