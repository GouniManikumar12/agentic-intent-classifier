# Agentic Intent Classifier v0.1

This repo contains a small, frozen baseline for classifying conversational queries into:

- `intent.type`
- `intent.decision_phase`
- `confidence`
- `commercial_score`
- fallback / policy / opportunity decisions

It is a demoable reference implementation for the Agentic Intent Taxonomy, not a production-grade model stack.

Generated model weights are intentionally not committed. Train the models locally before running inference or the demo API.

## What It Does

- predicts `intent.type` with the frozen `intent_type v0.1` classifier
- predicts `decision_phase` with the frozen `decision_phase v0.1` classifier
- composes both signals into a schema-aligned decision envelope
- applies a simple rule-based system layer for fallback, monetization eligibility, and opportunity strength

## What It Does Not Do

- it does not perform robust multi-turn context tracking
- it does not yet predict the full taxonomy, such as subtype or IAB content
- it does not guarantee high-confidence outputs on short or ambiguous prompts
- it does not replace policy review for sensitive or regulated categories

## Project Layout

- [combined_inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/combined_inference.py): combines both model heads into one envelope
- [demo_api.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/demo_api.py): local HTTP demo endpoint at `POST /classify`
- [inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference.py): `intent_type v0.1`
- [inference_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_decision_phase.py): `decision_phase v0.1`
- [training/train.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train.py): `intent_type` training
- [training/train_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train_decision_phase.py): `decision_phase` training
- [examples/demo_prompt_suite.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/demo_prompt_suite.json): fixed benchmark/demo prompts with current outputs
- [examples/composed_output_examples.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/composed_output_examples.md): short walkthrough examples
- [known_limitations.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/known_limitations.md): current gaps and caveats

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

## Current Status

This repo is intentionally frozen at:

- `intent_type v0.1`
- `decision_phase v0.1`

The next step is demo quality and system integration, not more ad hoc model tuning.
# agentic-intent-classifier
