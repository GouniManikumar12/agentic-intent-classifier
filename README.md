# Agentic Intent Classifier v0.6 Phase 4

This repo now contains the v0.1 modeling baseline, the Phase 1 systemization layer, the Phase 2 subtype expansion, and a Phase 4 full-taxonomy IAB content layer.

It classifies conversational queries into:

- `intent.type`
- `intent.subtype`
- `intent.decision_phase`
- `iab_content`
- `confidence`
- `commercial_score`
- fallback / policy / opportunity decisions

Phase 4 adds:

- taxonomy-backed IAB content mapping using every row from the local IAB TSV in `data/iab-content`
- a supervised `iab_content` classifier head trained on a deterministic full-taxonomy dataset
- classifier-first IAB routing with rule fallback when `iab_content` confidence is too low
- stable tier objects with ids, labels, mapping mode, and mapping confidence through tier4
- curated IAB mapping regression cases
- schema-safe IAB output inside the combined envelope

Generated model weights are intentionally not committed. Train the models locally before running inference or the demo API.

## What It Does

- predicts `intent.type` with the frozen `intent_type v0.1` classifier
- predicts `intent.subtype` with the new Phase 2 subtype classifier
- predicts `decision_phase` with the frozen `decision_phase v0.1` classifier
- predicts `iab_content` with a taxonomy-backed classifier head and falls back to rules when confidence is low
- calibrates confidence if calibration artifacts exist
- composes both signals into a schema-aligned decision envelope
- applies a rule-based system layer for fallback, monetization eligibility, and opportunity strength

## What It Does Not Do

- it does not perform robust multi-turn context tracking
- subtype quality is still materially weaker than `intent_type`
- the IAB classifier now covers the full taxonomy but is still trained on synthetic taxonomy-backed prompts rather than fully human-labeled real traffic
- combined scoring is still heuristic even though it is stronger than earlier phases
- it does not replace policy review for sensitive or regulated categories

## Project Layout

- [config.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/config.py): central labels, model paths, thresholds, and artifact locations
- [schemas.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/schemas.py): request/response schemas and validation rules
- [model_runtime.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/model_runtime.py): reusable model loading and calibrated inference
- [combined_inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/combined_inference.py): combines both model heads into one envelope
- [iab_taxonomy.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/iab_taxonomy.py): parses and indexes the local IAB taxonomy TSV
- [iab_mapping.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/iab_mapping.py): taxonomy-driven IAB rule fallback
- [demo_api.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/demo_api.py): local HTTP endpoint with validated requests and responses
- [inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference.py): `intent_type` inference wrapper
- [inference_subtype.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_subtype.py): `intent_subtype` inference wrapper
- [inference_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_decision_phase.py): `decision_phase` inference wrapper
- [inference_iab_content.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_iab_content.py): `iab_content` inference wrapper
- [training/train.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train.py): `intent_type` training
- [training/build_subtype_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_subtype_dataset.py): deterministic subtype dataset generation from existing corpora
- [training/train_subtype.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train_subtype.py): `intent_subtype` training
- [training/train_decision_phase.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train_decision_phase.py): `decision_phase` training
- [training/build_iab_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_iab_dataset.py): full-TSV `iab_content` dataset generation
- [training/train_iab_content.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/train_iab_content.py): configurable `iab_content` training
- [training/run_iab_full_pipeline.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/run_iab_full_pipeline.py): one-command IAB build/train/calibrate/eval pipeline
- [training/run_full_training_pipeline.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/run_full_training_pipeline.py): one-command all-head build/train/calibrate/eval pipeline
- [training/calibrate_confidence.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/calibrate_confidence.py): fits temperature scaling and confidence thresholds
- [evaluation/run_evaluation.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_evaluation.py): writes repeatable metrics and confusion matrices
- [evaluation/run_regression_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_regression_suite.py): executes structured known-failure checks
- [evaluation/run_iab_mapping_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_iab_mapping_suite.py): executes curated IAB mapping checks
- [evaluation/sweep_iab_threshold.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/sweep_iab_threshold.py): sweeps `iab_content` classifier thresholds against curated cases
- [examples/demo_prompt_suite.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/demo_prompt_suite.json): fixed benchmark/demo prompts with current outputs
- [examples/known_failure_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/known_failure_cases.json): structured must-fix vs acceptable-weakness regression cases
- [examples/iab_mapping_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/iab_mapping_cases.json): curated IAB mapping regression cases
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

Build the subtype dataset and train `intent_subtype`:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/build_subtype_dataset.py
../.venv/bin/python training/train_subtype.py
```

Retrain `decision_phase`:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/train_decision_phase.py
```

Build the separate full-intent-taxonomy augmentation dataset used by `intent_type`, `intent_subtype`, and `decision_phase` training:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/build_full_intent_taxonomy_dataset.py
```

Build the IAB dataset and train `iab_content`:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/build_iab_dataset.py
../.venv/bin/python training/train_iab_content.py
```

Run the full IAB pipeline in one command:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/run_iab_full_pipeline.py
```

For Google Colab or another GPU host, use larger batch sizes:

```bash
cd agentic-intent-classifier
python training/run_iab_full_pipeline.py \
  --epochs 2 \
  --train-batch-size 32 \
  --eval-batch-size 32
```

Run the full stack in one command:

```bash
cd agentic-intent-classifier
../.venv/bin/python training/run_full_training_pipeline.py
```

For Google Colab or another GPU host:

```bash
cd agentic-intent-classifier
python training/run_full_training_pipeline.py \
  --iab-epochs 2 \
  --iab-train-batch-size 32 \
  --iab-eval-batch-size 32
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

Run the structured known-failure regression suite on its own:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/run_regression_suite.py
```

Run the IAB mapping regression suite:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/run_iab_mapping_suite.py
```

Sweep `intent_type` fallback thresholds:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/sweep_intent_threshold.py
```

Sweep `iab_content` promotion thresholds:

```bash
cd agentic-intent-classifier
../.venv/bin/python evaluation/sweep_iab_threshold.py
```

Artifacts are written into:

- `artifacts/calibration/*.json`
- `artifacts/evaluation/latest/`
- `artifacts/evaluation/intent_threshold_sweep.json`
- `artifacts/evaluation/latest/iab_threshold_sweep.json`
- `artifacts/evaluation/latest/known_failure_regression.json`
- `artifacts/evaluation/latest/iab_mapping_regression.json`

## Current Status

Current modeling state:

- `intent_type v0.1`
- `decision_phase v0.1`
- `intent_subtype v0.1`
- hybrid `iab_content` classifier + rule fallback v0.1

The current focus after this Phase 4 slice is replacing synthetic IAB supervision with real labeled prompt-to-taxonomy examples and expanding calibration quality beyond the current synthetic prompt distribution.
# agentic-intent-classifier
