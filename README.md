# Agentic Intent Classifier

`agentic-intent-classifier` is a multi-head query classification stack for conversational traffic.

It currently produces:

- `intent.type`
- `intent.subtype`
- `intent.decision_phase`
- `iab_content`
- calibrated confidence per head
- combined fallback / policy / opportunity decisions

The repo is beyond the original v0.1 baseline. It now includes:

- shared config and label ownership
- reusable model runtime
- calibrated confidence and threshold gating
- combined inference with fallback/policy logic
- request/response validation in the demo API
- repeatable evaluation and regression suites
- full-TSV IAB taxonomy support through tier4
- a separate synthetic full-intent-taxonomy augmentation dataset for non-IAB heads
- a dedicated intent-type difficulty dataset and held-out benchmark with `easy`, `medium`, and `hard` cases
- a dedicated decision-phase difficulty dataset and held-out benchmark with `easy`, `medium`, and `hard` cases

Generated model weights are intentionally not committed.

## Current Taxonomy

### `intent.type`

- `informational`
- `exploratory`
- `commercial`
- `transactional`
- `support`
- `personal_reflection`
- `creative_generation`
- `chit_chat`
- `ambiguous`
- `prohibited`

### `intent.decision_phase`

- `awareness`
- `research`
- `consideration`
- `decision`
- `action`
- `post_purchase`
- `support`

### `intent.subtype`

- `education`
- `product_discovery`
- `comparison`
- `evaluation`
- `deal_seeking`
- `provider_selection`
- `signup`
- `purchase`
- `booking`
- `download`
- `contact_sales`
- `task_execution`
- `onboarding_setup`
- `troubleshooting`
- `account_help`
- `billing_help`
- `follow_up`
- `emotional_reflection`

### `iab_content`

- full label space is derived from every row in [data/iab-content/Content Taxonomy 3.0.tsv](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/data/iab-content/Content%20Taxonomy%203.0.tsv)
- output supports `tier1`, `tier2`, `tier3`, and optional `tier4`

## What The System Does

- runs four classifier heads:
  - `intent_type`
  - `intent_subtype`
  - `decision_phase`
  - `iab_content`
- applies calibration artifacts when present
- computes `commercial_score`
- applies fallback when confidence is too weak or policy-safe blocking is required
- emits a schema-validated combined envelope

## What The System Does Not Do

- it is not a multi-turn memory system
- it is not a production-optimized low-latency serving path
- it is not yet trained on large real-traffic human-labeled intent data
- combined decision logic is still heuristic, even though it is materially stronger than the original baseline

## Project Layout

- [config.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/config.py): labels, thresholds, artifact paths, model paths
- [model_runtime.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/model_runtime.py): shared calibrated inference runtime
- [combined_inference.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/combined_inference.py): composed system response
- [inference_intent_type.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/inference_intent_type.py): direct `intent_type` inference entrypoint
- [schemas.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/schemas.py): request/response validation
- [demo_api.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/demo_api.py): local validated API
- [iab_taxonomy.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/iab_taxonomy.py): full taxonomy parser/index
- [iab_mapping.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/iab_mapping.py): taxonomy-backed fallback mapper
- [training/build_full_intent_taxonomy_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_full_intent_taxonomy_dataset.py): separate synthetic intent augmentation dataset
- [training/build_intent_type_difficulty_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_intent_type_difficulty_dataset.py): extra `intent_type` augmentation plus held-out difficulty benchmark
- [training/build_decision_phase_difficulty_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_decision_phase_difficulty_dataset.py): extra `decision_phase` augmentation plus held-out difficulty benchmark
- [training/build_subtype_difficulty_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_subtype_difficulty_dataset.py): extra `intent_subtype` augmentation plus held-out difficulty benchmark
- [training/build_subtype_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_subtype_dataset.py): subtype dataset generation from existing corpora
- [training/build_iab_dataset.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/build_iab_dataset.py): full-TSV IAB dataset generation
- [training/run_full_training_pipeline.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/training/run_full_training_pipeline.py): full multi-head training/calibration/eval pipeline
- [evaluation/run_evaluation.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_evaluation.py): repeatable benchmark runner
- [evaluation/run_regression_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_regression_suite.py): known-failure regression runner
- [evaluation/run_iab_mapping_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_iab_mapping_suite.py): IAB regression runner
- [known_limitations.md](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/known_limitations.md): current gaps and caveats

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r agentic-intent-classifier/requirements.txt
```

## Inference

Run one query locally:

```bash
cd agentic-intent-classifier
python3 combined_inference.py "Which CRM should I buy for a 3-person startup?"
```

Run only the `intent_type` head:

```bash
cd agentic-intent-classifier
python3 inference_intent_type.py "best shoes under 100"
```

Run the demo API:

```bash
cd agentic-intent-classifier
python3 demo_api.py
```

Example request:

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

### Full local pipeline

```bash
cd agentic-intent-classifier
python3 training/run_full_training_pipeline.py \
  --iab-epochs 2 \
  --iab-train-batch-size 16 \
  --iab-eval-batch-size 16
```

This pipeline now does:

1. build separate full-intent-taxonomy augmentation data
2. build separate `intent_type` difficulty augmentation + benchmark
3. train `intent_type`
4. build subtype corpus
5. build separate `intent_subtype` difficulty augmentation + benchmark
6. train `intent_subtype`
7. build separate `decision_phase` difficulty augmentation + benchmark
8. train `decision_phase`
9. build full-TSV IAB data
10. train `iab_content`
11. calibrate all heads
12. run regression/evaluation unless `--skip-full-eval` is used

### Build datasets individually

Separate full-intent augmentation:

```bash
cd agentic-intent-classifier
python3 training/build_full_intent_taxonomy_dataset.py
```

Intent-type difficulty augmentation and benchmark:

```bash
cd agentic-intent-classifier
python3 training/build_intent_type_difficulty_dataset.py
```

Decision-phase difficulty augmentation and benchmark:

```bash
cd agentic-intent-classifier
python3 training/build_decision_phase_difficulty_dataset.py
```

Subtype difficulty augmentation and benchmark:

```bash
cd agentic-intent-classifier
python3 training/build_subtype_difficulty_dataset.py
```

Subtype dataset:

```bash
cd agentic-intent-classifier
python3 training/build_subtype_dataset.py
```

IAB dataset:

```bash
cd agentic-intent-classifier
python3 training/build_iab_dataset.py
```

### Train heads individually

```bash
cd agentic-intent-classifier
python3 training/train.py
python3 training/train_subtype.py
python3 training/train_decision_phase.py
python3 training/train_iab_content.py
```

### Calibration

```bash
cd agentic-intent-classifier
python3 training/calibrate_confidence.py --head all
```

## Evaluation

Full evaluation:

```bash
cd agentic-intent-classifier
python3 evaluation/run_evaluation.py
```

Known-failure regression:

```bash
cd agentic-intent-classifier
python3 evaluation/run_regression_suite.py
```

IAB regression:

```bash
cd agentic-intent-classifier
python3 evaluation/run_iab_mapping_suite.py
```

Threshold sweeps:

```bash
cd agentic-intent-classifier
python3 evaluation/sweep_intent_threshold.py
python3 evaluation/sweep_iab_threshold.py
```

Artifacts are written to:

- `artifacts/calibration/`
- `artifacts/evaluation/latest/`

## Google Colab

Use Colab for the full retraining pass if local memory is limited.

Clone once:

```bash
%cd /content
!git clone https://github.com/GouniManikumar12/agentic-intent-classifier.git
%cd /content/agentic-intent-classifier
```

If the repo is already cloned and you want the latest code, pull manually:

```bash
!git pull origin main
```

Full pipeline:

```bash
!python training/run_full_training_pipeline.py \
  --iab-epochs 2 \
  --iab-train-batch-size 32 \
  --iab-eval-batch-size 32
```

If full evaluation is too heavy for the current Colab runtime:

```bash
!python training/run_full_training_pipeline.py \
  --iab-epochs 2 \
  --iab-train-batch-size 32 \
  --iab-eval-batch-size 32 \
  --skip-full-eval
```

Then run eval separately after training:

```bash
!python evaluation/run_regression_suite.py
!python evaluation/run_iab_mapping_suite.py
!python evaluation/run_evaluation.py
```

## Current Known Baseline Benchmarks

The latest saved local evaluation artifact is [artifacts/evaluation/latest/summary.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/artifacts/evaluation/latest/summary.json).

Latest saved baseline:

- `intent_type` test accuracy: `0.9000`
- `decision_phase` test accuracy: `0.6897`
- `intent_subtype` test accuracy: `0.6212`
- `iab_content` test accuracy: `0.9583`
- combined demo fallback rate: `0.5333`
- known-failure regression: `8/9`
- must-fix regression: `6/6`
- IAB mapping regression: `9/9`

These are last-known saved metrics, not a guarantee for any newer unretrained code or weights.

## Latency Note

`combined_inference.py` is a debugging/offline path, not a production latency path.

Current production truth:

- per-request CLI execution is not a sub-50ms architecture
- production serving should use a long-lived API process with preloaded models
- if sub-50ms becomes a hard requirement, the serving path will need:
  - persistent loaded models
  - runtime optimization
  - likely fewer model passes or a shared multi-head model

## Current Status

Current repo status:

- full 10-class `intent.type` taxonomy is wired
- subtype and phase heads are present
- difficulty benchmarks are wired for `intent_type`, `intent_subtype`, and `decision_phase`
- full-TSV IAB taxonomy is wired through tier4
- separate full-intent augmentation dataset is in place
- evaluation/runtime memory handling is improved for large IAB splits

The main remaining gap is not basic infrastructure anymore. It is improving real-world robustness, especially for:

- `decision_phase`
- `intent_subtype`
- confidence quality on borderline commercial queries
- real-traffic supervision beyond synthetic data
