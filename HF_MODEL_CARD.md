---
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert-base-uncased
metrics:
- accuracy
- f1
tags:
- intent-classification
- multitask
- iab
- conversational-ai
- adtech
- calibrated-confidence
license: apache-2.0
---

# admesh/agentic-intent-classifier

Production-ready intent + IAB classifier bundle for conversational traffic.

Combines multitask intent modeling, supervised IAB content classification, and per-head confidence calibration to support safe monetization decisions in real time.

## Links

- Hugging Face: https://huggingface.co/admesh/agentic-intent-classifier
- GitHub: https://github.com/GouniManikumar12/agentic-intent-classifier

## What It Predicts

| Field | Description |
|---|---|
| `intent.type` | `commercial`, `informational`, `navigational`, `transactional` |
| `intent.subtype` | e.g. `product_discovery`, `comparison`, `how_to`, … |
| `intent.decision_phase` | `awareness`, `consideration`, `decision` |
| `iab_content` | IAB Content Taxonomy 3.0 tier1 / tier2 / tier3 labels |
| `component_confidence` | Per-head calibrated confidence with threshold flags |
| `system_decision` | Monetization eligibility, opportunity type, policy |

## Quick Start — `AdmeshIntentPipeline`

```python
import sys
from huggingface_hub import snapshot_download

# 1. Download the full bundle (models + calibration + code)
local_dir = snapshot_download(
    repo_id="admesh/agentic-intent-classifier",
    repo_type="model",
)
sys.path.insert(0, local_dir)

# 2. Import and instantiate
from pipeline import AdmeshIntentPipeline
clf = AdmeshIntentPipeline()

# 3. Classify
result = clf("Which laptop should I buy for college?")
import json
print(json.dumps(result, indent=2))
```

### One-liner using `from_pretrained`

```python
from pipeline import AdmeshIntentPipeline   # after sys.path is set, or in the bundle dir

clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
result = clf("I need a CRM for a 5-person startup")
```

### Batch inference

```python
results = clf([
    "Best running shoes under $100",
    "How to set up a CI/CD pipeline",
    "Buy noise-cancelling headphones",
])
for r in results:
    print(r["model_output"]["classification"]["intent"]["type"])
```

### Custom confidence thresholds

```python
result = clf(
    "Buy noise-cancelling headphones",
    threshold_overrides={"intent_type": 0.6, "intent_subtype": 0.35},
)
```

## Why Not `transformers.pipeline()` Directly?

This model uses three separate model files (multitask intent, IAB classifier, calibration JSONs) that HF's standard auto-loading expects as a single checkpoint. Using `AdmeshIntentPipeline` is the supported pattern — it wraps the full stack and handles model loading, calibration, and fallback logic automatically.

## Example Output

```json
{
  "model_output": {
    "classification": {
      "iab_content": {
        "taxonomy": "IAB Content Taxonomy",
        "taxonomy_version": "3.0",
        "tier1": {"id": "552", "label": "Style & Fashion"},
        "tier2": {"id": "579", "label": "Men's Fashion"},
        "mapping_mode": "exact",
        "mapping_confidence": 0.73
      },
      "intent": {
        "type": "commercial",
        "subtype": "product_discovery",
        "decision_phase": "consideration",
        "confidence": 0.9549,
        "commercial_score": 0.656
      }
    }
  },
  "system_decision": {
    "policy": {
      "monetization_eligibility": "allowed_with_caution",
      "eligibility_reason": "commercial_discovery_signal_present"
    },
    "opportunity": {"type": "soft_recommendation", "strength": "medium"}
  },
  "meta": {
    "system_version": "0.6.0-phase4",
    "calibration_enabled": true,
    "iab_mapping_is_placeholder": false
  }
}
```

## API Server Mode

```bash
cd "<local_dir>"
pip install -r requirements.txt
python3 demo_api.py
```

```bash
curl -sS -X POST http://127.0.0.1:8008/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"I need a CRM for a 5-person startup"}'
```

## Reproducible Revision

```python
local_dir = snapshot_download(
    repo_id="admesh/agentic-intent-classifier",
    repo_type="model",
    revision="0584798f8efee6beccd778b0afa06782ab5add60",
)
```

## Included Artifacts

| Path | Contents |
|---|---|
| `multitask_intent_model_output/` | DistilBERT multitask weights + tokenizer |
| `iab_classifier_model_output/` | IAB content classifier weights + tokenizer |
| `artifacts/calibration/` | Per-head temperature + threshold JSONs |
| `pipeline.py` | `AdmeshIntentPipeline` class |
| `combined_inference.py` | Core inference logic |

## Notes

- Use all three artifact folders together for full accuracy.
- For long-running production servers, instantiate the pipeline once and reuse it — models are cached in memory after the first call.
- `meta.iab_mapping_is_placeholder: true` means IAB artifacts were missing or skipped; train and calibrate IAB for full production accuracy.
