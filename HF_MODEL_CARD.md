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

- **Try it live**: [Interactive Demo](https://huggingface.co/spaces/manikumargouni/admesh_intent_classifier_demo)
- Hugging Face: https://huggingface.co/admesh/agentic-intent-classifier
- GitHub: https://github.com/GouniManikumar12/agentic-intent-classifier
- AdMesh: https://useadmesh.com
- Agentic Intent Protocol: https://agenticintentprotocol.com

## What It Predicts

| Field | Description |
|---|---|
| `intent.type` | `commercial`, `informational`, `navigational`, `transactional`, … |
| `intent.subtype` | `product_discovery`, `comparison`, `how_to`, … |
| `intent.decision_phase` | `awareness`, `consideration`, `decision`, … |
| `iab_content` | IAB Content Taxonomy 3.0 tier1 / tier2 / tier3 labels |
| `component_confidence` | Per-head calibrated confidence with threshold flags |
| `system_decision` | Monetization eligibility, opportunity type, policy |

---

## Deployment Options

### 0. Colab / Kaggle Quickstart (copy/paste)

```python
!pip -q install -U pip
!pip -q install -U "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
!pip -q install -U "transformers>=4.36.0" "huggingface_hub>=0.20.0" "safetensors>=0.4.0"
```

Restart the runtime after installs (**Runtime → Restart runtime**) so the new Torch version is actually used.

```python
from transformers import pipeline

clf = pipeline(
    "admesh-intent",
    model="admesh/agentic-intent-classifier",
    trust_remote_code=True,  # required (custom pipeline + multi-model bundle)
)

out = clf("Which laptop should I buy for college?")
print(out["meta"])
print(out["model_output"]["classification"]["intent"])
```

---

## Latency / inference timing (quick check)

The first call includes model/code loading. Warm up once, then measure:

```python
import time
q = "Which laptop should I buy for college?"

_ = clf("warm up")
t0 = time.perf_counter()
out = clf(q)
print(f"latency_ms={(time.perf_counter() - t0) * 1000:.1f}")
```

### 1. `transformers.pipeline()` — anywhere (Python)

```python
from transformers import pipeline

clf = pipeline(
    "admesh-intent",
    model="admesh/agentic-intent-classifier",
    trust_remote_code=True,
)

result = clf("Which laptop should I buy for college?")
```

Batch and custom thresholds:

```python
# batch
results = clf([
    "Best running shoes under $100",
    "How does TCP work?",
    "Buy noise-cancelling headphones",
])

# custom confidence thresholds
result = clf(
    "Buy headphones",
    threshold_overrides={"intent_type": 0.6, "intent_subtype": 0.35},
)
```

---

### 2. HF Inference Endpoints (managed, deploy to AWS / Azure / GCP)

1. Go to https://ui.endpoints.huggingface.co
2. **New Endpoint** → select `admesh/agentic-intent-classifier`
3. Framework: **PyTorch** — Task: **Text Classification**
4. Enable **"Load with trust_remote_code"**
5. Deploy

The endpoint serves the same `pipeline()` interface above via REST:

```bash
curl https://<your-endpoint>.endpoints.huggingface.cloud \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Which laptop should I buy for college?"}'
```

---

### 3. HF Spaces (Gradio / Streamlit demo)

```python
# app.py for a Gradio Space
import gradio as gr
from transformers import pipeline

clf = pipeline(
    "admesh-intent",
    model="admesh/agentic-intent-classifier",
    trust_remote_code=True,
)

def classify(text):
    return clf(text)

gr.Interface(fn=classify, inputs="text", outputs="json").launch()
```

---

### 4. Local / notebook via `snapshot_download`

```python
import sys
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="admesh/agentic-intent-classifier",
    repo_type="model",
)
sys.path.insert(0, local_dir)

from pipeline import AdmeshIntentPipeline
clf = AdmeshIntentPipeline()
result = clf("I need a CRM for a 5-person startup")
```

Or the one-liner factory:

```python
from pipeline import AdmeshIntentPipeline
clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
```

---

## Troubleshooting (avoid environment errors)

### `No module named 'combined_inference'` (or similar)

This means the Hub repo root is missing required Python files. Ensure these exist at the **root of the model repo** (same level as `pipeline.py`):

- `pipeline.py`, `config.json`, `config.py`
- `combined_inference.py`, `schemas.py`
- `model_runtime.py`, `multitask_runtime.py`, `multitask_model.py`
- `inference_intent_type.py`, `inference_subtype.py`, `inference_decision_phase.py`, `inference_iab_classifier.py`
- `iab_classifier.py`, `iab_taxonomy.py`

### `does not appear to have a file named model.safetensors`

Transformers requires a standard checkpoint at the repo root for `pipeline()` to initialize. This repo includes a **small dummy** `model.safetensors` + tokenizer files at the root for compatibility; the *real* production weights live in:

- `multitask_intent_model_output/`
- `iab_classifier_model_output/`
- `artifacts/calibration/`

---

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

## Reproducible Revision

```python
from huggingface_hub import snapshot_download
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
| `pipeline.py` | `AdmeshIntentPipeline` (transformers.Pipeline subclass) |
| `combined_inference.py` | Core inference logic |

## Notes

- `trust_remote_code=True` is required because this model uses a custom multi-head architecture that does not map to a single standard `AutoModel` checkpoint.
- `meta.iab_mapping_is_placeholder: true` means IAB artifacts were missing or skipped; train and calibrate IAB for full production accuracy.
- For long-running servers, instantiate once and reuse — models are cached in memory after the first call.
