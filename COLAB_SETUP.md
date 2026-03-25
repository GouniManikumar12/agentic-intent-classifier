# Google Colab setup — agentic-intent-classifier

## 1. Runtime

**Runtime → Change runtime type → GPU** (T4/L4/A100). Then verify:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Get the code

**Option A — clone (if the repo is public or you use a token):**

```python
!git clone <YOUR_REPO_URL> protocol
%cd protocol/agentic-intent-classifier
```

**Option B — upload:** Zip `agentic-intent-classifier/` (including `data/`, `examples/`, taxonomy TSV under `data/iab-content/` if you use IAB), unzip in Colab, then:

```python
%cd /content/agentic-intent-classifier
```

## 3. Install dependencies

```python
%pip install -q -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```python
%pip install -q torch transformers datasets accelerate scikit-learn numpy pandas safetensors
```

## 4. Optional: quieter TensorFlow / XLA logs

Run **before** importing `combined_inference` or anything that pulls TensorFlow:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
```

Harmless CUDA “already registered” lines may still appear; they do not mean training failed.

## 5. Optional: persist artifacts on Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy outputs to Drive after training, or symlink `multitask_intent_model_output` / `artifacts` / `iab_classifier_model_output` to a Drive folder.

## 6. Full pipeline (train + IAB + calibrate + verify + ONNX + smoke test)

From `agentic-intent-classifier/`:

```python
!python training/run_full_training_pipeline.py --skip-full-eval --complete
```

- `--skip-full-eval` avoids the heaviest eval pass (OOM on small RAM); remove when you have headroom.
- `--complete` = export multitask ONNX + `pipeline_verify.py` + one `combined_inference` query.

**Artifacts-only check (after copying weights in):**

```python
!python training/pipeline_verify.py
```

**Single query:**

```python
!python combined_inference.py "Which laptop should I buy for college?"
```

Check `meta.iab_mapping_is_placeholder`: `false` only if IAB was trained and calibration exists.

## 7. Minimal path (intent multitask + calibrate only)

If you only run multitask training and calibration in Colab (no full orchestrator):

```text
python training/train_multitask_intent.py
python training/calibrate_confidence.py --head intent_type
python training/calibrate_confidence.py --head intent_subtype
python training/calibrate_confidence.py --head decision_phase
```

Production “complete” stack still needs **IAB train + IAB calibrate** (see `run_full_training_pipeline.py`).

## 8. Working directory

Always `cd` to the folder that contains `config.py`, `training/`, and `data/`:

```python
import os
assert os.path.isfile("config.py"), "Wrong directory — cd into agentic-intent-classifier"
```
