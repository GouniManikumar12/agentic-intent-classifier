"""
AdmeshIntentPipeline — transformers.Pipeline subclass for
admesh/agentic-intent-classifier.

Because config.json declares "pt": [] the transformers pipeline() loader
skips AutoModel.from_pretrained() entirely and passes model=None straight
to this class.  All model loading is handled internally via combined_inference,
which resolves paths relative to __file__ so it works wherever HF downloads
the repo (Inference Endpoints, Spaces, local snapshot_download, etc.).

Supported HF deployment surfaces
---------------------------------
1. transformers.pipeline() direct call (trust_remote_code=True):

       from transformers import pipeline
       clf = pipeline(
           "admesh-intent",
           model="admesh/agentic-intent-classifier",
           trust_remote_code=True,
       )
       result = clf("Which laptop should I buy for college?")

2. HF Inference Endpoints — Standard (PyTorch, trust_remote_code=True):
   Deploy from https://ui.endpoints.huggingface.co — no custom container
   needed; HF loads this pipeline class automatically.

3. HF Spaces (Gradio / Streamlit):

       import sys
       from huggingface_hub import snapshot_download
       local_dir = snapshot_download("admesh/agentic-intent-classifier", repo_type="model")
       sys.path.insert(0, local_dir)
       from pipeline import AdmeshIntentPipeline
       clf = AdmeshIntentPipeline()
       result = clf("I need a CRM for a 5-person startup")

4. Anywhere via from_pretrained():

       from pipeline import AdmeshIntentPipeline
       clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

# ── try to import transformers.Pipeline; fall back gracefully if absent ───────
try:
    from transformers import Pipeline as _HFPipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _HFPipeline = object  # bare object as base when transformers is not installed
    _TRANSFORMERS_AVAILABLE = False


class AdmeshIntentPipeline(_HFPipeline):
    """
    Full intent + IAB classification pipeline.

    Inherits from ``transformers.Pipeline`` so it works natively with
    ``pipeline()``, HF Inference Endpoints (standard mode), and HF Spaces.

    When ``transformers`` is not installed it falls back to a plain callable
    class so the same code works in minimal environments too.

    Parameters
    ----------
    model:
        Ignored — we load all models internally.  Present only to satisfy
        the ``transformers.Pipeline`` interface when HF calls
        ``PipelineClass(model=None, ...)``.
    **kwargs:
        Forwarded to ``transformers.Pipeline.__init__`` if transformers is
        available, otherwise ignored.
    """

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self, model=None, tokenizer=None, **kwargs):
        # Ensure this repo's directory is on sys.path so all relative imports
        # in combined_inference / config / model_runtime resolve correctly.
        # Path(__file__) points to wherever HF cached the repo snapshot.
        _repo_dir = Path(__file__).resolve().parent
        if str(_repo_dir) not in sys.path:
            sys.path.insert(0, str(_repo_dir))

        if _TRANSFORMERS_AVAILABLE:
            import torch

            # transformers.Pipeline requires certain attributes to be set.
            # Because config.json has "pt": [] HF passes model=None here —
            # we satisfy the interface by setting the minimum required attrs
            # manually instead of calling super().__init__(model=None, ...)
            # which would raise inside infer_framework_load_model().
            self.task = kwargs.pop("task", "admesh-intent")
            self.model = model          # None — unused, kept for interface compat
            self.tokenizer = tokenizer  # None — unused
            self.feature_extractor = None
            self.image_processor = None
            self.modelcard = None
            self.framework = "pt"
            self.device = torch.device(kwargs.pop("device", "cpu"))
            self.torch_dtype = kwargs.pop("torch_dtype", None)
            self.binary_output = kwargs.pop("binary_output", False)
            self.call_count = 0
            self._batch_size = kwargs.pop("batch_size", 1)
            self._num_workers = kwargs.pop("num_workers", 0)
            self._preprocess_params: dict = {}
            self._forward_params: dict = {}
            self._postprocess_params: dict = {}
        # else: plain object, no init needed

        self._classify_fn = None  # lazy-loaded on first __call__

    # ── transformers.Pipeline abstract methods ────────────────────────────────
    # These are required by the ABC but our __call__ override bypasses them.
    # They are still implemented in case a caller invokes them directly.

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        if "threshold_overrides" in kwargs:
            forward_kwargs["threshold_overrides"] = kwargs["threshold_overrides"]
        if "force_iab_placeholder" in kwargs:
            forward_kwargs["force_iab_placeholder"] = kwargs["force_iab_placeholder"]
        return {}, forward_kwargs, {}

    def preprocess(self, inputs):
        return {"text": inputs if isinstance(inputs, str) else str(inputs)}

    def _forward(self, model_inputs, threshold_overrides=None, force_iab_placeholder=False):
        self._ensure_loaded()
        return self._classify_fn(
            model_inputs["text"],
            threshold_overrides=threshold_overrides,
            force_iab_placeholder=force_iab_placeholder,
        )

    def postprocess(self, model_outputs):
        return model_outputs

    # ── __call__ override ─────────────────────────────────────────────────────
    # We bypass Pipeline's preprocess→_forward→postprocess chain entirely so
    # we never touch self.model and keep full control over batching logic.

    def __call__(
        self,
        inputs: Union[str, list[str]],
        *,
        threshold_overrides: dict[str, float] | None = None,
        force_iab_placeholder: bool = False,
    ) -> Union[dict, list[dict]]:
        """
        Classify one or more query strings.

        Parameters
        ----------
        inputs:
            A single query string or a list of query strings.
        threshold_overrides:
            Optional per-head confidence threshold overrides, e.g.
            ``{"intent_type": 0.5, "iab_content": 0.3}``.
        force_iab_placeholder:
            Skip IAB classifier and return placeholder values (faster,
            no IAB accuracy).

        Returns
        -------
        dict or list[dict]:
            Full classification payload matching the combined_inference schema.
            Returns a single dict for a string input, list of dicts for a list.

        Examples
        --------
        ::

            clf = pipeline("admesh-intent", model="admesh/agentic-intent-classifier",
                           trust_remote_code=True)

            # single
            result = clf("Which laptop should I buy for college?")

            # batch
            results = clf(["Best running shoes", "How does TCP work?"])

            # custom thresholds
            result = clf("Buy headphones", threshold_overrides={"intent_type": 0.6})
        """
        self._ensure_loaded()

        single = isinstance(inputs, str)
        texts: list[str] = [inputs] if single else list(inputs)

        results = [
            self._classify_fn(
                text,
                threshold_overrides=threshold_overrides,
                force_iab_placeholder=force_iab_placeholder,
            )
            for text in texts
        ]
        return results[0] if single else results

    # ── warm-up / compile ─────────────────────────────────────────────────────

    def warm_up(self, compile: bool = False) -> "AdmeshIntentPipeline":
        """
        Pre-load all models and optionally compile them with torch.compile().

        Call once after instantiation so the first real request pays no
        model-load cost.  HF Inference Endpoints automatically sends a
        warm-up probe before routing live traffic, so this is optional there.

        Parameters
        ----------
        compile:
            If ``True``, call ``torch.compile()`` on the DistilBERT encoder
            and IAB classifier (requires PyTorch >= 2.0).  Gives ~15-30 %
            CPU speedup after the first traced call.
        """
        self._ensure_loaded()

        if compile:
            import torch  # noqa: PLC0415
            if not hasattr(torch, "compile"):
                import warnings
                warnings.warn(
                    "torch.compile() is not available (PyTorch >= 2.0 required). "
                    "Skipping.",
                    stacklevel=2,
                )
            else:
                from multitask_runtime import get_multitask_runtime  # noqa: PLC0415
                from model_runtime import get_head  # noqa: PLC0415

                rt = get_multitask_runtime()
                if rt._model is not None:
                    rt._model = torch.compile(rt._model)
                iab_head = get_head("iab_content")
                if iab_head._model is not None:
                    iab_head._model = torch.compile(iab_head._model)

        # Dry run — triggers any remaining lazy init (calibration JSON reads, etc.)
        self("warm up query for intent classification", force_iab_placeholder=True)
        return self

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "admesh/agentic-intent-classifier",
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> "AdmeshIntentPipeline":
        """
        Download the model bundle from HF Hub and return a ready-to-use instance.

        Parameters
        ----------
        repo_id:
            HF Hub model id.
        revision:
            Optional git commit hash to pin a specific release.
        token:
            Optional HF auth token for private repos.

        Example
        -------
        ::

            from pipeline import AdmeshIntentPipeline
            clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
            print(clf("I need a CRM for a 5-person startup"))
        """
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required. Install: pip install huggingface_hub"
            ) from exc

        kwargs: dict = {"repo_type": "model"}
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token

        local_dir = snapshot_download(repo_id=repo_id, **kwargs)
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        return cls()

    # ── internal ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._classify_fn is None:
            from combined_inference import classify_query  # noqa: PLC0415
            self._classify_fn = classify_query

    def __repr__(self) -> str:
        state = "loaded" if self._classify_fn is not None else "not yet loaded"
        return f"AdmeshIntentPipeline(classify_fn={state})"
