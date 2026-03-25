"""
AdmeshIntentPipeline — standard callable interface for admesh/agentic-intent-classifier.

Wraps combined_inference.classify_query() so the full intent + IAB
classification stack is usable via a single import, without fighting
with the transformers.Pipeline model-loading machinery.

Quick start (after snapshot_download):

    import sys
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download("admesh/agentic-intent-classifier", repo_type="model")
    sys.path.insert(0, local_dir)

    from pipeline import AdmeshIntentPipeline
    clf = AdmeshIntentPipeline()
    print(clf("Which laptop should I buy for college?"))

One-liner for Colab / scripts:

    clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
    print(clf("I need a CRM for a 5-person startup"))
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


class AdmeshIntentPipeline:
    """
    Intent + IAB classification pipeline for admesh/agentic-intent-classifier.

    Parameters
    ----------
    model_dir:
        Path to the local snapshot directory.  When constructed via
        ``from_pretrained()`` this is set automatically.  When
        constructing manually after adding the directory to ``sys.path``
        you can leave it as ``None``.
    """

    def __init__(self, model_dir: Union[str, Path, None] = None) -> None:
        if model_dir is not None:
            model_dir = Path(model_dir).resolve()
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))

        # Ensure the directory containing this file is also on the path so
        # combined_inference imports work when the pipeline is instantiated
        # from a sys.path-based import.
        _self_dir = Path(__file__).resolve().parent
        if str(_self_dir) not in sys.path:
            sys.path.insert(0, str(_self_dir))

        # Lazy-loaded reference — populated on first __call__.
        self._classify_fn = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "admesh/agentic-intent-classifier",
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> "AdmeshIntentPipeline":
        """
        Download the model bundle from Hugging Face Hub and return a
        ready-to-use pipeline instance.

        Parameters
        ----------
        repo_id:
            HF Hub model id, e.g. ``"admesh/agentic-intent-classifier"``.
        revision:
            Optional git revision / commit hash to pin a specific release.
        token:
            Optional HF auth token (needed for private repos).

        Example
        -------
        ::

            clf = AdmeshIntentPipeline.from_pretrained("admesh/agentic-intent-classifier")
            result = clf("Which laptop should I buy for college?")
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for from_pretrained(). "
                "Install it with: pip install huggingface_hub"
            ) from exc

        kwargs: dict = {"repo_type": "model"}
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token

        local_dir = snapshot_download(repo_id=repo_id, **kwargs)
        return cls(model_dir=local_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._classify_fn is None:
            from combined_inference import classify_query  # noqa: PLC0415
            self._classify_fn = classify_query

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
            If ``True``, skip IAB classifier and return placeholder
            values regardless of whether IAB artifacts are present.

        Returns
        -------
        dict or list[dict]:
            Full classification payload (same structure as
            ``combined_inference.classify_query``).  Returns a single dict
            when *inputs* is a string, or a list of dicts when a list is
            passed.

        Examples
        --------
        ::

            clf = AdmeshIntentPipeline.from_pretrained()

            # single query
            result = clf("Which laptop should I buy for college?")

            # batch
            results = clf([
                "Best running shoes under $100",
                "How to set up a CI/CD pipeline",
            ])

            # custom thresholds
            result = clf(
                "Buy noise-cancelling headphones",
                threshold_overrides={"intent_type": 0.6},
            )
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

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "loaded" if self._classify_fn is not None else "not yet loaded"
        return f"AdmeshIntentPipeline(classify_fn={loaded})"
