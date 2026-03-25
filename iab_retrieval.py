from __future__ import annotations

import json
import re
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import (
    IAB_RETRIEVAL_DEPTH_BONUS,
    IAB_RETRIEVAL_MODEL_MAX_LENGTH,
    IAB_RETRIEVAL_MODEL_NAME,
    IAB_RETRIEVAL_PREFIX_CONFIDENCE_THRESHOLDS,
    IAB_RETRIEVAL_TOP_K,
    IAB_TAXONOMY_EMBEDDINGS_PATH,
    IAB_TAXONOMY_NODES_PATH,
    IAB_TAXONOMY_VERSION,
    ensure_artifact_dirs,
)
from iab_taxonomy import IabNode, get_iab_taxonomy, path_to_label

RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "best",
    "buy",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "tonight",
    "what",
    "which",
    "with",
}
GTE_QWEN_QUERY_INSTRUCTION = "Given a user query, retrieve the most relevant IAB content taxonomy category."


def round_score(value: float) -> float:
    return round(float(value), 4)


def _normalize_keyword(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _keyword_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize_keyword(value).split()
        if token and token not in RETRIEVAL_STOPWORDS and len(token) > 1
    }


def _is_gte_qwen_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return "gte-qwen" in normalized


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(torch.all(attention_mask[:, -1] == 1))
    if left_padding:
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


def _node_keywords(node: IabNode) -> list[str]:
    keywords = {node.label, node.path_label}
    keywords.update(node.path)
    normalized = {_normalize_keyword(keyword) for keyword in keywords if keyword.strip()}
    return sorted(keyword for keyword in normalized if keyword)


def _node_retrieval_text(node: IabNode) -> str:
    keywords = _node_keywords(node)
    parts = [
        f"IAB category path: {node.path_label}",
        f"Canonical label: {node.label}",
        f"Tier depth: {node.level}",
    ]
    if len(node.path) > 1:
        parts.append(f"Parent path: {' > '.join(node.path[:-1])}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    return ". ".join(parts)


def _serialize_node(node: IabNode) -> dict:
    return {
        "unique_id": node.unique_id,
        "parent_id": node.parent_id,
        "label": node.label,
        "path": list(node.path),
        "path_label": node.path_label,
        "level": node.level,
        "keywords": _node_keywords(node),
        "retrieval_text": _node_retrieval_text(node),
    }


class LocalTextEmbedder:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._batch_size = 32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_gte_qwen = _is_gte_qwen_model(model_name)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self._is_gte_qwen,
            )
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            model_kwargs = {"trust_remote_code": self._is_gte_qwen}
            if self._device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self._model.to(self._device)
            self._model.eval()
        return self._model

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        return self._encode_texts(texts, batch_size=batch_size, treat_as_query=False)

    def encode_queries(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        return self._encode_texts(texts, batch_size=batch_size, treat_as_query=True)

    def _encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        treat_as_query: bool = False,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        effective_batch_size = batch_size or self._batch_size
        rows: list[torch.Tensor] = []
        for start in range(0, len(texts), effective_batch_size):
            batch_texts = texts[start : start + effective_batch_size]
            if treat_as_query and self._is_gte_qwen:
                batch_texts = [
                    f"Instruct: {GTE_QWEN_QUERY_INSTRUCTION}\nQuery: {text}"
                    for text in batch_texts
                ]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            if self._is_gte_qwen:
                pooled = _last_token_pool(hidden, inputs["attention_mask"])
            else:
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            rows.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
        return torch.cat(rows, dim=0)


@lru_cache(maxsize=1)
def get_iab_text_embedder() -> LocalTextEmbedder:
    return LocalTextEmbedder(IAB_RETRIEVAL_MODEL_NAME, IAB_RETRIEVAL_MODEL_MAX_LENGTH)


def build_iab_taxonomy_embedding_index(batch_size: int = 32) -> dict:
    ensure_artifact_dirs()
    taxonomy = get_iab_taxonomy()
    nodes = [_serialize_node(node) for node in taxonomy.nodes]
    embedder = get_iab_text_embedder()
    embeddings = embedder.encode_documents([node["retrieval_text"] for node in nodes], batch_size=batch_size)

    IAB_TAXONOMY_NODES_PATH.write_text(json.dumps(nodes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_name": embedder.model_name,
            "taxonomy_version": IAB_TAXONOMY_VERSION,
            "embedding_dim": int(embeddings.shape[1]),
            "node_count": len(nodes),
            "embeddings": embeddings,
        },
        IAB_TAXONOMY_EMBEDDINGS_PATH,
    )
    return {
        "taxonomy_version": IAB_TAXONOMY_VERSION,
        "model_name": embedder.model_name,
        "node_count": len(nodes),
        "embedding_dim": int(embeddings.shape[1]),
        "nodes_path": str(IAB_TAXONOMY_NODES_PATH),
        "embeddings_path": str(IAB_TAXONOMY_EMBEDDINGS_PATH),
    }


class IabEmbeddingRetriever:
    def __init__(self):
        self.taxonomy = get_iab_taxonomy()
        self.embedder = get_iab_text_embedder()
        self._nodes: list[dict] | None = None
        self._embeddings: torch.Tensor | None = None

    def _load_index(self) -> bool:
        if self._nodes is not None and self._embeddings is not None:
            return True
        if not IAB_TAXONOMY_NODES_PATH.exists() or not IAB_TAXONOMY_EMBEDDINGS_PATH.exists():
            return False

        nodes = json.loads(IAB_TAXONOMY_NODES_PATH.read_text(encoding="utf-8"))
        payload = torch.load(IAB_TAXONOMY_EMBEDDINGS_PATH, map_location="cpu")
        if payload.get("model_name") != IAB_RETRIEVAL_MODEL_NAME:
            return False
        if payload.get("taxonomy_version") != IAB_TAXONOMY_VERSION:
            return False

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if len(nodes) != embeddings.shape[0]:
            return False

        self._nodes = nodes
        self._embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        return True

    def ready(self) -> bool:
        return self._load_index()

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        return min(max((score + 1.0) / 2.0, 0.0), 1.0)

    def _candidate_from_index(self, score: float, index: int) -> dict:
        assert self._nodes is not None
        node = self._nodes[index]
        confidence = self._score_to_confidence(float(score))
        adjusted_confidence = confidence + (IAB_RETRIEVAL_DEPTH_BONUS * max(int(node["level"]) - 1, 0))
        return {
            "unique_id": node["unique_id"],
            "label": node["label"],
            "path": tuple(node["path"]),
            "path_label": node["path_label"],
            "level": int(node["level"]),
            "confidence": round_score(confidence),
            "adjusted_confidence": round_score(adjusted_confidence),
            "keywords": list(node.get("keywords", [])),
        }

    def _rerank_candidates(self, query_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        query_normalized = _normalize_keyword(query_text)
        query_tokens = _keyword_tokens(query_text)

        reranked = []
        for candidate in candidates:
            keyword_tokens = set()
            for keyword in candidate.get("keywords", []):
                keyword_tokens.update(_keyword_tokens(keyword))

            token_overlap = len(query_tokens & keyword_tokens)
            path_overlap = len(query_tokens & _keyword_tokens(candidate["path_label"]))
            lexical_bonus = min(0.04, (0.008 * token_overlap) + (0.004 * path_overlap))

            reranked.append(
                {
                    **candidate,
                    "token_overlap": token_overlap,
                    "path_overlap": path_overlap,
                    "lexical_bonus": round_score(lexical_bonus),
                    "rerank_score": round_score(candidate["adjusted_confidence"] + lexical_bonus),
                }
            )

        reranked.sort(
            key=lambda item: (
                item["rerank_score"],
                item["adjusted_confidence"],
                item["confidence"],
            ),
            reverse=True,
        )
        return reranked

    def _top_candidates_from_embedding(self, query_text: str, query_embedding: torch.Tensor) -> list[dict]:
        if not self._load_index():
            return []

        assert self._embeddings is not None

        scores = torch.mv(self._embeddings, query_embedding)
        top_k = min(IAB_RETRIEVAL_TOP_K, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k=top_k)

        candidates = [self._candidate_from_index(score, index) for score, index in zip(top_scores.tolist(), top_indices.tolist())]
        return self._rerank_candidates(query_text, candidates)

    def _top_candidates(self, text: str) -> list[dict]:
        if not self._load_index():
            return []
        query_embedding = self.embedder.encode_queries([text])[0]
        return self._top_candidates_from_embedding(text, query_embedding)

    def _select_path(self, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None

        top_candidate = candidates[0]
        top_path = tuple(top_candidate["path"])
        top_margin = round_score(
            top_candidate["confidence"] - candidates[1]["confidence"] if len(candidates) > 1 else top_candidate["confidence"]
        )
        prefix_support: dict[tuple[str, ...], float] = {}
        for depth in range(1, len(top_path) + 1):
            prefix = top_path[:depth]
            prefix_support[prefix] = max(
                candidate["confidence"]
                for candidate in candidates
                if tuple(candidate["path"][:depth]) == prefix
            )

        selected_path: tuple[str, ...] | None = None
        selected_threshold = 0.0
        for depth in range(1, len(top_path) + 1):
            threshold = IAB_RETRIEVAL_PREFIX_CONFIDENCE_THRESHOLDS.get(depth, 0.62)
            prefix = top_path[:depth]
            if prefix_support[prefix] >= threshold:
                selected_path = prefix
                selected_threshold = threshold
                continue
            break

        if selected_path is None:
            return None

        stopped_reason = "accepted" if selected_path == top_path else "parent_fallback"
        if len(top_path) > 1:
            ambiguous_sibling = any(
                tuple(candidate["path"][:-1]) == top_path[:-1]
                and (top_candidate["confidence"] - candidate["confidence"]) <= 0.03
                for candidate in candidates[1:]
            )
            if ambiguous_sibling:
                selected_path = top_path[:-1]
                selected_threshold = IAB_RETRIEVAL_PREFIX_CONFIDENCE_THRESHOLDS.get(len(selected_path), 0.62)
                stopped_reason = "ambiguous_sibling_parent_fallback"

        mapping_confidence = prefix_support[selected_path]
        return {
            "path": selected_path,
            "path_label": path_to_label(selected_path),
            "mapping_mode": "nearest_equivalent",
            "mapping_confidence": round_score(mapping_confidence),
            "confidence_threshold": round_score(selected_threshold),
            "top_candidate_confidence": round_score(top_candidate["confidence"]),
            "top_margin": top_margin,
            "stopped_reason": stopped_reason,
        }

    def predict(self, text: str) -> dict | None:
        candidates = self._top_candidates(text)
        return self._prediction_from_candidates(candidates)

    def _prediction_from_candidates(self, candidates: list[dict]) -> dict | None:
        selection = self._select_path(candidates)
        if selection is None:
            return None

        content = self.taxonomy.build_content_object(
            path=selection["path"],
            mapping_mode=selection["mapping_mode"],
            mapping_confidence=selection["mapping_confidence"],
        )
        return {
            "label": selection["path_label"],
            "confidence": selection["mapping_confidence"],
            "raw_confidence": selection["top_candidate_confidence"],
            "confidence_threshold": selection["confidence_threshold"],
            "calibrated": False,
            "meets_confidence_threshold": True,
            "content": content,
            "path": selection["path"],
            "mapping_mode": selection["mapping_mode"],
            "mapping_confidence": selection["mapping_confidence"],
            "source": "embedding_retrieval",
            "retrieval_model_name": IAB_RETRIEVAL_MODEL_NAME,
            "stopped_reason": selection["stopped_reason"],
            "top_margin": selection["top_margin"],
            "top_candidates": [
                {
                    **candidate,
                    "path": list(candidate["path"]),
                    "keywords": candidate["keywords"][:12],
                }
                for candidate in candidates
            ],
        }

    def predict_batch(self, texts: list[str], batch_size: int | None = None) -> list[dict | None]:
        if not texts:
            return []
        if not self._load_index():
            return [None for _ in texts]

        query_embeddings = self.embedder.encode_queries(texts, batch_size=batch_size)
        return [
            self._prediction_from_candidates(self._top_candidates_from_embedding(text, query_embedding))
            for text, query_embedding in zip(texts, query_embeddings)
        ]


@lru_cache(maxsize=1)
def get_iab_embedding_retriever() -> IabEmbeddingRetriever:
    return IabEmbeddingRetriever()


def predict_iab_content_retrieval(text: str) -> dict | None:
    retriever = get_iab_embedding_retriever()
    if not retriever.ready():
        return None
    return retriever.predict(text)


def predict_iab_content_retrieval_batch(texts: list[str], batch_size: int | None = None) -> list[dict | None]:
    retriever = get_iab_embedding_retriever()
    if not retriever.ready():
        return [None for _ in texts]
    return retriever.predict_batch(texts, batch_size=batch_size)
