from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from .config import IAB_TAXONOMY_GRAPH_PATH, IAB_TAXONOMY_PATH, IAB_TAXONOMY_VERSION  # type: ignore
except ImportError:
    from config import IAB_TAXONOMY_GRAPH_PATH, IAB_TAXONOMY_PATH, IAB_TAXONOMY_VERSION


@dataclass(frozen=True)
class IabNode:
    unique_id: str
    parent_id: str | None
    label: str
    path: tuple[str, ...]

    @property
    def level(self) -> int:
        return len(self.path)

    @property
    def path_label(self) -> str:
        return path_to_label(self.path)


class IabTaxonomy:
    def __init__(self, nodes: list[IabNode]):
        self.nodes = nodes
        self._path_index = {node.path: node for node in nodes}
        self._children_index: dict[tuple[str, ...], list[IabNode]] = {}
        self._level_index: dict[int, list[IabNode]] = {}
        for node in nodes:
            self._children_index.setdefault(node.path[:-1], []).append(node)
            self._level_index.setdefault(node.level, []).append(node)
        for children in self._children_index.values():
            children.sort(key=lambda item: item.path)
        for level_nodes in self._level_index.values():
            level_nodes.sort(key=lambda item: item.path)

    def get_node(self, path: tuple[str, ...]) -> IabNode:
        if path not in self._path_index:
            raise KeyError(f"Unknown IAB path: {path}")
        return self._path_index[path]

    def build_level(self, path: tuple[str, ...]) -> dict:
        node = self.get_node(path)
        return {"id": node.unique_id, "label": node.label}

    def has_path(self, path: tuple[str, ...]) -> bool:
        return path in self._path_index

    def immediate_children(self, prefix: tuple[str, ...]) -> list[IabNode]:
        return list(self._children_index.get(prefix, []))

    def siblings(self, path: tuple[str, ...]) -> list[IabNode]:
        node = self.get_node(path)
        return [candidate for candidate in self._children_index.get(path[:-1], []) if candidate.path != node.path]

    def level_nodes(self, level: int) -> list[IabNode]:
        return list(self._level_index.get(level, []))

    def to_training_graph(self) -> dict:
        nodes = []
        for node in self.nodes:
            child_nodes = self.immediate_children(node.path)
            sibling_nodes = self.siblings(node.path)
            nodes.append(
                {
                    "node_id": node.unique_id,
                    "parent_id": node.parent_id,
                    "level": node.level,
                    "label": node.label,
                    "path": list(node.path),
                    "path_label": node.path_label,
                    "child_ids": [child.unique_id for child in child_nodes],
                    "child_paths": [child.path_label for child in child_nodes],
                    "sibling_ids": [sibling.unique_id for sibling in sibling_nodes],
                    "sibling_paths": [sibling.path_label for sibling in sibling_nodes],
                    "canonical_surface_name": node.label,
                }
            )
        return {
            "taxonomy": "IAB Content Taxonomy",
            "taxonomy_version": IAB_TAXONOMY_VERSION,
            "node_count": len(nodes),
            "level_counts": {
                f"tier{level}": len(self.level_nodes(level))
                for level in range(1, 5)
            },
            "nodes": nodes,
        }

    def build_content_object(self, path: tuple[str, ...], mapping_mode: str, mapping_confidence: float) -> dict:
        if not path:
            raise ValueError("IAB path must not be empty")

        payload = {
            "taxonomy": "IAB Content Taxonomy",
            "taxonomy_version": IAB_TAXONOMY_VERSION,
            "tier1": self.build_level(path[:1]),
            "mapping_mode": mapping_mode,
            "mapping_confidence": round(float(mapping_confidence), 4),
        }
        if len(path) >= 2:
            payload["tier2"] = self.build_level(path[:2])
        if len(path) >= 3:
            payload["tier3"] = self.build_level(path[:3])
        if len(path) >= 4:
            payload["tier4"] = self.build_level(path[:4])
        return payload

    def build_content_object_from_label(
        self,
        path_label: str,
        mapping_mode: str,
        mapping_confidence: float,
    ) -> dict:
        return self.build_content_object(
            path=parse_path_label(path_label),
            mapping_mode=mapping_mode,
            mapping_confidence=mapping_confidence,
        )


def parse_path_label(path_label: str) -> tuple[str, ...]:
    path = tuple(part.strip() for part in path_label.split(">") if part.strip())
    if not path:
        raise ValueError("IAB path label must not be empty")
    return path


def path_to_label(path: tuple[str, ...]) -> str:
    if not path:
        raise ValueError("IAB path must not be empty")
    return " > ".join(path)


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)

    header = rows[1]
    data_rows = rows[2:]
    parsed = []
    for row in data_rows:
        padded = row + [""] * (len(header) - len(row))
        parsed.append(dict(zip(header, padded)))
    return parsed


@lru_cache(maxsize=1)
def get_iab_taxonomy() -> IabTaxonomy:
    nodes = []
    for row in _load_rows(IAB_TAXONOMY_PATH):
        path = tuple(
            value.strip()
            for key in ("Tier 1", "Tier 2", "Tier 3", "Tier 4")
            if (value := row.get(key, "").strip())
        )
        if not path:
            continue
        nodes.append(
            IabNode(
                unique_id=row["Unique ID"].strip(),
                parent_id=row["Parent"].strip() or None,
                label=row["Name"].strip(),
                path=path,
            )
        )
    return IabTaxonomy(nodes)


def write_training_graph(path: Path = IAB_TAXONOMY_GRAPH_PATH) -> Path:
    taxonomy = get_iab_taxonomy()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(taxonomy.to_training_graph(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
