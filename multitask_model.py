from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class MultiTaskLabelSizes:
    intent_type: int
    intent_subtype: int
    decision_phase: int


class MultiTaskIntentModel(nn.Module):
    def __init__(self, base_model_name: str, label_sizes: MultiTaskLabelSizes):
        super().__init__()
        self.base_model_name = base_model_name
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(getattr(self.encoder.config, "seq_classif_dropout", 0.2)))
        self.intent_type_head = nn.Linear(hidden_size, label_sizes.intent_type)
        self.intent_subtype_head = nn.Linear(hidden_size, label_sizes.intent_subtype)
        self.decision_phase_head = nn.Linear(hidden_size, label_sizes.decision_phase)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {
            "intent_type_logits": self.intent_type_head(pooled),
            "intent_subtype_logits": self.intent_subtype_head(pooled),
            "decision_phase_logits": self.decision_phase_head(pooled),
        }
