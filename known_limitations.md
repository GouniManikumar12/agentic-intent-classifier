# Known Limitations

## Model Limits

- `intent_type v0.1` is still conservative and often falls below the confidence threshold on otherwise sensible prompts.
- `decision_phase v0.1` is behaviorally useful but still weak in aggregate metrics.
- the composed output is gated by the weaker of the two model confidences, so fallback triggers often.

## Boundary Confusion

- `support` and `post_purchase` are cleaner than before, but still fragile on phrasing that mixes setup with a broken flow.
- `consideration` and `decision` can still collapse when a buying query is phrased like a general question.
- `awareness` and `research` remain close for “help me understand” style prompts.

## Intent-Type Gaps

- the current `intent_type` head does not have a dedicated `support` class, so some support-like prompts are forced into `personal_reflection`, `informational`, or fallback behavior.
- short contextual follow-ups are still sensitive to wording and may be over-fallbacked.
- price-seeking prompts can still flip between `commercial`, `informational`, and fallback depending on phrasing.

## System-Layer Caveats

- the current policy and opportunity logic is rule-based and deliberately simple.
- `commercial_score` is heuristic, not learned.
- fallback is intentionally conservative, which is useful for demos but suppresses opportunity signals.

## Product Limits

- no multi-turn memory is used in the combined classifier.
- no IAB content mapping is produced yet.
- no audit store, trace viewer, or UI is included beyond the local JSON demo endpoint.
