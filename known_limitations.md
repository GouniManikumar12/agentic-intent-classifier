# Known Limitations

## Model Limits

- `intent_type v0.1` now uses a manually tuned fallback threshold, but some ambiguous prompts still clear as non-safe intent labels.
- `intent_subtype v0.1` is useful for commercial patterns, but still weak on sparse support and price-seeking cases.
- `decision_phase v0.1` is behaviorally useful but still weak in aggregate metrics.
- `iab_content` now uses local embedding retrieval over taxonomy nodes, but exact-path quality still depends on how well the embedding model separates close sibling categories.
- the composed output is still gated by the weaker calibrated components, so action-style queries can over-fallback.

## Boundary Confusion

- `support` and `post_purchase` are cleaner than before, but still fragile on phrasing that mixes setup with a broken flow.
- `consideration` and `decision` can still collapse when a buying query is phrased like a general question.
- `awareness` and `research` remain close for “help me understand” style prompts.
- `deal_seeking` and `education` still blur on price questions phrased as general explanations.
- some broad software topics still need nearest-equivalent IAB mappings because the taxonomy does not have exact SaaS product nodes.
- CRM-style prompts are still a weak spot because retrieval can drift toward adjacent business software categories without a reranker.

## Intent-Type Gaps

- the expanded `intent_type` taxonomy now includes `support`, `exploratory`, `creative_generation`, `chit_chat`, and `prohibited`, but these new classes are still data-light until the next full retraining pass.
- short contextual follow-ups are still sensitive to wording and may be over-fallbacked.
- price-seeking prompts can still flip between `commercial`, `informational`, and fallback depending on phrasing.

## System-Layer Caveats

- the current policy and opportunity logic is still rule-based even though it now uses subtype.
- `commercial_score` is heuristic, not learned.
- the current `intent_type` threshold is manually sweep-tuned rather than learned jointly with the system layer.
- fallback is still intentionally conservative, which is useful for demos but suppresses some strong action signals.
- subtype is now part of the system decision, but the system still trusts `decision_phase` more than subtype on support safety.
- IAB routing uses the local 3.0 taxonomy TSV plus a local embedding index; exact path quality still depends on the retrieval model and the candidate text stored for each taxonomy node.

## Regression Tracking

- structured known-failure tracking now lives in [known_failure_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/known_failure_cases.json).
- acceptable weaknesses are now separated from `must_fix` behaviors and can be rerun via [run_regression_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_regression_suite.py).
- current acceptable weaknesses include signup over-fallback, price-seeking underclassification, and support subtype bleed into `emotional_reflection`.
- IAB behavior-lock checks now live in [iab_behavior_lock_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/iab_behavior_lock_cases.json) and can be rerun via [run_iab_mapping_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_iab_mapping_suite.py).
- curated IAB quality targets live in [iab_mapping_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/iab_mapping_cases.json), with broader target coverage in [iab_cross_vertical_mapping_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/iab_cross_vertical_mapping_cases.json) and [run_iab_quality_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_iab_quality_suite.py).

## Product Limits

- no multi-turn memory is used in the combined classifier.
- no real-traffic labeled IAB dataset is produced yet.
- no audit store, trace viewer, or UI is included beyond the local JSON demo endpoint.
