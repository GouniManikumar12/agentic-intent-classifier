# Known Limitations

## Model Limits

- `intent_type v0.1` now uses a manually tuned fallback threshold, but some ambiguous prompts still clear as non-safe intent labels.
- `intent_subtype v0.1` is useful for commercial patterns, but still weak on sparse support and price-seeking cases.
- `decision_phase v0.1` is behaviorally useful but still weak in aggregate metrics.
- `iab_content v0.1` is now a hybrid classifier-plus-fallback stack, but the supervised head is trained on synthetic taxonomy-backed prompts rather than real labeled traffic.
- the composed output is still gated by the weaker calibrated components, so action-style queries can over-fallback.

## Boundary Confusion

- `support` and `post_purchase` are cleaner than before, but still fragile on phrasing that mixes setup with a broken flow.
- `consideration` and `decision` can still collapse when a buying query is phrased like a general question.
- `awareness` and `research` remain close for “help me understand” style prompts.
- `deal_seeking` and `education` still blur on price questions phrased as general explanations.
- some broad software topics still need nearest-equivalent IAB mappings because the taxonomy does not have exact SaaS product nodes.
- CRM-style prompts are still one of the cleanest examples where the classifier alone can drift toward generic software and the rules layer has to rescue the final path.

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
- IAB routing uses the local 3.0 taxonomy TSV, a supervised `iab_content` head, and nearest-equivalent fallback rules; exact path quality still depends on synthetic training coverage plus keyword coverage.

## Regression Tracking

- structured known-failure tracking now lives in [known_failure_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/known_failure_cases.json).
- acceptable weaknesses are now separated from `must_fix` behaviors and can be rerun via [run_regression_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_regression_suite.py).
- current acceptable weaknesses include signup over-fallback, price-seeking underclassification, and support subtype bleed into `emotional_reflection`.
- IAB mapping checks now live in [iab_mapping_cases.json](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/examples/iab_mapping_cases.json) and can be rerun via [run_iab_mapping_suite.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/run_iab_mapping_suite.py).
- IAB classifier threshold tuning now lives in [sweep_iab_threshold.py](/Users/manikumargouni/Desktop/AdMesh/protocol/agentic-intent-classifier/evaluation/sweep_iab_threshold.py).

## Product Limits

- no multi-turn memory is used in the combined classifier.
- no real-traffic labeled IAB dataset is produced yet.
- no audit store, trace viewer, or UI is included beyond the local JSON demo endpoint.
