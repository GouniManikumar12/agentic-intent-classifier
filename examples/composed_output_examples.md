# Composed Output Examples

These examples show the end-to-end flow:

`query -> model_output -> system_decision`

## 1. Informational Awareness

Query:

```text
What is CRM software?
```

Observed behavior:

- classified as `intent.type=informational`
- classified as `decision_phase=awareness`
- fell back because confidence is below the demo threshold
- system decision sets `monetization_eligibility=not_allowed`

Why fallback happened:

- the phase is sensible, but the combined confidence is still too low for monetization decisions

## 2. Commercial Comparison

Query:

```text
Compare AI search monetization platforms for publishers
```

Observed behavior:

- classified as `intent.type=commercial`
- classified as `decision_phase=consideration`
- still falls back due low confidence
- opportunity remains `none` because fallback takes precedence

Why fallback happened:

- the underlying labels are directionally right, but the frozen baseline is intentionally conservative

## 3. Post-Purchase Setup

Query:

```text
How do I set up my new CRM?
```

Observed behavior:

- classified as `decision_phase=post_purchase`
- treated as low-confidence informational setup behavior
- system decision stays restricted

Why fallback happened:

- the system recognizes the lifecycle stage, but confidence is not high enough to remove fallback

## 4. Support / Sensitive

Query:

```text
I cannot log into my account
```

Observed behavior:

- classified as `decision_phase=support`
- intent-type falls into `personal_reflection` in `v0.1`
- fallback applies with `policy_default`
- monetization is blocked

Why fallback happened:

- this is a known limitation of the intent-type head and exactly the kind of case that should fail safe
