# Architecture & Design Decisions

Why the kit is shaped the way it is. Read alongside the source.

## Three independent capabilities, one small library
Evaluation, tracing, and guardrails are separable — you can use any one alone — but they share a
philosophy: **measurable, observable, safe**, with no external service required so they run in CI and
in tests. Each is built on small protocols so it extends without edits.

## Evaluation

### Scorers and the judge share one interface
A `Scorer` maps `(output, case) -> Score` in [0, 1] with a pass/fail. Deterministic scorers (exact,
contains, regex, Jaccard, json-valid) are cheap and reproducible — run them on every case. The
**LLM-as-judge** implements the *same* interface, so it drops into the same runner; it's the
expensive, nuanced complement for faithfulness/helpfulness that rules can't capture.

**Why one interface?** The runner, report, and gate don't care whether a score came from a regex or a
model — so you can mix cheap and expensive scorers freely and swap a `FakeJudge` in tests.

### The gate is the product
`EvalRunner` aggregates `CaseResult`s into an `EvalReport`; `gate()` raises `GateFailure` when the
pass rate misses the threshold. That's the **ship gate** interviewers ask you to design: a versioned
dataset + a numeric score + a regression alarm. Relevance/labels live in the dataset, thresholds in
the gate — both versioned, both diffable.

**Judge caveats (built into the docs on purpose):** LLM judges are biased — verbosity, position,
self-preference. Mitigations: an explicit rubric, schema-constrained verdicts, and pairing the judge
with cheap deterministic scorers so a single biased judgment can't pass a bad output alone.

## Tracing

An in-memory `Tracer` records nested **spans** (name, attributes, duration, tokens, cost) via a
context manager, producing an inspectable tree and roll-up totals. **No external dependency on
purpose** — in production you export the same spans to Langfuse / Arize Phoenix / OpenTelemetry; the
*shape* (spans with usage) is identical, so the mental model transfers directly.

## Guardrails

### Layered defense against "the new XSS"
Indirect prompt injection — a malicious instruction hidden in retrieved content — is the headline AI
security risk. The defense is **layered**, and a `Guard` does one of two things:
- **Redact** (transform): `PiiRedactionGuard` strips emails/phones/SSNs/cards.
- **Block** (flag): `PromptInjectionGuard`, `LengthGuard`, `BannedContentGuard`.

`GuardrailPipeline` runs input guards before the model and output guards after, **carrying
redactions forward** so later guards see sanitized text. A blocked input raises `GuardrailViolation`.

**Honest scope:** these are pattern-based — fast, deterministic, testable, and a real first layer, but
not sufficient alone. Production pairs them with a **model-based classifier** (Llama Guard, NeMo
Guardrails) as a second layer, and constrains/validates output (structured output, schema checks).
The pattern approach is documented as layer one, not the whole answer.

## How it retrofits the portfolio (Milestone 4 goal)
The kit is the eval/guardrail/observability layer for earlier milestones:
- `rag-knowledge-assistant`: judge **faithfulness**; gate the retrieval eval in CI; trace retrieve→generate.
- `agentic-workbench`: guard agent inputs for **injection** before tools run; trace the loop.
- `structured-extractor`: `JsonValid` + schema scorers as the quality gate.

## Trade-offs left open
- Model-based guardrails as the second layer.
- Span export to a real backend (Langfuse/Phoenix/OTel).
- Embedding-based semantic similarity; pairwise/preference judging + calibration.
