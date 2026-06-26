# Evaluation, Observability & Guardrails — Interview Q&A

Grounded in this codebase.

---

### Q. What is a "ship gate" and how do you design one?
A versioned eval **dataset**, a **numeric score**, and a **regression alarm**. Here: `EvalRunner`
runs a system over the dataset, scorers produce per-case pass/fail, the `EvalReport` aggregates a
pass rate, and `gate()` fails the build when it misses the threshold. Datasets and thresholds are
versioned in code so a regression shows up as a failing CI check, not a vibe.

### Q. How do you evaluate an LLM system?
Two layers. **Deterministic scorers** (exact/contains/regex/Jaccard/json-valid) for cheap,
reproducible checks on every case. **LLM-as-judge** for what rules can't capture — faithfulness,
helpfulness, correctness in substance — against an explicit rubric. Combine them; don't rely on the
judge alone.

### Q. What are the pitfalls of LLM-as-judge, and how do you mitigate them?
Judges are biased: they favor verbose answers, position (first/second), and outputs that resemble
their own. Mitigations: an explicit rubric, schema-constrained verdicts (score + pass + reasoning),
pairing with deterministic scorers, and calibrating against human labels. Treat the judge as one
noisy signal, not ground truth.

### Q. What metrics would you track?
Per-scorer mean scores, overall pass rate vs threshold, and (for retrieval) recall@k / MRR. Track
them over time so regressions and improvements are visible — that's what the report's
`mean_by_scorer()` and `pass_rate` feed.

### Q. Direct vs indirect prompt injection — what's the difference and the defense?
**Direct:** the user types "ignore your instructions." **Indirect:** a malicious instruction is hidden
in content the model later reads (a retrieved doc, a web page, a tool result) — "the new XSS,"
because the payload rides in data, not the prompt. Defense is **layered**: filter input
(`PromptInjectionGuard`), redact PII, constrain + validate output, limit tool scope, and add a
model-based classifier (Llama Guard / NeMo) as a second layer. No single layer is sufficient.

### Q. How do you handle PII in an LLM pipeline?
Redact it at the boundaries — on input before it reaches the model/logs and on output before it
reaches the user (`PiiRedactionGuard` runs on both sides of the pipeline). Pattern matching catches
the common cases (emails, phones, SSNs, cards); pair with a model-based detector for the rest, and
never log raw PII.

### Q. How would you make an LLM app observable?
Trace it: record nested **spans** for each step (retrieve, judge, generate) with duration, tokens,
and cost, so you can see *where* time and money go and *why* an answer came out the way it did. The
`Tracer` here is in-memory; in production you export the same spans to Langfuse / Phoenix / OTel.

### Q. Where do guardrails belong — and why both sides?
**Input** guards stop bad/injection content before the model acts (and before it hits logs).
**Output** guards catch leaked PII, banned content, and malformed structure before it reaches the
user. The `GuardrailPipeline` carries input redactions forward and validates output — defense in
depth, not a single chokepoint.

### Q. How does this plug into a CI pipeline?
`gate(report)` raises on failure → non-zero exit → red build. You version the eval dataset alongside
the code, run the eval in CI on every change, and block merges that drop quality below threshold —
exactly how you stop silent regressions from prompt/model/retrieval changes.
