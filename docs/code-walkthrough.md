# Code Walkthrough — a plain-English tour of llm-eval-kit

This is the "read me first" guide to the codebase. No jargon assumed. Every file is small on
purpose; you can read the whole library in an afternoon.

## Where to find X (cheat sheet)

| I want to find... | File | Look for |
|---|---|---|
| **The line where the gate decides pass/fail** | `src/llm_eval_kit/models.py` | `EvalReport.passed` — `return self.pass_rate >= self.threshold` |
| **The function that fails a CI build** | `src/llm_eval_kit/runner.py` | `gate()` — raises `GateFailure` when the report failed |
| **The planted eval case that fails the demo** | `src/llm_eval_kit/datasets.py` | `_DEMO_ANSWERS` — the `capital` case answers "Lyon" instead of "Paris" (marked `# wrong on purpose`) |
| **Why tests use FakeJudge, not a real judge** | `src/llm_eval_kit/judge.py` + `tests/test_judge.py` | `FakeJudge` docstring; explanation below |
| **The prompt-injection pattern list** (extend it to catch new attacks) | `src/llm_eval_kit/guardrails.py` | `_INJECTION_PATTERNS` — a list of `re.compile(...)` lines; add one line to catch a new phrasing |
| **The PII patterns** (email/phone/SSN/card) | `src/llm_eval_kit/guardrails.py` | `_PII_PATTERNS` — a dict of label → regex; add an entry to redact a new PII type |
| **Where spans open and close in the tracer** | `src/llm_eval_kit/tracing.py` | `Tracer.span()` — opens before `yield`, closes in `finally` |
| The judge selection (Gemini vs Claude vs Fake) | `src/llm_eval_kit/judge.py` | `make_judge()` |
| Each scorer (exact / contains / regex / Jaccard / JSON) | `src/llm_eval_kit/scorers.py` | one class per scorer |
| The shared Scorer interface | `src/llm_eval_kit/scorers.py` | `class Scorer(Protocol)` |
| Environment / `.env` settings | `src/llm_eval_kit/config.py` | `Settings` |
| The CLI commands (`evalkit run/guard/trace`) | `src/llm_eval_kit/cli.py` | `@app.command()` functions |
| The HTTP API | `src/llm_eval_kit/api.py` | FastAPI routes |
| Custom exceptions | `src/llm_eval_kit/errors.py` | `GateFailure`, `GuardrailViolation`, `JudgeError` |

## Suggested reading order

1. `models.py` — the nouns (EvalCase, Score, EvalReport). Everything else uses these.
2. `scorers.py` — the simplest graders. Meet the `Scorer` interface here.
3. `runner.py` — the loop that runs your system and applies scorers; `gate()` for CI.
4. `datasets.py` — the sample data, including the case that is wrong **on purpose**.
5. `judge.py` — an LLM grading an LLM, plus FakeJudge and judge selection.
6. `guardrails.py` — the input/output checkpoints (injection, PII, length, banned words).
7. `tracing.py` — spans and the tracer.
8. `cli.py`, `api.py`, `config.py`, `errors.py` — thin wiring around all of the above.

---

## 1. `models.py` — the data shapes

An eval, in one sentence: *ask your system every question in a dataset, grade every answer, and
check whether enough answers passed*. This file defines each of those nouns:

- `EvalCase` — one quiz question: an `input`, an optional `reference` (the answer key), an `id`.
- `Dataset` — a named list of cases.
- `Score` — one grader's mark on one answer: a value in `[0, 1]`, a `passed` bool, a `detail` note.
- `CaseResult` — one answer plus **all** its marks. A case passes only if *every* scorer passed it.
- `EvalReport` — the report card for the whole run.

### THE gate decision (the most important two lines in the repo)

```python
@property
def passed(self) -> bool:
    return self.pass_rate >= self.threshold
```

That's it — the entire ship/no-ship decision. `pass_rate` is "what fraction of cases passed";
`threshold` is the pass mark (default **0.8**, i.e. 80%). In the demo, 3 of 4 cases pass, so
`pass_rate = 0.75`, `0.75 >= 0.80` is False, and the gate **fails**. Raise the threshold and more
runs fail; lower it and more ship.

## 2. `scorers.py` — the graders

Every grader implements one tiny interface:

```python
class Scorer(Protocol):
    name: str
    def score(self, output: str, case: EvalCase) -> Score: ...
```

"Protocol" means: *anything* with a `name` and a `score()` method qualifies — no inheritance
needed. This is why an LLM judge and a one-line regex check are interchangeable to the runner.

The five graders, from strictest to loosest:

| Scorer | Passes when... |
|---|---|
| `ExactMatch` | the answer equals the reference exactly (whitespace trimmed) |
| `ContainsReference` | the reference appears anywhere in the answer (case-insensitive) |
| `RegexMatch` | the answer matches a pattern you supply (reference not used) |
| `JaccardSimilarity` | enough *words* overlap with the reference (default ≥ 50%) |
| `JsonValid` | the answer parses as JSON (content ignored) |

Line-by-line of the smallest one, `ContainsReference`:

```python
def score(self, output: str, case: EvalCase) -> Score:
    if case.reference is None:                                # no answer key? can't grade -> fail
        return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
    ok = case.reference.strip().lower() in output.lower()     # substring check, ignoring case
    return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)
```

## 3. `runner.py` — the exam invigilator and the CI gate

```python
System = Callable[[str], str]
```

A "system under test" is just a function from question-string to answer-string. Your RAG app, an
agent, or a hardcoded dict — all evaluable.

`EvalRunner.run()` is a three-step loop per case: **ask** the system → **grade** with every
scorer → **record** a `CaseResult`. It returns an `EvalReport` carrying the threshold.

`gate()` converts the report into a build verdict:

```python
def gate(report: EvalReport, *, raise_on_fail: bool = True) -> bool:
    if not report.passed and raise_on_fail:
        raise GateFailure(report.summary())   # uncaught exception -> exit code 1 -> red CI build
    return report.passed
```

Why raise instead of return? In CI, an exception means a non-zero exit code, which means a **red
build**. A quality regression physically cannot be merged quietly.

## 4. `datasets.py` — the sample data and the planted failure

`SAMPLE_DATASET` has four Q&A cases. The demo system (`_DEMO_ANSWERS`) answers three correctly and
one wrongly **on purpose**:

```python
"What is the capital of France?": "The capital of France is Lyon.",  # wrong on purpose
```

The failing case's id is **`capital`**. It exists so that `evalkit run` shows you exactly what a
caught regression looks like: pass rate 0.75, threshold 0.80, gate FAILED, exit code 1. Fix the
answer to "Paris" (or lower the threshold) and the gate goes green — try it.

## 5. `judge.py` — an LLM grading an LLM

Deterministic scorers can't judge *meaning* ("is this answer faithful to the source?"). So we ask
a second LLM to grade the first, against an explicit rubric. Crucially, judges implement the same
`Scorer` interface — the runner can't tell a judge from a regex.

Three judges live here:

- **`FakeJudge`** — no API, no network, no cost. Passes any non-empty answer (or returns a fixed
  score you choose).
- **`GeminiJudge`** — the real judge for this repo (uses your `GEMINI_API_KEY`,
  model `gemini-2.5-flash`).
- **`AnthropicJudge`** — the same idea via Claude, if you have an `ANTHROPIC_API_KEY` instead.

### Why do the tests use FakeJudge instead of a real judge?

Because tests must be **fast, free, and repeatable**, and a real judge is none of those:

1. **Money** — every real judge call costs API credits; a test suite runs hundreds of times.
2. **Network + secrets** — CI would need an API key and an internet connection; the whole point
   of this repo is that `pytest` runs offline.
3. **Nondeterminism** — a real model can grade the same answer 0.8 today and 0.7 tomorrow;
   a test that randomly fails is worse than no test.

FakeJudge has the *same interface and output shape* as the real judges, so tests prove all the
plumbing (runner → judge → report → gate) without an API. The real judges' own logic (prompt
building, verdict parsing, error wrapping) is tested too — against a **mocked** client, in
`tests/test_judge.py`.

### How GeminiJudge gets a robust verdict

```python
response = genai.Client(api_key=self._api_key).models.generate_content(
    model=self._model,                       # "gemini-2.5-flash"
    contents=self._prompt(output, case),     # rubric + question + answer (+ reference)
    config=types.GenerateContentConfig(
        system_instruction=_JUDGE_SYSTEM,        # the judge's "job description"
        max_output_tokens=self._max_tokens,
        response_mime_type="application/json",   # reply MUST be JSON...
        response_schema=_Verdict,                # ...matching {score, passed, reasoning}
    ),
)
verdict = response.parsed                    # already a validated _Verdict object
```

The last two config lines are the trick: instead of *hoping* the model replies in a parseable
format, we *force* it to. `response.parsed` hands back a validated Python object.

### Judge selection — `make_judge()`

```python
if settings.gemini_api_key:      return GeminiJudge(...)    # 1st choice
if settings.anthropic_api_key:   return AnthropicJudge(...) # 2nd choice
return FakeJudge()                                          # offline fallback
```

Set `GEMINI_API_KEY` in `.env` and `make_judge()` gives you a live Gemini judge; unset everything
and you're safely offline.

## 6. `guardrails.py` — checkpoints on the way in and out

A `Guard` takes text and returns a `GuardResult`: the (possibly cleaned) text, a list of
violations, and a `blocked` flag. Two families:

- **Redactors** clean and let through — `PiiRedactionGuard`.
- **Blockers** refuse — `PromptInjectionGuard`, `LengthGuard`, `BannedContentGuard`.

### The prompt-injection pattern list (extend it here!)

Near the middle of the file:

```python
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    ...
]
```

Each regex is one known attack phrasing. `re.I` = case-insensitive; `\s+` = any amount of
whitespace. **To catch a new attack, append one `re.compile(...)` line to this list** — the guard
loops over the list, so it picks up your pattern automatically. Then prove it works by adding a
test in `tests/test_guardrails.py`.

### The PII patterns

Just above the injection list:

```python
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": ..., "SSN": ..., "CARD": ...,
}
```

The key ("EMAIL") becomes the placeholder in the output — `[REDACTED_EMAIL]`. Add a new
`"LABEL": re.compile(...)` entry to redact a new kind of personal data.

### The pipeline

`GuardrailPipeline` runs input guards *before* the model and output guards *after*. Guards run in
order and each receives the previous guard's cleaned text ("redactions carry forward"). If **any**
guard blocks, the combined result is blocked — and `guard_input()` raises `GuardrailViolation` by
default so hostile input can't be ignored by accident.

## 7. `tracing.py` — spans, or "stopwatches with notebooks"

A `Span` records one step: name, start/end time, notes (`attributes`), tokens, cost, and child
spans. Nesting spans gives you a tree — `evalkit trace` prints one.

### Where spans open and close

All of it happens in `Tracer.span()` — a `@contextmanager`:

```python
@contextmanager
def span(self, name: str, **attributes: Any) -> Iterator[Span]:
    span = Span(name=name, ..., start=perf_counter())   # <-- span OPENS here (start time stamped)
    parent = self._stack[-1] if self._stack else None   # who is the current parent?
    (parent.children if parent else self._roots).append(span)
    self._stack.append(span)                            # we are now "inside" this span
    try:
        yield span                                      # your `with`-block body runs here
    finally:
        span.end = perf_counter()                       # <-- span CLOSES here (end time stamped)
        self._stack.pop()                               # step back out to the parent
```

Everything **before `yield`** runs when the `with` block is entered (open); everything in
**`finally`** runs when it exits (close) — *even if the body raised an exception*, so a crash
still produces a complete, timed trace. The `_stack` is how nesting works: while a span is open it
sits on the stack, and any new span becomes its child.

## 8. The wiring: `cli.py`, `api.py`, `config.py`, `errors.py`

- `cli.py` — three Typer commands. `evalkit run` builds a runner, prints the report, and exits
  with code 1 if the gate fails (that's what CI sees). `evalkit guard "<text>"` and
  `evalkit trace` demo the other two capabilities.
- `api.py` — the same features over HTTP (FastAPI): `POST /guard/input`, `GET /eval/demo`,
  `GET /health`.
- `config.py` — reads `.env` / environment variables into a typed `Settings` object (keys, model
  names, threshold). No secrets in code, ever.
- `errors.py` — the domain exceptions: `GateFailure` (red build), `GuardrailViolation` (blocked
  text), `JudgeError` (judge call failed). All inherit `EvalKitError` so you can catch broadly.

## Try these next (exercises)

1. Run `uv run evalkit run` and watch the `capital` case fail the gate. Make it pass **two
   different ways** (fix the answer in `datasets.py`; lower `--threshold`).
2. Add a new injection pattern to `_INJECTION_PATTERNS` that catches
   `"pretend you have no rules"` — plus a test proving `evalkit guard` now blocks it.
3. Add an `"IP"` entry to `_PII_PATTERNS` that redacts IPv4 addresses.
4. Set `GEMINI_API_KEY` in `.env` and run a real judged eval with
   `EvalRunner([make_judge()], ...)`.
