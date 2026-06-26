# Lessons Learned

Notes to my future self from building this (Milestone 4).

## Technical
- **One interface for cheap and expensive scorers is the key idea.** Making the LLM judge implement
  the same `Scorer` protocol as a regex means the runner, report, and gate don't care where a score
  came from — so you mix them freely and swap a `FakeJudge` in tests for free.
- **The gate is the deliverable, not the scorers.** A pass rate plus a threshold plus a non-zero exit
  is what actually stops regressions. Building the demo so it *fails* (one wrong answer) proved the
  alarm works — a passing demo would have proved nothing.
- **Guardrails are layered; say so.** Pattern matching is a real first layer and is fast/testable, but
  it's not the whole answer. Documenting "pair with a model classifier" is more credible than
  implying regexes solve injection.
- **Carry redactions forward in the pipeline.** Running guards in sequence and threading the
  (possibly redacted) text through means later guards see sanitized input — a small detail that makes
  the pipeline composable.
- **Tracing needs no service to be useful.** An in-memory span tree already answers "where did time
  and tokens go?" and maps 1:1 onto Langfuse/Phoenix/OTel later.

## Process
- **Make the failing case deliberate.** The demo's wrong answer (`capital → Lyon`) is the most
  important line in the dataset: it's what demonstrates the gate doing its job.
- **Offline-first again.** FakeJudge + deterministic scorers mean the whole kit — and 25 tests — runs
  with no keys, no network, no spend.

## If I did it again
- Add a model-based guardrail layer and a judge-calibration harness from the start.
- Retrofit the gate into the M1–M3 repos' CI immediately, not as a follow-up.
- Add span export to a real observability backend.
