# Midterm verification evidence

Verified at UTC: 2026-10-23T13:59:17.817341+00:00

## Results

- Project-defined cases: 12/12 passed.
- Unit checks: see unit_tests.txt (the runner prints its exact total).
- Answer-content regression scenarios: see answer_content_tests.txt (3/3 when successful).
- UI interactions: see ui_tests.txt; real Streamlit AppTest interactions, not a live browser recording.
- HTML ingestion reproduces the 66-passage corpus: True.
- API mode: NOT VERIFIED. All verification forced offline; no API quota used.

## Inputs and observed results

Full expected and actual text, citations and individual outcomes are in cases.json.

| Test | Expected | Actual mode | Outcome |
|---|---|---|---|
| T1 | Weekday opening times, cited to S1. | offline | PASS |
| T2 | Saturday status (open or closed) from S1. | offline | PASS |
| T3 | Clinic address (S2 or S3; both publish it). | offline | PASS |
| T4 | Contact details, cited to S2. | offline | PASS |
| T5 | Service summary, cited to S4. | offline | PASS |
| T6 | Explains how to book; must not claim to perform the booking. | offline | PASS |
| T7 | Paraphrase of T1; synonym handling, cited to S1. | offline | PASS |
| T8 | Resolves 'today' against published weekday hours, cited to S1. | offline | PASS |
| T9 | Refuses diagnosis and refers to medical staff. | blocked | PASS |
| T10 | Refuses to perform the booking and points to official guidance. | blocked | PASS |
| T11 | Answers that no such guarantee is published, gives contact route. | offline | PASS |
| T12 | Conflicting hours injected; must flag uncertainty, not assert. | offline | PASS |

## Limits of the evidence

These are team-defined regression checks, not a teacher-issued test suite or a clinical safety evaluation.
The original suite often checks keywords across both answers and retrieved passages; passing does not establish complete answer correctness.
T8 checks retrieval of a published schedule; it does not validate the real clock, holidays or current opening status.
T12 checks offline schedule conflict handling only. Online model generation and its citations remain unverified.
The safety filter is phrase-based and can miss rephrased medical requests. MCP and SKILL.md are not implemented.
The source snapshot can be stale, and the original source dates were not recorded.
Linux and Python 3.12 were tested here; the supplied Windows launch scripts were reviewed but not run on Windows.

## Reproduce

```sh
python -m pip install -r requirements-verified.txt
python verify_midterm.py
```

The command overwrites evidence files with the new run and returns a nonzero exit code on a failed check.
