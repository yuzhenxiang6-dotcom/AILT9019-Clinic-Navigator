"""
test_query.py - Offline verification of retrieval, safety filter, and fallback.

Run:  python test_query.py
No network and no API key are required: every test uses the offline path.

Covers the two demo scenarios:
  T1  a factual lookup that should retrieve the correct source passage
  T9  an out-of-scope request that should be blocked before any model call
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DASHSCOPE_FORCE_OFFLINE"] = "1"

import query  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def test_corpus_loads() -> None:
    print("\n[corpus]")
    try:
        passages = query.load_corpus()
        check("corpus is non-empty", len(passages) > 0)
        required = {"source_id", "heading", "url", "text"}
        missing = required - set(passages[0].keys())
        check("passages have required keys", not missing, f"missing {missing}")
    except Exception as exc:
        check("corpus is non-empty", False, str(exc))


def test_t1_factual_lookup() -> None:
    """T1: retrieve a specific fact. The schedule must be in the top results."""
    print("\n[T1] factual lookup: opening hours")
    q = "What are the weekday opening hours?"
    results = query.retrieve(q)
    check("returns results", len(results) > 0)

    blob = " ".join(p.get("text", "") for p in results).lower()
    # The live site publishes "Mon - Fri: 9:00am - 1:00pm ...".
    check("schedule is present", "9:00" in blob or "08:30" in blob,
          f"blob={blob[:120]}")
    check("weekday range is present", "mon" in blob and "fri" in blob)

    body, cited = query.build_offline_answer(results, q)
    check("answer mentions times", ":" in body, f"body={body[:80]}")
    check("citations returned", len(cited) > 0)


def test_heading_not_used_as_answer() -> None:
    """Regression: a page title must never become the offline answer."""
    print("\n[regression] heading vs content")
    passages = [
        {"text": "Main Clinic Opening Hours", "score": 9.0},
        {"text": "Friday | General | 08:30 - 13:00", "score": 5.0},
    ]
    best = query.best_substantive_passage(passages)
    check("skips the title", best is not None and "Friday" in best["text"])


def test_t9_safety_filter() -> None:
    """Out-of-scope requests must be blocked before any model call.

    Note the distinction the real corpus forced: asking HOW to book is an
    information question the site answers (test case T6), while commanding the
    system to book is refused (T10). The earlier version blocked on the bare
    word "appointment" and wrongly refused T6.
    """
    print("\n[T9] safety filter")
    blocked_queries = [
        "What are my symptoms?",
        "Can you diagnose this",
        "I want to book an appointment now",
        "Book me an appointment now.",
        "What medication should I take?",
        "I need emergency care",
    ]
    for q in blocked_queries:
        result = query.get_answer(q)
        check(
            f"blocked: {q[:38]!r}",
            result["mode"] == "blocked",
            f"got mode={result['mode']}",
        )
        check(
            f"no citations leaked: {q[:38]!r}",
            result["citations"] == [],
        )

    allowed = [
        "What are the opening hours?",
        "Where is the clinic located?",
        "How do I contact UHS?",
        # Informational booking questions must NOT be blocked.
        "How do I make a medical appointment?",
        "What is the procedure to book a dental appointment?",
        "Can I book online?",
    ]
    for q in allowed:
        result = query.get_answer(q)
        check(
            f"allowed: {q[:38]!r}",
            result["mode"] != "blocked",
            f"got mode={result['mode']}",
        )


def test_empty_query() -> None:
    print("\n[edge] empty input")
    result = query.get_answer("")
    check("empty query is handled", result["mode"] == "empty")


def test_never_raises() -> None:
    """The demo must not crash on any input."""
    print("\n[robustness] get_answer never raises")
    for q in ["", "   ", "?", "a", "opening hours", "!!!", "诊所开放时间"]:
        try:
            result = query.get_answer(q)
            ok = isinstance(result, dict) and "answer" in result and "mode" in result
            check(f"safe: {q[:24]!r}", ok)
        except Exception as exc:
            check(f"safe: {q[:24]!r}", False, f"raised {type(exc).__name__}: {exc}")


def test_offline_fallback_shape() -> None:
    """Without a key, the result must be offline mode with a usable passage."""
    print("\n[fallback] offline shape")
    result = query.get_answer("What are the opening hours on Friday?")
    check("mode is offline (no key here)", result["mode"] == "offline", f"mode={result['mode']}")
    check("answer is marked offline", result["answer"].startswith("[Offline Mode]"))
    check("a reason is reported", bool(result["reason"]))
    check("citations still provided", len(result["citations"]) > 0)


def main() -> int:
    print("=" * 62)
    print("UHS Retrieval QA - offline test suite")
    print("=" * 62)

    test_corpus_loads()
    test_t1_factual_lookup()
    test_heading_not_used_as_answer()
    test_t9_safety_filter()
    test_empty_query()
    test_never_raises()
    test_offline_fallback_shape()

    print("\n" + "=" * 62)
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    if FAILED:
        print("\nFailures:")
        for name, detail in FAILED:
            print(f"  - {name}  {detail}")
    print("=" * 62)

    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())