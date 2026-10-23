"""Reproduce the midterm evidence without credentials or network calls.

Run after installing requirements.txt: python verify_midterm.py
This intentionally verifies the offline workflow; it does not certify the API.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from collections import Counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    os.environ["DASHSCOPE_FORCE_OFFLINE"] = "1"
    env = dict(os.environ, DASHSCOPE_FORCE_OFFLINE="1")
    for key in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
        os.environ.pop(key, None)
    # Suppress machine-specific absolute paths in shareable logs.
    def clean(text: str) -> str:
        return text.replace(str(ROOT), "PROJECT_ROOT")

    executions = []
    def run(command: list[str], filename: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
        )
        (OUT / filename).write_text(
            "Command: python " + " ".join(command[1:]) + "\n"
            "Mode: forced offline; no paid API call\n"
            f"Exit code: {result.returncode}\n\n"
            + clean(result.stdout) + "\nSTDERR\n" + clean(result.stderr),
            encoding="utf-8",
        )
        executions.append({"command": command[1:], "exit_code": result.returncode,
                           "log": filename})
        return result

    suite = run([sys.executable, "run_tests.py"], "project_tests.txt")
    unit = run([sys.executable, "test_query.py"], "unit_tests.txt")
    run([sys.executable, "test_answer_content.py"], "answer_content_tests.txt")
    ui = run([sys.executable, "verify_ui.py"], "ui_tests.txt")
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "passages.json"
        ingestion = run([sys.executable, "ingest.py", "--out", str(rebuilt)],
                        "ingestion.txt")
        saved = json.loads((ROOT / "data/passages.json").read_text(encoding="utf-8"))
        reproduced = (ingestion.returncode == 0 and
                      json.loads(rebuilt.read_text(encoding="utf-8")) == saved)
    (OUT / "ingestion.txt").write_text(
        (OUT / "ingestion.txt").read_text(encoding="utf-8")
        + f"\nStructural equality with distributed corpus: {reproduced}\n",
        encoding="utf-8",
    )

    with contextlib.redirect_stderr(io.StringIO()):
        import query
        import run_tests
        cases = []
        for case in run_tests.CASES:
            passages = (run_tests.inject_conflicting_hours(query.load_corpus())
                        if case.inject_conflict else None)
            actual = query.get_answer(case.query, passages=passages)
            passed, failures = run_tests.evaluate(case, False, passages)
            cases.append({
                "id": case.case_id, "input": case.query,
                "expected": case.note,
                "expected_sources": list(case.expect_sources),
                "checks_expected_terms": list(case.expect_any),
                "actual_mode": actual["mode"], "actual_answer": actual["answer"],
                "actual_citations": [{"source_id": x.get("source_id"),
                                      "url": x.get("url"), "text": x.get("text")}
                                     for x in actual["citations"]],
                "passed": passed, "failures": failures,
            })
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    counts = dict(Counter(x["source_id"] for x in saved))
    hashes = {}
    for path in sorted((ROOT / "data/raw").glob("S*.html")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline; API generation not verified",
        "snapshot_origin": "HTML supplied in AILT9019(1).zip; not refreshed live",
        "original_retrieval_and_publication_dates": "not recorded in supplied data",
        "sources": [{"source_id": sid, "url": next(x["url"] for x in saved if x["source_id"] == sid),
                     "passage_count": count, "html_sha256": hashes.get(sid + ".html")}
                    for sid, count in counts.items()],
        "corpus_sha256": hashlib.sha256((ROOT / "data/passages.json").read_bytes()).hexdigest(),
    }
    (ROOT / "data/source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in
                ("beautifulsoup4", "rank-bm25", "openai", "python-dotenv", "streamlit")}
    (ROOT / "requirements-verified.txt").write_text(
        "# Direct dependencies used for the recorded Linux Python 3.12 run.\n"
        + "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    all_ok = (all(x["exit_code"] == 0 for x in executions) and reproduced
              and all(x["passed"] for x in cases))
    summary = {"verified_at_utc": manifest["verified_at_utc"], "python": platform.python_version(),
               "platform": platform.platform(), "direct_dependencies": versions,
               "mode": manifest["mode"], "passages": len(saved),
               "project_cases_passed": sum(x["passed"] for x in cases),
               "project_cases_total": len(cases), "corpus_rebuild_equal": reproduced,
               "executions": executions, "all_checks_passed": all_ok}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Midterm verification evidence", "", f"Verified at UTC: {manifest['verified_at_utc']}",
              "", "## Results", "", f"- Project-defined cases: {summary['project_cases_passed']}/12 passed.",
              "- Unit checks: see unit_tests.txt (the runner prints its exact total).",
              "- Answer-content regression scenarios: see answer_content_tests.txt (3/3 when successful).",
              "- UI interactions: see ui_tests.txt; real Streamlit AppTest interactions, not a live browser recording.",
              f"- HTML ingestion reproduces the {len(saved)}-passage corpus: {reproduced}.",
              "- API mode: NOT VERIFIED. All verification forced offline; no API quota used.",
              "", "## Inputs and observed results", "",
              "Full expected and actual text, citations and individual outcomes are in cases.json.", "",
              "| Test | Expected | Actual mode | Outcome |", "|---|---|---|---|"]
    for x in cases:
        report.append(f"| {x['id']} | {x['expected']} | {x['actual_mode']} | {'PASS' if x['passed'] else 'FAIL'} |")
    report += ["", "## Limits of the evidence", "",
               "These are team-defined regression checks, not a teacher-issued test suite or a clinical safety evaluation.",
               "The original suite often checks keywords across both answers and retrieved passages; passing does not establish complete answer correctness.",
               "T8 checks retrieval of a published schedule; it does not validate the real clock, holidays or current opening status.",
               "T12 checks offline schedule conflict handling only. Online model generation and its citations remain unverified.",
               "The safety filter is phrase-based and can miss rephrased medical requests. MCP and SKILL.md are not implemented.",
               "The source snapshot can be stale, and the original source dates were not recorded.",
               "Linux and Python 3.12 were tested here; the supplied Windows launch scripts were reviewed but not run on Windows.",
               "", "## Reproduce", "", "```sh", "python -m pip install -r requirements-verified.txt",
               "python verify_midterm.py", "```", "",
               "The command overwrites evidence files with the new run and returns a nonzero exit code on a failed check."]
    (OUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
