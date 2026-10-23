"""Regression checks prompted by observed incomplete offline answers."""
import os
os.environ["DASHSCOPE_FORCE_OFFLINE"] = "1"
import query
import run_tests

def main():
    services = query.get_answer("What medical service does it provide?")
    assert "vaccination" in services["answer"].lower(), services["answer"]
    assert "sutures" in services["answer"].lower(), services["answer"]
    assert any(p["source_id"] == "S4" for p in services["citations"])
    print("PASS service answer contains named services, not just an introductory title")

    booking = query.get_answer("How do I make a medical appointment?")
    assert "come in person" in booking["answer"].lower(), booking["answer"]
    assert booking["mode"] == "offline"
    assert any(p["source_id"] == "S3" for p in booking["citations"])
    print("PASS appointment question returns an actionable published route without booking")

    conflict = query.get_answer("Is the clinic open today?",
        passages=run_tests.inject_conflicting_hours(query.load_corpus()))
    assert "conflict" in conflict["answer"].lower()
    assert {"S1", "S9"}.issubset({p["source_id"] for p in conflict["citations"]})
    print("PASS conflicting-hours fallback cites both sides of the injected disagreement")
    print("Answer-content regressions: 3/3 passed")

if __name__ == "__main__":
    main()
