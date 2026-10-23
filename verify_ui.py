"""Exercise the actual Streamlit UI in offline mode (no API credentials)."""
import os
from pathlib import Path
os.environ["DASHSCOPE_FORCE_OFFLINE"] = "1"
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
    assert not app.exception, app.exception
    assert app.warning, "Persistent scope notice is missing"
    queries = [
        ("What are the weekday opening hours?", "[Offline Mode]", "Citations"),
        ("What illness do my symptoms mean?", "cannot diagnose", "No citations"),
        ("How do I make a medical appointment?", "[Offline Mode]", "Citations"),
        ("Book me an appointment now.", "cannot diagnose", "No citations"),
    ]
    for question, expected, citation_label in queries:
        app.text_input[0].set_value(question)
        app.button[0].click().run()
        assert not app.exception, app.exception
        visible = "\n".join(str(x.value) for x in app.markdown)
        visible += "\n" + "\n".join(str(x.value) for x in app.info)
        visible += "\n" + "\n".join(str(x.value) for x in app.subheader)
        assert expected in visible, (question, expected, visible)
        assert citation_label in visible, (question, citation_label)
        print("PASS", repr(question))
    print("UI scenarios: 4/4 passed; actual Streamlit inputs/buttons exercised in forced-offline mode.")


if __name__ == "__main__":
    main()
