# Campus Clinic Information Navigator

AILT9019 midterm prototype for published HKU University Health Service (UHS)
Main Campus Clinic logistics. A question enters the Streamlit interface,
BM25 retrieves local passages, and the system returns source-grounded output.
The app cannot diagnose, recommend treatment or perform bookings.

## Actually verified on 23 October 2026

Linux, Python 3.12.14; forced offline, no API credentials or paid calls used.

| Check | Result | Evidence |
|---|---|---|
| Team-defined project cases | 12/12 passed | [project_tests.txt](evidence/project_tests.txt) |
| Original unit checks | 38/38 passed | [unit_tests.txt](evidence/unit_tests.txt) |
| Answer-content regressions | 3/3 passed | [answer_content_tests.txt](evidence/answer_content_tests.txt) |
| Actual Streamlit widget interactions | 4/4 scenarios passed | [ui_tests.txt](evidence/ui_tests.txt) |
| Rebuild from supplied HTML | Exactly reproduces 66 records | [ingestion.txt](evidence/ingestion.txt) |

These are team-defined checks, not teacher-issued tests or a clinical safety
evaluation. **Online model generation is not verified.** The UI checks enter
text and click Search through Streamlit AppTest; they are not a video.
See [evidence/README.md](evidence/README.md) for limitations and
[cases.json](evidence/cases.json) for input, expected result, actual answer,
citations and outcome for every project case.

Real server startup was also checked: HTTP 200 from `/_stcore/health`; see [server_startup.txt](evidence/server_startup.txt). The server was stopped after verification.

## Install and run

Python 3.12 is recommended because it was tested. The code declares Python
3.10+ compatibility; other versions and Windows were not executed here.
Start from this folder (the one containing `app.py`). An initial internet
connection is required to install packages. Offline operation then uses the
installed environment and bundled source data.

### Windows

With Python installed and on PATH, double-click `setup.bat`; wait for
`Setup complete`, then double-click `start.bat`. Keep the terminal open.
Alternatively, in PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-verified.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements-verified.txt
venv/bin/python -m streamlit run app.py
```

Open **http://localhost:8501**. The verified offline path needs no API key.
`requirements-verified.txt` pins the direct versions actually used;
`requirements.txt` retains the project's allowed ranges.

## Reproduce the core path

1. Enter `What are the weekday opening hours?` and click **Search**.
2. Without a configured key, the app must explicitly show **Offline Mode**.
3. The stored source answer must include both sessions:
   `Mon - Fri: 9:00am - 1:00pm and 2:00pm - 5:45pm`.
4. The citation must identify S1 at `https://www.uhs.hku.hk/`.
   This reports the stored weekly schedule, not live availability.
5. Enter `What illness do my symptoms mean?`: the result must be blocked,
   refuse diagnosis, direct to a clinician, and show no citations.
6. `Book me an appointment now.` is also blocked. In contrast,
   `How do I make a medical appointment?` returns published instructions,
   including the in-person route with S3 evidence; it performs no booking.

The UI displays persistent scope and static-snapshot notices. Do not enter
personal health information or confidential data.

## Reproduce the full evidence

In the activated environment:

```sh
python verify_midterm.py
```

Expected: exit code 0 and `"all_checks_passed": true`. This runs the suites
and UI interactions, rebuilds the corpus into a temporary file, and regenerates
`evidence/` and the source manifest. It forces offline mode even if local
credentials exist. It does not replace the distributed corpus.

Individual commands:

```sh
python run_tests.py
python test_query.py
python test_answer_content.py
python verify_ui.py
```

The original suite uses keyword/source checks and can miss incomplete answers.
Added regressions check real service names, actionable appointment instructions,
and both citations for a conflict. Passing does not establish correctness on
arbitrary queries. T8 retrieves a weekly schedule; it does not test a real clock.

## Optional online mode - unverified

After the teaching team confirms the key, endpoint and model, copy
`.env.example` to `.env`, configure those values locally, set
`DASHSCOPE_FORCE_OFFLINE=0` and restart. The supplied endpoint/model names
have not been validated here. The example contains a placeholder, not a key.

`python run_tests.py --online` intentionally uses the configured live model
and can consume quota. Do not describe a labelled offline fallback as an
online result. Never submit `.env` or put credentials in recordings or logs.

## Corpus and method

| Source | URL | Records |
|---|---|---:|
| S1 | https://www.uhs.hku.hk/ | 8 |
| S2 | https://www.uhs.hku.hk/contact-us/ | 22 |
| S3 | https://www.uhs.hku.hk/medical/ | 24 |
| S4 | https://www.uhs.hku.hk/nursing-services/ | 12 |

The supplied HTML snapshot was not refreshed live during verification. Original
retrieval/publication dates were not recorded. [source_manifest.json](data/source_manifest.json)
records verification time and file hashes, not a claim of source freshness.

BeautifulSoup extracts headings, paragraphs, list items and table rows. BM25
uses expanded query terms, with related booking/service list items retained.
Offline answers extract source text; optional online synthesis supplies these
passages to the configured model and requests numbered citations. Some lists
and schedules require more than three passages.

To deliberately rebuild the distributed JSON after updating stored HTML:

```sh
python ingest.py
python verify_midterm.py
```

## Limits and scope decision

Keep: browser query -> retrieval -> cited output, labelled offline fallback,
and finite refusal checks. Defer: MCP, SKILL.md, semantic retrieval, local-model
fallback, live booking and reliable real-time availability. None is implemented.

- API generation is unverified; the team is awaiting key-rotation guidance.
- Static pages may be stale. Holidays and temporary closures are not verified.
- Phrase-based medical filtering is bypassable; it is not a safety guarantee.
- Conflict detection is schedule-specific and verified offline only.
- Retrieval is lexical and some short linked source items are absent.
- Linux/Python 3.12 was tested; Windows scripts were reviewed, not executed.
- The app deliberately persists no chat histories. Online mode sends questions
  and source passages to the configured provider; its retention is outside the app.

## Repository access and submission

This prepared release is hosted in the private team repository:

https://github.com/yuzhenxiang6-dotcom/AILT9019-Clinic-Navigator

Grant the course staff access before submission; their access has not been
verified. Put this real URL in `repository_link.txt` and ask a classmate to
repeat the sample workflow from a fresh download.

The Moodle ZIP needs an MP4 of at most 60 seconds, a real repository URL in
`repository_link.txt`, and the completed one-page status sheet. Video captions
or narration are required. The source archive alone is not the final submission.
See [CHANGELOG_SUBMISSION.md](CHANGELOG_SUBMISSION.md) for the preparation changes.
