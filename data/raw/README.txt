# Source HTML goes here

Place the saved source pages in this folder as:

    S1.html   https://www.uhs.hku.hk/
    S2.html   https://www.uhs.hku.hk/contact-us/
    S3.html   https://www.uhs.hku.hk/medical/
    S4.html   https://www.uhs.hku.hk/nursing-services/

`ingest.py` reads every `.html` file here and writes `data/passages.json`.
The source id in the corpus (S1, S2, ...) is taken from the filename, so the
naming convention matters:

    S1  opening hours        (T1, T2, T7, T8)
    S2  contact and address  (T3, T4)
    S3  medical appointments(T6)
    S4  nursing services     (T5)

NOTE: `https://www.uhs.hku.hk/our-services/` was tried first for S4 but the
page body is rendered client-side, so a plain save yields almost no text.
`/nursing-services/` carries a real server-rendered service list. If you swap
sources, re-run `python run_tests.py` to confirm the mapping still holds.

How to save a page: open it in the browser, then
Cmd+S (macOS) / Ctrl+S (Windows) and choose "Webpage, Complete" so that any
CSS and images are saved alongside the HTML.

To rebuild:

    python ingest.py

`data/passages.json` is committed to the repository, so this step is only
needed when the source pages change.