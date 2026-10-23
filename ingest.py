"""
ingest.py - Extract paragraphs from saved HTML pages into a retrieval corpus.

This script is a BUILD-TIME tool. It is run once by the developer to produce
`data/passages.json`. The end user (teammate recording the demo) never needs
to run it -- the generated JSON ships pre-built inside the project folder.

Usage
-----
    python ingest.py                 # process every .html in data/raw/
    python ingest.py --raw-dir DIR   # process a different folder
    python ingest.py --out out.json   # choose the output path

Extraction strategy
-------------------
Saved UHS pages are messy, so we pull from several block-level tags rather
than only <p>:

    h1..h6   headings, kept because a heading often carries the section name
             that a paragraph depends on for its meaning
    p        ordinary prose
    li       list items (departments, services, bullet facts)
    td/th    table cells -- IMPORTANT: UHS publishes clinic opening hours in
             <table> markup, not <p>. Patching only <p>/<li> silently drops
             the single most query-relevant content on the site.

Boilerplate (nav bars, footers, cookie banners, scripts) is removed before
extraction, and very short or link-only fragments are dropped afterwards.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError:
    sys.exit(
        "Missing dependency: beautifulsoup4\n"
        "Run:  pip install -r requirements.txt"
    )

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "passages.json"

# Tags whose text is chrome rather than content.
# NOTE: <header> and <footer> are intentionally NOT here. Modern WordPress
# themes use those tags for the main content wrapper, so removing them
# deletes the actual article text. On the real UHS pages that wiped out all
# 32 paragraphs and 13 table cells.
BOILERPLATE_TAGS = ("script", "style", "noscript", "form", "iframe")

# Class tokens that indicate chrome.
#
# These are deliberately SHORT and specific. The first version matched
# substrings like "header" and "footer", which on WordPress themes hits the
# main content wrapper and erases the whole page. The second version added
# "widget", which turned out to name Elementor's CONTENT container (16 real
# paragraphs on the UHS home page), not sidebar chrome.
BOILERPLATE_CLASS_TOKENS = frozenset("""
    nav navbar navigation breadcrumb breadcrumbs
    cookie cookies consent gdpr sidebar
""".split())

# Inline tags we turn into passages, in document order.
# NOTE: table cells are deliberately NOT here. A <td> like "Monday" or
# "08:30 - 17:00" is too short to survive MIN_CHARS on its own and, worse, is
# meaningless in isolation. Tables are handled a whole row at a time instead --
# see extract_tables().
CONTENT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li")

MIN_CHARS = 25           # drop stubs like "Home" or "Read more"
MAX_CHARS = 2000         # split anything longer so BM25 scores stay comparable

# Source id is derived from the filename stem: S1.html -> S1
SOURCE_ID_RE = re.compile(r"(S\d+)", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Collapse all whitespace runs into single spaces and strip."""
    return WS_RE.sub(" ", raw).strip()


def derive_source_id(path: Path, fallback_index: int) -> str:
    """Pull 'S1' out of 'S1.html'; fall back to positional numbering."""
    match = SOURCE_ID_RE.search(path.stem)
    return match.group(1).upper() if match else f"S{fallback_index}"


def strip_boilerplate(soup: BeautifulSoup) -> None:
    """Remove navigation chrome so it cannot pollute the corpus.

    Two hazards, both hit on real WordPress markup:

    1. `class` is not always a list. bs4 returns a plain string when the
       attribute appears once in an unusual form, and returns None for tags
       whose attributes were already destroyed by an earlier decompose().
       Joining blindly raised AttributeError on the first real page.
    2. Decomposing mutates the tree, so we snapshot the tags first rather than
       decomposing while iterating the live result set.
    """
    for tag_name in BOILERPLATE_TAGS:
        for tag in list(soup.find_all(tag_name)):
            tag.decompose()

    # Some sites wrap chrome in a div with a class hint instead of a <nav>.
    #
    # Safety rule: never delete a wrapper that actually contains content. If a
    # tag holds paragraphs, list items or table cells, it is the article, not
    # the furniture -- deleting it is what silently emptied the corpus.
    for tag in list(soup.find_all(class_=True)):
        try:
            raw = tag.get("class")
        except AttributeError:
            continue
        if raw is None:
            continue

        # Normalise: bs4 may hand back a list, a string, or nothing.
        classes = [raw] if isinstance(raw, str) else [str(c) for c in raw]

        # Split each class into hyphen/underscore segments and compare whole
        # segments, so "nav-links" matches "nav" while "navbar-primary" is
        # judged on its own merits rather than by substring.
        segments: set[str] = set()
        for cls in classes:
            for part in re.split(r"[-_\s]+", cls.lower()):
                if part:
                    segments.add(part)

        if not (segments & BOILERPLATE_CLASS_TOKENS):
            continue

        # It looks like chrome, but it holds real content. Keep it.
        if tag.find(["p", "li", "table"]):
            continue

        tag.decompose()


def is_useful(text: str, soup: BeautifulSoup, tag: Tag) -> bool:
    """Reject stubs and link-only fragments that pollute BM25 scoring."""
    if len(text) < MIN_CHARS:
        return False

    # A fragment that is nothing but a URL is not a passage.
    if re.fullmatch(r"[\s\W]*(https?://|www\.)\S+[\s\W]*", text):
        return False

    # Reject if it is mostly a single hyperlink -- navigation, not content.
    links = tag.find_all("a")
    if links:
        link_chars = sum(len(clean_text(a.get_text(" "))) for a in links)
        if link_chars > 0 and link_chars / max(len(text), 1) > 0.8:
            return False

    return True


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Break an oversized block on sentence-ish boundaries."""
    if len(text) <= limit:
        return [text]

    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: list[str] = []
    buffer = ""

    for part in parts:
        if len(buffer) + len(part) + 1 <= limit:
            buffer = f"{buffer} {part}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = part[:limit].strip()

    if buffer:
        chunks.append(buffer)
    return chunks or [text[:limit]]


def extract_heading_context(tags: list[Tag]) -> list[str]:
    """Return a running section label for each tag, based on the last heading.

    UHS paragraphs frequently read as bare fragments ('Mon-Fri 09:00-13:00')
    that are meaningless without the heading above them. We track the most
    recent heading and prefix it so each passage is self-contained.
    """
    context: list[str] = []
    current_heading = ""

    for tag in tags:
        if tag.name and tag.name.startswith("h"):
            current_heading = clean_text(tag.get_text(" "))
        context.append(current_heading)
    return context


def extract_tables(soup: BeautifulSoup) -> list[str]:
    """Flatten each <table> into one passage per <tr>.

    UHS publishes clinic opening hours as a table, where the meaning only
    exists in the row: day | session | time. Reading cells individually loses
    the association, so a row is rendered as "Day: Monday | Session: General |
    Time: 08:30 - 17:00" and kept as a single passage.
    """
    rows: list[str] = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [clean_text(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]

            if not cells:
                continue

            # A single-cell row is often a spanning header; keep it only if
            # it carries enough text to stand alone.
            if len(cells) == 1:
                if len(cells[0]) >= MIN_CHARS:
                    rows.append(cells[0])
                continue

            rows.append(" | ".join(cells))

    return rows


def parse_html(path: Path, source_id: str, url: str | None) -> list[dict[str, Any]]:
    """Parse one HTML file into a list of passage records."""
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [!] Cannot read {path.name}: {exc}", file=sys.stderr)
        return []

    soup = BeautifulSoup(html, "html.parser")
    strip_boilerplate(soup)

    # A <base href> or canonical link gives us a real citation URL.
    if url is None:
        base_tag = soup.find("base", href=True)
        if base_tag:
            url = base_tag["href"]
        else:
            canonical = soup.find("link", rel=lambda v: v and "canonical" in v)
            if canonical and canonical.get("href"):
                url = canonical["href"]

    tags = [t for t in soup.find_all(CONTENT_TAGS) if isinstance(t, Tag)]
    headings = extract_heading_context(tags)

    passages: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(display_heading: str, text: str) -> None:
        """Append a passage unless it duplicates one already stored."""
        fingerprint = display_heading.lower()
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        passages.append(
            {
                "source_id": source_id,
                "heading": display_heading,
                "url": url or f"(local file: {path.name})",
                "text": text,
            }
        )

    for tag, heading in zip(tags, headings):
        text = clean_text(tag.get_text(" "))
        if not is_useful(text, soup, tag):
            continue

        for chunk in split_long(text):
            # Attach the owning section so the passage stands on its own.
            if tag.name and tag.name.startswith("h"):
                add(chunk, chunk)
            elif heading and heading.lower() not in chunk.lower():
                add(f"{heading} | {chunk}", chunk)
            else:
                add(chunk, chunk)

    # Table rows are appended after the prose so heading context is already
    # established. Each row is self-describing, so no heading prefix needed.
    for row in extract_tables(soup):
        if len(row) >= MIN_CHARS:
            add(row, row)

    return passages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build data/passages.json from saved HTML files."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help=f"Folder of .html files (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                        help=f"Output JSON path (default: {DEFAULT_OUT_PATH})")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        sys.exit(
            f"No such folder: {raw_dir}\n"
            f"Save the source web pages as S1.html .. S4.html and put them there."
        )

    html_files = sorted(
        [p for p in raw_dir.glob("*.html")] + [p for p in raw_dir.glob("*.htm")],
        key=lambda p: p.name.lower(),
    )
    if not html_files:
        sys.exit(
            f"No .html files found in {raw_dir}\n"
            "Expected files such as S1.html, S2.html, S3.html, S4.html."
        )

    all_passages: list[dict[str, Any]] = []

    for index, html_path in enumerate(html_files, start=1):
        source_id = derive_source_id(html_path, index)
        print(f"[*] Parsing {html_path.name} -> source_id={source_id}")
        passages = parse_html(html_path, source_id, url=None)
        print(f"    -> {len(passages)} passages")
        all_passages.extend(passages)

    if not all_passages:
        sys.exit(
            "No passages were extracted.\n"
            "The saved pages probably contain no readable text. "
            "Re-save them from the browser with 'Webpage, Complete'."
        )

    # Stable ordering: by source, then document order.
    source_order = {derive_source_id(p, i): i
                    for i, p in enumerate(html_files, start=1)}
    all_passages.sort(key=lambda p: source_order.get(p["source_id"], 99))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(all_passages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_source: dict[str, int] = {}
    for passage in all_passages:
        by_source[passage["source_id"]] = by_source.get(passage["source_id"], 0) + 1

    print(f"\n[OK] Wrote {len(all_passages)} passages to {args.out}")
    for source_id, count in sorted(by_source.items()):
        print(f"     {source_id}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())