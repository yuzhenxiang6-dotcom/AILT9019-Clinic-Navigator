"""
query.py - Retrieval-augmented question answering over a local passage corpus.

Pipeline
--------
    query
      -> safety filter      (block out-of-scope requests)
      -> BM25 retrieval     (top-3 passages from data/passages.json)
      -> LLM synthesis      (Alibaba Cloud Bailian, OpenAI-compatible)
      -> answer + citations

Failure behaviour
-----------------
The demo must never crash. Every network call is wrapped, and any failure
degrades to returning the top retrieved passage verbatim with an
``[Offline Mode]`` prefix.

Critically, the fallback REPORTS WHY it fired. A missing key and a
misconfigured model both surface as "[Offline Mode]", but they are very
different bugs -- the second one fails silently and would leave you
believing the live path works when it never ran. The reason is therefore
printed to stderr and returned to the UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# --- Third-party imports, each guarded with an actionable message ------------
try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Missing dependency: python-dotenv\nRun:  pip install -r requirements.txt")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("Missing dependency: rank-bm25\nRun:  pip install -r requirements.txt")

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_ROOT / "data" / "passages.json"
ENV_PATH = PROJECT_ROOT / ".env"

TOP_K = 3
MAX_CONTEXT_CHARS = 6000

DEFAULT_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"

# --- Safety filter ----------------------------------------------------------
# SCOPE NOTE: this is a demonstration-stage keyword filter, NOT a safety
# guarantee. It is trivially bypassed by rephrasing. It exists so the demo has
# a predictable, inspectable behaviour on out-of-scope questions.
#
# IMPORTANT: booking terms are NOT blocked on their own. "How do I make a
# medical appointment?" (test case T6) is an INFORMATION question the site
# answers in its own words, and blocking it was a real bug -- the first
# version matched the bare substring "appointment" and wrongly refused it.
# Only an imperative REQUEST to perform the action is blocked.
BLOCKED_PATTERNS = (
    # Medical diagnosis / advice
    "diagnose", "diagnosis", "diagnostic",
    "symptom", "symptoms",
    "what's wrong", "whats wrong", "what is wrong",
    "should i take", "should i stop",
    "prescription", "medication", "dose", "dosage",
    # Acting on the user's behalf. Matched as whole phrases so that
    # "how do I book" stays answerable while "I want to book" does not.
    "book me", "book us", "book an appointment for", "book my",
    "reserve me", "reserve an appointment", "schedule me",
    "make a booking for", "appointment for me",
    "i want to book", "i'd like to book", "i would like to book",
    "i need to book", "can you book", "please book",
    # Urgent clinical need
    "emergency", "urgent care", "i need urgent",
)

# Informational phrasings that contain a blocked word but are legitimate
# questions about how the service works. Checked before BLOCKED_PATTERNS.
ALLOWED_PATTERNS = (
    "how do i", "how can i", "how to", "what is the procedure",
    "what are the", "what's the procedure", "what is the process",
    "can i", "am i able", "is it possible", "do i need",
    "where can i", "who can i", "when should i",
)


BLOCKED_RESPONSE = (
    "I can only answer questions about the information published on the "
    "retrieved source pages (hours, locations, contacts, services).\n\n"
    "This query asks for medical guidance or to perform a booking, which is "
    "outside the scope of this system. I cannot diagnose, prescribe, or make "
    "appointments on your behalf.\n\n"
    "Please contact UHS directly (General Enquiries: 3917 2501) or consult a "
    "qualified clinician. Booking guidance is published on the UHS website "
    "under Medical Appointment."
)

# --- Prompt -----------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant for the University Health Service.\n"
    "Answer ONLY using the numbered passages provided. If they do not contain "
    "the answer, reply exactly: NOT FOUND IN SOURCES.\n\n"
    "Rules:\n"
    "1. Never use outside knowledge. Never guess.\n"
    "2. Cite with the passage numbers in square brackets, e.g. [1] [2].\n"
    "3. Put every citation immediately after the sentence it supports.\n"
    "4. Be concise: at most three sentences.\n"
    "5. If passages disagree, say so and cite both."
)

USER_PROMPT_TEMPLATE = """Question: {query}

Passages:
{passages}

Answer using only the passages above. Cite supporting passages as [1], [2].
If the passages do not answer the question, reply: NOT FOUND IN SOURCES."""


def log(message: str) -> None:
    """Diagnostic output. Goes to stderr so it never pollutes the UI."""
    print(message, file=sys.stderr, flush=True)


def load_environment() -> None:
    """Load .env using an absolute path.

    Streamlit does not guarantee the current working directory is the project
    root, so a relative load_dotenv() silently finds nothing and the app
    appears to have no key. Resolving from __file__ removes that failure mode.
    """
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    else:
        log(f"[setup] No .env file at {ENV_PATH} (running without a key)")


load_environment()


# --- Tokenisation -----------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")
# Chinese text has no spaces, so add character bigrams for CJK runs.
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus CJK character bigrams."""
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)

    if CJK_RE.search(lowered):
        cjk_chars = CJK_RE.findall(lowered)
        tokens.extend(
            cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
        )
    return tokens


# --- Query expansion --------------------------------------------------------
# Users ask with words the corpus does not contain. These groups expand a query
# term into the corpus vocabulary it implies.
#
# Tuned against the REAL UHS pages, not invented text. Two rules learned the
# hard way:
#
#  1. Keep groups small. An early version mapped "clinic" to doctor/clinician/
#     physician/medical/staff, which added so much noise that "Where is the
#     clinic?" retrieved the complaints form instead of the address.
#  2. The site writes the schedule as "Mon - Fri", not as one row per day, so
#     expanding "weekday" into five separate day names pulls in unrelated pages
#     that merely mention "opening hours". Expand towards the RANGE instead.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    # The site writes the schedule as "Mon - Fri: 9:00am - 1:00pm", so map
    # onto its wording. Deliberately NOT adding "open"/"closed"/"hours" here:
    # doing so dragged in every page that merely says "Emergency during
    # Opening Hours", which outranked the real schedule.
    ("weekday", "weekdays", "mon", "fri"),
    ("weekend", "weekends", "saturday", "sunday", "closed"),
    ("saturday", "saturdays", "closed"),
    ("sunday", "sundays", "closed"),
    ("today", "mon", "fri", "saturday", "sunday", "closed"),
    ("contact", "telephone", "phone", "email", "enquiry", "call"),
    ("address", "located", "complex", "floor", "meng", "wah"),
    ("location", "located", "complex", "floor", "address", "where"),
    ("book", "booking", "appointment", "web"),
    ("appointment", "booking", "book", "web"),
    ("service", "consultation", "services"),
    ("medical", "clinic", "consultation"),
)

_SYNONYM_MAP: dict[str, tuple[str, ...]] = {}
for _group in SYNONYM_GROUPS:
    for _term in _group:
        _SYNONYM_MAP.setdefault(_term, tuple(
            t for t in _group if t != _term
        ))


def expand_query(query: str) -> str:
    """Append implied terms to the query so BM25 can reach the right passages.

    Expansions are added alongside the original words, never instead of them,
    so a term that genuinely appears in the corpus still dominates.

    Collective terms ("weekday") are NOT expanded when the query already names
    a specific day. Asking about "Friday" while the word "hours" drags in
    monday/tuesday/wednesday/thursday makes those rows compete with the Friday
    one and can push the actual answer out of the retrieved set entirely.
    """
    tokens = tokenize(query)
    named_days = {
        day for day in DAYS
        if re.search(rf"\b{day}s?\b", query, re.IGNORECASE)
    }
    # A collective word the user actually used ("weekday", "today") is a
    # legitimate request for all of its days, so it may expand freely. Only
    # an expansion the user did NOT ask for is suppressed.
    used_collective = {
        term for term in ("weekday", "weekdays", "weekend", "weekends",
                          "today", "now", "currently")
        if re.search(rf"\b{term}\b", query, re.IGNORECASE)
    }

    extra: list[str] = []

    for token in tokens:
        for synonym in _SYNONYM_MAP.get(token, ()):
            if synonym in extra:
                continue
            # Suppress a day introduced by a term the user did not use.
            if (synonym in DAYS and synonym not in named_days
                    and token not in used_collective):
                continue
            extra.append(synonym)

    return f"{query} {' '.join(extra)}" if extra else query


# --- Corpus -----------------------------------------------------------------
@lru_cache(maxsize=1)
def load_corpus() -> list[dict[str, Any]]:
    """Load and cache the passage corpus."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\n"
            "Run 'python ingest.py' once to build it from data/raw/*.html."
        )

    passages = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if not isinstance(passages, list) or not passages:
        raise ValueError(f"Corpus is empty or malformed: {CORPUS_PATH}")
    return passages


@lru_cache(maxsize=1)
def build_index() -> BM25Okapi:
    """Build the BM25 index over heading + text of every passage."""
    passages = load_corpus()
    corpus_tokens = [
        tokenize(f"{p.get('heading', '')} {p.get('text', '')}") for p in passages
    ]
    return BM25Okapi(corpus_tokens)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Return the top_k passages with a relevance score.

    top_k is treated as a floor, not a hard cap: a question spanning a set
    ("the weekday opening hours") needs every Monday-Friday row to be
    answerable, and truncating at three would silently return a partial
    schedule that looks complete.
    """
    passages = load_corpus()
    index = build_index()

    # Expand the query so vocabulary mismatches ("weekday" vs day names) still
    # reach the right passages.
    search_text = expand_query(query)

    query_tokens = tokenize(search_text)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    ranked = sorted(
        (score, i) for i, score in enumerate(scores) if score > 0
    )[::-1]

    if not ranked:
        # No lexical overlap: fall back to document order so the user still
        # sees something rather than an empty screen.
        log("[retrieve] No keyword overlap; returning first passages.")
        return [dict(p, score=0.0) for p in passages[:top_k]]

    # Promote passages that literally contain the distinctive query words.
    #
    # BM25 alone is length-normalised, so a long paragraph that happens to
    # contain one query word can outrank a short passage containing the exact
    # specific asked about. Rare terms carry far more intent than common ones,
    # so we reward exact hits of them.
    #
    # Only tokens the CORPUS actually contains qualify. A query word like
    # "weekday" that appears nowhere in the corpus can match no passage, so
    # counting it would dilute the boost instead of sharpening it.
    corpus_vocab: set[str] = set()
    for passage in passages:
        corpus_vocab.update(
            tokenize(f"{passage.get('heading', '')} {passage.get('text', '')}")
        )

    distinctive = {
        token for token in query_tokens
        if len(token) >= 4 and token not in STOPWORDS and token in corpus_vocab
    }

    def adjusted(index_: int, score: float) -> tuple[float, float, int]:
        text = f"{passages[index_].get('heading', '')} {passages[index_].get('text', '')}".lower()
        hits = sum(1 for token in distinctive if token in text)
        return (score + hits * EXACT_MATCH_BOOST, score, index_)

    # Rank by the boosted score, but keep the raw BM25 value for display so the
    # citation cards report a real relevance figure rather than a fabricated one.
    ranked = sorted((adjusted(i, s) for s, i in ranked), key=lambda x: (-x[0], x[2]))

    limit = top_k

    # A set-valued question needs one row per day it covers, otherwise the
    # answer is a partial schedule presented as if it were complete.
    wanted_days = days_requested(query)
    if len(wanted_days) > top_k:
        limit = len(wanted_days) + top_k

    results = []
    for _boosted, raw_score, index_ in ranked[:limit]:
        passage = dict(passages[index_])
        passage["score"] = float(raw_score)
        results.append(passage)
    # Short list items can score below their introductory heading. Include
    # the actual instructions/services rather than letting a title stand in
    # for an answer. Only copy records from the distributed source corpus.
    related = []
    if re.search(r"\b(how|procedure|process|ways)\b", query, re.IGNORECASE) and re.search(
        r"\b(appointment|booking|book)\b", query, re.IGNORECASE
    ):
        related = [p for p in passages if p.get("source_id") == "S3" and re.search(
            r"come in person|web booking|call\s+\d|please book your appointment",
            p.get("text", ""), re.IGNORECASE)]
    elif re.search(r"\bservices?\b", query, re.IGNORECASE):
        # Preserve the bounded list immediately following S4's service intro.
        service_rows = [p for p in passages if p.get("source_id") == "S4"]
        for i, p in enumerate(service_rows):
            if "provide the following services" in p.get("text", "").lower():
                related = service_rows[i + 1:i + 7]
                break
    present = {(p.get("source_id"), p.get("text")) for p in results}
    for p in related:
        key = (p.get("source_id"), p.get("text"))
        if key not in present:
            results.append(dict(p, score=0.0))
            present.add(key)
    return results


# --- Safety -----------------------------------------------------------------
def is_blocked(query: str) -> bool:
    """Keyword filter for out-of-scope queries (see SCOPE NOTE above).

    An informational phrasing ("how do I make an appointment") is allowed even
    when it mentions a sensitive topic, because the site documents those
    procedures and answering them is exactly the intended use. An imperative
    request ("book me an appointment") is refused.
    """
    lowered = query.lower()

    if any(pattern in lowered for pattern in ALLOWED_PATTERNS):
        # Still block an outright medical-advice request phrased as a question.
        if re.search(r"\b(symptom|diagnos|what('s| is) wrong)\b", lowered):
            return True
        return False

    return any(pattern in lowered for pattern in BLOCKED_PATTERNS)


NOT_FOUND_RESPONSE = (
    "This detail is not stated in the retrieved sources. "
    "The published pages do not mention it, so I cannot confirm it."
)


def _is_unanswerable(query: str, passages: list[dict[str, Any]]) -> bool:
    """Detect a yes/no question whose subject the corpus never addresses.

    "Does the clinic guarantee a five-minute wait?" retrieves the Location
    paragraph purely because it contains the words "five minute". Answering
    from it would be misleading -- the corpus simply does not address waiting
    times. The honest output is to say the information is absent and point at
    the contact route, rather than paraphrase an unrelated sentence.
    """
    lowered = query.lower()

    # Only applies to polar questions.
    if not re.search(r"\b(does|do|is|are|can|will|has|have)\b", lowered):
        return False

    # Find the phrase the question is really asking about.
    match = re.search(
        r"\b(?:guarantee|guaranteed|promise|offer|provide|have|has)\s+"
        r"(?:a|an|the)?\s*([a-z\- ]{3,40}?)\s*\??$",
        lowered,
    )
    subject = match.group(1).strip() if match else ""

    if not subject:
        return False

    # Require the subject to be absent from every retrieved passage, while
    # still requiring SOME overlap with the query (otherwise we would reject
    # merely because retrieval failed).
    #
    # The whole normalised phrase is tested, not individual tokens: "five-minute
    # wait" must be judged as a phrase. Tokenising it yields {five, minute},
    # and "five" occurs in a Location passage ("a five minute walk from the
    # Library"), which would wrongly make the subject look covered.
    phrase = re.sub(r"[^a-z0-9 ]+", " ", subject.lower())
    phrase = re.sub(r"\s+", " ", phrase).strip()

    blob = " ".join(
        f"{p.get('heading','')} {p.get('text','')}".lower() for p in passages
    )
    blob_norm = re.sub(r"[^a-z0-9 ]+", " ", blob)
    blob_norm = re.sub(r"\s+", " ", blob_norm)

    if not phrase:
        return False

    if phrase in blob_norm:
        return False

    # Fall back to a looser test: if every content word of the phrase is
    # present together somewhere, treat it as covered.
    words = [w for w in phrase.split() if w not in STOPWORDS and len(w) >= 3]
    if words and all(w in blob_norm for w in words):
        return False

    return True


DAY_ABBREVIATIONS = {
    "mon": "monday", "tue": "tuesday", "wed": "wednesday",
    "thu": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday",
}


def _row_days(text: str) -> list[str]:
    """Return every day a passage makes a claim about.

    Handles the two shapes a schedule appears in:
      * one row per day        -- "Saturday | Closed | Except public holidays"
      * a day range            -- "Mon - Fri: 9:00am - 1:00pm"

    The range form is what the live UHS site actually uses, so a parser that
    only understood the first shape would find no conflict and no weekday
    row at all.
    """
    days: list[str] = []

    # Range form: "Mon - Fri", "Mon-Fri", "Mon – Fri".
    range_match = re.match(
        r"\s*(mon|tue|wed|thu|fri|sat|sun)\w*\s*[-–—]\s*(mon|tue|wed|thu|fri|sat|sun)\w*",
        text, re.IGNORECASE,
    )
    if range_match:
        start = DAY_ABBREVIATIONS[range_match.group(1).lower()[:3]]
        end = DAY_ABBREVIATIONS[range_match.group(2).lower()[:3]]
        if start in DAYS and end in DAYS:
            span = DAYS[DAYS.index(start): DAYS.index(end) + 1]
            return list(span)

    # Single leading day: "Monday | ..." or "Saturdays are closed".
    single = re.match(
        r"\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text, re.IGNORECASE,
    )
    if single:
        return [single.group(1).lower()]

    # Otherwise, any day named anywhere in the text.
    found: list[str] = []
    for match in re.finditer(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text, re.IGNORECASE,
    ):
        day = match.group(1).lower()
        if day not in found:
            found.append(day)
    return found


def _row_key(text: str) -> str:
    """Return the leading day name of a table row, if it has one."""
    match = re.match(r"\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                     text, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def days_requested(query: str) -> list[str]:
    """Which weekdays does this question ask about?

    Collective wording ("weekday", "weekend", "today") expands to the set of
    days it stands for, so both retrieval and answer assembly agree on what
    needs to be covered.
    """
    if re.search(r"\bweekdays?\b", query, re.IGNORECASE):
        return ["monday", "tuesday", "wednesday", "thursday", "friday"]
    if re.search(r"\bweekends?\b", query, re.IGNORECASE):
        return ["saturday", "sunday"]
    if re.search(r"\btoday\b|\bnow\b|\bcurrently\b", query, re.IGNORECASE):
        return list(DAYS)
    # Allow plurals: "Saturdays" is how people actually ask.
    return [d for d in DAYS if re.search(rf"\b{d}s?\b", query, re.IGNORECASE)]


def build_offline_answer(
    passages: list[dict[str, Any]],
    query: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Compose a readable offline answer from the retrieved passages.

    Some questions are answered by ONE passage ("Is it open on Saturday?").
    Others are answered by a SET ("What are the weekday opening hours?" needs
    every Monday-Friday row, and showing just one row is misleading). When the
    query names multiple days, or asks for a general schedule, we present all
    matching day rows together.
    """
    informative = [p for p in passages if passage_is_informative(p)]

    if re.search(r"\b(how|procedure|process|ways)\b", query, re.IGNORECASE) and re.search(
        r"\b(appointment|booking|book)\b", query, re.IGNORECASE
    ):
        routes = [p for p in passages if p.get("source_id") == "S3" and re.search(
            r"come in person|web booking|call\s+\d|please book your appointment",
            p.get("text", ""), re.IGNORECASE)]
        if routes:
            return ("Published appointment instructions:\n" +
                    "\n".join("- " + p["text"].strip() for p in routes), routes)
    if re.search(r"\bservices?\b", query, re.IGNORECASE):
        service_rows = [p for p in passages if p.get("source_id") == "S4" and re.search(
            r"general health advice|removal of sutures|enquiries and advice on vaccination|"
            r"administration of vaccination|administration of injections|supervision of anti",
            p.get("text", ""), re.IGNORECASE)]
        if service_rows:
            return ("Published nursing services (S4):\n" +
                    "\n".join("- " + p["text"].strip() for p in service_rows), service_rows)

    wanted_days = days_requested(query)

    # Collect schedule lines and the days each one covers. A range line such as
    # "Mon - Fri: 9:00am - 1:00pm" covers five days at once and must not be
    # emitted five times.
    schedule: list[tuple[str, list[str]]] = []
    for passage in informative:
        text = passage.get("text", "")
        days = _row_days(text)
        if not days:
            continue
        looks_like_schedule = bool(
            re.search(r"\d{1,2}[:.]\d{2}", text)
            or re.search(r"\b(open|closed|closed)\b", text, re.IGNORECASE)
        )
        if not looks_like_schedule:
            continue
        if wanted_days and not (set(days) & set(wanted_days)):
            continue
        schedule.append((text.strip(), days))

    if schedule:
        lines = "\n".join(f"- {text}" for text, _ in schedule)
        return (
            "Based on the published schedule:\n" + lines,
            [p for p in informative if p.get("text", "").strip()
             in {t for t, _ in schedule}],
        )

    best = best_substantive_passage(passages, query)
    if best is not None:
        if _is_unanswerable(query, passages):
            # The corpus genuinely does not address this. Say so, and offer
            # the contact route instead of borrowing an unrelated sentence.
            contact = _find_contact(passages)
            suffix = (f" You can confirm by contacting the clinic: {contact}"
                      if contact else
                      " You can confirm by contacting the clinic directly.")
            return NOT_FOUND_RESPONSE + suffix, [best]
        return best.get("text", "").strip(), [best]

    # Nothing informative: show whatever was retrieved, honestly labelled.
    snippets = "\n".join(f"- {(p.get('text') or '').strip()}" for p in passages)
    return f"Closest matching sections:\n{snippets}", passages


def _find_contact(passages: list[dict[str, Any]]) -> str:
    """Find a contact detail in the corpus to offer as a next step."""
    for passage in passages:
        text = passage.get("text", "")
        if re.search(r"telephone|phone|enquiry|email|fax|\d{4}\s?\d{4}|@", text,
                     re.IGNORECASE):
            return text.strip()[:220]
    return ""


# --- LLM --------------------------------------------------------------------
def get_api_key() -> str | None:
    for var in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(var, "").strip()
        if value:
            return value
    return None


MIN_PASSAGE_CHARS = 15  # below this a passage is a stub, not a statement

# Weight added per exact hit of a distinctive (rare) query token. Small enough
# that it reorders near-ties rather than overwhelming the BM25 signal.
EXACT_MATCH_BOOST = 0.35

# Common words carry little intent, so they never earn the exact-match bonus.
STOPWORDS = frozenset("""
    what when where which who whom whose why how does did do is are was were
    the a an and or of to in on at for with from by as be been being this
    that these those there here it its i you we they he she can could would
    should about into over under tell give show find get know please any some
    open opening hours hour time times day days week month year
""".split())

# Weekday order, used when presenting a schedule.
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def passage_is_informative(passage: dict[str, Any]) -> bool:
    """Whether a passage can stand alone as an offline answer.

    Length alone is a poor test: a table row like "Friday | General | 08:30 -
    13:00" is only ~29 characters yet answers the question completely, while a
    bare title can be much longer and answer nothing. So we reject only true
    stubs -- anything shorter than the floor, or a heading with no sentence
    punctuation and no data separator ("|") in it.
    """
    text = (passage.get("text") or "").strip()
    if len(text) < MIN_PASSAGE_CHARS:
        return False

    # A "table row" carries real data via the pipe separator.
    if "|" in text:
        return True

    # Otherwise it must read as a sentence, not a title fragment.
    return text.endswith((".", "!", "?", ":", ")")) or len(text.split()) >= 8


def best_substantive_passage(
    passages: list[dict[str, Any]],
    query: str = "",
) -> dict[str, Any] | None:
    """Pick the passage that best answers the query on its own.

    Page titles and section headings rank well in BM25 because they repeat the
    query terms, but they are useless as an offline answer ("Main Clinic
    Opening Hours" tells the user nothing they did not already ask), so those
    are skipped.

    Among the remaining passages we prefer the one covering the most
    DISTINCTIVE query terms, not simply the top-ranked one. Asking about
    "opening hours on Friday" also retrieves the Appointments paragraph, which
    mentions "opening" but answers a different question; the row naming Friday
    is the actual answer even though it ranks lower.
    """
    candidates = [p for p in passages if passage_is_informative(p)]
    if not candidates:
        return None

    tokens = {
        token for token in tokenize(query)
        if len(token) >= 4 and token not in STOPWORDS
    }
    if not tokens:
        return candidates[0]

    def coverage(passage: dict[str, Any]) -> tuple[int, float]:
        text = f"{passage.get('heading', '')} {passage.get('text', '')}".lower()
        hits = sum(1 for token in tokens if token in text)
        return (hits, float(passage.get("score") or 0.0))

    return max(candidates, key=coverage)


def build_context(passages: list[dict[str, Any]]) -> str:
    """Render the retrieved passages into the numbered prompt context."""
    blocks: list[str] = []
    used = 0

    for i, passage in enumerate(passages, start=1):
        heading = passage.get("heading") or passage.get("source_id", "")
        chunk = f"[{i}] ({passage.get('source_id', '?')}) {heading}\n{passage.get('text', '')}"
        if used + len(chunk) > MAX_CONTEXT_CHARS:
            break
        blocks.append(chunk)
        used += len(chunk)

    return "\n\n".join(blocks)


def call_llm(query: str, passages: list[dict[str, Any]]) -> str:
    """Query the Bailian endpoint. Raises on any failure."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: openai\nRun:  pip install -r requirements.txt"
        ) from exc

    key = get_api_key()
    if not key:
        raise RuntimeError("No API key found in .env (DASHSCOPE_API_KEY).")

    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = float(os.getenv("DASHSCOPE_TIMEOUT", "20"))
    forced = os.getenv("DASHSCOPE_FORCE_OFFLINE", "0").strip().lower() in ("1", "true", "yes")

    if forced:
        raise RuntimeError("Offline mode forced by DASHSCOPE_FORCE_OFFLINE=1.")

    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=1)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                query=query, passages=build_context(passages)
            )},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The model returned an empty response.")
    return content.strip()


# --- Conflict detection -----------------------------------------------------
def detect_conflict(passages: list[dict[str, Any]], query: str) -> str:
    """Flag disagreement between sources about the same fact.

    The naive version of this -- "two passages mention opening hours, so they
    must conflict" -- is wrong, and produced false positives on ordinary
    queries: an Appointments paragraph and a Location paragraph both contain
    the word "open" without making any claim about the current status.

    A real conflict requires two passages that make INCOMPATIBLE claims about
    the SAME day:

      * both give different time ranges for the same weekday, or
      * one says a day is closed while the other says it is open.

    Anything less ambiguous than that is not reported as a conflict.
    """
    if not re.search(r"\b(open|opening|hours|close|closed|today|schedule|weekend|weekday)\b",
                     query, re.IGNORECASE):
        return ""

    # claim[day] = (source_id, is_closed, frozenset_of_times)
    claims: dict[str, list[tuple[str, bool, frozenset[str]]]] = {}

    for passage in passages:
        text = passage.get("text", "")
        days = _row_days(text)
        if not days:
            continue

        times = frozenset(
            re.findall(r"\d{1,2}[:.]\d{2}\s*[-–]\s*\d{1,2}[:.]\d{2}", text)
        )
        if not times and not re.search(r"\b(open|closed)\b", text, re.IGNORECASE):
            continue

        is_closed = bool(re.search(r"\bclosed\b|\bnot open\b", text, re.IGNORECASE))
        for day in days:
            claims.setdefault(day, []).append(
                (str(passage.get("source_id", "?")), is_closed, times)
            )

    for day, entries in claims.items():
        if len(entries) < 2:
            continue

        # Collapse duplicate claims from the SAME source: a page restating
        # itself is not a conflict, and "Mon - Fri" overlapping itself across
        # five days must not register as five disagreements.
        by_source: dict[str, set[tuple[bool, frozenset[str]]]] = {}
        for source_id, is_closed, times in entries:
            by_source.setdefault(source_id, set()).add((is_closed, times))

        if len(by_source) < 2:
            continue

        statuses = {closed for claims_set in by_source.values()
                    for closed, _ in claims_set}
        timesets = {times for claims_set in by_source.values()
                    for _, times in claims_set if times}

        if len(statuses) > 1 or len(timesets) > 1:
            sources = sorted(by_source)
            return (
                f"The sources give conflicting {day} information, so I cannot "
                f"state the current status with confidence (disagreement between "
                f"{' and '.join(sources)}). Please confirm on the "
                "official UHS page or call the clinic directly."
            )

    # Prose conflicts. A source may claim a schedule without using a day-row
    # layout, e.g. "open every day including weekends, never closed on
    # Saturday". That directly contradicts a table saying Saturday is closed,
    # so compare prose claims against the table claims on the days they name.
    prose: list[tuple[str, str, bool]] = []
    for passage in passages:
        text = passage.get("text", "")
        if _row_key(text):
            continue  # already handled above
        if not re.search(r"\b(open|closed|never)\b", text, re.IGNORECASE):
            continue
        if not re.search(r"\b(every day|all days|including weekends?|24 hours|"
                         r"never closed|always open)\b", text, re.IGNORECASE):
            continue
        prose.append((str(passage.get("source_id", "?")), text,
                      bool(re.search(r"\bclosed\b|\bnever\b", text, re.IGNORECASE))))

    for source_id, text, _ in prose:
        says_open_all = bool(re.search(
            r"\b(every day|all days|24 hours)\b", text, re.IGNORECASE))
        if not says_open_all:
            continue
        for day, entries in claims.items():
            if day in ("saturday", "sunday") and any(e[1] for e in entries):
                others = sorted({e[0] for e in entries} | {source_id})
                return (
                    f"The sources conflict on {day}: one states the clinic is "
                    f"closed while another claims otherwise "
                    f"({', '.join(others)}). I cannot state the current status "
                    "with confidence -- please confirm on the official UHS page "
                    "or call the clinic directly."
                )

    return ""


# --- Public API -------------------------------------------------------------
def get_answer(query: str, passages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Answer a query. Never raises; always returns a result dict.

    Parameters
    ----------
    query : str
        The user's question.
    passages : list[dict] | None
        Optional pre-retrieved passages. Supplying them bypasses BM25, which
        the test suite uses to run the contradictory-corpus scenario (T12)
        without mutating the on-disk corpus.

    Returns
    -------
    {
        "answer":    str,           # answer text, possibly prefixed
        "citations": [passage, ...], # up to TOP_K supporting passages
        "mode":      "online"|"offline"|"blocked"|"empty",
        "reason":    str,           # why offline mode fired
    }
    """
    query = (query or "").strip()

    if not query:
        return {
            "answer": "Please type a question.",
            "citations": [],
            "mode": "empty",
            "reason": "Empty query.",
        }

    if is_blocked(query):
        return {
            "answer": BLOCKED_RESPONSE,
            "citations": [],
            "mode": "blocked",
            "reason": "Query matched the out-of-scope filter.",
        }

    if passages is None:
        try:
            passages = retrieve(query)
        except Exception as exc:
            log(f"[ERROR] Retrieval failed: {type(exc).__name__}: {exc}")
            return {
                "answer": (
                    "Could not read the passage corpus. "
                    "Run 'python ingest.py' to rebuild data/passages.json."
                ),
                "citations": [],
                "mode": "empty",
                "reason": f"Retrieval error: {exc}",
            }

    if not passages:
        return {
            "answer": "No matching passages were found in the corpus.",
            "citations": [],
            "mode": "empty",
            "reason": "Retrieval returned nothing.",
        }

    try:
        answer = call_llm(query, passages)
        return {
            "answer": answer,
            "citations": passages,
            "mode": "online",
            "reason": "",
        }
    except Exception as exc:
        # The important distinction: was this expected (no key) or a real
        # misconfiguration (404 / 401 / timeout)? Surface both.
        detail = f"{type(exc).__name__}: {exc}"
        log("=" * 62)
        log("[OFFLINE MODE] The live API call did not succeed.")
        log(f"[OFFLINE MODE] Reason: {detail}")
        if not get_api_key():
            log("[OFFLINE MODE] No key is configured - this is the expected "
                "behaviour when .env is missing.")
        else:
            log("[OFFLINE MODE] A key IS configured, so this is a real "
                "failure. Check DASHSCOPE_MODEL and DASHSCOPE_BASE_URL in .env.")
        log("=" * 62)

        # Detect contradictory sources so the fallback hedges instead of
        # asserting a status the corpus does not agree on (test case T12).
        conflict_note = detect_conflict(passages, query)
        body, cited = build_offline_answer(passages, query)

        if conflict_note:
            fallback = (
                f"[Offline Mode] {conflict_note}\n\n"
                "Published passages that differ:\n" +
                "\n".join("- " + p.get("text", "").strip() for p in passages) + "\n\n" +
                f"(Live model unavailable -- reason: {detail})"
            )
        else:
            fallback = (
                f"[Offline Mode] {body}\n\n"
                f"(Live model unavailable -- reason: {detail})"
            )

        # Cite the passages actually used, but never fewer than the retrieved
        # set when nothing could be narrowed down.
        return {
            "answer": fallback,
            "citations": passages if conflict_note else (cited or passages),
            "mode": "offline",
            "reason": detail,
        }


if __name__ == "__main__":
    # Quick console check:  python query.py "opening hours"
    demo_query = sys.argv[1] if len(sys.argv) > 1 else "opening hours"
    result = get_answer(demo_query)
    print(f"\nQUERY : {demo_query}")
    print(f"MODE  : {result['mode']}")
    if result["reason"]:
        print(f"REASON: {result['reason']}")
    print(f"\n{result['answer']}")
    print(f"\nCitations: {len(result['citations'])}")
    for i, c in enumerate(result["citations"], start=1):
        print(f"  S{i} [{c.get('source_id')}] {c.get('heading', '')[:70]}")
