"""Job Alert Bot - Phase 5 three-stage pipeline.

Pipeline:
1. TITLE EXCLUDE FILTER. Drop jobs whose title matches any title-level
   exclude keyword. Cheap, instant.
2. FETCH JOB PAGE + DESCRIPTION FILTER. Fetch each survivor's full job
   posting page. Extract description text. Drop jobs whose description
   contains NO description_include keyword. Medium cost (one HTTP request
   per surviving job).
3. AI SCORING. Send title + truncated description to Groq. Expensive
   (one API call per stage-2 survivor). Truncated to keep token cost
   predictable.

JS-rendered page handling: if description extraction yields nothing, the
job still passes through to AI scoring with title only. Better to score
with limited info than silently drop it.

Seen-once caching: every URL that reaches the title filter is recorded in
the snapshot (scored, unscored, or filtered-out). Any URL already in the
snapshot is skipped entirely on later runs - never re-fetched, re-scored,
or re-emailed. A job is emailed exactly once: with its score if Groq scored
it that run, or "unscored" if the daily quota was already used up.

Required env vars (from GitHub Secrets):
- GROQ_API_KEY, CV_TEXT          -> enable AI scoring
- RESEND_API_KEY, DIGEST_EMAIL_TO -> enable email
- FORCE_EMAIL=true               -> send email even on empty digest
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import markdown as md_lib
import requests
from bs4 import BeautifulSoup


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
URLS_FILE = REPO_ROOT / "urls.json"
KEYWORDS_FILE = REPO_ROOT / "keywords.json"
SNAPSHOTS_DIR = REPO_ROOT / "snapshots"
DIGEST_DIR = REPO_ROOT / "digest"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CV_TEXT = os.environ.get("CV_TEXT", "").strip()

RESEND_API_URL = "https://api.resend.com/emails"
RESEND_FROM = "Job Alert Bot <onboarding@resend.dev>"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
DIGEST_EMAIL_TO = os.environ.get("DIGEST_EMAIL_TO", "").strip()
FORCE_EMAIL = os.environ.get("FORCE_EMAIL", "").lower() in ("true", "1", "yes")

REPO_URL = "https://github.com/VaibhavJ97/job-alert-bot"

JOB_PATTERNS = re.compile(
    r"(/jobs?/[^/\s]+|/career/[^/\s]+|/careers/[^/\s]+|"
    r"/karriere/[^/\s]+|/stelle/[^/\s]+|/stellen/[^/\s]+|"
    r"/stellenanzeige/[^/\s]+|/stellenangebote/[^/\s]+|"
    r"/angebote/[^/\s]*[?&]id=\d|"
    r"/position/[^/\s]+|/vacanc[^/\s]+|/openings/[^/\s]+|"
    r"/offers?/[^/\s]+|/jobdetail/|jobid=|/job_|"
    r"/recruiting/[^/\s]+|smartrecruiters\.com/.+/.+|"
    r"workday\.com/.+/job/|greenhouse\.io/.+/jobs/|"
    r"lever\.co/.+/[a-z0-9-]+)",
    re.IGNORECASE,
)

EXCLUDE_LINK_PATTERNS = re.compile(
    r"(\.css|\.js|\.png|\.jpg|\.jpeg|\.gif|\.pdf|\.svg|\.ico|"
    r"\.woff|\.woff2|\.ttf|"
    r"#|mailto:|tel:|javascript:|"
    r"/jobs/?$|/career/?$|/careers/?$|/karriere/?$|/stellen/?$)",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

TIMEOUT = 25
GROQ_TIMEOUT = 30
GROQ_DELAY_SECONDS = 3.0
GROQ_BACKOFF_SECONDS = 15
GROQ_MAX_ATTEMPTS = 3

# Circuit breaker: a sentinel score_job returns when the Groq daily quota is
# exhausted (429 even after all retries). After this many such results in a
# row, the run stops calling Groq entirely and emails the rest unscored,
# instead of sleeping ~3 min per remaining job.
RATE_LIMITED = "RATE_LIMITED"
QUOTA_GIVEUP_THRESHOLD = 2

DESCRIPTION_MAX_CHARS = 2000          # truncation before sending to AI
DESCRIPTION_FETCH_DELAY = 1.0         # pause between job-page fetches


# ----------------------------------------------------------------------
# Config loaders
# ----------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s or "site"


def load_urls() -> list[dict]:
    if not URLS_FILE.exists():
        print(f"ERROR: {URLS_FILE} not found.", file=sys.stderr)
        return []
    try:
        with URLS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: urls.json is not valid JSON: {e}", file=sys.stderr)
        return []
    enabled = []
    for s in data.get("sites", []):
        if isinstance(s, dict) and s.get("url") and s.get("name") and s.get("enabled", True):
            enabled.append(s)
    return enabled


def load_keywords() -> tuple[list[str], list[str], list[str]]:
    """Return (title_include, title_exclude, description_include)."""
    if not KEYWORDS_FILE.exists():
        return [], [], []
    try:
        with KEYWORDS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"WARN: keywords.json invalid JSON: {e}", file=sys.stderr)
        return [], [], []
    title_incl = [str(k).lower().strip() for k in data.get("include", []) if str(k).strip()]
    title_excl = [str(k).lower().strip() for k in data.get("exclude", []) if str(k).strip()]
    desc_incl = [str(k).lower().strip() for k in data.get("description_include", []) if str(k).strip()]
    return title_incl, title_excl, desc_incl


def title_passes_filter(
    title: str, includes: list[str], excludes: list[str]
) -> tuple[bool, str]:
    t = title.lower()
    for kw in excludes:
        if kw in t:
            return False, f"title excluded by '{kw}'"
    if not includes:
        return True, "no title include filter"
    for kw in includes:
        if kw in t:
            return True, f"title matched '{kw}'"
    return False, "no title include keyword matched"


def description_passes_filter(
    desc: str | None, includes: list[str]
) -> tuple[bool, str, str | None]:
    """Return (passed, reason, matched_keyword).

    Rules:
    - If includes is empty -> always pass (no filter).
    - If desc is None or empty (couldn't fetch) -> pass with reason 'no description'.
      The job will be scored with title-only.
    - Otherwise desc must contain at least one include keyword.
    """
    if not includes:
        return True, "no description filter", None
    if not desc:
        return True, "no description (will score title-only)", None
    d = desc.lower()
    for kw in includes:
        if kw in d:
            return True, f"description matched '{kw}'", kw
    return False, "no description keyword matched", None


# ----------------------------------------------------------------------
# Fetch + parse
# ----------------------------------------------------------------------

def fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "").lower()
        if "html" not in ct and "xml" not in ct:
            print(f"  skipped: content-type {ct}", file=sys.stderr)
            return None
        return r.text
    except requests.exceptions.RequestException as e:
        print(f"  FETCH FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return None


JUNK_TITLE_WORDS = {
    "apply", "apply now", "apply here", "bewerben", "jetzt bewerben",
    "mehr", "more", "read more", "learn more", "details", "view", "view job",
    "view details", "weiterlesen", "ansehen", "zur stelle", "zum job",
    "jobboerse", "jobborse", "open", "oeffnen", "next", "previous",
    "weiter", "zurueck", "ende", "show more", "see more",
}


def _clean_title(text: str) -> str:
    """Normalise anchor/heading text into a usable job title."""
    t = re.sub(r"\s+", " ", text or "").strip()
    # strip leading menu arrows / bullets (e.g. the FZ Juelich "> Science")
    t = re.sub(r"^[\u25b8\u25be\u25b6\u25bc\u2023\u203a\u00bb>\u2013\u2022\s\-]+", "", t)
    # strip search-result counters like "6. Ergebnis:" / "12. Result:"
    t = re.sub(r"^\d+\.\s*(Ergebnis|Result)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _is_junk_title(t: str) -> bool:
    if not t:
        return True
    low = t.lower().strip(" .:-")
    if low in JUNK_TITLE_WORDS:
        return True
    if len(t) < 3:                       # "1", "Se"
        return True
    if not re.search(r"[A-Za-z\u00C0-\u017F]", t):   # no letters at all
        return True
    return False


def _recover_title(a) -> str:
    """When an anchor's own text is junk (e.g. an 'Apply' button), find the real
    job title nearby: a title/aria-label attribute, or a heading / title-class
    element inside the same posting block."""
    for attr in ("aria-label", "title"):
        cand = _clean_title(a.get(attr, ""))
        if not _is_junk_title(cand):
            return cand
    node = a
    for _ in range(4):
        node = node.parent
        if node is None:
            break
        target = (
            node.find(["h1", "h2", "h3", "h4", "h5"])
            or node.find(attrs={"data-qa": re.compile(r"(name|title)", re.I)})
            or node.find(attrs={"class": re.compile(r"(title|name|posting)", re.I)})
        )
        if target is not None:
            cand = _clean_title(target.get_text(" ", strip=True))
            if not _is_junk_title(cand):
                return cand
    return ""


def extract_job_links(base_url: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    jobs: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(base_url, href).split("#", 1)[0].rstrip("/")
        # Treat ".../apply" as the posting itself, so an Apply button and the
        # title link de-duplicate to a single entry pointing at the posting.
        if full.endswith("/apply"):
            full = full[: -len("/apply")]
        if not full or full in seen:
            continue
        if EXCLUDE_LINK_PATTERNS.search(full) or not JOB_PATTERNS.search(full):
            continue
        seen.add(full)
        heading = (a.find(["h1", "h2", "h3", "h4", "h5"])
                   or a.find(attrs={"data-qa": re.compile(r"(name|title)", re.I)}))
        raw = heading.get_text(" ", strip=True) if heading else a.get_text(" ", strip=True)
        title = _clean_title(raw)
        if _is_junk_title(title):
            recovered = _recover_title(a)
            title = recovered or (title if title else "(no title)")
            if _is_junk_title(title):
                title = "(no title)"
        jobs.append({"url": full, "title": title[:250]})
    return jobs


# ----------------------------------------------------------------------
# ATS API handler
#
# Many "JS-rendered" career pages are just front-ends for an ATS that also
# publishes the same jobs as a public JSON feed. When a site URL belongs to
# one of these platforms we read the feed directly: it returns title, link
# AND description in one request, so these sites work without a browser and
# skip the per-job description fetch entirely.
#   Ashby:      api.ashbyhq.com/posting-api/job-board/<org>
#   Greenhouse: boards-api.greenhouse.io/v1/boards/<token>/jobs?content=true
#   Lever:      api.lever.co/v0/postings/<org>?mode=json
# ----------------------------------------------------------------------

ATS_FETCH_FAIL = "ATS_FETCH_FAIL"   # sentinel: recognized ATS, but feed errored


def _ats_platform(url: str) -> tuple[str | None, str | None]:
    """Return (platform, org) if url is a recognized ATS board, else (None, None)."""
    p = urlparse(url)
    host = p.netloc.lower()
    seg = [s for s in p.path.split("/") if s]
    if not seg:
        return None, None
    if "ashbyhq.com" in host:
        return "ashby", seg[0]
    if "greenhouse.io" in host and seg[0] not in ("embed",):
        return "greenhouse", seg[0]
    if "lever.co" in host:
        return "lever", seg[0]
    return None, None


def _html_to_text(html_str: str | None) -> str | None:
    """Turn an HTML (or HTML-entity-escaped) description into plain text."""
    if not html_str:
        return None
    txt = html_lib.unescape(html_str)
    if "&lt;" in txt or "&gt;" in txt or "&amp;" in txt:   # double-escaped
        txt = html_lib.unescape(txt)
    text = BeautifulSoup(txt, "lxml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def fetch_ats_api(url: str):
    """If url is a recognized ATS board, return a list of
    {url, title, description} dicts (description may be None). Returns None if
    the url is not an ATS board, or ATS_FETCH_FAIL if the feed request errored.
    """
    platform, org = _ats_platform(url)
    if not platform:
        return None

    if platform == "ashby":
        api = f"https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=false"
    elif platform == "greenhouse":
        api = f"https://boards-api.greenhouse.io/v1/boards/{org}/jobs?content=true"
    else:  # lever
        api = f"https://api.lever.co/v0/postings/{org}?mode=json"

    try:
        r = requests.get(api, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  ATS API FAIL ({platform}): {type(e).__name__}: {e}",
              file=sys.stderr)
        return ATS_FETCH_FAIL

    jobs: list[dict] = []
    seen: set[str] = set()

    if platform == "ashby":
        raw = data.get("jobs", []) if isinstance(data, dict) else []
        for j in raw:
            if not j.get("isListed", True):
                continue
            u = (j.get("jobUrl") or "").split("#", 1)[0].rstrip("/")
            t = _clean_title(j.get("title") or "")
            if not u or not t or u in seen:
                continue
            seen.add(u)
            desc = j.get("descriptionPlain") or _html_to_text(j.get("descriptionHtml"))
            jobs.append({"url": u, "title": t[:250], "description": desc})
    elif platform == "greenhouse":
        raw = data.get("jobs", []) if isinstance(data, dict) else []
        for j in raw:
            u = (j.get("absolute_url") or "").split("#", 1)[0].rstrip("/")
            t = _clean_title(j.get("title") or "")
            if not u or not t or u in seen:
                continue
            seen.add(u)
            jobs.append({"url": u, "title": t[:250],
                         "description": _html_to_text(j.get("content"))})
    else:  # lever - top-level JSON list
        raw = data if isinstance(data, list) else []
        for j in raw:
            u = (j.get("hostedUrl") or "").split("#", 1)[0].rstrip("/")
            t = _clean_title(j.get("text") or "")
            if not u or not t or u in seen:
                continue
            seen.add(u)
            desc = j.get("descriptionPlain") or _html_to_text(j.get("description"))
            jobs.append({"url": u, "title": t[:250], "description": desc})

    return jobs


def fetch_job_description(url: str) -> str | None:
    """Fetch one job posting page and extract description text.

    Returns the cleaned text, or None if the page couldn't be fetched
    or contained no meaningful content.
    """
    html = fetch_html(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "lxml")

    # Strip non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    # Try main content containers in order of preference
    container = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(
            r"(content|description|job|posting|detail|main)", re.I))
        or soup.find("body")
    )
    if container is None:
        return None

    text = container.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    # If text is suspiciously short, treat as no content (likely JS-rendered)
    if len(text) < 100:
        return None
    return text


def load_snapshot(slug: str) -> tuple[bool, dict[str, dict]]:
    p = SNAPSHOTS_DIR / f"{slug}.json"
    if not p.exists():
        return False, {}
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return True, {j["url"]: j for j in data.get("jobs", []) if "url" in j}
    except Exception as e:
        print(f"  snapshot read error: {e}", file=sys.stderr)
        return False, {}


def save_snapshot(slug: str, jobs: list[dict]) -> None:
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    p = SNAPSHOTS_DIR / f"{slug}.json"
    payload = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# AI scoring
# ----------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """You are a strict job-match evaluator.

Given a candidate's CV and a job posting (title + optional description), output a 0-100 match score with brief reasoning.

Scoring rubric:
- 80-100: Strong direct match. Skills, domain, and seniority align well.
- 60-79: Good fit. Most skills align, manageable learning curve.
- 40-59: Moderate. Some skill overlap but role is a stretch.
- 20-39: Weak. Few overlapping skills, major reskilling needed.
- 0-19: Poor or unrelated.

Be strict. Reserve 80+ for genuinely matching roles.
Penalize roles that obviously require senior experience the candidate lacks.
Penalize roles requiring strong German fluency (candidate is A2).
If description is provided, weight description content over title.

Output ONLY valid JSON in this exact form:
{"score": 73, "reasoning": "Direct geosciences + Python fit per description; junior level matches."}
"""


def score_job(title: str, description: str | None = None) -> dict | None:
    if not GROQ_API_KEY or not CV_TEXT:
        return None

    parts = [
        f"CANDIDATE CV:\n{CV_TEXT}",
        f"\nJOB TITLE: {title}",
    ]
    if description:
        truncated = description[:DESCRIPTION_MAX_CHARS]
        parts.append(f"\nJOB DESCRIPTION (truncated):\n{truncated}")
    else:
        parts.append("\nJOB DESCRIPTION: (not available, score from title only)")
    parts.append("\nRespond with JSON only.")
    user_prompt = "\n".join(parts)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    api_headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
        try:
            r = requests.post(
                GROQ_API_URL,
                headers=api_headers,
                json=payload,
                timeout=GROQ_TIMEOUT,
            )

            if r.status_code == 429:
                wait_s = GROQ_BACKOFF_SECONDS * attempt
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = max(wait_s, int(float(retry_after)))
                    except ValueError:
                        pass
                wait_s = min(wait_s, 60)
                if attempt < GROQ_MAX_ATTEMPTS:
                    print(
                        f"  rate limited (attempt {attempt}/{GROQ_MAX_ATTEMPTS}), "
                        f"waiting {wait_s}s..."
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"  score fail for '{title[:60]}': rate limit exhausted "
                    f"after {GROQ_MAX_ATTEMPTS} attempts"
                )
                return RATE_LIMITED

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            return {
                "score": max(0, min(100, int(result.get("score", 0)))),
                "reasoning": str(result.get("reasoning", "")).strip()[:300],
            }
        except Exception as e:
            if attempt < GROQ_MAX_ATTEMPTS:
                print(
                    f"  score error (attempt {attempt}/{GROQ_MAX_ATTEMPTS}) "
                    f"for '{title[:50]}': {e} - retrying"
                )
                time.sleep(5)
                continue
            print(f"  score fail for '{title[:60]}': {e}", file=sys.stderr)
            return None
    return None


def score_badge(score: int) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 60:
        return "GOOD"
    if score >= 40:
        return "MEDIUM"
    if score >= 20:
        return "WEAK"
    return "POOR"


# ----------------------------------------------------------------------
# Email
# ----------------------------------------------------------------------

EMAIL_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 720px; margin: 0 auto; padding: 24px; color: #24292f;
       line-height: 1.55; background: #ffffff; }
h1 { color: #1a1a1a; border-bottom: 1px solid #d0d7de; padding-bottom: 12px;
     margin-top: 0; font-size: 24px; }
h2 { color: #24292f; margin-top: 32px; font-size: 18px;
     border-bottom: 1px solid #eaeef2; padding-bottom: 6px; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
ul { padding-left: 22px; }
li { margin-bottom: 10px; }
strong { color: #1a1a1a; }
p { margin: 12px 0; }
.footer { color: #57606a; font-size: 0.85em; margin-top: 36px;
          padding-top: 16px; border-top: 1px solid #d0d7de; }
"""


def markdown_to_html_email(md_content: str) -> str:
    body = md_lib.markdown(md_content, extensions=["extra"])
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{EMAIL_CSS}</style>
</head>
<body>
{body}
<div class="footer">
Job Alert Bot - generated automatically.
<a href="{REPO_URL}">View source on GitHub</a>
</div>
</body>
</html>"""


def send_email_digest(subject: str, html_body: str, text_body: str) -> bool:
    if not RESEND_API_KEY or not DIGEST_EMAIL_TO:
        return False
    try:
        r = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": RESEND_FROM,
                "to": [DIGEST_EMAIL_TO],
                "subject": subject,
                "html": html_body,
                "text": text_body,
            },
            timeout=20,
        )
        if r.status_code >= 400:
            print(f"Email send failed: HTTP {r.status_code}: {r.text[:200]}",
                  file=sys.stderr)
            return False
        print(f"Email sent to {DIGEST_EMAIL_TO}")
        return True
    except Exception as e:
        print(f"Email send error: {e}", file=sys.stderr)
        return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    sites = load_urls()
    if not sites:
        print("No enabled sites in urls.json.")
        return 0

    title_incl, title_excl, desc_incl = load_keywords()
    print(
        f"Title filter: {len(title_incl)} include, {len(title_excl)} exclude. "
        f"Description filter: {len(desc_incl)} include."
    )

    scoring_enabled = bool(GROQ_API_KEY and CV_TEXT)
    if scoring_enabled:
        print(f"AI scoring: ENABLED (model {GROQ_MODEL}, "
              f"{GROQ_DELAY_SECONDS}s between calls)")
    else:
        missing = []
        if not GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not CV_TEXT:
            missing.append("CV_TEXT")
        print(f"AI scoring: DISABLED (missing env: {', '.join(missing)})")

    email_enabled = bool(RESEND_API_KEY and DIGEST_EMAIL_TO)
    if email_enabled:
        print(f"Email delivery: ENABLED (to {DIGEST_EMAIL_TO})")
        if FORCE_EMAIL:
            print("  FORCE_EMAIL is set - will send even if no new jobs")
    else:
        missing = []
        if not RESEND_API_KEY:
            missing.append("RESEND_API_KEY")
        if not DIGEST_EMAIL_TO:
            missing.append("DIGEST_EMAIL_TO")
        print(f"Email delivery: DISABLED (missing env: {', '.join(missing)})")

    today = dt.date.today().isoformat()
    DIGEST_DIR.mkdir(exist_ok=True)

    digest_lines = [f"# Job Alert Digest - {today}", ""]
    total_new = 0
    total_scored_now = 0
    quota_exhausted = False          # circuit breaker: stop scoring once tripped
    consecutive_rate_limited = 0
    total_failed = 0
    total_seeded = 0
    total_title_filtered = 0
    total_desc_filtered = 0
    total_desc_fetched = 0
    total_desc_missing = 0
    all_report: list[dict] = []
    failed_sites: list[str] = []

    for site in sites:
        name = site["name"]
        url = site["url"]
        slug = slugify(name)
        print(f"\n[{name}] {url}")

        # Try the ATS JSON feed first (Ashby/Greenhouse/Lever). If this URL
        # isn't an ATS board, fall back to fetching + scraping the HTML page.
        api_jobs = fetch_ats_api(url)
        if api_jobs == ATS_FETCH_FAIL:
            total_failed += 1
            failed_sites.append(name)
            continue
        if api_jobs is not None:
            current_jobs = api_jobs           # each already carries a description
            print(f"  found {len(current_jobs)} jobs via ATS API")
        else:
            html = fetch_html(url)
            if html is None:
                total_failed += 1
                failed_sites.append(name)
                continue
            current_jobs = extract_job_links(url, html)
            print(f"  found {len(current_jobs)} job-like links")

        had_snapshot, prev_by_url = load_snapshot(slug)

        # ---- Stage 1: title filter ----
        title_passing: list[dict] = []
        title_filtered_count = 0
        for j in current_jobs:
            ok, _r = title_passes_filter(j["title"], title_incl, title_excl)
            if ok:
                title_passing.append(j)
            else:
                title_filtered_count += 1
        total_title_filtered += title_filtered_count
        if title_filtered_count:
            print(f"  title filter dropped {title_filtered_count}, "
                  f"{len(title_passing)} survive")

        # Build merged list: keep cached entries, identify new URLs
        merged: list[dict] = []
        new_jobs: list[dict] = []
        for j in title_passing:
            prev = prev_by_url.get(j["url"])
            if prev is not None:
                # Seen on a previous run - already emailed (scored or unscored)
                # or already filtered out. Never fetch/score/email it again.
                merged.append(prev)
            else:
                # Brand-new URL - fetch description, filter, score, email once.
                new_jobs.append(j)

        # ---- Stage 2: description fetch + filter (only for new URLs) ----
        desc_passing: list[dict] = []
        desc_filtered_this_site = 0
        if new_jobs:
            print(f"  fetching descriptions for {len(new_jobs)} new job(s)...")
            for j in new_jobs:
                if "description" in j:
                    # Came from the ATS feed - description already in hand,
                    # no extra request needed.
                    desc = j.get("description")
                else:
                    desc = fetch_job_description(j["url"])
                    time.sleep(DESCRIPTION_FETCH_DELAY)
                    total_desc_fetched += 1
                    if desc is None:
                        total_desc_missing += 1

                ok, _reason, _kw = description_passes_filter(desc, desc_incl)
                if ok:
                    j["description"] = desc  # may be None - that's fine
                    desc_passing.append(j)
                else:
                    desc_filtered_this_site += 1
                    # Remember it so this URL is never re-fetched on later runs.
                    merged.append({"url": j["url"], "title": j["title"],
                                   "filtered": True})
        total_desc_filtered += desc_filtered_this_site
        if desc_filtered_this_site:
            print(f"  description filter dropped {desc_filtered_this_site}, "
                  f"{len(desc_passing)} survive")

        # ---- Stage 3: AI scoring (optional, with quota circuit breaker) ----
        if desc_passing and scoring_enabled and not quota_exhausted:
            print(f"  scoring {len(desc_passing)} job(s)...")
            for j in desc_passing:
                result = score_job(j["title"], j.get("description"))
                if result == RATE_LIMITED:
                    consecutive_rate_limited += 1
                    if consecutive_rate_limited >= QUOTA_GIVEUP_THRESHOLD:
                        quota_exhausted = True
                        print(
                            f"  Groq quota looks exhausted "
                            f"({consecutive_rate_limited} rate-limited in a row); "
                            "skipping scoring for the rest of this run. Remaining "
                            "jobs are kept unscored and scored on a later run."
                        )
                        break
                elif result is not None:
                    j["score"] = result["score"]
                    j["reasoning"] = result["reasoning"]
                    total_scored_now += 1
                    consecutive_rate_limited = 0
                time.sleep(GROQ_DELAY_SECONDS)
        elif quota_exhausted and desc_passing:
            print(f"  skipping scoring for {len(desc_passing)} job(s) "
                  "(Groq quota exhausted earlier this run)")

        # Never persist the description payload in the snapshot.
        for j in desc_passing:
            j.pop("description", None)

        # Add new jobs to merged + mark as new (scored or not).
        new_urls_added: list[str] = []
        for j in desc_passing:
            merged.append(j)
            new_urls_added.append(j["url"])

        # ---- Collect what to report for this site ----
        # desc_passing = the jobs that cleared title + description filters this
        # run (scored or not). On a first run we report all of them (seed); on
        # later runs these are exactly the new postings.
        if not had_snapshot:
            total_seeded += 1
        else:
            total_new += len(desc_passing)
        for j in desc_passing:
            all_report.append({
                "site_name": name,
                "title": j.get("title", "(no title)"),
                "url": j["url"],
                "score": j.get("score"),
            })

        save_snapshot(slug, merged)

    # ------------------------------------------------------------------
    # Build the email: one block per company, jobs sorted best-first, showing
    # only the score - no AI reasoning. Companies with no matches are omitted.
    # ------------------------------------------------------------------
    by_company: dict[str, list[dict]] = {}
    for j in all_report:
        by_company.setdefault(j["site_name"], []).append(j)

    def _job_key(j):
        sc = j.get("score")
        return (0, -sc) if sc is not None else (1, 0)

    def _company_key(item):
        _name, jobs_ = item
        scored = [x["score"] for x in jobs_ if x.get("score") is not None]
        return (0, -max(scored)) if scored else (1, _name.lower())

    for company, jobs_ in sorted(by_company.items(), key=_company_key):
        jobs_.sort(key=_job_key)
        digest_lines.append(f"## {company}")
        digest_lines.append("")
        for i, j in enumerate(jobs_, 1):
            sc = j.get("score")
            rating = f"{sc}/100" if sc is not None else "unscored"
            digest_lines.append(f"{i}. [{j['title']}]({j['url']}): **{rating}**")
        digest_lines.append("")

    if failed_sites:
        digest_lines.append("## Could not fetch (will retry next run)")
        digest_lines.append("")
        digest_lines.append(", ".join(failed_sites))
        digest_lines.append("")

    summary = (
        f"**Summary:** {len(all_report)} jobs | {total_new} new | "
        f"{total_seeded} seeded | {total_scored_now} scored | "
        f"{total_title_filtered} title-filtered | "
        f"{total_desc_filtered} desc-filtered | "
        f"{total_failed} fetch failures"
    )
    digest_lines.insert(2, summary)
    digest_lines.insert(3, "")

    digest_path = DIGEST_DIR / f"{today}.md"
    digest_md = "\n".join(digest_lines)
    digest_path.write_text(digest_md, encoding="utf-8")

    print()
    print("=" * 60)
    print(summary.replace("**", ""))
    print(f"Digest written to: {digest_path}")

    # ---- Email delivery ----
    if email_enabled:
        should_send = FORCE_EMAIL or len(all_report) > 0 or total_seeded > 0
        if should_send:
            n = len(all_report)
            if n > 0:
                subject = f"Job Alerts: {n} match{'es' if n != 1 else ''} - {today}"
            else:
                subject = f"Job Alerts: test - {today}"
            html_body = markdown_to_html_email(digest_md)
            send_email_digest(subject, html_body, digest_md)
        else:
            print("Email skipped: no matching jobs to report")

    return 0


if __name__ == "__main__":
    sys.exit(main())
