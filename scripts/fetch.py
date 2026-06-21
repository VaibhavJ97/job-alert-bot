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

Score caching: a URL with an existing score in the snapshot is never
re-fetched, re-filtered, or re-scored. Only NEW URLs trigger the full
pipeline. Editing keywords.json triggers re-evaluation of currently-
filtered jobs but not cached-scored ones.

Required env vars (from GitHub Secrets):
- GROQ_API_KEY, CV_TEXT          -> enable AI scoring
- RESEND_API_KEY, DIGEST_EMAIL_TO -> enable email
- FORCE_EMAIL=true               -> send email even on empty digest
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

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
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

TIMEOUT = 25
GROQ_TIMEOUT = 30
GROQ_DELAY_SECONDS = 3.0
GROQ_BACKOFF_SECONDS = 15
GROQ_MAX_ATTEMPTS = 3

DESCRIPTION_MAX_CHARS = 2000          # truncation before sending to AI
DESCRIPTION_FETCH_DELAY = 1.0         # pause between job-page fetches

LISTING_PAGE_DELAY = 1.0              # pause between listing-page fetches
MAX_LISTING_PAGES = 10               # safety cap on pagination depth per site


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


def extract_job_links(base_url: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    jobs: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(base_url, href).split("#", 1)[0].rstrip("/")
        if not full or full in seen:
            continue
        if EXCLUDE_LINK_PATTERNS.search(full) or not JOB_PATTERNS.search(full):
            continue
        seen.add(full)
        title = a.get_text(" ", strip=True) or "(no title)"
        title = re.sub(r"\s+", " ", title)[:250]
        jobs.append({"url": full, "title": title})
    return jobs


NEXT_PAGE_TEXT = re.compile(
    r"^\s*(next|next page|weiter|n\u00e4chste|naechste|n\u00e4chste seite|"
    r"more|mehr|older|\u00e4lter|aelter|\u203a|\u00bb|>|>>|\u2192)\s*$",
    re.IGNORECASE,
)


def find_next_page(base_url: str, soup: "BeautifulSoup", visited: set[str]) -> str | None:
    """Best-effort discovery of a 'next page' link in server-rendered pagination.

    Checks rel=next, then an aria-label hinting 'next', then any anchor whose
    visible text IS a next indicator. Returns an absolute URL or None.
    JS-paginated sites expose no such link, so we simply stop at page 1.
    """
    tag = soup.find(attrs={"rel": "next"})
    if tag and tag.get("href"):
        cand = urljoin(base_url, tag["href"]).split("#", 1)[0].rstrip("/")
        if cand and cand not in visited:
            return cand

    a = soup.find(
        "a",
        attrs={"aria-label": re.compile(r"(next|weiter|n\u00e4chste|naechste)", re.I)},
        href=True,
    )
    if a and a.get("href"):
        cand = urljoin(base_url, a["href"]).split("#", 1)[0].rstrip("/")
        if cand and cand not in visited:
            return cand

    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True)
        if txt and NEXT_PAGE_TEXT.match(txt):
            cand = urljoin(base_url, a["href"]).split("#", 1)[0].rstrip("/")
            if cand and cand not in visited:
                return cand
    return None


def crawl_listing(start_url: str) -> tuple[list[dict], bool]:
    """Fetch a career listing and follow pagination across ALL pages.

    Returns (jobs, fetch_failed). fetch_failed is True only when the very
    first page cannot be fetched. Job links are de-duplicated across pages.
    Pagination is capped at MAX_LISTING_PAGES to avoid runaway loops.
    """
    jobs_by_url: dict[str, dict] = {}
    visited: set[str] = set()
    next_url: str | None = start_url
    pages = 0
    first_ok = False

    while next_url and pages < MAX_LISTING_PAGES:
        nu = next_url.split("#", 1)[0].rstrip("/")
        if nu in visited:
            break
        visited.add(nu)
        html = fetch_html(nu)
        pages += 1
        if html is None:
            break
        first_ok = True
        for j in extract_job_links(nu, html):
            jobs_by_url.setdefault(j["url"], j)
        soup = BeautifulSoup(html, "lxml")
        nxt = find_next_page(nu, soup, visited)
        if nxt:
            print(f"  page {pages}: {len(jobs_by_url)} links so far, following next page")
            time.sleep(LISTING_PAGE_DELAY)
            next_url = nxt
        else:
            next_url = None

    return list(jobs_by_url.values()), (not first_ok)


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
                return None

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
    total_pending = 0
    total_failed = 0
    total_seeded = 0
    total_title_filtered = 0
    total_desc_filtered = 0
    total_desc_fetched = 0
    total_desc_missing = 0

    # Every job we will report in THIS run, across all sites. No caps -
    # the full list goes into the digest/email, scored first then unscored.
    all_report_jobs: list[dict] = []
    site_status: list[str] = []

    for site in sites:
        name = site["name"]
        url = site["url"]
        slug = slugify(name)
        print(f"\n[{name}] {url}")

        # ---- Stage 0: crawl ALL listing pages for this site ----
        current_jobs, fetch_failed = crawl_listing(url)
        if fetch_failed:
            total_failed += 1
            site_status.append(f"- **{name}**: fetch failed (will retry next run)")
            continue

        had_snapshot, prev_by_url = load_snapshot(slug)
        print(f"  found {len(current_jobs)} job-like links across all pages")

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

        # Keep already-scored URLs cached (don't re-report); collect new URLs.
        merged: list[dict] = []
        new_jobs: list[dict] = []
        for j in title_passing:
            prev = prev_by_url.get(j["url"])
            if prev and "score" in prev:
                merged.append({
                    "url": j["url"],
                    "title": j["title"],
                    "score": prev["score"],
                    "reasoning": prev.get("reasoning", ""),
                })
            else:
                new_jobs.append(j)

        # ---- Stage 2: description fetch + keyword filter (new URLs only) ----
        desc_passing: list[dict] = []
        desc_filtered_this_site = 0
        if new_jobs:
            print(f"  fetching descriptions for {len(new_jobs)} new job(s)...")
            for j in new_jobs:
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
        total_desc_filtered += desc_filtered_this_site
        if desc_filtered_this_site:
            print(f"  description filter dropped {desc_filtered_this_site}, "
                  f"{len(desc_passing)} survive")

        # desc_passing is now the EMAIL-READY set for this site: everything
        # that cleared the title + description filters. Scoring below is
        # best-effort enrichment only - it never adds or removes a job.

        # ---- Stage 3: AI scoring (optional, best-effort) ----
        scored_now_site = 0
        if desc_passing and scoring_enabled:
            print(f"  scoring {len(desc_passing)} job(s)...")
            for j in desc_passing:
                result = score_job(j["title"], j.get("description"))
                if result is not None:
                    j["score"] = result["score"]
                    j["reasoning"] = result["reasoning"]
                    total_scored_now += 1
                    scored_now_site += 1
                j.pop("description", None)  # never persisted in snapshot
                time.sleep(GROQ_DELAY_SECONDS)
        else:
            for j in desc_passing:
                j.pop("description", None)

        # All filter-passing new jobs join merged (scored or not). Unscored
        # ones are saved WITHOUT a score, so the next run re-detects them as
        # new and retries scoring - no fake placeholder score is ever cached.
        for j in desc_passing:
            merged.append(j)

        pending_site = sum(1 for j in desc_passing if "score" not in j)
        total_pending += pending_site

        # ---- Decide what to report in the email for this site ----
        if not had_snapshot:
            total_seeded += 1
            report = list(desc_passing)
            site_status.append(
                f"- **{name}**: seeded {len(report)} matches "
                f"({scored_now_site} scored, {pending_site} pending), "
                f"{title_filtered_count} title-filtered, "
                f"{desc_filtered_this_site} desc-filtered"
            )
        else:
            report = list(desc_passing)
            total_new += len(report)
            site_status.append(
                f"- **{name}**: {len(report)} new "
                f"({scored_now_site} scored, {pending_site} pending)"
            )

        for j in report:
            all_report_jobs.append({**j, "site_name": name})

        save_snapshot(slug, merged)

    # ------------------------------------------------------------------
    # ONE complete, uncapped list: scored first (best -> worst), then every
    # unscored job. Nothing is truncated to a top-N.
    # ------------------------------------------------------------------
    scored_jobs = [j for j in all_report_jobs if j.get("score") is not None]
    unscored_jobs = [j for j in all_report_jobs if j.get("score") is None]
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)

    if scored_jobs or unscored_jobs:
        digest_lines.append(f"## All matches ({len(all_report_jobs)}) - best first")
        digest_lines.append("")
        for j in scored_jobs:
            sc = j["score"]
            why = j.get("reasoning") or ""
            digest_lines.append(
                f"- **{sc}/100** [{score_badge(sc)}] **{j['site_name']}**: "
                f"[{j['title']}]({j['url']})"
            )
            if why:
                digest_lines.append(f"    {why}")
        for j in unscored_jobs:
            digest_lines.append(
                f"- **[unscored]** **{j['site_name']}**: "
                f"[{j['title']}]({j['url']}) "
                "- not yet scored; scored on a later run"
            )
        digest_lines.append("")

    # Per-site status footer (counts + fetch failures)
    if site_status:
        digest_lines.append("## Sites")
        digest_lines.append("")
        digest_lines.extend(site_status)
        digest_lines.append("")

    summary = (
        f"**Summary:** {len(all_report_jobs)} jobs in this email | "
        f"{total_new} new | {total_seeded} seeded | "
        f"{total_scored_now} scored | "
        f"{total_pending} pending (filter-matched, unscored) | "
        f"{total_title_filtered} title-filtered | "
        f"{total_desc_filtered} desc-filtered | "
        f"{total_desc_fetched} desc fetches ({total_desc_missing} unavailable) | "
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
        have_jobs = len(all_report_jobs) > 0
        should_send = FORCE_EMAIL or have_jobs
        if should_send:
            n = len(all_report_jobs)
            if n > 0:
                plural = "es" if n != 1 else ""
                subject = f"Job Alerts: {n} match{plural} - {today}"
            else:
                subject = f"Job Alerts: test - {today}"
            html_body = markdown_to_html_email(digest_md)
            send_email_digest(subject, html_body, digest_md)
        else:
            print("Email skipped: no matching jobs to report")

    return 0


if __name__ == "__main__":
    sys.exit(main())
