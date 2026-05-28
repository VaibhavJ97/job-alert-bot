"""Job Alert Bot - Phase 3 scraper with AI scoring.

Reads urls.json, fetches each enabled site, extracts links that look
like job postings, scores each job against the CV using Groq, writes a
Markdown digest sorted by match score.

Design principles:
- The user edits only urls.json. Everything else is auto-managed.
- One broken URL never breaks the rest of the run.
- One failed Groq score never breaks the rest of the scoring.
- Scores are cached in snapshots - a given job URL is scored only once.
- Removing a URL from urls.json simply stops fetching it.

In GitHub Actions, GROQ_API_KEY and CV_TEXT are injected from repo
secrets. If either is missing, the scraper still runs but skips scoring.
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

import requests
from bs4 import BeautifulSoup


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
URLS_FILE = REPO_ROOT / "urls.json"
SNAPSHOTS_DIR = REPO_ROOT / "snapshots"
DIGEST_DIR = REPO_ROOT / "digest"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CV_TEXT = os.environ.get("CV_TEXT", "").strip()

JOB_PATTERNS = re.compile(
    r"(/jobs?/[^/\s]+|/career/[^/\s]+|/careers/[^/\s]+|"
    r"/karriere/[^/\s]+|/stelle/[^/\s]+|/stellen/[^/\s]+|"
    r"/position/[^/\s]+|/vacanc[^/\s]+|/openings/[^/\s]+|"
    r"/offers?/[^/\s]+|/jobdetail/|jobid=|/job_|"
    r"/recruiting/[^/\s]+|smartrecruiters\.com/.+/.+|"
    r"workday\.com/.+/job/|greenhouse\.io/.+/jobs/|"
    r"lever\.co/.+/[a-z0-9-]+)",
    re.IGNORECASE,
)

EXCLUDE_PATTERNS = re.compile(
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
GROQ_DELAY_SECONDS = 1.0  # respectful pause between API calls


# ----------------------------------------------------------------------
# Helpers
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
        if EXCLUDE_PATTERNS.search(full) or not JOB_PATTERNS.search(full):
            continue
        seen.add(full)
        title = a.get_text(" ", strip=True) or "(no title)"
        title = re.sub(r"\s+", " ", title)[:250]
        jobs.append({"url": full, "title": title})
    return jobs


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

Given a candidate's CV and a job title, output a 0-100 match score with brief reasoning.

Scoring rubric:
- 80-100: Strong direct match. Skills, domain, and seniority align well.
- 60-79: Good fit. Most skills align, manageable learning curve.
- 40-59: Moderate. Some skill overlap but role is a stretch.
- 20-39: Weak. Few overlapping skills, major reskilling needed.
- 0-19: Poor or unrelated.

Be strict. Reserve 80+ for genuinely matching roles.
Penalize roles that obviously require senior experience the candidate lacks.
Penalize roles requiring strong German fluency (candidate is A2).

Output ONLY valid JSON in this exact form:
{"score": 73, "reasoning": "Direct geosciences + Python fit, but senior level may be a reach."}
"""


def score_job(title: str) -> dict | None:
    """Score one job title against CV_TEXT via Groq.

    Returns {"score": int 0-100, "reasoning": str} or None on failure.
    """
    if not GROQ_API_KEY or not CV_TEXT:
        return None

    user_prompt = (
        f"CANDIDATE CV:\n{CV_TEXT}\n\n"
        f"JOB TITLE: {title}\n\n"
        "Respond with JSON only."
    )

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            timeout=GROQ_TIMEOUT,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
        score = int(result.get("score", 0))
        reasoning = str(result.get("reasoning", "")).strip()
        return {
            "score": max(0, min(100, score)),
            "reasoning": reasoning[:300],
        }
    except Exception as e:
        print(f"  score fail for '{title[:60]}': {e}", file=sys.stderr)
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
# Main
# ----------------------------------------------------------------------

def main() -> int:
    sites = load_urls()
    if not sites:
        print("No enabled sites in urls.json.")
        return 0

    scoring_enabled = bool(GROQ_API_KEY and CV_TEXT)
    if scoring_enabled:
        print(f"AI scoring: ENABLED (model {GROQ_MODEL})")
    else:
        missing = []
        if not GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not CV_TEXT:
            missing.append("CV_TEXT")
        print(f"AI scoring: DISABLED (missing env: {', '.join(missing)})")

    today = dt.date.today().isoformat()
    DIGEST_DIR.mkdir(exist_ok=True)

    digest_lines = [f"# Job Alert Digest - {today}", ""]
    total_new = 0
    total_scored_now = 0
    total_failed = 0
    total_seeded = 0
    new_jobs_for_top_list: list[dict] = []

    for site in sites:
        name = site["name"]
        url = site["url"]
        slug = slugify(name)
        print(f"\n[{name}] {url}")

        html = fetch_html(url)
        if html is None:
            total_failed += 1
            digest_lines.append(f"## {name}")
            digest_lines.append("")
            digest_lines.append("FETCH FAILED. Will retry next run.")
            digest_lines.append("")
            continue

        current_jobs = extract_job_links(url, html)
        had_snapshot, prev_by_url = load_snapshot(slug)
        print(f"  found {len(current_jobs)} job-like links")

        # Merge: keep cached scores forward
        merged: list[dict] = []
        new_urls: list[str] = []
        for j in current_jobs:
            prev = prev_by_url.get(j["url"])
            if prev and "score" in prev:
                merged.append({
                    "url": j["url"],
                    "title": j["title"],
                    "score": prev["score"],
                    "reasoning": prev.get("reasoning", ""),
                })
            else:
                merged.append(dict(j))
                if j["url"] not in prev_by_url:
                    new_urls.append(j["url"])

        # Score everything missing a score
        unscored = [j for j in merged if "score" not in j]
        if unscored and scoring_enabled:
            print(f"  scoring {len(unscored)} unscored job(s)...")
            for j in unscored:
                result = score_job(j["title"])
                if result is not None:
                    j["score"] = result["score"]
                    j["reasoning"] = result["reasoning"]
                    total_scored_now += 1
                time.sleep(GROQ_DELAY_SECONDS)

        if not had_snapshot:
            total_seeded += 1
            digest_lines.append(f"## {name}")
            digest_lines.append("")
            digest_lines.append(
                f"First run. Seeded {len(merged)} current postings as baseline. "
                "New postings will be flagged as NEW from next run on."
            )
            digest_lines.append("")
            # Even on seed run, show the top-scoring ones so user gets value immediately
            scored = [j for j in merged if "score" in j]
            if scored:
                top_seeded = sorted(scored, key=lambda x: x["score"], reverse=True)[:5]
                digest_lines.append("Top matches found at this site:")
                for j in top_seeded:
                    sc = j["score"]
                    why = j.get("reasoning") or ""
                    digest_lines.append(
                        f"- **{sc}/100** [{score_badge(sc)}] [{j['title']}]({j['url']})"
                    )
                    if why:
                        digest_lines.append(f"    {why}")
                digest_lines.append("")
        else:
            new_jobs = [j for j in merged if j["url"] in new_urls]
            if new_jobs:
                total_new += len(new_jobs)
                new_jobs_for_top_list.extend(
                    [{**j, "site_name": name} for j in new_jobs]
                )
                digest_lines.append(f"## {name} - {len(new_jobs)} new")
                digest_lines.append("")
                new_sorted = sorted(
                    new_jobs,
                    key=lambda x: x.get("score", -1),
                    reverse=True,
                )
                for j in new_sorted:
                    sc = j.get("score")
                    if sc is not None:
                        badge = score_badge(sc)
                        why = j.get("reasoning") or ""
                        digest_lines.append(
                            f"- **{sc}/100** [{badge}] [{j['title']}]({j['url']})"
                        )
                        if why:
                            digest_lines.append(f"    {why}")
                    else:
                        digest_lines.append(f"- [{j['title']}]({j['url']})")
                digest_lines.append("")

        save_snapshot(slug, merged)

    # Top-of-digest "best new matches across all sites"
    if new_jobs_for_top_list:
        new_jobs_for_top_list.sort(
            key=lambda x: x.get("score", -1),
            reverse=True,
        )
        top_n = new_jobs_for_top_list[:10]
        top_block = ["## Top new matches across all sites", ""]
        for j in top_n:
            sc = j.get("score")
            if sc is None:
                continue
            why = j.get("reasoning") or ""
            top_block.append(
                f"- **{sc}/100** [{score_badge(sc)}] **{j['site_name']}**: "
                f"[{j['title']}]({j['url']})"
            )
            if why:
                top_block.append(f"    {why}")
        top_block.append("")
        digest_lines[2:2] = top_block

    summary = (
        f"**Summary:** {total_new} new posting(s) | "
        f"{total_scored_now} scored this run | "
        f"{total_seeded} site(s) seeded | "
        f"{total_failed} fetch failure(s)"
    )
    digest_lines.insert(2, summary)
    digest_lines.insert(3, "")

    digest_path = DIGEST_DIR / f"{today}.md"
    digest_path.write_text("\n".join(digest_lines), encoding="utf-8")

    print()
    print("=" * 60)
    print(summary.replace("**", ""))
    print(f"Digest written to: {digest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
