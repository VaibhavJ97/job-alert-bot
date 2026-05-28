"""Job Alert Bot - Phase 1 scraper.

Reads urls.json, fetches each enabled site, extracts links that look
like job postings, compares against the previous snapshot, and writes a
Markdown digest with anything new.

Design principles:
- The user edits only urls.json. Everything else is auto-managed.
- One broken URL never breaks the rest of the run.
- Removing a URL from urls.json simply stops fetching it. The old
  snapshot stays harmlessly in snapshots/ until you delete it.
- A new URL is "seeded" on first run (snapshot saved, nothing reported
  as new) so you don't get spammed by 100 existing postings.

Run locally:
    pip install -r requirements.txt
    python scripts/fetch.py

Outputs:
    snapshots/<site-slug>.json   -> per-site state
    digest/<YYYY-MM-DD>.md       -> today's new postings
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
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

# URL fragments that strongly suggest a single-job posting page
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

# Anything matching these is definitely NOT a single job posting
EXCLUDE_PATTERNS = re.compile(
    r"(\.css|\.js|\.png|\.jpg|\.jpeg|\.gif|\.pdf|\.svg|\.ico|"
    r"\.woff|\.woff2|\.ttf|"
    r"#|mailto:|tel:|javascript:|"
    r"/jobs/?$|/career/?$|/careers/?$|/karriere/?$|/stellen/?$)",
    re.IGNORECASE,
)

# A normal browser User-Agent. Anti-bot WAFs block UAs that identify as
# bots, even at one fetch per 6 hours per site, so a generic browser UA
# is necessary. This is a personal job-search tool checking pages the
# user would manually visit anyway - the volume is the same as a human
# refreshing tabs occasionally.
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

TIMEOUT = 25  # seconds


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def slugify(name: str) -> str:
    """Turn a site name into a safe filename."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s or "site"


def load_urls() -> list[dict]:
    """Load urls.json. Returns enabled sites only."""
    if not URLS_FILE.exists():
        print(f"ERROR: {URLS_FILE} not found.", file=sys.stderr)
        return []
    try:
        with URLS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: urls.json is not valid JSON: {e}", file=sys.stderr)
        return []
    sites = data.get("sites", [])
    enabled = []
    for s in sites:
        if not isinstance(s, dict):
            continue
        if not s.get("url") or not s.get("name"):
            continue
        if s.get("enabled", True):
            enabled.append(s)
    return enabled


def fetch_html(url: str) -> str | None:
    """Fetch a URL. Returns HTML on success, None on any failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        # Some sites return non-HTML. Skip if so.
        ct = r.headers.get("Content-Type", "").lower()
        if "html" not in ct and "xml" not in ct:
            print(f"  skipped: content-type {ct}", file=sys.stderr)
            return None
        return r.text
    except requests.exceptions.RequestException as e:
        print(f"  FETCH FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def extract_job_links(base_url: str, html: str) -> list[dict]:
    """Find <a> tags that look like single-job postings.

    Returns a list of {url, title} dicts, deduplicated by URL.
    """
    soup = BeautifulSoup(html, "lxml")
    jobs: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        full = urljoin(base_url, href)

        # Strip URL fragments (#section) to avoid duplicates
        full = full.split("#", 1)[0].rstrip("/")
        if not full:
            continue

        if EXCLUDE_PATTERNS.search(full):
            continue
        if not JOB_PATTERNS.search(full):
            continue
        if full in seen:
            continue
        seen.add(full)

        title = a.get_text(" ", strip=True) or "(no title)"
        title = re.sub(r"\s+", " ", title)[:250]

        jobs.append({"url": full, "title": title})

    return jobs


def load_snapshot(slug: str) -> tuple[bool, set[str]]:
    """Load previous URLs for this site.

    Returns (existed, set_of_urls). existed=False means first run for
    this site (will be seeded, not reported as new).
    """
    p = SNAPSHOTS_DIR / f"{slug}.json"
    if not p.exists():
        return False, set()
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return True, {j["url"] for j in data.get("jobs", []) if "url" in j}
    except Exception as e:
        print(f"  snapshot read error: {e}", file=sys.stderr)
        return False, set()


def save_snapshot(slug: str, jobs: list[dict]) -> None:
    """Save current job list as the snapshot."""
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    p = SNAPSHOTS_DIR / f"{slug}.json"
    payload = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    sites = load_urls()
    if not sites:
        print("No enabled sites in urls.json.")
        return 0

    today = dt.date.today().isoformat()
    DIGEST_DIR.mkdir(exist_ok=True)

    digest_lines = [
        f"# Job Alert Digest - {today}",
        "",
    ]
    total_new = 0
    total_failed = 0
    total_seeded = 0

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
        had_snapshot, previous_urls = load_snapshot(slug)

        print(f"  found {len(current_jobs)} job-like links")

        if not had_snapshot:
            print(f"  first run - seeding snapshot, not reporting as new")
            total_seeded += 1
            digest_lines.append(f"## {name}")
            digest_lines.append("")
            digest_lines.append(
                f"First run. Seeded {len(current_jobs)} current postings as "
                f"baseline. New postings will be reported from next run on."
            )
            digest_lines.append("")
        else:
            new_jobs = [j for j in current_jobs if j["url"] not in previous_urls]
            print(f"  {len(new_jobs)} new since last run")
            if new_jobs:
                total_new += len(new_jobs)
                digest_lines.append(f"## {name} - {len(new_jobs)} new")
                digest_lines.append("")
                for j in new_jobs:
                    digest_lines.append(f"- [{j['title']}]({j['url']})")
                digest_lines.append("")

        save_snapshot(slug, current_jobs)

    # Summary at the top
    summary = (
        f"**Summary:** {total_new} new posting(s) across "
        f"{len(sites) - total_failed - total_seeded} active sites. "
        f"{total_seeded} site(s) seeded. {total_failed} fetch failure(s)."
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
