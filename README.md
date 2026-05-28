# Job Alert Bot

> Personal job-search assistant. Watches a list of company career pages and tells me when new postings appear. Built by Vaibhav Jaiswal as part of a portfolio of AI-assisted projects.

**Status:** Phase 1 of 4. Local run works. Cron, AI scoring, and email digest come in Phases 2-4.

## What it does (Phase 1)

1. Reads `urls.json` (the only file I edit).
2. Fetches each career page.
3. Extracts links that look like single job postings.
4. Diffs against the previous snapshot for that site.
5. Writes a Markdown digest of new postings into `digest/<date>.md`.

If a site fails (403, 404, network error, anti-bot wall), it is logged and skipped. The other sites keep working. Removing a URL from `urls.json` simply stops fetching it. The first run for a new site is treated as a baseline and reports nothing as new.

## How to add or remove a career site

Open `urls.json`. The shape is:

```json
{
  "sites": [
    {
      "name": "Some Company",
      "url": "https://example.com/careers",
      "enabled": true
    }
  ]
}
```

To add a site: paste a new block. To remove a site: delete its block, or set `"enabled": false` to keep it in the file but skip it. No other file needs to change.

## Run it locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/VaibhavJ97/job-alert-bot.git
cd job-alert-bot
pip install -r requirements.txt
python scripts/fetch.py
```

After the first run, you will see:

- `snapshots/<site>.json` - per-site state, one file per site
- `digest/<YYYY-MM-DD>.md` - today's report

Run it again the next day. New postings (if any) show up in the new digest.

## Repo layout

```
job-alert-bot/
├── urls.json              <- I edit this
├── requirements.txt
├── scripts/
│   └── fetch.py           <- the scraper
├── snapshots/             <- auto-generated per-site state
└── digest/                <- auto-generated dated reports
```

## Why a normal browser User-Agent

Anti-bot WAFs (Cloudflare, Akamai) block User-Agents that identify as bots, even for one fetch per six hours. Since this is a personal tool checking pages I would manually visit anyway, the volume is identical to a human refreshing tabs. The User-Agent in `scripts/fetch.py` is a generic Chrome-on-Linux string.

If a site still blocks the request, that site uses heavier protection (JS challenge, fingerprinting). Options:

- Switch the site to its RSS feed if one exists (Phase 2 will auto-detect)
- Drop the site from `urls.json`
- Use the site's official email-alert feature

I will not add a headless browser to bypass anti-bot. That crosses from "polite scraping" into adversarial.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Local scraper, snapshot diff, Markdown digest | DONE |
| 2 | Auto-detect RSS feeds, GitHub Actions cron, robots.txt check | TODO |
| 3 | AI scoring (0-100 match vs my CV) using Groq Llama 3.1 | TODO |
| 4 | Email digest via Resend.com, tiny web UI to view past digests | TODO |

## Privacy: how my CV and email stay out of this public repo

This repo is public but my personal data never reaches it.

- `cv.md`, `.env`, and `secrets.json` are listed in `.gitignore`. Git refuses to commit them.
- Phase 3 onwards, my CV content lives in a **GitHub Secret** named `CV_TEXT`. Secrets are encrypted at rest, never visible to anyone (including me) after I save them, and only available to the GitHub Actions cron job at runtime.
- Phase 4 onwards, the recipient email and the Resend API key live in GitHub Secrets named `DIGEST_EMAIL_TO` and `RESEND_API_KEY`.
- The CV I store as a Secret is a matching-relevant version: skills, education, experience, languages. It does not contain phone number, address, date of birth, or photo.

The `.env.example` file in the repo shows which environment variables the project expects, without any real values. To run locally, copy it to `.env` (which is gitignored) and fill in your own values.

## How this was built - AI-pair-programming disclosure

This project was built with AI-assisted development workflows. The product idea, the architecture, and every decision are mine; AI accelerated the execution.

What was mine:

- The product idea: a personal job watcher I can configure with a single URL list
- The design constraint: one config file, isolated per-site failures, first-run seeding to avoid spam
- The legal stance: no auto-apply, polite scraping, no anti-bot bypass
- Every line review before commit

What AI accelerated:

- Anthropic Claude as primary pair-programmer for the scraper architecture and the URL-extraction patterns
- GitHub Copilot for inline suggestions

## Cost

Zero. Runs on free tiers across the board:

- GitHub Actions: free tier (way more than the 4 runs/day we need)
- Groq API (Phase 3): free tier
- Resend.com (Phase 4): 100 emails/day free
- No database needed - state lives in the repo as JSON

## Contact

- Portfolio: https://vaibhavj97.vercel.app
- GitHub: https://github.com/VaibhavJ97
