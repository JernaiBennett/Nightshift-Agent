# Nightshift Job Intelligence Agent

A CLI agent that reads your resume, analyzes job postings with Claude AI, identifies skill gaps, recommends courses to close them, and tracks your applications — all from the terminal.

## Features

- **Resume parsing** — Point it at your PDF or text resume; Claude extracts your skills, experience, education, and certifications automatically
- **URL job extraction** — Pass a job posting URL and Claude scrapes and parses the title, company, and full description for you
- **Skill gap analysis** — Claude compares the job requirements against your profile and scores how well you match (0–100)
- **Course recommendations** — For each gap, Claude suggests specific courses with provider, estimated time, and priority (immediate / short-term / long-term)
- **Resume keyword hints** — Exact keywords from the job description you should add to your resume to pass ATS screening
- **Application tracking** — Track every job through its full lifecycle: saved → applied → interviewing → offer / rejected / withdrawn
- **Prompt caching** — Your resume and system instructions are cached across analyses, keeping API costs low

## Setup

```bash
# 1. Clone the repo
gh repo clone JernaiBennett/Nightshift-Agent
cd Nightshift-Agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key (get one at console.anthropic.com)
export ANTHROPIC_API_KEY=sk-ant-...       # Mac/Linux
$env:ANTHROPIC_API_KEY="sk-ant-..."      # Windows PowerShell
```

## Commands

### Load your resume
```bash
python agent.py setup-resume /path/to/resume.pdf
python agent.py setup-resume /path/to/resume.txt
```
Claude reads your resume file and saves a structured profile to `~/.nightshift/resume.json`. Supports `.pdf`, `.txt`, and `.md`.

---

### Add a job

**From a URL (auto-extracts everything):**
```bash
python agent.py add-job https://jobs.lever.co/company/job-id
```

**Manually (paste the description):**
```bash
python agent.py add-job
```
Prompts for company, title, and URL, then lets you paste the job description. Press **Enter on a blank line** to finish.

---

### Analyze a job
```bash
python agent.py analyze 1
python agent.py analyze 1 --force   # re-run even if already analyzed
```
Runs Claude gap analysis on the saved job. Shows:
- Match score and summary
- Your strengths relative to the role
- Skill gaps ranked by importance (critical / important / nice-to-have)
- Course recommendations with provider and estimated time
- Keywords to add to your resume
- Application advice

---

### List all jobs
```bash
python agent.py list
python agent.py list --status applied   # filter by status
```
Available statuses: `saved`, `applied`, `interviewing`, `offer`, `rejected`, `withdrawn`

---

### Update application status
```bash
python agent.py update-status 1 applied
python agent.py update-status 1 interviewing --notes "Phone screen scheduled for Friday"
```

---

### View a job
```bash
python agent.py view 1
```
Shows full details and the stored analysis for a job.

## Data Storage

All data is stored locally on your machine:

| File | Contents |
|------|----------|
| `~/.nightshift/resume.json` | Your parsed skills profile |
| `~/.nightshift/jobs.db` | SQLite database of all tracked jobs |

## Requirements

- Python 3.9+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
