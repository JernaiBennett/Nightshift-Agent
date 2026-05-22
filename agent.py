#!/usr/bin/env python3
"""Job Intelligence Agent — resume gap analysis, course recommendations, application tracking."""

import base64
import html as html_lib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

import anthropic
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(help="Job Intelligence Agent: gap analysis, course suggestions, application tracking.")
console = Console()

MODEL = "claude-sonnet-4-6"
DATA_DIR = Path.home() / ".nightshift"
RESUME_FILE = DATA_DIR / "resume.json"
DB_FILE = DATA_DIR / "jobs.db"

RESUME_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "current_title": {"type": "string"},
        "years_experience": {"type": "string"},
        "summary": {"type": "string"},
        "technical_skills": {"type": "array", "items": {"type": "string"}},
        "languages": {"type": "array", "items": {"type": "string"}},
        "frameworks": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "cloud_platforms": {"type": "array", "items": {"type": "string"}},
        "soft_skills": {"type": "array", "items": {"type": "string"}},
        "education": {"type": "array", "items": {"type": "string"}},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "notable_projects": {"type": "array", "items": {"type": "string"}},
        "preferred_roles": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "name", "current_title", "years_experience", "summary",
        "technical_skills", "languages", "frameworks", "tools",
        "cloud_platforms", "soft_skills", "education", "certifications",
        "notable_projects", "preferred_roles",
    ],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTIONS = """\
You are an expert career intelligence analyst and skills advisor. Your role is to:

1. Deeply analyze job descriptions to extract required and preferred skills, technologies, experience levels, and competencies.
2. Compare job requirements against the candidate's profile to identify genuine skill gaps.
3. Provide realistic match scoring based on how well the candidate's background aligns.
4. Recommend specific, actionable learning resources (courses, certifications, projects) to close skill gaps.
5. Give honest, constructive career advice tailored to the specific role.

Guidelines:
- Be specific and practical, not generic.
- Prioritize skill gaps by their importance to getting the job.
- Recommend real courses from providers like Coursera, Udemy, edX, LinkedIn Learning, Pluralsight, A Cloud Guru, etc.
- Estimate realistic time-to-competency for each recommendation.
- Identify exact keywords from the job description that should appear in the resume.
- Match score should reflect a realistic probability of passing initial screening.

Always return valid JSON matching the provided schema exactly.
"""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "match_score": {
            "type": "integer",
            "description": "0-100 score reflecting how well the candidate matches the job requirements"
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentence honest assessment of the fit"
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific skills and experiences the candidate already has that match"
        },
        "skill_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "importance": {
                        "type": "string",
                        "enum": ["critical", "important", "nice_to_have"]
                    },
                    "your_level": {"type": "string"},
                    "required_level": {"type": "string"}
                },
                "required": ["skill", "importance", "your_level", "required_level"],
                "additionalProperties": False
            }
        },
        "learning_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "course_name": {"type": "string"},
                    "provider": {"type": "string"},
                    "url_hint": {"type": "string"},
                    "estimated_time": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["immediate", "short_term", "long_term"]
                    }
                },
                "required": ["skill", "course_name", "provider", "estimated_time", "priority"],
                "additionalProperties": False
            }
        },
        "resume_keywords_to_add": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact keywords from the job description to incorporate into the resume"
        },
        "application_advice": {
            "type": "string",
            "description": "Specific advice on whether/how to apply and how to position the application"
        }
    },
    "required": [
        "match_score", "summary", "strengths", "skill_gaps",
        "learning_recommendations", "resume_keywords_to_add", "application_advice"
    ],
    "additionalProperties": False
}

JOB_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["company", "title", "description"],
    "additionalProperties": False,
}

STATUS_COLORS = {
    "saved": "cyan",
    "applied": "blue",
    "interviewing": "yellow",
    "offer": "green",
    "rejected": "red",
    "withdrawn": "dim",
}

VALID_STATUSES = list(STATUS_COLORS.keys())


def get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'saved',
            match_score INTEGER,
            analysis TEXT,
            notes TEXT,
            applied_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html_lib.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def fetch_job_from_url(url: str) -> Optional[dict]:
    """Fetch a job posting URL and use Claude to extract company, title, and description."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        console.print(f"[red]Could not fetch URL: {exc}[/red]")
        return None

    page_text = _strip_html(resp.text)[:12000]  # cap to avoid excess tokens

    client = anthropic.Anthropic()
    with console.status("[bold blue]Extracting job details from page...[/bold blue]"):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": (
                    "Extract the job posting details from this page text. "
                    "Return the full job description as-is (do not summarize).\n\n"
                    f"---\n{page_text}\n---"
                ),
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "json_schema": {"name": "job_extraction", "schema": JOB_EXTRACTION_SCHEMA},
                }
            },
        )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def load_resume() -> dict:
    if not RESUME_FILE.exists():
        console.print("[red]No resume found. Run [bold]setup-resume[/bold] first.[/red]")
        raise typer.Exit(1)
    return json.loads(RESUME_FILE.read_text())


def analyze_job_with_claude(job_description: str, resume: dict) -> dict:
    resume_text = json.dumps(resume, indent=2)
    client = anthropic.Anthropic()

    with console.status("[bold blue]Analyzing job with Claude...[/bold blue]"):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_INSTRUCTIONS,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"## Candidate Profile\n\n```json\n{resume_text}\n```",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze this job description and return a JSON analysis "
                        "matching the schema exactly:\n\n"
                        f"---\n{job_description}\n---"
                    ),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "job_analysis",
                        "schema": ANALYSIS_SCHEMA,
                    },
                }
            },
        )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def print_analysis(analysis: dict, company: str, title: str) -> None:
    score = analysis["match_score"]
    score_color = "green" if score >= 70 else "yellow" if score >= 50 else "red"

    console.print()
    console.print(Panel(
        f"[bold]{title}[/bold] @ [bold]{company}[/bold]\n"
        f"Match Score: [{score_color}][bold]{score}/100[/bold][/{score_color}]\n\n"
        f"{analysis['summary']}",
        title="[bold blue]Job Analysis[/bold blue]",
        border_style="blue",
    ))

    if analysis["strengths"]:
        console.print("\n[bold green]Your Strengths[/bold green]")
        for s in analysis["strengths"]:
            console.print(f"  [green]✓[/green] {s}")

    if analysis["skill_gaps"]:
        console.print("\n[bold red]Skill Gaps[/bold red]")
        gap_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        gap_table.add_column("Skill")
        gap_table.add_column("Importance")
        gap_table.add_column("Your Level")
        gap_table.add_column("Required Level")
        for gap in analysis["skill_gaps"]:
            imp = gap["importance"]
            imp_color = "red" if imp == "critical" else "yellow" if imp == "important" else "cyan"
            gap_table.add_row(
                gap["skill"],
                f"[{imp_color}]{imp}[/{imp_color}]",
                gap["your_level"],
                gap["required_level"],
            )
        console.print(gap_table)

    if analysis["learning_recommendations"]:
        console.print("\n[bold yellow]Learning Recommendations[/bold yellow]")
        for rec in analysis["learning_recommendations"]:
            pri = rec["priority"]
            pri_color = "red" if pri == "immediate" else "yellow" if pri == "short_term" else "cyan"
            url_hint = rec.get("url_hint", "")
            url_part = f"\n    [dim]{url_hint}[/dim]" if url_hint else ""
            console.print(
                f"  [{pri_color}][{pri}][/{pri_color}] [bold]{rec['skill']}[/bold]: "
                f"{rec['course_name']} — {rec['provider']} ({rec['estimated_time']})"
                f"{url_part}"
            )

    if analysis["resume_keywords_to_add"]:
        console.print("\n[bold magenta]Keywords to Add to Resume[/bold magenta]")
        console.print("  " + " · ".join(f"[magenta]{k}[/magenta]" for k in analysis["resume_keywords_to_add"]))

    console.print("\n[bold]Application Advice[/bold]")
    console.print(Panel(analysis["application_advice"], border_style="dim"))


def parse_resume_file(file_path: Path) -> dict:
    """Send a resume file to Claude and extract structured profile data."""
    client = anthropic.Anthropic()
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        raw = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")
        content = [
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": raw},
            },
            {
                "type": "text",
                "text": "Extract all information from this resume and return it as structured JSON matching the schema exactly.",
            },
        ]
    else:
        text = file_path.read_text(errors="replace")
        content = (
            f"Extract all information from this resume and return it as structured JSON "
            f"matching the schema exactly.\n\n---\n{text}\n---"
        )

    with console.status("[bold blue]Reading resume with Claude...[/bold blue]"):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "resume_profile",
                        "schema": RESUME_EXTRACTION_SCHEMA,
                    },
                }
            },
        )

    text_block = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_block)


@app.command("setup-resume")
def setup_resume(
    file: Optional[Path] = typer.Argument(None, help="Path to your resume file (PDF or TXT)"),
):
    """Load your resume from a file — PDF or plain text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if file is None:
        console.print("[red]Please provide your resume file:[/red]")
        console.print("  python agent.py setup-resume /path/to/resume.pdf")
        console.print("  python agent.py setup-resume /path/to/resume.txt")
        raise typer.Exit(1)

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    suffix = file.suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        console.print("[red]Supported formats: .pdf, .txt, .md[/red]")
        raise typer.Exit(1)

    resume = parse_resume_file(file)

    RESUME_FILE.write_text(json.dumps(resume, indent=2))

    console.print(Panel(
        f"[bold]{resume.get('name', 'Unknown')}[/bold] — {resume.get('current_title', '')}\n"
        f"Experience: {resume.get('years_experience', '')}\n"
        f"Skills found: {len(resume.get('technical_skills', []) + resume.get('languages', []) + resume.get('frameworks', []))} technical, "
        f"{len(resume.get('certifications', []))} certifications\n"
        f"Education: {len(resume.get('education', []))} entries",
        title="[bold green]Resume Parsed[/bold green]",
        border_style="green",
    ))
    console.print(f"[green]Profile saved to {RESUME_FILE}[/green]")


@app.command("add-job")
def add_job(
    url: Optional[str] = typer.Argument(None, help="Job posting URL — auto-extracts details"),
):
    """Add a job from a URL (auto-extracts title + description) or enter manually."""
    company, title, description = "", "", ""

    if url:
        extracted = fetch_job_from_url(url)
        if extracted:
            company = extracted["company"]
            title = extracted["title"]
            description = extracted["description"]

            console.print(Panel(
                f"[bold]Company:[/bold] {company}\n"
                f"[bold]Title:[/bold]   {title}\n"
                f"[bold]Preview:[/bold] {description[:200]}{'...' if len(description) > 200 else ''}",
                title="[bold green]Extracted from URL[/bold green]",
                border_style="green",
            ))

            override = typer.confirm("Looks good?", default=True)
            if not override:
                company = typer.prompt("Company", default=company)
                title = typer.prompt("Job title", default=title)
        else:
            console.print("[yellow]Falling back to manual entry.[/yellow]")

    if not description:
        if not company:
            company = typer.prompt("Company")
        if not title:
            title = typer.prompt("Job title")
        if not url:
            url = typer.prompt("Job URL (optional)", default="")
        console.print("\nPaste the job description. Press [bold]Enter on a blank line[/bold] when done:")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        description = "\n".join(lines).strip()

    if not description:
        console.print("[red]No description provided.[/red]")
        raise typer.Exit(1)

    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (company, title, url, description, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'saved', ?, ?)",
            (company, title, url or None, description, now, now),
        )
        job_id = cur.lastrowid

    console.print(f"\n[green]Job saved as ID {job_id}.[/green] Run [bold]analyze {job_id}[/bold] to get gap analysis.")


@app.command("analyze")
def analyze(
    job_id: int = typer.Argument(..., help="Job ID to analyze"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-analyze even if already analyzed"),
):
    """Run Claude gap analysis on a saved job."""
    resume = load_resume()

    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            console.print(f"[red]Job ID {job_id} not found.[/red]")
            raise typer.Exit(1)

        if row["analysis"] and not force:
            console.print("[yellow]Already analyzed. Use --force to re-analyze.[/yellow]")
            analysis = json.loads(row["analysis"])
            print_analysis(analysis, row["company"], row["title"])
            return

        analysis = analyze_job_with_claude(row["description"], resume)

        conn.execute(
            "UPDATE jobs SET analysis = ?, match_score = ?, updated_at = ? WHERE id = ?",
            (json.dumps(analysis), analysis["match_score"], datetime.utcnow().isoformat(), job_id),
        )

    print_analysis(analysis, row["company"], row["title"])


@app.command("list")
def list_jobs(
    status: Optional[str] = typer.Option(None, help=f"Filter by status: {', '.join(VALID_STATUSES)}"),
):
    """List all tracked jobs."""
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()

    if not rows:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title="Job Applications", box=box.ROUNDED, show_lines=False)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Company")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Added")

    for r in rows:
        color = STATUS_COLORS.get(r["status"], "white")
        score = str(r["match_score"]) if r["match_score"] is not None else "—"
        added = r["created_at"][:10]
        table.add_row(
            str(r["id"]),
            r["company"],
            r["title"],
            f"[{color}]{r['status']}[/{color}]",
            score,
            added,
        )

    console.print(table)


@app.command("update-status")
def update_status(
    job_id: int = typer.Argument(...),
    status: str = typer.Argument(..., help=f"New status: {', '.join(VALID_STATUSES)}"),
    notes: str = typer.Option("", help="Optional notes"),
):
    """Update the application status for a job."""
    if status not in VALID_STATUSES:
        console.print(f"[red]Invalid status. Choose from: {', '.join(VALID_STATUSES)}[/red]")
        raise typer.Exit(1)

    now = datetime.utcnow().isoformat()
    applied_date = now[:10] if status == "applied" else None

    with get_db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            console.print(f"[red]Job ID {job_id} not found.[/red]")
            raise typer.Exit(1)

        if applied_date:
            conn.execute(
                "UPDATE jobs SET status = ?, notes = ?, applied_date = ?, updated_at = ? WHERE id = ?",
                (status, notes or None, applied_date, now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ?, notes = ?, updated_at = ? WHERE id = ?",
                (status, notes or None, now, job_id),
            )

    color = STATUS_COLORS.get(status, "white")
    console.print(f"[green]Job {job_id} updated to [{color}]{status}[/{color}].[/green]")


@app.command("view")
def view_job(job_id: int = typer.Argument(...)):
    """View full details and analysis for a job."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        console.print(f"[red]Job ID {job_id} not found.[/red]")
        raise typer.Exit(1)

    color = STATUS_COLORS.get(row["status"], "white")
    console.print(Panel(
        f"[bold]{row['title']}[/bold] @ [bold]{row['company']}[/bold]\n"
        f"Status: [{color}]{row['status']}[/{color}]"
        + (f"\nURL: {row['url']}" if row["url"] else "")
        + (f"\nApplied: {row['applied_date']}" if row["applied_date"] else "")
        + (f"\nNotes: {row['notes']}" if row["notes"] else ""),
        title=f"Job #{row['id']}",
        border_style="blue",
    ))

    if row["analysis"]:
        analysis = json.loads(row["analysis"])
        print_analysis(analysis, row["company"], row["title"])
    else:
        console.print("[dim]No analysis yet. Run [bold]analyze {job_id}[/bold] to get one.[/dim]")


if __name__ == "__main__":
    app()
