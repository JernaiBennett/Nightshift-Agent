#!/usr/bin/env python3
"""Job Intelligence Agent — resume gap analysis, course recommendations, application tracking."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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


@app.command("setup-resume")
def setup_resume():
    """Interactively build or update your resume/skills profile."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if RESUME_FILE.exists():
        existing = json.loads(RESUME_FILE.read_text())
        console.print("[yellow]Existing resume found. Press Enter to keep current values.[/yellow]\n")

    def prompt(field: str, default: str = "") -> str:
        current = existing.get(field, default)
        hint = f" [dim](current: {current})[/dim]" if current else ""
        val = typer.prompt(f"{field}{hint}", default=current)
        return val

    resume = {
        "name": prompt("name"),
        "current_title": prompt("current_title"),
        "years_experience": prompt("years_experience"),
        "summary": prompt("summary"),
    }

    console.print("\n[bold]Skills[/bold] (comma-separated):")
    for category in ["technical_skills", "languages", "frameworks", "tools", "cloud_platforms", "soft_skills"]:
        current = ", ".join(existing.get(category, []))
        raw = typer.prompt(f"  {category}", default=current)
        resume[category] = [s.strip() for s in raw.split(",") if s.strip()]

    console.print("\n[bold]Education[/bold] (e.g. 'BS Computer Science, State University, 2020'):")
    current_edu = "\n".join(existing.get("education", []))
    raw_edu = typer.prompt("  education entries (one per line, use \\n)", default=current_edu)
    resume["education"] = [e.strip() for e in raw_edu.split(r"\n") if e.strip()]

    console.print("\n[bold]Certifications[/bold] (comma-separated):")
    current_certs = ", ".join(existing.get("certifications", []))
    raw_certs = typer.prompt("  certifications", default=current_certs)
    resume["certifications"] = [c.strip() for c in raw_certs.split(",") if c.strip()]

    console.print("\n[bold]Recent Projects / Achievements[/bold] (comma-separated, brief descriptions):")
    current_proj = ", ".join(existing.get("notable_projects", []))
    raw_proj = typer.prompt("  notable_projects", default=current_proj)
    resume["notable_projects"] = [p.strip() for p in raw_proj.split(",") if p.strip()]

    resume["preferred_roles"] = [r.strip() for r in typer.prompt(
        "\nPreferred role types (comma-separated)", default=", ".join(existing.get("preferred_roles", []))
    ).split(",") if r.strip()]

    RESUME_FILE.write_text(json.dumps(resume, indent=2))
    console.print(f"\n[green]Resume saved to {RESUME_FILE}[/green]")


@app.command("add-job")
def add_job(
    company: str = typer.Option(..., prompt=True),
    title: str = typer.Option(..., prompt=True),
    url: str = typer.Option("", prompt="Job URL (optional)"),
):
    """Add a new job and paste the description (end with a line containing only '---')."""
    console.print("\nPaste the job description below. End with a line containing only [bold]---[/bold]:")
    lines = []
    while True:
        line = input()
        if line.strip() == "---":
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
