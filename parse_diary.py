#!/usr/bin/env python3
"""
parse_diary.py — Jiva meditation diary parser
Usage: python parse_diary.py <diary_file> --run <N>
Output: <diary_file_stem>.json in the same directory
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Section keyword classifier ─────────────────────────────────────────────────
# Order matters: check more specific anchors first
SECTION_KEYWORDS = [
    ("finger",       ["finger"]),
    ("insights",     ["insight", "realization"]),
    ("distractions", ["distract", "thought", "pulled"]),
    ("experience",   ["experience", "jiva"]),
    ("entry",        ["entry", "diary"]),
]

def classify_header(line: str) -> str | None:
    """Return canonical section name if line is a section header, else None."""
    # Must match **...:  ** pattern (bold header ending in colon)
    if not re.match(r'^\*\*.+:\*\*\s*$', line.strip()):
        return None
    lower = line.lower()
    for name, keywords in SECTION_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return name
    return None  # bold header but unrecognized — treat as entry content

# ── Timestamp parsing ──────────────────────────────────────────────────────────
OUTER_RE = re.compile(r'^===\s+Session\s+(\d+)\s+[—\-]+\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+===')
INNER_RE = re.compile(r'^\*\*Session\s+\d+\s+[—\-]+\s+(\d{4}-\d{2}-\d{2}\s+[\d:]+)\*\*')

def parse_timestamp(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def fmt_iso(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None

# ── Session block parser ───────────────────────────────────────────────────────
def parse_block(block: str, run: int) -> dict | None:
    """Parse a single session block (text between --- delimiters)."""
    lines = block.splitlines()

    outer_time = None
    inner_time = None
    session_num = None

    sections = {k: [] for k in ["entry", "insights", "distractions", "finger", "experience"]}
    current_section = None

    for line in lines:
        # Outer header
        m = OUTER_RE.match(line)
        if m:
            session_num = int(m.group(1))
            outer_time = parse_timestamp(m.group(2))
            continue

        # Inner header — store but don't rely on for timing
        m = INNER_RE.match(line)
        if m:
            inner_time = parse_timestamp(m.group(1))
            continue

        # Section header detection
        section = classify_header(line)
        if section is not None:
            current_section = section
            continue

        # Skip blank lines at the top before any section is active
        if current_section is None:
            continue

        sections[current_section].append(line)

    if session_num is None or outer_time is None:
        return None  # not a real session block

    # Clean up accumulated text
    def clean(lines_list):
        text = "\n".join(lines_list).strip()
        # Remove LaTeX arrow artifacts
        text = text.replace(r"$\rightarrow$", "→").replace(r"\rightarrow", "→")
        # Strip trailing/leading blank lines
        return text if text else None

    return {
        "session":          session_num,
        "outer_time":       fmt_iso(outer_time),
        "inner_time":       fmt_iso(inner_time),
        "elapsed_minutes":  None,   # filled in after all sessions parsed
        "sections": {
            "entry":        clean(sections["entry"]),
            "insights":     clean(sections["insights"]),
            "distractions": clean(sections["distractions"]),
            "finger":       clean(sections["finger"]),
            "experience":   clean(sections["experience"]),
        }
    }

# ── Elapsed time pass ─────────────────────────────────────────────────────────
def fill_elapsed(sessions: list) -> None:
    """Compute elapsed_minutes as outer[N] - outer[N-1]. Session 1 stays None."""
    for i in range(1, len(sessions)):
        prev = sessions[i-1]["outer_time"]
        curr = sessions[i]["outer_time"]
        if prev and curr:
            dt_prev = datetime.fromisoformat(prev)
            dt_curr = datetime.fromisoformat(curr)
            delta = (dt_curr - dt_prev).total_seconds() / 60
            sessions[i]["elapsed_minutes"] = round(delta, 1)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Parse a Jiva meditation diary into JSON.")
    ap.add_argument("diary_file", help="Path to diary text file")
    ap.add_argument("--run", type=int, required=True, help="Run number (e.g. 5)")
    args = ap.parse_args()

    diary_path = Path(args.diary_file)
    if not diary_path.exists():
        raise SystemExit(f"File not found: {diary_path}")

    raw = diary_path.read_text(encoding="utf-8")

    # Split on --- session delimiters
    blocks = re.split(r'\n---+\n', raw)

    sessions = []
    for block in blocks:
        if block.strip():
            result = parse_block(block, args.run)
            if result:
                sessions.append(result)

    # Sort by session number in case blocks are out of order
    sessions.sort(key=lambda s: s["session"])
    fill_elapsed(sessions)

    output = {
        "run":         args.run,
        "source_file": diary_path.name,
        "sessions":    sessions
    }

    out_path = diary_path.with_suffix(".json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Parsed {len(sessions)} sessions → {out_path}")

if __name__ == "__main__":
    main()