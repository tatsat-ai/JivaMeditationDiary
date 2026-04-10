#!/usr/bin/env python3
"""
load_diary.py — Load a parsed Jiva diary JSON into the SQLite database.
Schema is created automatically on first run.

Usage: python load_diary.py <json_file> [--db <path>]
       --db defaults to ./jiva.db

Prerequisite: Insert the Runs record manually before loading:
  sqlite3 jiva.db "INSERT INTO Runs VALUES(5,'jiva_diary_run5.txt',14,'gemma4-e2b',0.8,2048,'baseline run');"
"""

import json
import sqlite3
import argparse
from pathlib import Path

SECTION_KEYS = ['entry', 'insights', 'distractions', 'finger', 'experience']

def check_schema(con):
    tables = {row[0] for row in 
              con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    required = {'Runs', 'Sessions', 'Sections', 'Similarities'}
    missing = required - tables
    if missing:
        raise SystemExit(
            f"Missing tables: {missing}\n"
            f"Run jiva_schema.sql in DB Browser first, or:\n"
            f"  sqlite3 jiva.db < japa_schema.sql\n"        )
    
# ── Loader ─────────────────────────────────────────────────────────────────────

def load(json_path: Path, db_path: Path):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    run_id = data['run']

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    check_schema(con)

    # Verify the Runs record exists
    row = con.execute("SELECT run_id FROM Runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise SystemExit(
            f"\nRuns record for run_id={run_id} not found.\n"
            f"Insert it first, e.g.:\n"
            f"  sqlite3 {db_path} \"INSERT INTO Runs VALUES"
            f"({run_id},'{data['source_file']}',<n_sessions>,"
            f"'<model>',<temp>,<max_tokens>,'<notes>');\"\n"
        )

    sessions_loaded = 0
    sections_loaded = 0

    for sess in data['sessions']:
        snum = sess['session']

        # Insert Session — skip if already exists (idempotent re-run)
        try:
            con.execute("""
                INSERT INTO Sessions(run_id, session_num, outer_time, inner_time, elapsed_minutes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                run_id,
                snum,
                sess.get('outer_time'),
                sess.get('inner_time'),
                sess.get('elapsed_minutes'),
            ))
            sessions_loaded += 1
        except sqlite3.IntegrityError:
            print(f"  Session {snum} already exists — skipping")
            continue

        # Insert Sections
        sections = sess.get('sections', {})
        for key in SECTION_KEYS:
            text = sections.get(key)
            con.execute("""
                INSERT INTO Sections(run_id, session_num, section, text)
                VALUES (?, ?, ?, ?)
            """, (run_id, snum, key, text))
            if text:
                sections_loaded += 1

    con.commit()
    con.close()

    print(f"Run {run_id}: loaded {sessions_loaded} sessions, "
          f"{sections_loaded} non-null sections → {db_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Load Jiva diary JSON into SQLite.")
    ap.add_argument("json_file", help="Parsed diary JSON file")
    ap.add_argument("--db", default="jiva.db", help="SQLite database path (default: jiva.db)")
    args = ap.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        raise SystemExit(f"File not found: {json_path}")

    load(json_path, Path(args.db))

if __name__ == "__main__":
    main()