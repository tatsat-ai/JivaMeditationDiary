#!/usr/bin/env python3
"""
report_japa.py — Query the Similarities table and produce analysis reports.

Usage:
    python report_japa.py [--db <path>] [--method tfidf|st] [--zscore <float>]

    --db        Path to SQLite database (default: ./jiva.db)
    --method    Similarity method to report on (default: st)
    --zscore    Z-score threshold for outlier detection (default: -1.5)

Output:
    - Plain text summary printed to stdout
    - Three CSV files written alongside the database:
        convergence_arc.csv
        cross_run_alignment.csv
        outliers.csv
"""

import sqlite3
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────────────────────

DB_DEFAULT    = './japa_schema.db'
SECTIONS      = ['entry', 'insights', 'distractions', 'finger', 'experience']
SECTION_ORDER = {s: i for i, s in enumerate(SECTIONS)}

# ── Database ───────────────────────────────────────────────────────────────────

def get_connection(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

# ── Query helpers ──────────────────────────────────────────────────────────────

def fetch_within_run_scores(con, method: str) -> list[dict]:
    """
    All within-run pairs for the same section type.
    Returns both consecutive and session-1-vs-N pairs —
    distinguished at analysis time by session numbers.
    """
    rows = con.execute("""
        SELECT
            a.run_id,
            a.section,
            a.session_num  AS session_a,
            b.session_num  AS session_b,
            sim.score,
            r.model,
            r.temperature,
            r.max_tokens,
            r.n_sessions
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        JOIN Runs r     ON r.run_id = a.run_id
        WHERE sim.method  = ?
          AND a.run_id    = b.run_id
          AND a.section   = b.section
        ORDER BY a.run_id, a.section, a.session_num, b.session_num
    """, (method,)).fetchall()
    return [dict(r) for r in rows]

def fetch_cross_run_scores(con, method: str) -> list[dict]:
    """
    All cross-run pairs for the same section type and session number.
    """
    rows = con.execute("""
        SELECT
            a.section,
            a.session_num  AS session_num,
            a.run_id       AS run_a,
            b.run_id       AS run_b,
            sim.score
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        WHERE sim.method      = ?
          AND a.run_id       != b.run_id
          AND a.section       = b.section
          AND a.session_num   = b.session_num
        ORDER BY a.section, a.session_num, a.run_id, b.run_id
    """, (method,)).fetchall()
    return [dict(r) for r in rows]

def fetch_section_ids(con) -> dict[tuple, int]:
    """Map (run_id, session_num, section) -> Sections.id"""
    rows = con.execute("SELECT id, run_id, session_num, section FROM Sections").fetchall()
    return {(r['run_id'], r['session_num'], r['section']): r['id'] for r in rows}

# ── Report 1: Convergence Arc ─────────────────────────────────────────────────

def report_convergence(within_rows: list[dict], method: str) -> tuple[list, str]:
    """
    Session-1 baseline vs every other session within the same run.
    Score represents drift from the opening session.
    """
    # Filter to session_1 vs session_N pairs only (session_a == 1)
    #baseline_pairs = [r for r in within_rows if r['session_a'] == 1 and r['session_b'] > 1]
    # Correct — consecutive steps only
    baseline_pairs = [r for r in within_rows if r['session_b'] == r['session_a'] + 1]
    # Group by (run_id, section)
    arcs = defaultdict(list)
    for r in baseline_pairs:
        arcs[(r['run_id'], r['section'])].append(r)

    csv_rows = []
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  CONVERGENCE ARC  (method={method}, consecutive session similarity)")
    lines.append(f"  Score approaching 1.0 = sessions becoming identical = attractor lock-in")
    lines.append(f"{'='*70}")

    for section in SECTIONS:
        lines.append(f"\n  Section: {section.upper()}")
        lines.append(f"  {'Run':<6} {'Model':<18} {'Sess':<6} {'Score':>7}  Arc")

        for run_id in sorted({r['run_id'] for r in baseline_pairs}):
            key = (run_id, section)
            if key not in arcs:
                continue
            pairs = sorted(arcs[key], key=lambda r: r['session_b'])
            model = pairs[0]['model'] or 'unknown'
            scores = [r['score'] for r in pairs]

            for i, r in enumerate(pairs):
                bar = spark_bar(r['score'])
                lines.append(f"  {run_id:<6} {model:<18} {r['session_b']:<6} {r['score']:>7.4f}  {bar}")
                csv_rows.append({
                    'run_id':     run_id,
                    'model':      model,
                    'section':    section,
                    'session_b':    r['session_b'],
                    'score':      round(r['score'], 6),
                    'method':     method,
                })
            # Final score summary
            if scores:
                lines.append(f"  {'':<6} {'':18} {'mean':<6} {sum(scores)/len(scores):>7.4f}")

    return csv_rows, "\n".join(lines)

# ── Report 2: Cross-Run Alignment ─────────────────────────────────────────────

def report_cross_run(cross_rows: list[dict], method: str) -> tuple[list, str]:
    """
    Average pairwise similarity across runs at each session number,
    per section type. High score = attractor convergence across runs.
    """
    # Group by (section, session_num) — average all run-pair scores
    groups = defaultdict(list)
    for r in cross_rows:
        groups[(r['section'], r['session_num'])].append(r['score'])

    csv_rows = []
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  CROSS-RUN ATTRACTOR ALIGNMENT  (method={method})")
    lines.append(f"{'='*70}")
    lines.append("  High score = runs converging to same attractor at that session\n")

    for section in SECTIONS:
        lines.append(f"  Section: {section.upper()}")
        lines.append(f"  {'Sess':<6} {'Avg Score':>10}  {'Min':>7}  {'Max':>7}  {'N pairs':>8}  Arc")

        session_nums = sorted({k[1] for k in groups if k[0] == section})
        for snum in session_nums:
            scores = groups.get((section, snum), [])
            if not scores:
                continue
            avg  = sum(scores) / len(scores)
            mn   = min(scores)
            mx   = max(scores)
            bar  = spark_bar(avg)
            lines.append(f"  {snum:<6} {avg:>10.4f}  {mn:>7.4f}  {mx:>7.4f}  {len(scores):>8}  {bar}")
            csv_rows.append({
                'section':     section,
                'session_num': snum,
                'avg_score':   round(avg, 6),
                'min_score':   round(mn, 6),
                'max_score':   round(mx, 6),
                'n_pairs':     len(scores),
                'method':      method,
            })
        lines.append("")

    return csv_rows, "\n".join(lines)

# ── Report 3: Outlier Detection ────────────────────────────────────────────────

def report_outliers(within_rows: list[dict], method: str,
                    zscore_threshold: float) -> tuple[list, str]:
    """
    Within-run, consecutive session pairs.
    Flag sessions where the score is below zscore_threshold std devs
    from the run+section mean — these are the one-offs worth reading.
    """
    import math

    # Filter to consecutive pairs (session_b == session_a + 1)
    consec = [r for r in within_rows if r['session_b'] == r['session_a'] + 1]

    # Group by (run_id, section) to compute mean and std
    groups = defaultdict(list)
    for r in consec:
        groups[(r['run_id'], r['section'])].append(r)

    csv_rows = []
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  OUTLIER SESSIONS  (method={method}, Z < {zscore_threshold})")
    lines.append(f"{'='*70}")
    lines.append("  These consecutive-session drops are worth reading qualitatively.\n")

    found_any = False
    for section in SECTIONS:
        section_lines = []
        for run_id in sorted({r['run_id'] for r in consec}):
            key = (run_id, section)
            if key not in groups:
                continue
            pairs  = groups[key]
            scores = [r['score'] for r in pairs]
            if len(scores) < 2:
                continue

            mean = sum(scores) / len(scores)
            var  = sum((s - mean)**2 for s in scores) / len(scores)
            std  = math.sqrt(var) if var > 0 else 0

            for r in pairs:
                z = (r['score'] - mean) / std if std > 0 else 0
                if z < zscore_threshold:
                    model = r['model'] or 'unknown'
                    section_lines.append(
                        f"  Run {run_id} ({model})  "
                        f"S{r['session_a']}→S{r['session_b']}  "
                        f"score={r['score']:.4f}  Z={z:.2f}"
                    )
                    csv_rows.append({
                        'run_id':    run_id,
                        'model':     model,
                        'section':   section,
                        'session_a': r['session_a'],
                        'session_b': r['session_b'],
                        'score':     round(r['score'], 6),
                        'z_score':   round(z, 4),
                        'method':    method,
                    })
                    found_any = True

        if section_lines:
            lines.append(f"  Section: {section.upper()}")
            lines.extend(section_lines)
            lines.append("")

    if not found_any:
        lines.append(f"  No outliers found at Z < {zscore_threshold}.")

    return csv_rows, "\n".join(lines)

# ── Spark bar ──────────────────────────────────────────────────────────────────

def spark_bar(score: float, width: int = 20) -> str:
    """Simple ASCII progress bar for scores in [0, 1]."""
    filled = round(score * width)
    return '[' + '█' * filled + '·' * (width - filled) + ']'

# ── CSV writer ─────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Japa similarity report generator.")
    ap.add_argument("--db",     default=DB_DEFAULT,  help="SQLite DB path")
    ap.add_argument("--method", default="st",
                    choices=["st", "tfidf"],          help="Similarity method (default: st)")
    ap.add_argument("--zscore", default=-1.5, type=float,
                                                      help="Outlier Z-score threshold (default: -1.5)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    con = get_connection(db_path)

    print(f"\nJapa Meditation Experiment — Similarity Report")
    print(f"Database : {db_path}")
    print(f"Method   : {args.method}")
    print(f"Z-score  : {args.zscore}")

    within_rows = fetch_within_run_scores(con, args.method)
    cross_rows  = fetch_cross_run_scores(con, args.method)

    # ── Report 1: Convergence arc
    csv1, text1 = report_convergence(within_rows, args.method)
    print(text1)

    # ── Report 2: Cross-run alignment
    csv2, text2 = report_cross_run(cross_rows, args.method)
    print(text2)

    # ── Report 3: Outliers
    csv3, text3 = report_outliers(within_rows, args.method, args.zscore)
    print(text3)

    # ── Write CSVs
    print("\nWriting CSV files...")
    base = db_path.parent
    write_csv(base / f"convergence_arc_{args.method}.csv",      csv1)
    write_csv(base / f"cross_run_alignment_{args.method}.csv",  csv2)
    write_csv(base / f"outliers_{args.method}.csv",             csv3)

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()