#!/usr/bin/env python3
"""
analyze_japa.py — Compute TF-IDF and sentence transformer similarity scores
and write them to the Similarities table in jiva.db.

Usage:
    python analyze_japa.py [--db <path>] [--recompute]

    --db         Path to SQLite database (default: ./jiva.db)
    --recompute  Clear and recompute all similarity scores (default: skip existing)

Requires:
    pip install scikit-learn sentence-transformers --break-system-packages

Pair types computed:
    1. Within-run, same section, consecutive sessions (convergence arc)
    2. Within-run, same section, session 1 vs every other session (drift from baseline)
    3. Cross-run, same section, same session number (attractor alignment)
"""

import sqlite3
import argparse
import itertools
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────

SECTIONS     = ['entry', 'insights', 'distractions', 'finger', 'experience']
ST_MODEL     = 'all-MiniLM-L6-v2'
DB_DEFAULT   = './japa_schema.db'

# ── Database helpers ───────────────────────────────────────────────────────────

def get_connection(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    return con

def load_sections(con) -> list[dict]:
    """Load all non-null sections with their IDs."""
    rows = con.execute("""
        SELECT s.id, s.run_id, s.session_num, s.section, s.text
        FROM Sections s
        WHERE s.text IS NOT NULL
        ORDER BY s.run_id, s.session_num, s.section
    """).fetchall()
    return [dict(r) for r in rows]

def existing_pairs(con, method: str) -> set[tuple[int,int]]:
    """Return set of (section_a, section_b) already in Similarities for this method."""
    rows = con.execute(
        "SELECT section_a, section_b FROM Similarities WHERE method = ?", (method,)
    ).fetchall()
    return {(r[0], r[1]) for r in rows}

def insert_similarities(con, pairs: list[tuple[int,int,str,float]]):
    """Bulk insert (section_a, section_b, method, score) tuples."""
    con.executemany("""
        INSERT INTO Similarities(section_a, section_b, method, score)
        VALUES (?, ?, ?, ?)
    """, pairs)
    con.commit()

# ── Pair generation ────────────────────────────────────────────────────────────

def build_pairs(sections: list[dict]) -> list[tuple[int,int]]:
    """
    Build all pairs to compute, enforcing section_a < section_b.
    Three types:
      1. Within-run, same section type, consecutive sessions
      2. Within-run, same section type, session 1 vs all others
      3. Cross-run, same section type, same session number
    """
    # Index: (run_id, session_num, section) -> row dict
    idx = {(r['run_id'], r['session_num'], r['section']): r for r in sections}

    # Group by (run_id, section)
    from collections import defaultdict
    by_run_section = defaultdict(list)
    for r in sections:
        by_run_section[(r['run_id'], r['section'])].append(r)

    # Group by (section, session_num) for cross-run
    by_section_session = defaultdict(list)
    for r in sections:
        by_section_session[(r['section'], r['session_num'])].append(r)

    pairs = set()

    for (run_id, section), rows in by_run_section.items():
        rows_sorted = sorted(rows, key=lambda r: r['session_num'])

        # Type 1: consecutive sessions
        for i in range(len(rows_sorted) - 1):
            a, b = rows_sorted[i]['id'], rows_sorted[i+1]['id']
            pairs.add((min(a,b), max(a,b)))

        # Type 2: session 1 vs all others
        session1 = next((r for r in rows_sorted if r['session_num'] == 1), None)
        if session1:
            for r in rows_sorted:
                if r['session_num'] != 1:
                    a, b = session1['id'], r['id']
                    pairs.add((min(a,b), max(a,b)))

    # Type 3: cross-run, same section, same session number
    for (section, session_num), rows in by_section_session.items():
        for r1, r2 in itertools.combinations(rows, 2):
            if r1['run_id'] != r2['run_id']:
                a, b = r1['id'], r2['id']
                pairs.add((min(a,b), max(a,b)))

    return sorted(pairs)

# ── Vectorization ──────────────────────────────────────────────────────────────

def compute_tfidf(sections: list[dict]) -> dict[int, np.ndarray]:
    """Return dict of section_id -> TF-IDF vector (dense numpy array)."""
    ids   = [r['id'] for r in sections]
    texts = [r['text'] for r in sections]
    vec   = TfidfVectorizer(
                strip_accents='unicode',
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
    matrix = vec.fit_transform(texts).toarray()
    return {sid: matrix[i] for i, sid in enumerate(ids)}

def compute_st(sections: list[dict]) -> dict[int, np.ndarray]:
    """Return dict of section_id -> sentence transformer embedding."""
    print(f"  Loading sentence transformer model: {ST_MODEL}")
    model  = SentenceTransformer(ST_MODEL)
    ids    = [r['id'] for r in sections]
    texts  = [r['text'] for r in sections]
    print(f"  Encoding {len(texts)} sections...")
    embeds = model.encode(texts, show_progress_bar=True,
                          batch_size=32, convert_to_numpy=True)
    return {sid: embeds[i] for i, sid in enumerate(ids)}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ── Main ───────────────────────────────────────────────────────────────────────

def run_method(con, method: str, vectors: dict[int, np.ndarray],
               pairs: list[tuple[int,int]], recompute: bool):

    if recompute:
        con.execute("DELETE FROM Similarities WHERE method = ?", (method,))
        con.commit()
        skip = set()
        print(f"  [{method}] Cleared existing scores.")
    else:
        skip = existing_pairs(con, method)
        print(f"  [{method}] Skipping {len(skip)} existing pairs.")

    todo = [(a, b) for a, b in pairs if (a, b) not in skip
                                      and a in vectors and b in vectors]
    print(f"  [{method}] Computing {len(todo)} pairs...")

    batch = []
    for a_id, b_id in todo:
        score = cosine(vectors[a_id], vectors[b_id])
        batch.append((a_id, b_id, method, round(score, 6)))

    insert_similarities(con, batch)
    print(f"  [{method}] Done — {len(batch)} scores written.")

def main():
    ap = argparse.ArgumentParser(description="Compute japa similarity scores.")
    ap.add_argument("--db",        default=DB_DEFAULT, help="SQLite DB path")
    ap.add_argument("--recompute", action="store_true",
                    help="Clear and recompute all scores")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    con = get_connection(db_path)

    print("Loading sections from database...")
    sections = load_sections(con)
    print(f"  {len(sections)} non-null sections across all runs.")

    if not sections:
        raise SystemExit("No sections found. Load diary JSON files first.")

    print("Building pair list...")
    pairs = build_pairs(sections)
    print(f"  {len(pairs)} unique pairs to evaluate.")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("\nComputing TF-IDF vectors...")
    tfidf_vecs = compute_tfidf(sections)
    run_method(con, 'tfidf', tfidf_vecs, pairs, args.recompute)

    # ── Sentence Transformers ─────────────────────────────────────────────────
    print("\nComputing sentence transformer vectors...")
    st_vecs = compute_st(sections)
    run_method(con, 'st', st_vecs, pairs, args.recompute)

    con.close()
    print("\nAll done.")

if __name__ == "__main__":
    main()