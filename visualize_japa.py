#!/usr/bin/env python3
"""
visualize_japa.py — Plotly visualization of japa meditation similarity data.

Produces four self-contained HTML files:
    convergence_arc_{method}.html
    cross_run_alignment_{method}.html
    outliers_{method}.html
    section_heatmap_{method}.html

Click any data point to read the full section text in a panel at the bottom.

Usage:
    python visualize_japa.py [--db <path>] [--method tfidf|st] [--out <dir>]
    [--zscore <float>]
"""

import sqlite3
import argparse
import math
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Constants ──────────────────────────────────────────────────────────────────

SECTIONS  = ['insights', 'distractions', 'finger', 'experience', 'entry']
TEMPLATE  = 'plotly_dark'
DB_DEFAULT = './japa_schema.db'

# ── Click panel (injected into every HTML file) ────────────────────────────────

CLICK_PANEL_JS = """
<div id="text-panel" style="
    display:none;
    position:fixed;
    bottom:0; left:0; right:0;
    max-height:38vh;
    overflow-y:auto;
    background:#1e1e2e;
    color:#cdd6f4;
    padding:16px 24px;
    font-family:monospace;
    font-size:13px;
    line-height:1.7;
    border-top:2px solid #89b4fa;
    z-index:9999;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.5);
">
  <div style="display:flex; justify-content:space-between;
              align-items:center; margin-bottom:10px;">
    <span id="text-panel-title"
          style="color:#89b4fa; font-weight:bold; font-size:14px;"></span>
    <button onclick="document.getElementById('text-panel').style.display='none'"
            style="background:#f38ba8; border:none; color:#1e1e2e;
                   font-size:13px; font-weight:bold; padding:3px 10px;
                   border-radius:4px; cursor:pointer;">✕ close</button>
  </div>
  <div id="text-panel-body" style="white-space:pre-wrap;"></div>
</div>

<script>
(function() {
  var plots = document.querySelectorAll('.plotly-graph-div');
  plots.forEach(function(plot) {
    plot.on('plotly_click', function(data) {
      var pt = data.points[0];
      if (!pt.customdata) return;
      var cd   = pt.customdata;
      var title = cd[0];   // pre-formatted title string
      var body  = cd[1];   // section text
      if (!body || body === 'null') body = '(no text recorded for this section)';
      document.getElementById('text-panel-title').textContent = title;
      document.getElementById('text-panel-body').textContent  = body;
      document.getElementById('text-panel').style.display = 'block';
      document.getElementById('text-panel').scrollTop = 0;
    });
  });
})();
</script>
"""

def inject_panel(html: str) -> str:
    return html.replace('</body>', CLICK_PANEL_JS + '\n</body>')

def write_html(fig, path: Path):
    html = fig.to_html(include_plotlyjs='cdn')
    path.write_text(inject_panel(html), encoding='utf-8')
    print(f'  → {path}')

# ── Database ───────────────────────────────────────────────────────────────────

def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def run_labels(con) -> dict[int, str]:
    rows = con.execute(
        "SELECT run_id, model, temperature, max_tokens FROM Runs"
    ).fetchall()
    return {
        r['run_id']: f"Run {r['run_id']} · {r['model'] or 'unknown'} "
                     f"· t={r['temperature']} · {r['max_tokens']}tok"
        for r in rows
    }

def section_texts(con) -> dict[tuple, str]:
    """Map (run_id, session_num, section) -> text."""
    rows = con.execute(
        "SELECT run_id, session_num, section, text FROM Sections"
    ).fetchall()
    return {(r['run_id'], r['session_num'], r['section']): r['text'] or ''
            for r in rows}

# ── Chart 1: Convergence Arc ───────────────────────────────────────────────────

def chart_convergence(con, method: str, out_dir: Path,
                      labels: dict, texts: dict):
    rows = con.execute("""
        SELECT a.run_id, a.section,
               a.session_num AS session_a,
               b.session_num AS session_b,
               sim.score
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        WHERE sim.method    = ?
          AND a.run_id      = b.run_id
          AND a.section     = b.section
          AND b.session_num = a.session_num + 1
        ORDER BY a.run_id, a.section, a.session_num
    """, (method,)).fetchall()

    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r['section']][r['run_id']].append(r)

    fig = make_subplots(
        rows=len(SECTIONS), cols=1,
        subplot_titles=[s.upper() for s in SECTIONS],
        shared_xaxes=True,
        vertical_spacing=0.055,
    )

    run_ids = sorted({r['run_id'] for r in rows})
    colors  = px.colors.qualitative.Plotly

    for s_idx, section in enumerate(SECTIONS):
        for r_idx, run_id in enumerate(run_ids):
            pts = sorted(data[section].get(run_id, []),
                         key=lambda r: r['session_a'])
            if not pts:
                continue

            xs, ys, cdata = [], [], []
            for p in pts:
                xs.append(p['session_b'])
                ys.append(p['score'])
                # customdata: [title, text_of_session_b]
                title = (f"Run {run_id}  ·  {section.upper()}  "
                         f"·  S{p['session_a']}→S{p['session_b']}  "
                         f"·  score={p['score']:.4f}")
                text  = texts.get((run_id, p['session_b'], section), '')
                cdata.append([title, text])

            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode='lines+markers',
                    name=labels.get(run_id, f'Run {run_id}'),
                    legendgroup=str(run_id),
                    showlegend=(s_idx == 0),
                    line=dict(color=colors[r_idx % len(colors)], width=2),
                    marker=dict(size=7),
                    customdata=cdata,
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        '<i>click to read text</i>'
                        '<extra></extra>'
                    ),
                ),
                row=s_idx + 1, col=1
            )

        fig.add_hline(y=1.0, line_dash='dot',
                      line_color='rgba(255,255,255,0.15)',
                      row=s_idx + 1, col=1)
        fig.update_yaxes(range=[0.2, 1.05], row=s_idx + 1, col=1,
                         title_text='similarity')

    fig.update_layout(
        template=TEMPLATE,
        title=dict(
            text='Convergence Arc — Consecutive Session Similarity<br>'
                 '<sup>Score → 1.0 = attractor lock-in  ·  '
                 'Click any point to read that session\'s text</sup>',
            font=dict(size=18)
        ),
        height=280 * len(SECTIONS),
        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='left', x=0),
    )
    fig.update_xaxes(title_text='Session', row=len(SECTIONS), col=1)

    write_html(fig, out_dir / f'convergence_arc_{method}.html')

# ── Chart 2: Cross-Run Alignment ───────────────────────────────────────────────

def chart_cross_run(con, method: str, out_dir: Path, texts: dict):
    rows = con.execute("""
        SELECT a.section, a.session_num,
               a.run_id  AS run_a,
               b.run_id  AS run_b,
               sim.score
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        WHERE sim.method    = ?
          AND a.run_id     != b.run_id
          AND a.section     = b.section
          AND a.session_num = b.session_num
        ORDER BY a.section, a.session_num
    """, (method,)).fetchall()

    # Aggregate per (section, session_num)
    agg = defaultdict(list)
    raw = defaultdict(list)   # (section, session_num) -> list of (run_a, run_b, score)
    for r in rows:
        key = (r['section'], r['session_num'])
        agg[key].append(r['score'])
        raw[key].append((r['run_a'], r['run_b'], r['score']))

    fig = make_subplots(
        rows=len(SECTIONS), cols=1,
        subplot_titles=[s.upper() for s in SECTIONS],
        shared_xaxes=True,
        vertical_spacing=0.055,
    )
    colors = px.colors.qualitative.Bold

    for s_idx, section in enumerate(SECTIONS):
        keys = sorted([k for k in agg if k[0] == section],
                      key=lambda k: k[1])
        if not keys:
            continue

        xs   = [k[1]                              for k in keys]
        avgs = [sum(agg[k]) / len(agg[k])         for k in keys]
        mins = [min(agg[k])                        for k in keys]
        maxs = [max(agg[k])                        for k in keys]
        color = colors[s_idx % len(colors)]

        # Shade band
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=maxs + mins[::-1],
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.12)'),
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip',
            ),
            row=s_idx + 1, col=1
        )

        # Build customdata: title + both texts for the pair with highest score
        cdata = []
        for k in keys:
            pairs = raw[k]
            best  = max(pairs, key=lambda p: p[2])
            ra, rb, sc = best
            t_a = texts.get((ra, k[1], section), '')
            t_b = texts.get((rb, k[1], section), '')
            title = (f"{section.upper()}  ·  Session {k[1]}  "
                     f"·  avg={sum(agg[k])/len(agg[k]):.4f}  "
                     f"·  best pair: R{ra} vs R{rb} ({sc:.4f})")
            body  = (f"─── Run {ra}, Session {k[1]} ───\n{t_a}\n\n"
                     f"─── Run {rb}, Session {k[1]} ───\n{t_b}")
            cdata.append([title, body])

        fig.add_trace(
            go.Scatter(
                x=xs, y=avgs,
                mode='lines+markers',
                name=section.upper(),
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=7),
                customdata=cdata,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    '<i>click to compare best-matching run pair</i>'
                    '<extra></extra>'
                ),
            ),
            row=s_idx + 1, col=1
        )
        fig.update_yaxes(range=[0.0, 1.05], row=s_idx + 1, col=1,
                         title_text='avg similarity')

    fig.update_layout(
        template=TEMPLATE,
        title=dict(
            text='Cross-Run Attractor Alignment — Same Section, Same Session<br>'
                 '<sup>High score = runs converging to same attractor  ·  '
                 'Band = min/max range  ·  Click to compare best-matching pair</sup>',
            font=dict(size=18)
        ),
        height=260 * len(SECTIONS),
        showlegend=False,
    )
    fig.update_xaxes(title_text='Session', row=len(SECTIONS), col=1)

    write_html(fig, out_dir / f'cross_run_alignment_{method}.html')

# ── Chart 3: Outliers ──────────────────────────────────────────────────────────

def chart_outliers(con, method: str, out_dir: Path,
                   labels: dict, texts: dict, zscore_threshold: float):
    rows = con.execute("""
        SELECT a.run_id, a.section,
               a.session_num AS session_a,
               b.session_num AS session_b,
               sim.score
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        WHERE sim.method    = ?
          AND a.run_id      = b.run_id
          AND a.section     = b.section
          AND b.session_num = a.session_num + 1
        ORDER BY a.run_id, a.section, a.session_num
    """, (method,)).fetchall()

    groups = defaultdict(list)
    for r in rows:
        groups[(r['run_id'], r['section'])].append(dict(r))

    normal  = defaultdict(list)
    outliers = []

    for (run_id, section), pts in groups.items():
        scores = [p['score'] for p in pts]
        mean   = sum(scores) / len(scores)
        var    = sum((s - mean)**2 for s in scores) / len(scores)
        std    = math.sqrt(var) if var > 0 else 0
        for p in pts:
            z = (p['score'] - mean) / std if std > 0 else 0
            p['z'] = z
            if z < zscore_threshold:
                outliers.append(p)
            else:
                normal[run_id].append(p)

    fig = go.Figure()
    colors  = px.colors.qualitative.Plotly
    run_ids = sorted({r['run_id'] for r in rows})

    for r_idx, run_id in enumerate(run_ids):
        pts = normal[run_id]
        if not pts:
            continue
        cdata = []
        for p in pts:
            title = (f"Run {run_id}  ·  {p['section'].upper()}  "
                     f"·  S{p['session_a']}→S{p['session_b']}  "
                     f"·  score={p['score']:.4f}  Z={p['z']:.2f}")
            text  = texts.get((run_id, p['session_b'], p['section']), '')
            cdata.append([title, text])

        fig.add_trace(go.Scatter(
            x=[f"S{p['session_a']}→{p['session_b']}" for p in pts],
            y=[p['score'] for p in pts],
            mode='markers',
            name=labels.get(run_id, f'Run {run_id}'),
            marker=dict(size=7, color=colors[r_idx % len(colors)], opacity=0.7),
            customdata=cdata,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '<i>click to read text</i><extra></extra>'
            ),
        ))

    if outliers:
        cdata = []
        for p in outliers:
            title = (f"⚠ OUTLIER  Run {p['run_id']}  ·  "
                     f"{p['section'].upper()}  "
                     f"·  S{p['session_a']}→S{p['session_b']}  "
                     f"·  score={p['score']:.4f}  Z={p['z']:.2f}")
            text  = texts.get((p['run_id'], p['session_b'], p['section']), '')
            cdata.append([title, text])

        fig.add_trace(go.Scatter(
            x=[f"S{p['session_a']}→{p['session_b']}" for p in outliers],
            y=[p['score'] for p in outliers],
            mode='markers+text',
            name=f'Outlier Z < {zscore_threshold}',
            marker=dict(size=15, symbol='diamond',
                        color='yellow',
                        line=dict(color='white', width=1)),
            text=[f"R{p['run_id']} {p['section']}" for p in outliers],
            textposition='top center',
            textfont=dict(size=9, color='yellow'),
            customdata=cdata,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '<i>click to read text</i><extra></extra>'
            ),
        ))

    fig.add_hline(y=1.0, line_dash='dot',
                  line_color='rgba(255,255,255,0.15)',
                  annotation_text='perfect similarity')

    fig.update_layout(
        template=TEMPLATE,
        title=dict(
            text=f'Session Transition Scores — Outliers Highlighted '
                 f'(Z < {zscore_threshold})<br>'
                 '<sup>Yellow diamonds = unexpected drops  ·  '
                 'Click any point to read that session\'s text</sup>',
            font=dict(size=18)
        ),
        xaxis_title='Session Transition',
        yaxis_title='Cosine Similarity',
        height=620,
        xaxis=dict(tickangle=45),
    )

    write_html(fig, out_dir / f'outliers_{method}.html')

# ── Chart 4: Section Heatmap ───────────────────────────────────────────────────

def chart_heatmap(con, method: str, out_dir: Path,
                  labels: dict, texts: dict):
    rows = con.execute("""
        SELECT a.section,
               a.run_id        AS run_a,
               b.run_id        AS run_b,
               AVG(sim.score)  AS avg_score
        FROM Similarities sim
        JOIN Sections a ON a.id = sim.section_a
        JOIN Sections b ON b.id = sim.section_b
        WHERE sim.method    = ?
          AND a.run_id     != b.run_id
          AND a.section     = b.section
          AND a.session_num = b.session_num
        GROUP BY a.section, a.run_id, b.run_id
    """, (method,)).fetchall()

    run_ids = sorted({r['run_a'] for r in rows} | {r['run_b'] for r in rows})
    short   = {rid: f'R{rid}' for rid in run_ids}

    fig = make_subplots(
        rows=1, cols=len(SECTIONS),
        subplot_titles=[s.upper() for s in SECTIONS],
        horizontal_spacing=0.04,
    )

    sec_data = defaultdict(dict)
    for r in rows:
        sec_data[r['section']][(r['run_a'], r['run_b'])] = r['avg_score']

    for s_idx, section in enumerate(SECTIONS):
        n      = len(run_ids)
        matrix = [[None]*n for _ in range(n)]
        cdata  = [[None]*n for _ in range(n)]

        for i, ra in enumerate(run_ids):
            for j, rb in enumerate(run_ids):
                if ra == rb:
                    score = 1.0
                    t_a = texts.get((ra, 1, section), '')
                    body  = f'─── Run {ra}, Session 1 (self-comparison) ───\n{t_a}'
                else:
                    score = (sec_data[section].get((ra, rb)) or
                             sec_data[section].get((rb, ra)))
                    t_a = texts.get((ra, 1, section), '')
                    t_b = texts.get((rb, 1, section), '')
                    body = (f'─── Run {ra}, Session 1 ───\n{t_a}\n\n'
                            f'─── Run {rb}, Session 1 ───\n{t_b}')

                matrix[i][j] = score
                title = (f"{section.upper()}  ·  "
                         f"R{ra} vs R{rb}  ·  "
                         f"avg={score:.4f}" if score else
                         f"{section.upper()}  ·  R{ra} vs R{rb}  ·  no data")
                cdata[i][j] = [title, body]

        tick_labels = [short[r] for r in run_ids]

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=tick_labels,
                y=tick_labels,
                colorscale='Viridis',
                zmin=0.3, zmax=1.0,
                showscale=(s_idx == len(SECTIONS) - 1),
                customdata=cdata,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    '<i>click to compare Session 1 texts</i>'
                    '<extra></extra>'
                ),
            ),
            row=1, col=s_idx + 1
        )

    fig.update_layout(
        template=TEMPLATE,
        title=dict(
            text='Cross-Run Similarity Heatmap — Average Score Across All Sessions<br>'
                 '<sup>Brighter = more similar attractor  ·  '
                 'Click any cell to compare Session 1 texts</sup>',
            font=dict(size=18)
        ),
        height=480,
    )

    write_html(fig, out_dir / f'section_heatmap_{method}.html')

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Japa visualization generator.')
    ap.add_argument('--db',     default=DB_DEFAULT)
    ap.add_argument('--method', default='st', choices=['st', 'tfidf'])
    ap.add_argument('--out',    default=None)
    ap.add_argument('--zscore', default=-1.5, type=float)
    args = ap.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out) if args.out else db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f'Database not found: {db_path}')

    con = connect(db_path)
    labels = run_labels(con)
    texts  = section_texts(con)

    print(f'\nGenerating visualizations — method={args.method}')
    print(f'Output: {out_dir}\n')

    chart_convergence(con, args.method, out_dir, labels, texts)
    chart_cross_run(  con, args.method, out_dir,         texts)
    chart_outliers(   con, args.method, out_dir, labels, texts, args.zscore)
    chart_heatmap(    con, args.method, out_dir, labels, texts)

    con.close()
    print('\nDone. Open HTML files in any browser.')

if __name__ == '__main__':
    main()