#!/usr/bin/env python3
"""Figure 1: the SLMJury setup as two mirrored lanes (closed- and open-ended) with their results.

Drawn with matplotlib at print size, in points, from results/analysis; every number is asserted against the
paper. Needs Source Sans Pro and Source Code Pro (TeX Live). Writes results/figures/1_overview.pdf (+ .png).
"""
from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

import style

NAME = 'overview'
FILE = style.file_stem(NAME)     # 1_overview
W = 453.6                        # \textwidth in pt (6.30 in); the canvas unit is the printed point
PDF_META = {'Creator': 'SLMJury', 'Producer': None, 'CreationDate': None}


class ToolMissing(RuntimeError):
    """Source Sans Pro or Source Code Pro is not installed."""


# Fonts: one sans family, monospace for example text, Computer Modern for math
FONT_FILES = ['SourceSansPro-Regular.otf', 'SourceSansPro-Semibold.otf', 'SourceSansPro-RegularIt.otf',
              'SourceSansPro-Bold.otf', 'SourceCodePro-Regular.otf']
MONO = 'Source Code Pro'
RC = {'font.family': 'Source Sans Pro', 'pdf.fonttype': 42, 'mathtext.fontset': 'cm', 'axes.unicode_minus': False}
_FONTS_ADDED = []


def register_fonts():
    """Add the TeX Live font files to matplotlib once; raise ToolMissing if kpsewhich cannot find them."""
    if _FONTS_ADDED:
        return
    kpse = shutil.which('kpsewhich')
    found = {}
    if kpse:
        out = subprocess.run([kpse, *FONT_FILES], capture_output=True, text=True, timeout=20).stdout
        found = {Path(p).name: p for p in out.splitlines() if p.strip()}
    missing = [f for f in FONT_FILES if f not in found]
    if missing:
        raise ToolMissing(f'font files not found with kpsewhich: {", ".join(missing)}')
    for f in FONT_FILES:
        font_manager.fontManager.addfont(found[f])
    # fontTools warns about these files' old header timestamps on every save; the warning is harmless
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    _FONTS_ADDED.append(True)


# Colours (the print figure has no dark variant)
NAVY = '#1F3A5F'
NAVY_PALE = '#E1E8F3'
NAVY_LIGHT = '#BFCFEA'
MUTED = '#4A5A70'
INK = '#1E2733'
FAINT = '#9AA7B8'
OPEN_GREEN = '#5E8A55'
OPEN_PALE = '#E8EFE3'
OPEN_LIGHT = '#C6D7BE'
CHECK = '#5E8F4F'
FAMILY_COLOR = {'Llama': '#B5615C', 'Qwen2.5': '#4E9A8F', 'Qwen3': '#4F75A8',
                'Phi-4': '#C8913F', 'Large': '#7E6BA0'}
FAMILY_SHAPE = {'Llama': 'triangle', 'Qwen2.5': 'square', 'Qwen3': 'circle',
                'Phi-4': 'diamond', 'Large': 'hexagon'}
DOMAIN_COLOR = {'math': '#D9A441', 'sci.': '#7FA36E', 'gen.': '#8E98A8'}
# Shapes look different in weight at the same radius; these even them out.
SHAPE_SCALE = {'triangle': 1.22, 'square': 0.9, 'circle': 1.0, 'diamond': 1.18, 'hexagon': 1.05}


# Data

rows = style.load
D = {}          # every number the figure prints, filled by draw()


half_up = style.half_up


def load_data():
    """All figure numbers, read from the CSVs and asserted against the paper."""
    judges = []
    for r in rows('large_judges', 'small_vs_large.csv'):
        if r['type'] == 'Ladder':
            continue
        judges.append({'key': r['judge'], 'name': r['display'], 'type': r['type'],
                       'family': style.family_of(r['judge']), 'params': float(r['params_b']),
                       'acc10': float(r['acc_t10']) if r['acc_t10'] else None,
                       'best': float(r['best_acc']), 'best_budget': int(r['best_budget']),
                       'rho': float(r['summeval_rho']), 'n': int(r['n_items'])})
    small = [j for j in judges if j['type'] == 'SLM']
    large = [j for j in judges if j['type'] == 'Large']
    assert len(small) == 16 and len(large) == 6
    assert all(j['family'] == 'Large' for j in large)
    counts = {f: sum(j['family'] == f for j in small) for f in ('Llama', 'Qwen2.5', 'Qwen3', 'Phi-4')}
    assert counts == {'Llama': 3, 'Qwen2.5': 3, 'Qwen3': 5, 'Phi-4': 5}, counts
    assert min(j['params'] for j in small) == 0.6 and max(j['params'] for j in small) == 14
    assert min(j['params'] for j in large) == 70 and max(j['params'] for j in large) == 122
    assert all(j['n'] == 64824 for j in judges)

    cost = {(r['model'], int(r['max_tokens'])): float(r['cost_multiple_vs_ref'])
            for r in rows('scaling_cost', 'cost_ratio.csv')}
    for j in judges:
        j['cost'] = cost[(j['key'], j['best_budget'])]

    by_key = {j['key']: j for j in judges}
    phi, qwen = by_key['phi4-14b'], by_key['qwen3.5-122b']
    assert str(half_up(phi['best'])) == '90.16' and phi['best_budget'] == 10
    assert str(half_up(qwen['best'])) == '90.30' and qwen['best_budget'] == 10
    tost = [r for r in rows('significance', 'tost_equivalence.csv')
            if {r['key_a'], r['key_b']} == {'phi4-14b', 'qwen3.5-122b'}]
    assert len(tost) == 1 and tost[0]['tost_equivalent'] == 'True' and tost[0]['tost_margin_pp'] == '0.5'

    # Serving cost of large judges that give a quick verdict, relative to a 14B judge at B=10
    quick_large = [cost[(j['key'], 10)] for j in large if (j['key'], 10) in cost]
    assert cost[('phi4-14b', 10)] == 1.0
    cost_lo, cost_hi = int(half_up(min(quick_large), 0)), int(half_up(max(quick_large), 0))
    assert (cost_lo, cost_hi) == (9, 32)

    summeval_n = {int(r['summeval_n_total']) for r in rows('main_results', 'open_ended.csv')}
    assert summeval_n == {1600}
    mt_items = max(int(r['n_pooled_items']) for r in rows('significance', 'mtbench_common_reference.csv'))
    assert mt_items == 160  # 80 conversations x 2 students
    personas = [c for c in rows('main_results', 'personas.csv')[0]
                if c not in ('judge', 'max_tokens', 'base', 'range', 'best_persona', 'worst_persona')]
    assert len(personas) == 6
    datasets = [c for c in rows('main_results', 'per_dataset.csv')[0] if c not in ('judge', 'max_tokens', 'avg')]
    assert len(datasets) == 8
    juries = rows('main_results', 'juries.csv')
    assert all(len(r['panel'].split('+')) == 3 for r in juries)

    return {'judges': judges, 'small': small, 'large': large, 'phi': phi, 'qwen': qwen,
            'cost_lo': cost_lo, 'cost_hi': cost_hi, 'n_items': 64824, 'summeval_n': 1600,
            'mtbench_n': 80, 'n_personas': len(personas), 'n_datasets': len(datasets),
            'jury_size': 3, 'phi_acc': str(half_up(phi['best'])), 'qwen_acc': str(half_up(qwen['best']))}


# Drawing helpers

def canvas(height):
    fig = plt.figure(figsize=(W / 72, height / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def glyph(ax, x, y, family, r=3.0, filled=True, lw=0.75, z=5, edge=None):
    """Family marker: shape and colour by family, filled for B=10, open for B=8,192 only."""
    shape = FAMILY_SHAPE[family]
    col = FAMILY_COLOR[family]
    r = r * SHAPE_SCALE[shape]
    face = col if filled else 'white'
    kw = dict(facecolor=face, edgecolor=edge or col, lw=lw, zorder=z)
    if shape == 'circle':
        p = Circle((x, y), r, **kw)
    else:
        n, rot = {'triangle': (3, 90), 'square': (4, 45), 'diamond': (4, 90), 'hexagon': (6, 90)}[shape]
        pts = [(x + r * math.cos(math.radians(rot + 360 * k / n)),
                y + r * math.sin(math.radians(rot + 360 * k / n)) - (0.25 * r if shape == 'triangle' else 0))
               for k in range(n)]
        p = Polygon(pts, closed=True, **kw)
    ax.add_patch(p)
    return p


def size_radius(params, lo=2.0, hi=3.5):
    """Marker radius grows with log parameter count (0.6B to 122B)."""
    t = (math.log10(params) - math.log10(0.6)) / (math.log10(122) - math.log10(0.6))
    return lo + (hi - lo) * t


def box(ax, x, y, w, h, fc='white', ec=NAVY, lw=0.7, r=3.0, ls='-', z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def halo(ax, x, y, r_core, color=NAVY, z=2):
    """Soft concentric rings marking where the figure starts and ends."""
    for k, a in ((2.0, 0.10), (1.55, 0.18)):
        ax.add_patch(Circle((x, y), r_core * k, fc=color, ec='none', alpha=a, zorder=z))


def line(ax, pts, color=NAVY, lw=0.8, ls='-', z=3, arrow=True, head=3.2, alpha=1.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='butt', alpha=alpha)
    if arrow:
        arrowhead(ax, pts[-2], pts[-1], color, head, z, alpha)


def arrowhead(ax, p0, p1, color, size=3.2, z=3, alpha=1.0):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d = math.hypot(dx, dy) or 1
    ux, uy = dx / d, dy / d
    tip = p1
    base = (tip[0] - ux * size, tip[1] - uy * size)
    half = size * 0.48
    pts = [tip, (base[0] - uy * half, base[1] + ux * half), (base[0] + uy * half, base[1] - ux * half)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec='none', zorder=z + 0.1, alpha=alpha))


def curve(ax, p0, p1, color=NAVY, lw=0.8, z=3, arrow=True, ls='-', bend=0.5, head=3.2):
    """Horizontal S-curve from p0 to p1 (cubic Bezier)."""
    dx = (p1[0] - p0[0]) * bend
    verts = [p0, (p0[0] + dx, p0[1]), (p1[0] - dx, p1[1]), p1]
    path = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fc='none', ec=color, lw=lw, ls=ls, zorder=z, capstyle='butt'))
    if arrow:
        arrowhead(ax, (p1[0] - 1, p1[1]), p1, color, head, z)


def wave(ax, x0, x1, y, amp=1.6, period=4.2, color=NAVY, lw=0.8, z=3):
    """A small sine trace that stands for a long reasoning chain."""
    n = 200
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
    ys = [y + amp * math.sin(2 * math.pi * (x - x0) / period) for x in xs]
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle='round')


def check_mark(ax, x, y, s=3.0, color=CHECK, lw=1.0, z=6):
    ax.plot([x - s, x - s * 0.3, x + s], [y, y - s * 0.75, y + s * 0.95], color=color, lw=lw,
            zorder=z, solid_capstyle='round', solid_joinstyle='round')


# Layout grid (pt). Every element below is placed from these values.
H = 184
MID = H / 2                                  # mirror axis


def mirror(y):
    return 2 * MID - y


LANE_X0, LANE_X1 = 28, 338
START_X = 14                                 # start node; its halo clears the canvas edge and the lanes
LANE_TOP = (96, 180)                         # closed lane; the open lane is its mirror
C_Y = 124                                    # closed flow line
O_Y = mirror(C_Y)                            # open flow line
TITLE_Y = 172                                # lane title centre line (closed), mirrored for open
CARD_X0, CARD_X1, CARD_HALF = 34, 104, 21
INNER_X0, INNER_X1 = CARD_X0 + 2.5, CARD_X1 - 2.5
CHIP_Y, CHIP_LABEL_Y = 160, 151              # dataset row (closed), mirrored for open
ROW_DY = 11.5                                # card row spacing
SYM_X, VAL_X = 39.5, 48
POOL_X0, POOL_X1 = 112, 202
FORK_X, TRACK_X0, TRACK_X1, TRACK_DY = 207, 214, 230, 12
COL_X0, COL_W = 239, 41                      # verdict column; the score dots share its centre
COL_CX = COL_X0 + COL_W / 2
NODE_X, NODE_R = 293, 5.5                    # oracle and rho nodes
PANEL_X0, PANEL_X1 = 344, 452
PANEL_CX = (PANEL_X0 + PANEL_X1) / 2
CAPTION_DY = (6.5, 14.5)                     # caption lines above a result box's bottom edge
HEADER_DY = 8.0                              # first header line below a result box's top edge
FAMILY_ORDER = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4']
STRAT_W, STRAT_GAP, STRAT_Y0, STRAT_Y1 = 37, 4, 151, 176   # strategy boxes (closed lane)
WAVE_AMP, WAVE_PERIOD = 1.7, 3.3              # the reasoning (B=8,192) segment in both lanes
TIP_ANGLE = 32                               # where the reference arrows meet rho, either side of straight down
BAND_LABEL_Y = 9.5                          # labels along the open lane's bottom band
PERSONA_COLORS = ['#C8913F', '#B5615C', '#8E98A8', '#7FA36E', '#4E9A8F', '#7E6BA0']

REG = {}        # anchors as drawn, for the alignment check
LABELS = []     # (text, x, centre y, ha) as placed, for the rendered-text check


def mark(name, **kw):
    REG[name] = kw


def t(ax, x, y, s, size=7, color=INK, ha='center', weight='normal', family=None, z=6, check=True):
    """Text whose cap height is centred on y, so labels on one line share a baseline."""
    kw = {'family': family} if family else {}
    ax.text(x, y - 0.34 * size, s, fontsize=size, color=color, ha=ha, va='baseline', fontweight=weight,
            zorder=z, **kw)
    if check and '$' not in s:
        LABELS.append((s, x, y, ha, size))


def lanes(ax):
    y0, y1 = LANE_TOP
    box(ax, LANE_X0, y0, LANE_X1 - LANE_X0, y1 - y0, fc=NAVY_PALE, ec=NAVY_LIGHT, lw=0.7, ls=(0, (3, 2)), r=5, z=0)
    box(ax, LANE_X0, mirror(y1), LANE_X1 - LANE_X0, y1 - y0, fc=OPEN_PALE, ec=OPEN_LIGHT, lw=0.7, ls=(0, (3, 2)),
        r=5, z=0)
    mark('lane_closed', y0=y0, y1=y1)
    mark('lane_open', y0=mirror(y1), y1=mirror(y0))
    t(ax, CARD_X0, TITLE_Y, 'closed-ended', size=7.5, color=NAVY, ha='left', weight='semibold')
    t(ax, CARD_X0, mirror(TITLE_Y), 'open-ended', size=7.5, color=OPEN_GREEN, ha='left', weight='semibold')
    mark('title_closed', x=CARD_X0, y=TITLE_Y)
    mark('title_open', x=CARD_X0, y=mirror(TITLE_Y))


def start_node(ax):
    halo(ax, START_X, MID, 6.5)
    ax.add_patch(Circle((START_X, MID), 6.5, fc=NAVY, ec='none', zorder=4))
    ax.text(START_X, MID, r'$\mathcal{D}$', fontsize=8.5, color='white', ha='center', va='center_baseline', zorder=5)
    curve(ax, (START_X + 6.5, MID + 1.5), (CARD_X0, C_Y), color=NAVY, lw=0.9, bend=0.6)
    curve(ax, (START_X + 6.5, MID - 1.5), (CARD_X0, O_Y), color=OPEN_GREEN, lw=0.9, bend=0.6)
    mark('start_to_closed', x=CARD_X0, y=C_Y)
    mark('start_to_open', x=CARD_X0, y=O_Y)


def card(ax, cy, edge):
    box(ax, CARD_X0, cy - CARD_HALF, CARD_X1 - CARD_X0, 2 * CARD_HALF, fc='white', ec=edge, lw=0.6, r=2.5, z=2)
    line(ax, [(CARD_X1, cy), (POOL_X0, cy)], color=NAVY if edge == NAVY_LIGHT else OPEN_GREEN, lw=0.9, head=3.0)


def closed_inputs(ax):
    # Eight benchmark chips span the card's inner width: equal pitch inside groups, equal gaps between them
    size, pitch = 5.6, 7.0
    groups = (('math', 3), ('sci.', 2), ('gen.', 3))
    gap = (CARD_X1 - CARD_X0 - 8 * size - 5 * (pitch - size)) / 2
    x = CARD_X0
    xs = []
    for dom, n in groups:
        x0 = x
        for _ in range(n):
            box(ax, x, CHIP_Y - size / 2, size, size, fc=DOMAIN_COLOR[dom], ec='none', r=1.0, z=3)
            xs.append(x)
            x += pitch
        t(ax, (x0 + x - pitch + size) / 2, CHIP_LABEL_Y, dom, color=MUTED)
        x += gap - (pitch - size)
    mark('chips', xs=xs, right=xs[-1] + size)
    card(ax, C_Y, NAVY_LIGHT)
    mark('card_closed', y0=C_Y - CARD_HALF, y1=C_Y + CARD_HALF)
    for dy, sym, val, col in ((ROW_DY, 'q', '2x+6=14', INK), (0, 'g', 'x=4', INK),
                              (-ROW_DY, 'r', '…so x=10', '#A0524D')):
        t(ax, SYM_X, C_Y + dy, f'${sym}$', size=8, color=MUTED, ha='left')
        t(ax, VAL_X, C_Y + dy, val, color=col, ha='left', family=MONO)


def open_inputs(ax):
    for x, ha, name, n in ((CARD_X0, 'left', 'SummEval', f"{D['summeval_n']:,}"),
                           (CARD_X1, 'right', 'MT-Bench', str(D['mtbench_n']))):
        t(ax, x, mirror(CHIP_LABEL_Y), name, ha=ha)
        t(ax, x, mirror(CHIP_Y), n, color=MUTED, ha=ha)
    mark('datasets_open', name_y=mirror(CHIP_LABEL_Y), count_y=mirror(CHIP_Y))
    card(ax, O_Y, OPEN_LIGHT)
    mark('card_open', y0=O_Y - CARD_HALF, y1=O_Y + CARD_HALF)
    # Source article drawn as text lines, its summary, and the human score
    t(ax, SYM_X, O_Y + ROW_DY, '$x$', size=8, color=MUTED, ha='left')
    for w, dy in ((INNER_X1 - VAL_X - 2, 2.0), (INNER_X1 - VAL_X - 12, -2.0)):
        ax.add_patch(Rectangle((VAL_X, O_Y + ROW_DY + dy - 0.75), w, 1.5, fc='#B9C2CE', ec='none', zorder=4))
    t(ax, SYM_X, O_Y, '$y$', size=8, color=MUTED, ha='left')
    t(ax, VAL_X, O_Y, 'Port closed.', ha='left', family=MONO)
    t(ax, SYM_X, O_Y - ROW_DY, '$h$', size=8, color=MUTED, ha='left')
    r = 1.9
    for k in range(5):
        ax.add_patch(Circle((VAL_X + r + 6.2 * k, O_Y - ROW_DY), r, fc=OPEN_GREEN if k < 4 else 'white',
                            ec=OPEN_GREEN, lw=0.6, zorder=4))
    mark('human_dots', left=VAL_X)


def judge_pool(ax):
    # Content lines, centre to centre, then centred on the mirror axis
    steps = [('title', 0), ('sep1', 7), ('small', 7), ('Llama', 11), ('Qwen2.5', 11), ('Qwen3', 11),
             ('Phi-4', 11), ('sep2', 8.5), ('large', 7), ('hex', 11), ('sep3', 8.5), ('key1', 7), ('key2', 10)]
    total = sum(s for _, s in steps)
    y, Y = MID + total / 2, {}
    for name, s in steps:
        y -= s
        Y[name] = y
    Y = {k: v + steps[0][1] for k, v in Y.items()}
    pad = 9
    y0, y1 = Y['key2'] - pad, Y['title'] + pad
    box(ax, POOL_X0, y0, POOL_X1 - POOL_X0, y1 - y0, fc='white', ec=NAVY, lw=0.9, r=3.5, z=3)
    mark('pool', y0=y0, y1=y1)
    xl, xr = POOL_X0 + 5, POOL_X1 - 5
    t(ax, (POOL_X0 + POOL_X1) / 2, Y['title'], r'judge pool $\mathcal{J}$', size=7.5, color=NAVY, weight='semibold')
    for s in ('sep1', 'sep2', 'sep3'):
        ax.plot([xl - 1, xr + 1], [Y[s]] * 2, color=NAVY_LIGHT, lw=0.6, zorder=5)
    t(ax, xl, Y['small'], f"{len(D['small'])} small", size=7.5, color=NAVY, weight='semibold', ha='left')
    t(ax, xr, Y['small'], '0.6–14B', color=MUTED, ha='right')
    t(ax, xl, Y['large'], f"{len(D['large'])} large", size=7.5, color=NAVY, weight='semibold', ha='left')
    t(ax, xr, Y['large'], '70–122B', color=MUTED, ha='right')
    pitch, last_x = 9.6, xr - 3.5
    marks, pitches = {}, []
    for fam in FAMILY_ORDER:
        t(ax, xl, Y[fam], fam, ha='left')
        members = sorted([j for j in D['small'] if j['family'] == fam], key=lambda j: j['params'])
        xs = [last_x - pitch * (len(members) - 1 - k) for k in range(len(members))]
        for gx, j in zip(xs, members):
            glyph(ax, gx, Y[fam], fam, r=size_radius(j['params']), filled=j['best_budget'] == 10, z=6)
            marks[j['key']] = (gx, Y[fam])
        pitches.append(xs)
    larges = sorted(D['large'], key=lambda j: j['params'])
    xs = [last_x - pitch * (len(larges) - 1 - k) for k in range(len(larges))]
    for gx, j in zip(xs, larges):
        glyph(ax, gx, Y['hex'], 'Large', r=size_radius(j['params']), filled=j['best_budget'] == 10, z=6)
        marks[j['key']] = (gx, Y['hex'])
    pitches.append(xs)
    mark('pool_rows', rows=pitches)
    for key in ('phi4-14b', 'qwen3.5-122b'):
        ax.add_patch(Circle(marks[key], 4.7, fc='none', ec=NAVY, lw=0.6, zorder=7))
    for key, filled, label in (('key1', True, r'best at $B=10$'), ('key2', False, r'best at $B=8{,}192$')):
        ax.add_patch(Circle((xl + 2.3, Y[key]), 2.3, fc=MUTED if filled else 'white', ec=MUTED, lw=0.7, zorder=6))
        t(ax, xl + 7.5, Y[key], label, color=MUTED, ha='left')
    return Y


def persona_icon(ax, x, y):
    for k, col in enumerate(PERSONA_COLORS):
        ax.add_patch(Circle((x - 4 + 4 * (k % 3), y + 2 - 4 * (k // 3)), 1.45, fc=col, ec='none', zorder=5))


def jury_icon(ax, x, y):
    for dy in (4, 0, -4):
        ax.plot([x - 4, x + 4], [y + dy, y], color=NAVY_LIGHT, lw=0.6, zorder=4)
        ax.add_patch(Circle((x - 4, y + dy), 1.4, fc=NAVY, ec='none', zorder=5))
    ax.add_patch(Circle((x + 4, y), 1.8, fc=CHECK, ec='none', zorder=5))


def debate_icon(ax, x, y):
    y -= (3.9 + 1.4 - (3.9 * 0.5 + 1.4)) / 2          # nodes span +5.3 to -3.35; centre that span on y
    ax.add_patch(Circle((x, y), 3.9, fc='none', ec=NAVY_LIGHT, lw=0.6, zorder=4))
    for a in (90, 210, 330):
        ax.add_patch(Circle((x + 3.9 * math.cos(math.radians(a)), y + 3.9 * math.sin(math.radians(a))), 1.4,
                            fc=NAVY, ec='none', zorder=5))


def strategies(ax, pool_y):
    # Optional small-judge variants on a dashed branch: equal boxes, equal gaps, centred between pool and lane edge
    items = ((f"{D['n_personas']} personas", persona_icon), (f"jury of {D['jury_size']}", jury_icon),
             ('debate', debate_icon))
    w, gap, yb, yt = STRAT_W, STRAT_GAP, STRAT_Y0, STRAT_Y1
    total = 3 * w + 2 * gap
    x = (POOL_X1 + LANE_X1) / 2 - total / 2
    ax.plot([POOL_X1, POOL_X1 + 3, POOL_X1 + 3, x], [pool_y, pool_y, (yb + yt) / 2, (yb + yt) / 2],
            color=FAINT, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
    xs = []
    for label, icon in items:
        box(ax, x, yb, w, yt - yb, fc='white', ec=FAINT, lw=0.6, ls=(0, (2, 1.5)), r=2.5, z=3)
        icon(ax, x + w / 2, yb + 17.5)
        t(ax, x + w / 2, yb + 6.5, label, color=MUTED)
        xs.append(x)
        x += w + gap
    mark('strategies', xs=xs, w=w, left=xs[0], right=xs[-1] + w)


def budget_fork(ax):
    ax.plot([POOL_X1, FORK_X], [C_Y, C_Y], color=NAVY, lw=0.9, zorder=3)
    ax.add_patch(Circle((FORK_X, C_Y), 1.6, fc=NAVY, ec='none', zorder=4))
    up, dn = C_Y + TRACK_DY, C_Y - TRACK_DY
    curve(ax, (FORK_X, C_Y), (TRACK_X0, up), color=NAVY, lw=0.9, arrow=False, bend=0.6)
    ax.plot([TRACK_X0, TRACK_X1], [up, up], color=NAVY, lw=0.9, zorder=3)
    curve(ax, (FORK_X, C_Y), (TRACK_X0, dn), color=NAVY, lw=0.9, arrow=False, bend=0.6)
    wave(ax, TRACK_X0, TRACK_X1, dn, amp=WAVE_AMP, period=WAVE_PERIOD, color=NAVY, lw=0.75)
    mid = (TRACK_X0 + TRACK_X1) / 2
    t(ax, mid, up + 7, r'$B=10$', size=7.5, color=NAVY)
    t(ax, mid, dn - 7, r'$B=8{,}192$', size=7.5, color=NAVY)
    curve(ax, (TRACK_X1, up), (COL_X0, C_Y), color=NAVY, lw=0.9, bend=0.6, arrow=False)
    curve(ax, (TRACK_X1, dn), (COL_X0, C_Y), color=NAVY, lw=0.9, bend=0.6, head=2.8)
    mark('wave_closed', x0=TRACK_X0, x1=TRACK_X1, y=dn, amp=WAVE_AMP, period=WAVE_PERIOD)


def verdicts(ax):
    for dy, word in ((12, 'Correct'), (0, 'Incorrect'), (-12, 'Undefined')):
        hit = word == 'Incorrect'
        box(ax, COL_X0, C_Y + dy - 4.6, COL_W, 9.2, fc=NAVY if hit else 'white', ec=NAVY if hit else NAVY_LIGHT,
            lw=0.6, r=1.8, z=3)
        t(ax, COL_CX, C_Y + dy, word, color='white' if hit else MUTED, family=MONO)
    mark('col_closed', cx=COL_CX, ys=[C_Y + 12, C_Y, C_Y - 12])
    ax.plot([COL_X0 + COL_W, NODE_X - NODE_R], [C_Y, C_Y], color=NAVY, lw=0.9, zorder=3)
    ax.add_patch(Circle((NODE_X, C_Y), NODE_R, fc='white', ec=CHECK, lw=0.9, zorder=4))
    check_mark(ax, NODE_X, C_Y - 0.3, s=2.6, color=CHECK, lw=1.0)
    t(ax, NODE_X, C_Y + 11, 'oracle', color=MUTED)
    t(ax, NODE_X, C_Y - 11, '10≠4', color=MUTED, family=MONO)
    mark('node_closed', x=NODE_X, y=C_Y, r=NODE_R)
    line(ax, [(NODE_X + NODE_R, C_Y), (PANEL_X0, C_Y)], color=NAVY, lw=0.9, head=3.0)
    mark('arrow_closed', y=C_Y)


def rating_card(ax, x0, y_top, edge):
    """A small card: a title line and two 1-5 rating rows with one mark filled in each."""
    w, h = CARD_W, CARD_H
    box(ax, x0, y_top - h, w, h, fc='white', ec=edge, lw=0.7, r=1.6, z=6)
    ax.plot([x0 + 2.4, x0 + w - 4.2], [y_top - 3.6] * 2, color='#B9C2CE', lw=1.2, zorder=7, solid_capstyle='round')
    for dy, filled in ((8.5, 4), (13.2, 3)):
        for k in range(5):
            ax.add_patch(Circle((x0 + w / 2 + MARK_PITCH * (k - 2), y_top - dy), MARK_R,
                                fc=OPEN_GREEN if k == filled - 1 else 'white', ec=edge, lw=0.45, zorder=8))


# Reference illustrations: a figure plus a rating card each, the cards facing; shared sizes and top line
FIG_H, CARD_W, CARD_H, GAP = 24, 14, 17, 3
FIG_W = FIG_H / 2 * math.sqrt(3)               # width of the pointy-top hexagon of height FIG_H
SCENE_W = FIG_W + GAP + CARD_W
MARK_R, MARK_PITCH = 1.05, 2.5


def human_scene(ax, x0, yb):
    """A rater (left) and the rating sheet they fill in (right)."""
    fx = x0 + FIG_W / 2
    head_r = 4.0
    ax.add_patch(Circle((fx, yb + FIG_H - head_r), head_r, fc=MUTED, ec='none', zorder=5))
    shoulder_h = FIG_H - 2 * head_r - 1.6
    torso = Ellipse((fx, yb), FIG_W, 2 * shoulder_h, fc=MUTED, ec='none', zorder=5)
    ax.add_patch(torso)
    torso.set_clip_path(Rectangle((fx - FIG_W, yb), 2 * FIG_W, FIG_H, transform=ax.transData))
    cx0 = x0 + FIG_W + GAP
    rating_card(ax, cx0, yb + FIG_H, MUTED)
    return {'left': x0, 'right': x0 + SCENE_W, 'bottom': yb, 'fig_x': fx, 'card_x': cx0 + CARD_W / 2,
            'card_top': yb + FIG_H}


def llm_scene(ax, x0, yb):
    """The large-model glyph (right) writing its rating into a speech bubble (left)."""
    fx = x0 + CARD_W + GAP + FIG_W / 2
    r = FIG_H / 2
    hy = yb + r
    purple = FAMILY_COLOR['Large']
    pts = [(fx + r * math.cos(math.radians(90 + 60 * k)), hy + r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    ax.add_patch(Polygon(pts, closed=True, fc=purple, ec=purple, lw=0.6, zorder=5))
    for dy, w in ((3.4, 9.0), (0, 6.0), (-3.4, 7.8)):
        ax.plot([fx - 4.5, fx - 4.5 + w], [hy + dy] * 2, color='white', lw=1.1, zorder=6, solid_capstyle='round')
    rating_card(ax, x0, yb + FIG_H, purple)
    # Speech-bubble tail from the card's right edge to the model; its fill hides the card edge it crosses
    ex = x0 + CARD_W
    tail = [(ex - 0.6, yb + 13.0), (ex - 0.6, yb + 9.5), (fx - FIG_W / 2 + 0.2, yb + 9.0)]
    ax.add_patch(Polygon(tail, closed=True, fc='white', ec='none', zorder=7))
    ax.plot([ex, tail[2][0], ex], [tail[0][1], tail[2][1], tail[1][1]], color=purple, lw=0.7, zorder=7)
    return {'left': x0, 'right': x0 + SCENE_W, 'bottom': yb, 'fig_x': fx, 'card_x': x0 + CARD_W / 2,
            'card_top': yb + FIG_H}


def scoring(ax):
    # Open-ended judges only reason (B=8,192): a straight path with the closed lane's wave drawn inline
    ax.plot([POOL_X1, TRACK_X0], [O_Y, O_Y], color=OPEN_GREEN, lw=0.9, zorder=3)
    wave(ax, TRACK_X0, TRACK_X1, O_Y, amp=WAVE_AMP, period=WAVE_PERIOD, color=OPEN_GREEN, lw=0.75)
    t(ax, (TRACK_X0 + TRACK_X1) / 2, O_Y - 7, r'$B=8{,}192$', size=7.5, color=OPEN_GREEN)
    mark('wave_open', x0=TRACK_X0, x1=TRACK_X1, y=O_Y, amp=WAVE_AMP, period=WAVE_PERIOD, label_dy=7)
    # Score 1-5 as five dots on the flow line, centred on the verdict column; the judge gives 4
    dot_r, pitch = 2.4, 7.0
    xs = [COL_CX + pitch * (k - 2) for k in range(5)]
    line(ax, [(TRACK_X1, O_Y), (xs[0] - dot_r, O_Y)], color=OPEN_GREEN, lw=0.9, head=2.8)
    mark('open_path', ys=[O_Y, O_Y, O_Y], x_end=xs[0] - dot_r)
    for score, x in zip((1, 2, 3, 4, 5), xs):
        ax.add_patch(Circle((x, O_Y), dot_r, fc=OPEN_GREEN if score == 4 else 'white', ec=OPEN_GREEN,
                            lw=0.7, zorder=4))
    t(ax, xs[0], O_Y + 8.5, '1', color=MUTED)
    t(ax, xs[-1], O_Y + 8.5, '5', color=MUTED)
    mark('dots_open', xs=xs, y=O_Y)
    line(ax, [(xs[-1] + dot_r, O_Y), (NODE_X - NODE_R, O_Y)], color=OPEN_GREEN, lw=0.9, head=2.8)
    ax.add_patch(Circle((NODE_X, O_Y), NODE_R, fc='white', ec=OPEN_GREEN, lw=0.9, zorder=4))
    ax.text(NODE_X, O_Y, r'$\rho$', fontsize=8.5, color=OPEN_GREEN, ha='center', va='center_baseline', zorder=5)
    mark('node_open', x=NODE_X, y=O_Y, r=NODE_R)
    # Reference scores (human for SummEval, an LLM for MT-Bench), spread evenly between pool and lane edge
    label_y = BAND_LABEL_Y
    yb = label_y + 6.5
    space = (LANE_X1 - POOL_X1 - 2 * SCENE_W) / 3
    scenes = []
    x0 = POOL_X1 + space
    for scene, lab, side in ((human_scene, 'human', -1), (llm_scene, 'LLM reference', 1)):
        g = scene(ax, x0, yb)
        t(ax, x0 + SCENE_W / 2, label_y, lab, color=MUTED)
        scenes.append(g)
        x0 += SCENE_W + space
    # Orthogonal connectors: a riser from each card's top centre to one bus, then a single entry into rho
    gap0 = 1.2
    bus_y = (scenes[0]['card_top'] + gap0 + O_Y - NODE_R) / 2
    risers = [((g['card_x'], g['card_top'] + gap0), (g['card_x'], bus_y)) for g in scenes]
    for a_pt, b_pt in risers:
        ax.plot([a_pt[0], b_pt[0]], [a_pt[1], b_pt[1]], color=MUTED, lw=0.6, zorder=3, solid_capstyle='butt')
    bus_x0, bus_x1 = min(r[0][0] for r in risers), max(NODE_X, max(r[0][0] for r in risers))
    ax.plot([bus_x0, bus_x1], [bus_y, bus_y], color=MUTED, lw=0.6, zorder=3, solid_capstyle='projecting')
    entry = ((NODE_X, bus_y), (NODE_X, O_Y - NODE_R))
    line(ax, list(entry), color=MUTED, lw=0.6, head=2.4)
    mark('scenes', scenes=scenes, space=space, label_y=label_y)
    mark('connectors', risers=risers, bus_y=bus_y, bus=(bus_x0, bus_x1), entry=entry)
    line(ax, [(NODE_X + NODE_R, O_Y), (PANEL_X0, O_Y)], color=OPEN_GREEN, lw=0.9, head=3.0)
    mark('arrow_open', y=O_Y)


def result_box(ax, top, edge):
    y0 = top - (LANE_TOP[1] - LANE_TOP[0])
    box(ax, PANEL_X0, y0, PANEL_X1 - PANEL_X0, top - y0, fc='white', ec=edge, lw=0.9, r=3.5, z=1)
    return y0


def caption(ax, y0, lines):
    for dy, s in zip(CAPTION_DY[::-1], lines):
        t(ax, PANEL_CX, y0 + dy, s)


ACC_LO, ACC_HI = 87, 90.5


def closed_result(ax):
    top = LANE_TOP[1]
    y0 = result_box(ax, top, NAVY)
    mark('box_closed', y0=y0, y1=top)
    caption(ax, y0, ['Phi-4 equivalent to the best LLM', 'within ±0.5 points'])
    head_y = top - HEADER_DY
    bin_y, lo_y, hi_y = y0 + 22, y0 + 29.5, top - 24

    def acc_y(a):
        return lo_y + (a - ACC_LO) / (ACC_HI - ACC_LO) * (hi_y - lo_y)
    ax_x = 360
    ax.plot([ax_x, ax_x], [lo_y, hi_y], color=MUTED, lw=0.6, zorder=2)
    for v in (88, 90):
        ax.plot([ax_x - 2, ax_x], [acc_y(v)] * 2, color=MUTED, lw=0.6, zorder=2)
        t(ax, ax_x - 3.5, acc_y(v), str(v), color=MUTED, ha='right')
    for d in (2.8, 4.2):
        ax.plot([ax_x - 2, ax_x + 2], [lo_y - d - 0.7, lo_y - d + 0.7], color=MUTED, lw=0.6)
    t(ax, ax_x - 3.5, bin_y, f'<{ACC_LO}', color=MUTED, ha='right')
    cols = {'SLM': 380, 'Large': 423}
    fixed = {'phi4-14b': (cols['SLM'], acc_y(D['phi']['best'])), 'qwen3.5-122b': (cols['Large'], acc_y(D['qwen']['best']))}
    (px, py), (qx, qy) = fixed['phi4-14b'], fixed['qwen3.5-122b']
    placed = list(fixed.values())
    for kind, away in (('SLM', -1), ('Large', 1)):
        js = sorted([j for j in D['judges'] if j['type'] == kind], key=lambda j: -j['best'])
        low = [j for j in js if j['best'] < ACC_LO]
        for j in js:
            if j['key'] in fixed:
                gx, gy = fixed[j['key']]
            elif j['best'] >= ACC_LO:
                gy = acc_y(j['best'])
                # Dodge sideways, never into the space between the pair and the end node
                for off in (0, away * 6, -away * 6, away * 12, -away * 12, away * 18):
                    gx = cols[kind] + off
                    between = px < gx < qx and gy > acc_y(89.2)
                    if not between and all((gx - a) ** 2 + (gy - b) ** 2 > 5.6 ** 2 for a, b in placed):
                        break
                else:
                    raise RuntimeError(f"no free spot for {j['key']}")
                placed.append((gx, gy))
            else:
                k = low.index(j)
                gx, gy = (ax_x + 7 + 5.8 * k if kind == 'SLM' else cols[kind]), bin_y
            glyph(ax, gx, gy, j['family'], r=2.1, filled=j['best_budget'] == 10, z=5)
    # End node: Phi-4 and Qwen3.5-122B, equivalent within +/-0.5 points
    ex, ey = (px + qx) / 2, (py + qy) / 2
    halo(ax, ex, ey, 4.2, z=1.5)
    ax.plot([px + 3, ex - 4.2], [py, ey], color=NAVY, lw=0.7, zorder=3)
    ax.plot([ex + 4.2, qx - 3], [ey, qy], color=NAVY, lw=0.7, zorder=3)
    ax.add_patch(Circle((ex, ey), 4.2, fc=NAVY, ec='none', zorder=4))
    ax.text(ex, ey, r'$\approx$', fontsize=8.5, color='white', ha='center', va='center_baseline', zorder=5)
    for cx, name, acc in ((px, 'Phi-4', D['phi_acc']), (qx, 'Qwen3.5-122B', D['qwen_acc'])):
        t(ax, cx, head_y, name, size=7.5, weight='semibold')
        t(ax, cx, head_y - 8.5, f'{acc}%', size=7.5)
    mark('closed_header', y=head_y, xs=[px, qx], cols=[cols['SLM'], cols['Large']])
    mark('ladder', block=(ax_x - 12, cols['Large'] + 13.2 + 2.7))


def open_result(ax):
    top = mirror(LANE_TOP[0])
    y0 = result_box(ax, top, OPEN_GREEN)
    mark('box_open', y0=y0, y1=top)
    pairs = rows('significance', 'summeval_paired.csv')
    slms = list(dict.fromkeys(r['slm_key'] for r in pairs))
    larges = list(dict.fromkeys(r['large_key'] for r in pairs))
    assert len(slms) == 5 and len(larges) == 6 and len(pairs) == 30
    wins = sum(r['verdict'] in ('SLM better', 'not distinguishable') for r in pairs)
    assert wins == 25
    caption(ax, y0, ['SLMs match or beat LLMs', f'in {wins} of {len(pairs)} pairs'])
    t(ax, PANEL_CX, top - HEADER_DY, r'SummEval $\rho$', size=7.5, color=OPEN_GREEN, weight='semibold')
    by_key = {j['key']: j for j in D['judges']}
    c, row_gap = 6.6, 6.0
    block_w = row_gap + 6 * c                         # row glyph column plus the grid
    gx0 = PANEL_CX - block_w / 2 + row_gap
    # Grid block (column glyphs included) centred between the caption and the title
    block_h = 5 * c + 3.4 + 1.7
    gy0 = (y0 + CAPTION_DY[1] + 2.5 + top - HEADER_DY - 2.6) / 2 - block_h / 2
    for k, lk in enumerate(larges):
        glyph(ax, gx0 + k * c + c / 2, gy0 + 5 * c + 3.4, 'Large', r=1.7, filled=by_key[lk]['best_budget'] == 10,
              lw=0.55)
    for i, sk in enumerate(slms):
        y = gy0 + (4 - i) * c + c / 2
        glyph(ax, gx0 - row_gap + 1.9, y, by_key[sk]['family'], r=1.9, filled=by_key[sk]['best_budget'] == 10,
              lw=0.6)
        for k, lk in enumerate(larges):
            v = next(r['verdict'] for r in pairs if r['slm_key'] == sk and r['large_key'] == lk)
            fc = {'SLM better': OPEN_GREEN, 'not distinguishable': OPEN_LIGHT, 'large better': 'white'}[v]
            ec = FAMILY_COLOR['Large'] if v == 'large better' else 'white'
            box(ax, gx0 + k * c + 0.45, gy0 + (4 - i) * c + 0.45, c - 0.9, c - 0.9, fc=fc, ec=ec, lw=0.6, r=0.6, z=4)
    mark('grid', left=gx0 - row_gap, right=gx0 + 6 * c, y0=gy0, y1=gy0 + block_h,
         room=(y0 + CAPTION_DY[1] + 2.5, top - HEADER_DY - 2.6))


# Alignment check

def check_alignment(page, tol=0.5):
    """Alignment problems from the drawn anchors and, given the PDF page, from the rendered labels.
    Returns (geometric relations, labels checked, problems)."""
    R = REG
    eq = [  # (description, a, b) that must be equal
        ('lanes mirror: closed top vs open bottom', R['lane_closed']['y1'], mirror(R['lane_open']['y0'])),
        ('lanes mirror: closed bottom vs open top', R['lane_closed']['y0'], mirror(R['lane_open']['y1'])),
        ('titles mirror', R['title_closed']['y'], mirror(R['title_open']['y'])),
        ('titles share left edge', R['title_closed']['x'], R['title_open']['x']),
        ('cards mirror (top/bottom)', R['card_closed']['y1'], mirror(R['card_open']['y0'])),
        ('dataset row: chips vs counts', CHIP_Y, mirror(R['datasets_open']['count_y'])),
        ('dataset row: chip labels vs names', CHIP_LABEL_Y, mirror(R['datasets_open']['name_y'])),
        ('chips span the card width (left)', R['chips']['xs'][0], CARD_X0),
        ('chips span the card width (right)', R['chips']['right'], CARD_X1),
        ('title, chips and card share a left edge', R['title_closed']['x'], CARD_X0),
        ('start curves mirror', R['start_to_closed']['y'], mirror(R['start_to_open']['y'])),
        ('pool centred on the mirror axis', (R['pool']['y0'] + R['pool']['y1']) / 2, MID),
        ('waves share x start', R['wave_closed']['x0'], R['wave_open']['x0']),
        ('waves share x end', R['wave_closed']['x1'], R['wave_open']['x1']),
        ('open path horizontal (pool to wave)', R['open_path']['ys'][0], O_Y),
        ('open path horizontal (wave to dots)', R['open_path']['ys'][2], O_Y),
        ('open wave centred on the flow line', R['wave_open']['y'], O_Y),
        ('waves share amplitude', R['wave_open']['amp'], R['wave_closed']['amp']),
        ('waves share period', R['wave_open']['period'], R['wave_closed']['period']),
        ('score dots centred on the verdict column', R['dots_open']['xs'][2], R['col_closed']['cx']),
        ('score dots on the open flow line', R['dots_open']['y'], O_Y),
        ('oracle and rho share x', R['node_closed']['x'], R['node_open']['x']),
        ('oracle and rho mirror', R['node_closed']['y'], mirror(R['node_open']['y'])),
        ('oracle and rho share radius', R['node_closed']['r'], R['node_open']['r']),
        ('panel arrows mirror', R['arrow_closed']['y'], mirror(R['arrow_open']['y'])),
        ('result boxes mirror (top)', R['box_closed']['y1'], mirror(R['box_open']['y0'])),
        ('result boxes mirror (bottom)', R['box_closed']['y0'], mirror(R['box_open']['y1'])),
        ('result boxes match lanes', R['box_closed']['y0'], R['lane_closed']['y0']),
        ('strategies centred between pool and lane edge', (R['strategies']['left'] + R['strategies']['right']) / 2,
         (POOL_X1 + LANE_X1) / 2),
        ('pair: left margin equals the gap', R['scenes']['scenes'][0]['left'] - POOL_X1,
         R['scenes']['scenes'][1]['left'] - R['scenes']['scenes'][0]['right']),
        ('pair: right margin equals the gap', LANE_X1 - R['scenes']['scenes'][1]['right'],
         R['scenes']['scenes'][1]['left'] - R['scenes']['scenes'][0]['right']),
        ('pair shares a baseline', R['scenes']['scenes'][0]['bottom'], R['scenes']['scenes'][1]['bottom']),
        ('pair cards share a top line', R['scenes']['scenes'][0]['card_top'], R['scenes']['scenes'][1]['card_top']),
        ('pair mirrored about its own centre (figures)',
         (R['scenes']['scenes'][0]['fig_x'] + R['scenes']['scenes'][1]['fig_x']) / 2, (POOL_X1 + LANE_X1) / 2),
        ('pair mirrored about its own centre (cards)',
         (R['scenes']['scenes'][0]['card_x'] + R['scenes']['scenes'][1]['card_x']) / 2, (POOL_X1 + LANE_X1) / 2),
        ('riser 1 vertical', R['connectors']['risers'][0][0][0], R['connectors']['risers'][0][1][0]),
        ('riser 2 vertical', R['connectors']['risers'][1][0][0], R['connectors']['risers'][1][1][0]),
        ('riser 1 starts at its card top centre', R['connectors']['risers'][0][0][0], R['scenes']['scenes'][0]['card_x']),
        ('riser 2 starts at its card top centre', R['connectors']['risers'][1][0][0], R['scenes']['scenes'][1]['card_x']),
        ('risers end on the bus (1)', R['connectors']['risers'][0][1][1], R['connectors']['bus_y']),
        ('risers end on the bus (2)', R['connectors']['risers'][1][1][1], R['connectors']['bus_y']),
        ('bus reaches the entry', R['connectors']['bus'][1], NODE_X),
        ('single entry centred on rho', R['connectors']['entry'][0][0], NODE_X),
        ('entry vertical', R['connectors']['entry'][0][0], R['connectors']['entry'][1][0]),
        ('entry tip on rho bottom', R['connectors']['entry'][1][1], O_Y - NODE_R),
        ('bus midway between card tops and rho', R['connectors']['bus_y'] - (R['scenes']['scenes'][0]['card_top'] + 1.2),
         (O_Y - NODE_R) - R['connectors']['bus_y']),
        ('grid block centred in its box', (R['grid']['left'] + R['grid']['right']) / 2, PANEL_CX),
        ('grid block centred between caption and title', (R['grid']['y0'] + R['grid']['y1']) / 2, sum(R['grid']['room']) / 2),
        ('closed header over its columns (Phi-4)', R['closed_header']['xs'][0], R['closed_header']['cols'][0]),
        ('closed header over its columns (Qwen3.5)', R['closed_header']['xs'][1], R['closed_header']['cols'][1]),
        ('header lines at the same depth in both boxes', R['box_closed']['y1'] - R['closed_header']['y'], HEADER_DY),
        ('human dots start at the value column', R['human_dots']['left'], VAL_X),
    ]
    gaps = [('chip pitch inside groups', [R['chips']['xs'][i + 1] - R['chips']['xs'][i] for i in (0, 1, 3, 5, 6)]),
            ('gaps between chip groups', [R['chips']['xs'][3] - R['chips']['xs'][2], R['chips']['xs'][5] - R['chips']['xs'][4]]),
            ('strategy box pitch', [b - a for a, b in zip(R['strategies']['xs'], R['strategies']['xs'][1:])]),
            ('verdict chip pitch', [a - b for a, b in zip(R['col_closed']['ys'], R['col_closed']['ys'][1:])]),
            ('score dot pitch', [b - a for a, b in zip(R['dots_open']['xs'], R['dots_open']['xs'][1:])])]
    for k, xs in enumerate(R['pool_rows']['rows']):
        gaps.append((f'pool glyph pitch, row {k + 1}', [b - a for a, b in zip(xs, xs[1:])]))
    gaps.append(('pool rows share their right end', [xs[-1] for xs in R['pool_rows']['rows']]))
    problems = []
    cn = R['connectors']
    if (O_Y - NODE_R) - cn['bus_y'] < 5:
        problems.append('entry arrow shorter than 5 pt')
    if O_Y - 2.4 - cn['bus_y'] < 3:
        problems.append('bus too close to the score dots')
    for desc, a, b in eq:
        if abs(a - b) > tol:
            problems.append(f'{desc}: {a:.2f} vs {b:.2f}')
    for desc, vals in gaps:
        if max(vals) - min(vals) > tol:
            problems.append(f'{desc}: {", ".join(f"{v:.2f}" for v in vals)}')
    if page is None:
        return len(eq) + len(gaps), 0, problems
    # Rendered text: centred labels centred on their anchor, aligned labels on their edge, shared baselines
    spans = [s for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()]
    n_text = 0
    for s_text, x, y, ha, size in LABELS:
        cand = [s for s in spans if s['text'].strip() == s_text]
        if not cand:
            problems.append(f'label not found as one span: {s_text!r}')
            continue
        sp = min(cand, key=lambda s: abs((s['bbox'][0] + s['bbox'][2]) / 2 - x) + abs(s['origin'][1] - (H - y)))
        x0, x1 = sp['bbox'][0], sp['bbox'][2]
        got = {'center': (x0 + x1) / 2, 'left': x0, 'right': x1}[ha]
        base = H - sp['origin'][1]
        n_text += 1
        if abs(got - x) > tol:
            problems.append(f'text {s_text!r} {ha} edge at {got:.2f}, anchor {x:.2f}')
        if abs(base - (y - 0.34 * size)) > tol:
            problems.append(f'text {s_text!r} baseline {base:.2f}, expected {y - 0.34 * size:.2f}')
    baseline_groups = [('math', 'sci.', 'gen.'), ('SummEval', 'MT-Bench'), ('1,600', '80'), ('human', 'LLM reference'),
                       ('1', '5'),
                       ('6 personas', 'jury of 3', 'debate'), ('Phi-4', 'Qwen3.5-122B'), ('90.16%', '90.30%'),
                       ('16 small', '0.6–14B'), ('6 large', '70–122B'), ('oracle',)]
    placed = {s_text: (x, y) for s_text, x, y, ha, size in LABELS}
    for group in baseline_groups:
        bases = []
        for s_text in group:
            x, y = placed[s_text]
            cand = [s for s in spans if s['text'].strip() == s_text]
            sp = min(cand, key=lambda s: abs(H - s['origin'][1] - y) + abs((s['bbox'][0] + s['bbox'][2]) / 2 - x))
            bases.append(H - sp['origin'][1])
        if max(bases) - min(bases) > tol:
            problems.append(f'baseline group {group}: {", ".join(f"{b:.2f}" for b in bases)}')
    return len(eq) + len(gaps), n_text, problems


def check_text(page):
    """Rendered-text problems: type below the 6 pt floor and labels that overlap."""
    spans = []
    for b in page.get_text('dict')['blocks']:
        for li in b.get('lines', []):
            for s in li['spans']:
                if s['text'].strip():
                    spans.append((s['bbox'], s['size'], s['text']))
    probs = [f'text below {style.FS_MIN} pt: {t!r} ({size:.1f} pt)' for _, size, t in spans if size < style.FS_MIN - 1e-6]
    for i in range(len(spans)):
        for k in range(i + 1, len(spans)):
            a, b = spans[i][0], spans[k][0]
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            # span boxes include line spacing, so require the overlap to reach into the glyph cores
            if w > 0.6 and h > 0.42 * min(spans[i][1], spans[k][1]):
                probs.append(f'overlapping text: {spans[i][2]!r} / {spans[k][2]!r}')
    return probs


def check_pdf(page):
    """Print-PDF problems: page size and fonts (all embedded, none Type 3)."""
    probs = []
    if abs(page.rect.width - W) > 0.01 or abs(page.rect.height - H) > 0.01:
        probs.append(f'page {page.rect.width:.2f} x {page.rect.height:.2f} pt, expected {W} x {H}')
    for f in page.get_fonts(full=True):
        if f[2] == 'Type3':
            probs.append(f'Type 3 font: {f[3]}')
        elif f[1] == 'n/a':
            probs.append(f'font not embedded: {f[3]}')
    return probs


def check(pdf_bytes):
    """Every layout problem of a rendered overview PDF; the PDF checks need PyMuPDF."""
    try:
        import fitz
    except ImportError:
        print('  overview: PyMuPDF not installed, rendered-text and font checks skipped')
        n_geo, n_text, probs = check_alignment(None)
    else:
        page = fitz.open(stream=pdf_bytes, filetype='pdf')[0]
        n_geo, n_text, probs = check_alignment(page)
        probs += check_pdf(page) + check_text(page)
    print(f'  overview: {W / 72:.2f} x {H / 72:.2f} in, {n_geo} alignment relations, {n_text} placed labels, '
          f'{len(probs)} layout problem(s)')
    for p in probs:
        print(f'   !! {p}')
    return probs


# Build

def draw():
    """Draw the figure on a fresh canvas and return it."""
    REG.clear()
    LABELS.clear()
    D.clear()
    D.update(load_data())
    fig, ax = canvas(H)
    lanes(ax)
    start_node(ax)
    closed_inputs(ax)
    open_inputs(ax)
    pool_y = judge_pool(ax)
    strategies(ax, pool_y['small'])
    budget_fork(ax)
    verdicts(ax)
    scoring(ax)
    closed_result(ax)
    open_result(ax)
    return fig


def build(formats=('pdf', 'png'), out_dir=None, verbose=True):
    """Write 1_overview.<fmt> for each of `formats` to out_dir (default results/figures).
    Returns (written files, layout problems)."""
    register_fonts()
    out = Path(out_dir) if out_dir else style.paper_dir()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    # The figure keeps its own settings (matplotlib defaults plus RC), so the data figures' theme does not leak in
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(RC)
        fig = draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', metadata=PDF_META)
        for ext in formats:
            target = out / f'{FILE}.{ext}'
            if ext == 'pdf':
                target.write_bytes(buf.getvalue())
            elif ext == 'png':
                fig.savefig(target, dpi=style.PREVIEW_DPI)
            elif ext == 'svg':
                fig.savefig(target, metadata={'Date': None})
            else:
                fig.savefig(target)
            written.append(target)
        plt.close(fig)
    probs = check(buf.getvalue())
    if verbose:
        for w in written:
            print(f'  wrote {w}')
    return written, probs


def main():
    try:
        _, probs = build()
    except ToolMissing as e:
        sys.exit(f'overview: {e}')
    return 1 if probs else 0


if __name__ == '__main__':
    sys.exit(main())
