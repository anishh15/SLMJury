"""Shared style for the SLMJury data figures; every figure module does ``from style import *``.

Sets the font, colours, type scale and output rules, and holds the shared helpers. Figures are authored at
printed size (6.30 in full width, 3.03 in one column), with type in printed points.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import matplotlib.patches as mpatches                 # noqa: E402
import matplotlib.transforms as mtransforms           # noqa: E402
import matplotlib.ticker as mticker                   # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.text import Text                      # noqa: E402
import numpy as np                                    # noqa: E402

from slmjury.analysis import paths                    # noqa: E402

# ------------------------------------------------------------ size, type ----
W_FULL = 6.30      # full text width (inches)
W_COL = 3.03       # one column
FS_LABEL = 8       # axis labels
FS_TICK = 7        # tick labels
FS_LEGEND = 7      # legend
FS_ANNOT = 6.5     # point, bar and value labels, short notes
FS_TITLE = 8       # panel titles
FS_MIN = 6         # nothing smaller than this


def _find_font():
    """Register TeX Gyre Heros if a TeX installation has it, else return the
    first standard sans-serif font matplotlib knows about."""
    kpse = shutil.which('kpsewhich')
    if kpse:
        try:
            out = subprocess.run([kpse, 'texgyreheros-regular.otf'],
                                 capture_output=True, text=True,
                                 timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ''
        if out and os.path.isfile(out):
            folder = os.path.dirname(out)
            for face in ('regular', 'bold', 'italic', 'bolditalic'):
                path = os.path.join(folder, f'texgyreheros-{face}.otf')
                if os.path.isfile(path):
                    font_manager.fontManager.addfont(path)
            return 'TeX Gyre Heros'
    known = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'):
        if name in known:
            return name
    return 'DejaVu Sans'


FONT = _find_font()

# --------------------------------------------------------------- colour -----
def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


def shade(c, t):
    """t > 0 lightens toward white, t < 0 darkens toward black."""
    return mix(c, '#FFFFFF', t) if t >= 0 else mix(c, '#000000', -t)


# Colours that do not depend on the theme.
TEAL = '#5E9C94'
TEAL_LIGHT = '#BFE0DA'
TEAL_PALE = '#E3F1EE'
GREEN = '#6A9A5B'        # good / agree
AMBER = '#D9A441'
RUST = '#B5715C'

# One colour and one marker per family in every figure; members differ by shade or line style, not hue.
FAMILY_MARKER = {'Llama': '^', 'Qwen2.5': 's', 'Qwen3': 'o', 'Phi-4': 'D',
                 'Large': 'h'}
FAMILY_ORDER = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4', 'Large']
FAMILY_LABEL = {'Llama': 'Llama 3.x', 'Qwen2.5': 'Qwen2.5', 'Qwen3': 'Qwen3',
                'Phi-4': 'Phi-4', 'Large': 'Large (70B–122B)'}
# One markersize prints the five shapes at different visual weights; these
# factors make them read as the same size.
MARKER_SCALE = {'^': 1.10, 's': 0.86, 'o': 1.0, 'D': 0.80, 'h': 1.05,
                'v': 1.10}

# light is the print theme; dark keeps the layout and family colours on the README's deep-space background.
THEMES = {
    'light': {
        'PAPER': 'white',        # background, marker halos, bar and cell edges
        'INK': '#1E2733',        # text
        'CELL_INK': '#1E2733',   # dark text on a light heatmap cell
        'MUTED': '#5B6575',      # secondary text, leaders
        'GRID': '#E3E7ED',
        'AXIS': '#4A5361',       # spines and tick marks
        'NAVY': '#1F3A5F',
        'NAVY_MID': '#3F5F8C',
        'NAVY_LIGHT': '#BFCFEA',
        'NAVY_PALE': '#E1E8F3',
        'SLATE': '#8A94A6',      # neutral, baseline, mixed families
        'UNDER': '#E4E7EC',      # heatmap cells below the colour scale
        'FAMILY_COLOR': {'Llama': '#B5615C', 'Qwen2.5': '#4E9A8F',
                         'Qwen3': '#4F75A8', 'Phi-4': '#C8913F',
                         'Large': '#7E6BA0'},
        'SEQ': ['#FFFFFF', '#E1E8F3', '#BFCFEA', '#7F9AC0', '#3F5F8C',
                '#1F3A5F'],
        'DIV': [shade(TEAL, -0.25), TEAL, TEAL_LIGHT, '#FFFFFF', '#BFCFEA',
                '#3F5F8C', '#1F3A5F'],
    },
    'dark': {
        'PAPER': '#050D1E',
        'INK': '#F4F7FC',
        'CELL_INK': '#0B1B33',
        'MUTED': '#A9BED8',
        'GRID': '#1B2C4F',
        'AXIS': '#A9B4C4',
        'NAVY': '#9CCBFA',
        'NAVY_MID': '#2E7DF0',
        'NAVY_LIGHT': '#1B3A6B',
        'NAVY_PALE': '#0F1D3D',
        'SLATE': '#93A3BC',
        'UNDER': '#14213B',
        'FAMILY_COLOR': {'Llama': '#C9736D', 'Qwen2.5': '#5AAFA3',
                         'Qwen3': '#6C93C8', 'Phi-4': '#D6A04D',
                         'Large': '#9A87BE'},
        'SEQ': ['#0A1530', '#0F2A5C', '#1A55B8', '#2E7DF0', '#38C6EA',
                '#C4F1FD'],
        'DIV': ['#9AD8CE', '#5AAFA3', '#1F4A45', '#0A1530', '#16407F',
                '#2E7DF0', '#9CCBFA'],
    },
}
THEME = 'light'
PAPER = INK = CELL_INK = MUTED = GRID = AXIS = None
NAVY = NAVY_MID = NAVY_LIGHT = NAVY_PALE = SLATE = UNDER = None
FAMILY_COLOR = {}


def deepen(c, t):
    """A stronger shade of c for text set against the background: darker by
    t on the light theme, lighter by t on the dark one."""
    return shade(c, -t if THEME == 'light' else t)


def tint(c, t=0.55):
    """Light tint of a colour (second condition on top of family)."""
    return shade(c, t)


def seq_cmap(name='slm_seq'):
    """Sequential scale for accuracy: background to navy."""
    return LinearSegmentedColormap.from_list(name, THEMES[THEME]['SEQ'])


def div_cmap(name='slm_div'):
    """Diverging scale centred on zero: teal (negative), navy (positive)."""
    return LinearSegmentedColormap.from_list(name, THEMES[THEME]['DIV'])


def text_on(rgb):
    """White or dark text for a cell of colour rgb (relative luminance)."""
    r, g, b = to_rgb(rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if lum < 0.5 else CELL_INK


def _rc():
    return {
        'font.family': FONT,
        'font.size': FS_TICK,
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT,
        'mathtext.it': f'{FONT}:italic',
        'mathtext.bf': f'{FONT}:bold',
        'mathtext.sf': FONT,
        'mathtext.default': 'it',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'path',
        'svg.hashsalt': 'slmjury',
        'axes.labelsize': FS_LABEL,
        'axes.labelcolor': INK,
        'axes.titlesize': FS_TITLE,
        'axes.titleweight': 'normal',
        'axes.titlecolor': INK,
        'axes.titlelocation': 'left',
        'axes.titlepad': 4,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.axisbelow': True,
        'axes.facecolor': PAPER,
        'axes.labelpad': 2.5,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'xtick.color': AXIS,
        'ytick.color': AXIS,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'xtick.minor.size': 1.4,
        'ytick.minor.size': 1.4,
        'xtick.major.pad': 2.0,
        'ytick.major.pad': 2.0,
        'legend.fontsize': FS_LEGEND,
        'legend.frameon': False,
        'legend.handletextpad': 0.4,
        'legend.borderaxespad': 0.4,
        'legend.borderpad': 0.3,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,
        'legend.labelspacing': 0.3,
        'lines.linewidth': 1.1,
        'lines.markersize': 4.0,
        'lines.solid_capstyle': 'round',
        'lines.dash_capstyle': 'butt',
        'patch.linewidth': 0.6,
        'text.color': INK,
        'figure.facecolor': PAPER,
        'figure.dpi': 200,
        'savefig.dpi': 600,
        'hatch.linewidth': 0.6,
        'hatch.color': PAPER,
    }


def set_theme(name):
    """Switch every colour constant and the matplotlib settings to a theme
    ('light' or 'dark'). Re-import figure modules afterwards."""
    global THEME
    if name not in THEMES:
        raise ValueError(f'unknown theme {name!r}; expected one of {sorted(THEMES)}')
    THEME = name
    spec = THEMES[name]
    globals().update({k: (dict(v) if isinstance(v, dict) else v)
                      for k, v in spec.items() if k not in ('SEQ', 'DIV')})
    plt.rcParams.update(_rc())


set_theme('light')

MINUS = '−'
EN = '–'
TIMES = '×'

# -------------------------------------------------------------- numbers -----


def half_up(v, dp=2):
    """Round once, half away from zero, from the full-precision value.

    A CSV string is rounded from its own text. A float is first cut to six decimals (the CSVs' precision),
    so binary noise cannot move it across a rounding boundary.
    """
    if isinstance(v, str):
        d = Decimal(v)
    else:
        d = Decimal(repr(round(float(v), 6)))
    q = Decimal(1).scaleb(-dp)
    return d.quantize(q, rounding=ROUND_HALF_UP)


def fmt(v, dp=2, sign=False):
    """Number for print: half up once, true minus, optional explicit plus.
    A value that rounds to zero prints unsigned."""
    d = half_up(v, dp)
    s = f'{abs(d):,.{dp}f}'
    if d < 0:
        s = MINUS + s
    elif sign and d > 0:
        s = '+' + s
    return s


def tick_fmt(dp=0, sign=False):
    def _f(x, _pos):
        if abs(x) < 1e-12:
            x = 0.0
        return fmt(x, dp, sign=sign)
    return mticker.FuncFormatter(_f)


# ---------------------------------------------------------------- judges ----
DISPLAY = {
    'llama3.2-1b': 'Llama-3.2-1B', 'llama3.2-3b': 'Llama-3.2-3B',
    'llama3.1-8b': 'Llama-3.1-8B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B',
    'qwen2.5-3b': 'Qwen2.5-3B', 'qwen2.5-7b': 'Qwen2.5-7B',
    'qwen3-0.6b': 'Qwen3-0.6B', 'qwen3-1.7b': 'Qwen3-1.7B',
    'qwen3-4b': 'Qwen3-4B', 'qwen3-8b': 'Qwen3-8B', 'qwen3-14b': 'Qwen3-14B',
    'qwen3-32b': 'Qwen3-32B',
    'phi4-14b': 'Phi-4', 'phi4mi-3.8b': 'Phi-4-mini',
    'phi4r-14b': 'Phi-4-Reasoning', 'phi4rp-14b': 'Phi-4-Reasoning-Plus',
    'phi4mr-3.8b': 'Phi-4-mini-Reasoning',
    'llama3.3-70b': 'Llama-3.3-70B', 'qwen2.5-72b': 'Qwen2.5-72B',
    'gpt-oss-120b': 'GPT-OSS-120B', 'qwen3.5-122b': 'Qwen3.5-122B',
    'glm-4.5-air': 'GLM-4.5-Air',
    'deepseek-r1-distill-llama-70b': 'DeepSeek-R1-Distill-70B',
}

PARAMS = {
    'llama3.2-1b': 1, 'llama3.2-3b': 3, 'llama3.1-8b': 8,
    'qwen2.5-1.5b': 1.5, 'qwen2.5-3b': 3, 'qwen2.5-7b': 7,
    'qwen3-0.6b': 0.6, 'qwen3-1.7b': 1.7, 'qwen3-4b': 4, 'qwen3-8b': 8,
    'qwen3-14b': 14, 'qwen3-32b': 32,
    'phi4mi-3.8b': 3.8, 'phi4mr-3.8b': 3.8, 'phi4-14b': 14, 'phi4r-14b': 14,
    'phi4rp-14b': 14,
    'llama3.3-70b': 70, 'deepseek-r1-distill-llama-70b': 70,
    'qwen2.5-72b': 72, 'glm-4.5-air': 106, 'gpt-oss-120b': 120,
    'qwen3.5-122b': 122,
}
LARGE = {'llama3.3-70b', 'deepseek-r1-distill-llama-70b', 'qwen2.5-72b',
         'glm-4.5-air', 'gpt-oss-120b', 'qwen3.5-122b'}
LADDER = {'qwen3-32b'}     # the Qwen3 scaling-ladder judge


def family_of(j):
    if j in LARGE:
        return 'Large'
    if j.startswith('llama'):
        return 'Llama'
    if j.startswith('qwen2.5'):
        return 'Qwen2.5'
    if j.startswith('qwen3'):      # qwen3.5-122b is caught by LARGE above
        return 'Qwen3'
    if j.startswith('phi4'):
        return 'Phi-4'
    raise KeyError(j)


def dn(j):
    return DISPLAY[j]


def dn_tag(j):
    """Display name, with "(ladder)" after the Qwen3 scaling-ladder judge."""
    return DISPLAY[j] + (' (ladder)' if j in LADDER else '')


def jcolor(j):
    return FAMILY_COLOR[family_of(j)]


def jmarker(j):
    return FAMILY_MARKER[family_of(j)]


# ------------------------------------------------------------------ data ----

def load(experiment, name):
    """Rows of results/analysis/<experiment>/<name> as dicts of strings."""
    path = paths.require(paths.results_dir() / 'analysis' / experiment / name,
                         f'results/analysis/{experiment}/{name}')
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


DATASETS = ['gsm8k', 'gsm_plus', 'math', 'arc_easy', 'arc_challenge',
            'hellaswag', 'winogrande', 'truthfulqa']
DS_SHORT = {'gsm8k': 'GSM8K', 'gsm_plus': 'GSM+', 'math': 'MATH',
            'arc_easy': 'ARC-E', 'arc_challenge': 'ARC-C',
            'hellaswag': 'HSwag', 'winogrande': 'WGrnd', 'truthfulqa': 'TQA'}
DOMAINS = [('Math', ['gsm8k', 'gsm_plus', 'math']),
           ('Science', ['arc_easy', 'arc_challenge']),
           ('General', ['hellaswag', 'winogrande', 'truthfulqa'])]

# ----------------------------------------------------------- construction ---


def new_figure(w, h):
    return plt.figure(figsize=(w, h))


def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def style_ax(ax, grid='y', spines=('left', 'bottom')):
    for s in ('left', 'right', 'top', 'bottom'):
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_zorder(4.5)     # drawn over bars that start on it
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.5, zorder=0)
    ax.tick_params(which='both', direction='out')
    return ax


def panel_title(ax, text, pad=4):
    """'(a) Short title', left-aligned over the axes."""
    return ax.set_title(text, loc='left', fontsize=FS_TITLE, color=INK,
                        pad=pad)


def _visible_size(s, filled):
    """Markersize so filled and open marks look about the same size: a
    filled mark loses half its 0.5-pt edge inside, an open one gains half its
    0.95-pt coloured edge outside. The open mark is shrunk only a little so
    small ones keep a visible hole."""
    return s + 0.6 if filled else s - 0.3


def judge_mark(ax, x, y, j=None, family=None, filled=True, size=4.2,
               color=None, z=5, edge=None, **kw):
    """One judge: family colour and shape; filled (B=10) or open (B=8,192)."""
    f = family or family_of(j)
    c = color or FAMILY_COLOR[f]
    m = kw.pop('marker', FAMILY_MARKER[f])
    s = size * MARKER_SCALE.get(m, 1.0)
    if filled:
        face, ec, ew = c, edge or PAPER, 0.5
    else:
        face, ec, ew = PAPER, edge or c, 0.95
    s = _visible_size(s, filled)
    return ax.plot([x], [y], linestyle='none', marker=m, markersize=s,
                   markerfacecolor=face, markeredgecolor=ec,
                   markeredgewidth=ew, zorder=z, **kw)[0]


def family_handles(families, filled=True, size=4.2, labels=None):
    hs = []
    for f in families:
        m = FAMILY_MARKER[f]
        c = FAMILY_COLOR[f]
        hs.append(Line2D([], [], linestyle='none', marker=m,
                         markersize=_visible_size(
                             size * MARKER_SCALE.get(m, 1.0), filled),
                         markerfacecolor=c if filled else PAPER,
                         markeredgecolor=PAPER if filled else c,
                         markeredgewidth=0.5 if filled else 0.95,
                         label=(labels or FAMILY_LABEL)[f]))
    return hs


def budget_handles(color=None, size=4.2, marker='o'):
    """Legend keys for filled (B=10) and open (B=8,192) marks."""
    color = color or MUTED
    s = size * MARKER_SCALE.get(marker, 1.0)
    return [Line2D([], [], linestyle='none', marker=marker,
                   markersize=_visible_size(s, True),
                   markerfacecolor=color, markeredgecolor=PAPER,
                   markeredgewidth=0.5, label='$B{=}10$'),
            Line2D([], [], linestyle='none', marker=marker,
                   markersize=_visible_size(s, False),
                   markerfacecolor=PAPER, markeredgecolor=color,
                   markeredgewidth=0.95, label='$B{=}8{,}192$')]


def family_patches(families, labels=None):
    return [mpatches.Patch(color=FAMILY_COLOR[f], label=(labels or FAMILY_LABEL)[f])
            for f in families]


# ----------------------------------------------------- overlap geometry -----

def _px(fig, pt):
    return pt * fig.dpi / 72.0


def drawn_texts(fig):
    """Every visible, non-empty Text the renderer draws (tick labels only for
    ticks inside the view)."""
    fig.canvas.draw()
    tick_texts, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_texts.update({id(t.label1), id(t.label2)})
            try:
                for t in axis._update_ticks():
                    for lab in (t.label1, t.label2):
                        if lab.get_visible():
                            drawn.add(id(lab))
            except Exception:
                pass
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in tick_texts and id(t) not in drawn:
            continue
        if t.get_gid() == 'skip-overlap':
            continue
        out.append(t)
    return out


def marker_boxes(fig, shrink=0.85):
    """Display boxes of every data marker (Line2D markers and scatters)."""
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        clip = ax.get_window_extent(r)
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_gid() == 'skip-overlap':
                continue
            m = ln.get_marker()
            if m in (None, 'None', '', ' ') or ln.get_markersize() <= 0:
                continue
            xy = ln.get_xydata()
            if len(xy) == 0:
                continue
            pts = ln.get_transform().transform(xy)
            half = _px(fig, ln.get_markersize()) / 2 * shrink
            for px, py in pts:
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                if ln.get_clip_on() and not (
                        clip.x0 - half <= px <= clip.x1 + half
                        and clip.y0 - half <= py <= clip.y1 + half):
                    continue
                boxes.append(mtransforms.Bbox.from_extents(
                    px - half, py - half, px + half, py + half))
        for col in ax.collections:
            if not isinstance(col, PathCollection) or not col.get_visible():
                continue
            if col.get_gid() == 'skip-overlap':
                continue
            offs = col.get_offsets()
            if len(offs) == 0:
                continue
            pts = col.get_offset_transform().transform(offs)
            sizes = col.get_sizes()
            for k, (px, py) in enumerate(pts):
                s = sizes[k % len(sizes)] if len(sizes) else 20
                half = _px(fig, math.sqrt(s)) / 2 * shrink
                boxes.append(mtransforms.Bbox.from_extents(
                    px - half, py - half, px + half, py + half))
    return boxes


def _segments(fig, ax):
    """Display-space polylines of data lines, error bars and arrows."""
    paths_ = []
    for ln in ax.get_lines():
        if not ln.get_visible() or ln.get_gid() in ('skip-overlap', 'leader'):
            continue
        if ln.get_linestyle() in ('None', ' ', '', 'none'):
            continue
        xy = ln.get_xydata()
        if len(xy) < 2:
            continue
        paths_.append((ln.get_transform().transform(xy),
                       max(ln.get_linewidth(), 0.5)))
    for col in ax.collections:
        if not isinstance(col, LineCollection) or not col.get_visible():
            continue
        if col.get_gid() == 'skip-overlap':
            continue
        lw = col.get_linewidths()
        lw = float(lw[0]) if len(lw) else 0.8
        for seg in col.get_segments():
            if len(seg) >= 2:
                paths_.append((col.get_transform().transform(seg), lw))
    for t in ax.texts:
        ap = getattr(t, 'arrow_patch', None)
        if ap is None:
            continue
        try:
            polys = ap.get_path().to_polygons(closed_only=False)
            tr = ap.get_transform()
        except Exception:
            continue
        # Leaders are thin; arrows count at their drawn width. The path is
        # split at its move-to codes (shaft and head are separate pieces).
        for poly in polys:
            if len(poly) >= 2:
                paths_.append((tr.transform(poly),
                               0.45 if t.get_gid() == 'leader' else 0.8))
    return paths_


def line_boxes(fig, ax, step_pt=1.5):
    r = fig.canvas.get_renderer()
    clip = ax.get_window_extent(r)
    step = _px(fig, step_pt)
    boxes = []
    for pts, lw in _segments(fig, ax):
        half = _px(fig, lw) / 2 + 0.5
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if not all(np.isfinite([x0, y0, x1, y1])):
                continue
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for k in range(n + 1):
                px = x0 + (x1 - x0) * k / n
                py = y0 + (y1 - y0) * k / n
                if not (clip.x0 - 1 <= px <= clip.x1 + 1
                        and clip.y0 - 1 <= py <= clip.y1 + 1):
                    continue
                boxes.append(mtransforms.Bbox.from_extents(
                    px - half, py - half, px + half, py + half))
    return boxes


def bar_boxes(fig, ax):
    """Display boxes of bars (Rectangle patches), not of bands or cells."""
    r = fig.canvas.get_renderer()
    clip = ax.get_window_extent(r)
    boxes = []
    for p in ax.patches:
        if not p.get_visible() or p.get_gid() in ('band', 'cell',
                                                   'skip-overlap'):
            continue
        if not isinstance(p, mpatches.Rectangle):
            continue
        bb = p.get_window_extent(r)
        if p.get_clip_on():
            bb = mtransforms.Bbox.intersection(bb, clip)
            if bb is None:
                continue
        if bb.width > 0.5 and bb.height > 0.5:
            boxes.append(bb)
    return boxes


def _legends(fig):
    return list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                if ax.get_legend() is not None]


def legend_boxes(fig):
    r = fig.canvas.get_renderer()
    return [(lg, lg.get_window_extent(r)) for lg in _legends(fig)]


def _legend_texts(fig):
    ids = set()
    for lg in _legends(fig):
        ids.update(id(t) for t in lg.get_texts())
        if lg.get_title() is not None:
            ids.add(id(lg.get_title()))
    return ids


def text_poly(t, r, shrink=0.0):
    """The text's true (possibly rotated) rectangle in display space, as a
    4x2 array, shrunk by `shrink` pixels on every side."""
    bb = t.get_window_extent(r)
    rot = t.get_rotation() % 360
    if abs(rot) < 1e-6:
        b = bb.padded(-shrink)
        return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1],
                         [b.x0, b.y1]])
    t.set_rotation(0)
    b0 = t.get_window_extent(r).padded(-shrink)
    t.set_rotation(rot)
    th = math.radians(rot)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c = np.array([[b0.x0, b0.y0], [b0.x1, b0.y0], [b0.x1, b0.y1],
                  [b0.x0, b0.y1]]) @ R.T
    # place the rotated rectangle so its bounding box matches the real one
    full = t.get_window_extent(r)
    cx = (c[:, 0].min() + c[:, 0].max()) / 2
    cy = (c[:, 1].min() + c[:, 1].max()) / 2
    return c + np.array([(full.x0 + full.x1) / 2 - cx,
                         (full.y0 + full.y1) / 2 - cy])


def _box_poly(b):
    return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])


def poly_overlap(a, b):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def _hits(poly, pbb, boxes):
    """True if the polygon (with axis-aligned bbox pbb) overlaps any box."""
    for m in boxes:
        if not pbb.overlaps(m):
            continue
        if poly_overlap(poly, _box_poly(m)):
            return True
    return False


def check_overlaps(fig, tol_pt=0.5, verbose=True):
    """Return a list of layout problems (strings) found in ``fig``.

    Checks text on text, text on a marker, line or bar, a legend on data or text, anything off the canvas and type
    below FS_MIN. Text with gid 'cell' skips the mark checks; legend texts are checked through the legend's box.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tol = _px(fig, tol_pt)
    leg_ids = _legend_texts(fig)
    texts = [t for t in drawn_texts(fig) if id(t) not in leg_ids]
    boxes = []
    for t in texts:
        try:
            poly = text_poly(t, r, shrink=tol)
            pbb = mtransforms.Bbox.from_extents(poly[:, 0].min(), poly[:, 1].min(),
                                                poly[:, 0].max(), poly[:, 1].max())
            boxes.append((t, t.get_window_extent(r), poly, pbb))
        except RuntimeError:
            continue
    probs = []
    W, H = fig.get_size_inches() * fig.dpi
    for t, bb, _, _ in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            probs.append(f'off-canvas: {t.get_text()!r}')
    for i in range(len(boxes)):
        ti, _, pi, bi = boxes[i]
        for k in range(i + 1, len(boxes)):
            tk, _, pk, bk = boxes[k]
            if bi.overlaps(bk) and poly_overlap(pi, pk):
                probs.append(f'text/text: {ti.get_text()!r} x {tk.get_text()!r}')
    marks = marker_boxes(fig)
    lines, bars = [], []
    for ax in fig.axes:
        lines += line_boxes(fig, ax)
        bars += bar_boxes(fig, ax)
    for t, _, poly, pbb in boxes:
        if t.get_gid() == 'cell':
            continue
        if _hits(poly, pbb, marks):
            probs.append(f'text/marker: {t.get_text()!r}')
        if _hits(poly, pbb, lines):
            probs.append(f'text/line: {t.get_text()!r}')
        if _hits(poly, pbb, bars):
            probs.append(f'text/bar: {t.get_text()!r}')
    for lg, lb in legend_boxes(fig):
        b = lb.padded(-tol)
        if any(b.overlaps(m) for m in marks):
            probs.append('legend/marker')
        if any(b.overlaps(m) for m in lines):
            probs.append('legend/line')
        if any(b.overlaps(m) for m in bars):
            probs.append('legend/bar')
        for t, _, poly, pbb in boxes:
            if b.overlaps(pbb) and poly_overlap(_box_poly(b), poly):
                probs.append(f'legend/text: {t.get_text()!r}')
        if lb.x0 < -1 or lb.y0 < -1 or lb.x1 > W + 1 or lb.y1 > H + 1:
            probs.append('legend off-canvas')
    for t in drawn_texts(fig):
        if t.get_fontsize() < FS_MIN - 1e-6:
            probs.append(f'font below {FS_MIN} pt: {t.get_text()!r}')
    if verbose:
        for p in probs:
            print('   OVERLAP', p)
    return probs


# ------------------------------------------------------ label placement -----

def place_labels(ax, items, candidates, pad_pt=1.0, extra=(), leader_pt=10.0,
                 bounds=None):
    """Label points so no label touches another label, a marker, a line, a bar, a legend or a box in ``extra``.

    ``items`` are dicts with x, y, text and optional color, size and candidates (per item, (dx_pt, dy_pt, ha, va)).
    The first clear candidate inside ``bounds`` (default the axes) wins; a leader is drawn at ``leader_pt`` or more.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = _px(fig, pad_pt)
    obstacles = marker_boxes(fig, shrink=1.0) + line_boxes(fig, ax)
    for a in fig.axes:
        obstacles += bar_boxes(fig, a)
    obstacles += [lb for _, lb in legend_boxes(fig)]
    obstacles += [t.get_window_extent(r) for t in drawn_texts(fig)]
    obstacles += list(extra)
    area = bounds or ax.get_window_extent(r)
    placed = []
    for it in items:
        cands = it.get('candidates') or candidates
        chosen = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(it['text'], xy=(it['x'], it['y']), xytext=(dx, dy),
                            textcoords='offset points', ha=ha, va=va,
                            fontsize=it.get('size', FS_ANNOT),
                            color=it.get('color', INK), zorder=8,
                            annotation_clip=False)
            bb = t.get_window_extent(r).padded(pad)
            inside = (bb.x0 >= area.x0 - 1 and bb.x1 <= area.x1 + 1 and
                      bb.y0 >= area.y0 - 1 and bb.y1 <= area.y1 + 1)
            ok = inside and not any(bb.overlaps(o) for o in obstacles)
            if ok and math.hypot(dx, dy) >= leader_pt:
                lead = _leader_boxes(fig, ax, t.get_window_extent(r),
                                     it['x'], it['y'])
                ok = not any(lb.overlaps(o) for lb in lead for o in obstacles)
            if ok:
                chosen = (t, dx, dy)
                break
            t.remove()
        if chosen is None:
            dx, dy, ha, va = cands[0]
            t = ax.annotate(it['text'], xy=(it['x'], it['y']), xytext=(dx, dy),
                            textcoords='offset points', ha=ha, va=va,
                            fontsize=it.get('size', FS_ANNOT),
                            color=it.get('color', INK), zorder=8,
                            annotation_clip=False)
            chosen = (t, dx, dy)
            print(f'   place_labels: no clear spot for {it["text"]!r}')
        t, dx, dy = chosen
        if math.hypot(dx, dy) >= leader_pt:
            obstacles += _leader_boxes(fig, ax, t.get_window_extent(r),
                                       it['x'], it['y'])
            add_leader(ax, t, it['x'], it['y'])
        obstacles.append(t.get_window_extent(r).padded(pad))
        placed.append(t)
    return placed


def _leader_boxes(fig, ax, bb, x, y, shrink_pt=3.2, step_pt=1.5):
    """Sample boxes along the leader a label at `bb` would get to (x, y),
    leaving out the ends (the label itself and the point's own marker)."""
    px, py = ax.transData.transform((x, y))
    ex = min(max(px, bb.x0), bb.x1)
    ey = min(max(py, bb.y0), bb.y1)
    L = math.hypot(px - ex, py - ey)
    a0, a1 = _px(fig, 1.5), L - _px(fig, shrink_pt + 1.0)
    out = []
    if a1 <= a0:
        return out
    n = max(1, int((a1 - a0) / _px(fig, step_pt)))
    h = _px(fig, 0.4)
    for k in range(n + 1):
        u = (a0 + (a1 - a0) * k / n) / L
        qx, qy = ex + (px - ex) * u, ey + (py - ey) * u
        out.append(mtransforms.Bbox.from_extents(qx - h, qy - h, qx + h, qy + h))
    return out


def add_leader(ax, t, x, y, shrink_pt=3.2):
    """Thin grey leader from the label's nearest edge to the point."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(r)
    px, py = ax.transData.transform((x, y))
    ex = min(max(px, bb.x0), bb.x1)
    ey = min(max(py, bb.y0), bb.y1)
    lx, ly = ax.transData.inverted().transform((ex, ey))
    ax.annotate('', xy=(x, y), xytext=(lx, ly), textcoords='data',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.45,
                                shrinkA=1.0, shrinkB=shrink_pt),
                zorder=7, gid='leader')


def swarm_offsets(xs_px, d_px, band_px):
    """Vertical offsets (display px) so points closer than d_px in x stack
    into a small column inside +-band_px, ordered by x within each cluster."""
    xs_px = np.asarray(xs_px, dtype=float)
    order = np.argsort(xs_px)
    off = np.zeros(len(xs_px))
    if len(xs_px) == 0:
        return off
    clusters, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if xs_px[b] - xs_px[a] < d_px:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    for c in clusters:
        k = len(c)
        if k == 1:
            continue
        half = min(band_px, (k - 1) * d_px / 2)
        for i, idx in enumerate(c):
            off[idx] = -half + 2 * half * i / (k - 1)
    return off


def beeswarm(xs_px, d_px, band_px):
    """Greedy beeswarm: each point (in x order) takes the smallest vertical
    offset at which it is at least d_px from every placed point; if a point
    cannot fit inside +-band_px the whole row falls back to swarm_offsets."""
    xs_px = np.asarray(xs_px, dtype=float)
    order = np.argsort(xs_px)
    off = np.zeros(len(xs_px))
    placed = []
    steps = [0.0]
    k = 1
    while k * d_px * 0.5 <= band_px + 1e-9:
        steps += [k * d_px * 0.5, -k * d_px * 0.5]
        k += 1
    for idx in order:
        x = xs_px[idx]
        for o in steps:
            if all((x - px) ** 2 + (o - py) ** 2 >= d_px ** 2 for px, py in placed):
                off[idx] = o
                placed.append((x, o))
                break
        else:
            return swarm_offsets(xs_px, d_px, band_px)
    return off


def ring(dist=6.0, diag=4.5):
    """Default candidate offsets (points) for place_labels, nearest first."""
    c = [(dist, 0, 'left', 'center'), (-dist, 0, 'right', 'center'),
         (0, dist, 'center', 'bottom'), (0, -dist, 'center', 'top'),
         (diag, diag, 'left', 'bottom'), (diag, -diag, 'left', 'top'),
         (-diag, diag, 'right', 'bottom'), (-diag, -diag, 'right', 'top')]
    far = []
    for k in (1.8, 2.6, 3.4):
        far += [(dx * k, dy * k, ha, va) for dx, dy, ha, va in c]
    return c + far


# ------------------------------------------------------------- file names --
# Output files carry the figure's number in the paper: <number>_<name>.
FIGURE_NUMBERS = {
    'overview': 1, 'factor_decomposition': 2, 'budget_sweep': 3,
    'persona_sensitivity': 4, 'strategy_comparison': 5, 'dataset_heatmap': 6,
    'scaling_curve': 7, 'token_budget': 8, 'overthinking_delta': 9,
    'instruction_following': 10, 'prompt_sensitivity': 11, 'stress_variants': 12,
    'summeval_small_vs_large': 13, 'paradigm_divergence': 14, 'slm_vs_large': 15,
}


def file_stem(name):
    """File name (without extension) for a figure: '2_factor_decomposition'."""
    return f'{FIGURE_NUMBERS[name]}_{name}'


# ------------------------------------------------------------------ output --
# What finish() writes. make_figures.py sets these through configure(); a
# figure script run on its own writes the print PDF and PNG to
# results/figures/.
SVG_LIMIT = 400_000      # bytes; a larger README image is written as PNG
ASSET_PNG_DPI = 200      # README PNGs: about twice screen resolution
PREVIEW_DPI = 300
_OUT = {'paper': True, 'formats': ('pdf', 'png'), 'paper_dir': None,
        'asset_dir': None, 'check': True}
WRITTEN = []             # files written by finish(), for the caller's report


def configure(paper=True, formats=('pdf', 'png'), paper_dir=None,
              asset_dir=None, check=True):
    """Choose what finish() writes. paper: write the print figure in
    `formats` to `paper_dir` (default results/figures). asset_dir: also write
    a README image <number>_<name>-<theme>.svg (or .png when the SVG is too large)."""
    _OUT.update(paper=paper, formats=tuple(formats), paper_dir=paper_dir,
                asset_dir=asset_dir, check=check)


def paper_dir():
    d = _OUT['paper_dir'] or (paths.results_dir() / 'figures')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path, **kw):
    fig.savefig(path, bbox_inches=None, pad_inches=0, facecolor=PAPER, **kw)
    WRITTEN.append(Path(path))


def save_asset(fig, name, asset_dir):
    """README image for the current theme: SVG if small enough, else PNG."""
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    svg = asset_dir / f'{file_stem(name)}-{THEME}.svg'
    png = asset_dir / f'{file_stem(name)}-{THEME}.png'
    fig.savefig(svg, bbox_inches=None, pad_inches=0, facecolor=PAPER,
                metadata={'Date': None})
    if svg.stat().st_size <= SVG_LIMIT:
        png.unlink(missing_ok=True)
        WRITTEN.append(svg)
    else:
        svg.unlink()
        fig.savefig(png, bbox_inches=None, pad_inches=0, facecolor=PAPER,
                    dpi=ASSET_PNG_DPI)
        WRITTEN.append(png)


def finish(fig, name):
    """Run the overlap check, then write the figure as configured. Returns
    the list of layout problems."""
    probs = check_overlaps(fig) if _OUT['check'] else []
    if probs:
        print(f'  !! {name}: {len(probs)} layout problem(s)')
    if _OUT['paper'] and THEME == 'light':
        out = paper_dir()
        for ext in _OUT['formats']:
            if ext == 'pdf':
                _save(fig, out / f'{file_stem(name)}.pdf',
                      metadata={'Creator': 'SLMJury', 'Producer': None,
                                'CreationDate': None})
            elif ext == 'png':
                _save(fig, out / f'{file_stem(name)}.png', dpi=PREVIEW_DPI)
            elif ext == 'svg':
                _save(fig, out / f'{file_stem(name)}.svg', metadata={'Date': None})
            else:
                _save(fig, out / f'{file_stem(name)}.{ext}')
    if _OUT['asset_dir']:
        save_asset(fig, name, _OUT['asset_dir'])
    plt.close(fig)
    return probs


__all__ = [n for n in dir() if not n.startswith('_')]
