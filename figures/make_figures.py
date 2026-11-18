#!/usr/bin/env python3
"""Draw every figure from the analysis CSVs in results/analysis/."""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import style
import overview

NOTES = """\
For each figure this writes the print version to results/figures/ (PDF at
its printed size plus a 300-dpi PNG preview by default) in the light print
theme. For each data figure it also writes the README image to
assets/figures/<number>_<name>-dark.svg in the dark theme: the same layout on
the README's deep-space background. A README image is written as PNG (about
200 dpi) instead when its SVG would be larger than 400 KB. The overview
(1_overview) is a print figure only and has no README image.

After drawing, each print PDF is checked for its page size (6.30 in or
3.03 in wide, within 0.1 in of its set height), for Type 3 fonts and for
fonts other than the one style.py chose. Layout problems found while drawing
(overlapping text, labels on marks, text off the canvas, type below 6 pt)
are listed at the end. The exit code is 1 if a figure fails or has a
problem.

examples, from the repository root:
  python figures/make_figures.py                       # every figure
  python figures/make_figures.py --list
  python figures/make_figures.py budget_sweep overview # only these
  python figures/make_figures.py --formats pdf,svg     # print formats

The overview needs the Source Sans Pro and Source Code Pro fonts from a TeX
installation; without them it is skipped with a message. The PDF checks need
PyMuPDF and are skipped without it.
Results are read from results/ (or $SLMJURY_RESULTS).
"""

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / 'assets' / 'figures'

# (name, printed width in inches, printed height in inches)
FIGURES = [
    ('factor_decomposition', style.W_FULL, 2.45),
    ('budget_sweep', style.W_FULL, 3.05),
    ('persona_sensitivity', style.W_FULL, 2.85),
    ('strategy_comparison', style.W_COL, 2.82),
    ('dataset_heatmap', style.W_FULL, 2.55),
    ('scaling_curve', style.W_FULL, 2.70),
    ('token_budget', style.W_FULL, 2.75),
    ('overthinking_delta', style.W_COL, 3.30),
    ('instruction_following', style.W_FULL, 2.55),
    ('prompt_sensitivity', style.W_FULL, 3.35),
    ('stress_variants', style.W_FULL, 3.60),
    ('summeval_small_vs_large', style.W_FULL, 3.60),
    ('paradigm_divergence', style.W_FULL, 3.00),
    ('slm_vs_large', style.W_FULL, 3.35),
]
OVERVIEW = overview.NAME
NAMES = [OVERVIEW] + [n for n, _, _ in FIGURES]


def run_figure(name):
    """Import (or re-import, so it picks up the current theme) and draw one
    figure. Returns its layout problems."""
    module = f'fig_{name}'
    if module in sys.modules:
        mod = importlib.reload(sys.modules[module])
    else:
        mod = importlib.import_module(module)
    return mod.main() or []


def check_pdf(path, width, height):
    """Problems with a print PDF's size and fonts, or None without PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return None
    doc = fitz.open(path)
    w, h = doc[0].rect.width / 72, doc[0].rect.height / 72
    fonts = {f[3].split('+')[-1] for p in doc for f in p.get_fonts(full=True)}
    types = {f[2] for p in doc for f in p.get_fonts(full=True)}
    probs = []
    if abs(w - width) > 0.01:
        probs.append(f'width {w:.2f} in, expected {width:.2f}')
    if abs(h - height) > 0.1 + 1e-6:
        probs.append(f'height {h:.2f} in, expected {height:.2f}')
    if 'Type3' in types:
        probs.append('Type 3 font')
    family = style.FONT.replace(' ', '')
    other = sorted(f for f in fonts if not f.startswith(family))
    if other:
        probs.append(f'fonts other than {style.FONT}: {other}')
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Draw the figures from results/analysis.', epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('figures', nargs='*', metavar='NAME',
                    help='figures to draw (default: all); see --list')
    ap.add_argument('--list', action='store_true', help='print the figure names and stop')
    ap.add_argument('--formats', default='pdf,png',
                    help='comma-separated print formats for results/figures/ '
                         '(default: pdf,png)')
    args = ap.parse_args()

    if args.list:
        for n in NAMES:
            print(n)
        return 0
    unknown = [n for n in args.figures if n not in NAMES]
    if unknown:
        ap.error(f'unknown figure(s): {", ".join(unknown)}; see --list')
    names = args.figures or NAMES
    formats = [f.strip().lower() for f in args.formats.split(',') if f.strip()]
    data_figs = [(n, w, h) for n, w, h in FIGURES if n in names]

    failed, layout, skipped = [], [], []
    t0 = time.perf_counter()
    print(f'font: {style.FONT}')

    # Light pass: print versions only.
    style.set_theme('light')
    style.configure(paper=True, formats=formats)
    for name, _, _ in data_figs:
        try:
            probs = run_figure(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        layout += [f'{name}: {p}' for p in probs]

    # Dark pass: README images (same layout, so no second check).
    style.set_theme('dark')
    style.configure(paper=False, asset_dir=ASSETS, check=False)
    for name, _, _ in data_figs:
        if name in failed:
            continue
        try:
            run_figure(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    style.set_theme('light')

    if OVERVIEW in names:
        try:
            written, probs = overview.build(formats=formats, verbose=False)
            style.WRITTEN.extend(written)
            layout += [f'{OVERVIEW}: {p}' for p in probs]
        except overview.ToolMissing as e:
            print(f'  overview skipped: {e}')
            skipped.append(OVERVIEW)
        except Exception:
            traceback.print_exc()
            failed.append(OVERVIEW)

    print()
    if 'pdf' in formats:
        out = style.paper_dir()
        for name, w, h in data_figs:
            if name in failed:
                continue
            probs = check_pdf(out / f'{style.file_stem(name)}.pdf', w, h)
            if probs is None:
                print('PDF checks skipped (PyMuPDF not installed)')
                break
            print(f'  {name:26} {"; ".join(probs) if probs else "ok"}')
            layout += [f'{name}: {p}' for p in probs]

    print(f'\nwrote {len(style.WRITTEN)} file(s) in {time.perf_counter() - t0:.1f}s')
    for p in style.WRITTEN:
        try:
            shown = p.resolve().relative_to(REPO)
        except ValueError:
            shown = p
        print(f'  {shown}')
    if skipped:
        print(f'skipped: {", ".join(skipped)}')
    if layout:
        print(f'\n{len(layout)} layout problem(s):')
        for p in layout:
            print(f'  {p}')
    if failed:
        print(f'\nfailed: {", ".join(failed)}')
    return 1 if failed or layout else 0


if __name__ == '__main__':
    sys.exit(main())
