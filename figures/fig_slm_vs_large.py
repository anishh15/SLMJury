"""Accuracy against judge size: accuracy levels off near 8-14B.

Every judge of 3B or more at its best budget. Reads results/analysis/large_judges/small_vs_large.csv and
main_results/individual.csv; writes slm_vs_large (6.30 x 3.35 in).
"""
import math

from matplotlib.legend_handler import HandlerTuple

from style import *   # noqa: F401,F403

NAME = 'slm_vs_large'
FLOOR = 85.5
MARGIN = 0.5
MS = 5.4          # small judges and the ladder judge
MS_LARGE = 6.0    # large judges


def main():
    rows = {r['judge']: r for r in load('large_judges', 'small_vs_large.csv')}
    e0 = {}
    for r in load('main_results', 'individual.csv'):
        e0.setdefault(r['judge'], {})[r['max_tokens']] = r['accuracy']
    pts = []
    for j, r in rows.items():
        p = float(r['params_b'])
        if p < 3:
            continue
        assert r['complete_8ds'] == 'True'
        acc = r['best_acc']
        if j in e0:     # six-decimal source for the small judges
            acc = e0[j][r['best_budget']]
            assert abs(float(acc) - float(r['best_acc'])) < 1e-3
        assert r['best_budget'] in ('10', '8192')
        pts.append({'j': j, 'p': p, 'acc': acc,
                    'filled': r['best_budget'] == '10'})
    assert len(pts) == 19
    phi = float(e0['phi4-14b']['10'])

    # Spread judges of (nearly) one size symmetrically in log space.
    pts.sort(key=lambda q: q['p'])
    clusters = []
    for q in pts:
        if clusters and math.log10(q['p']) - math.log10(clusters[-1][0]['p']) < 0.05:
            clusters[-1].append(q)
        else:
            clusters.append([q])
    for g in clusters:
        c = sum(math.log10(q['p']) for q in g) / len(g)
        g.sort(key=lambda q: -float(q['acc']))
        for i, q in enumerate(g):
            q['x'] = 10 ** (c + (i - (len(g) - 1) / 2) * 0.034)

    fig = new_figure(W_FULL, 3.35)
    ax = axes_in(fig, 0.42, 0.36, 5.82, 2.58)
    style_ax(ax, grid='y')
    ax.set_xscale('log')
    ax.set_xlim(2.55, 175)
    ax.set_ylim(FLOOR, 90.9)
    ax.axhspan(phi - MARGIN, phi + MARGIN, color=NAVY_PALE, lw=0, zorder=0.6,
               gid='band')
    ax.axhline(phi, color=NAVY_MID, lw=0.7, ls=(0, (4, 2.5)), zorder=1)

    above, floor = [], []
    for q in pts:
        a = float(q['acc'])
        size = MS_LARGE if q['j'] in LARGE else MS
        if a < FLOOR:
            floor.append(q)
            judge_mark(ax, q['x'], FLOOR + 0.09, q['j'], size=size, z=6,
                       filled=q['filled'], clip_on=False)
        else:
            above.append(q)
            judge_mark(ax, q['x'], a, q['j'], size=size, z=6,
                       filled=q['filled'])

    ax.set_xticks([3, 4, 8, 14, 32, 70, 120])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([86, 87, 88, 89, 90])
    ax.set_xlabel('Judge parameters (billions, log scale)')
    ax.set_ylabel('Accuracy at best budget (%)')

    # Legend, two rows above the plot: families; then budget and the band.
    leg1 = ax.legend(handles=family_handles(FAMILY_ORDER, size=MS - 0.6),
                     loc='lower left', bbox_to_anchor=(0.0, 1.0 + 0.17 / 2.58),
                     ncol=5, borderaxespad=0.0, handletextpad=0.3,
                     columnspacing=1.4)
    ax.add_artist(leg1)
    band = (mpatches.Patch(facecolor=NAVY_PALE, edgecolor='none'),
            Line2D([], [], color=NAVY_MID, lw=0.7, ls=(0, (4, 2.5))))
    hs = budget_handles(color=SLATE, size=MS - 0.6)
    hs[0].set_label('$B{=}10$ (best budget)')
    hs[1].set_label('$B{=}8{,}192$ (best budget)')
    ax.legend(handles=hs + [band],
              labels=[h.get_label() for h in hs] +
              [f'Phi-4 (dashed) {chr(177)}{fmt(MARGIN, 1)} points: '
               'equivalence margin'],
              handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
              loc='lower left', bbox_to_anchor=(0.0, 1.0 + 0.01 / 2.58),
              ncol=3, borderaxespad=0.0, handletextpad=0.3, columnspacing=1.4)

    # Labels for every judge on the plot.
    items = []
    for q in sorted(above, key=lambda q: -float(q['acc'])):
        items.append({'x': q['x'], 'y': float(q['acc']), 'text': dn_tag(q['j']),
                      'color': deepen(FAMILY_COLOR['Large'], 0.2)
                      if q['j'] in LARGE else INK})
    # The floor note, drawn first so the labels keep clear of it.
    parts = [f"{dn(q['j'])} {fmt(q['acc'], 2)}"
             for q in sorted(floor, key=lambda q: -float(q['acc']))]
    note = (f'Below {fmt(FLOOR, 1)}% (on the axis floor): {parts[0]},\n'
            + ', '.join(parts[1:]))
    ax.text(0.30, 0.13, note, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FS_ANNOT, color=MUTED, linespacing=1.25, zorder=7)
    place_labels(ax, items, ring(6.5, 5.0), pad_pt=0.8, leader_pt=9.0)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
