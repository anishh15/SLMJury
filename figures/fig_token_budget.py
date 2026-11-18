"""Accuracy at B=10 and B=8,192 for the 16 judges that run at both budgets (13 small, 3 large).

Reads results/analysis/main_results/individual.csv and large_judges/small_vs_large.csv; writes
token_budget (6.30 x 2.75 in).
"""
from style import *   # noqa: F401,F403

NAME = 'token_budget'


def main():
    acc = {}
    for r in load('main_results', 'individual.csv'):
        acc.setdefault(r['judge'], {})[r['max_tokens']] = float(r['accuracy'])
    for r in load('large_judges', 'small_vs_large.csv'):
        if r['type'] == 'Large' and r['acc_t10'] not in ('', 'None'):
            acc[r['judge']] = {'10': float(r['acc_t10']),
                               '8192': float(r['acc_t8192'])}
    judges = sorted((j for j in acc if '10' in acc[j] and '8192' in acc[j]),
                    key=lambda j: acc[j]['10'])
    assert len(judges) == 16, judges

    fig = new_figure(W_FULL, 2.75)
    ax = axes_in(fig, 0.42, 0.62, 5.82, 1.88)
    style_ax(ax, grid='y')
    x = np.arange(len(judges))
    w = 0.38
    cols = [jcolor(j) for j in judges]
    ax.bar(x - w / 2, [acc[j]['10'] for j in judges], w, color=cols,
           edgecolor=PAPER, linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, [acc[j]['8192'] for j in judges], w,
           color=[tint(c, 0.58) for c in cols], edgecolor=PAPER,
           linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([dn(j) for j in judges], rotation=35, ha='right',
                       rotation_mode='anchor')
    ax.tick_params(axis='x', length=0, pad=2.5)
    ax.set_xlim(-0.6, len(judges) - 0.4)
    ax.set_ylim(30, 100)
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylabel('Accuracy (%)')
    fams = [f for f in FAMILY_ORDER if any(family_of(j) == f for j in judges)]
    hs = family_patches(fams)
    hs += [mpatches.Patch(color=SLATE, label='$B{=}10$ (solid)'),
           mpatches.Patch(color=tint(SLATE, 0.58), label='$B{=}8{,}192$ (light)')]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=len(hs), handlelength=1.0, columnspacing=1.1,
              borderaxespad=0.25, handletextpad=0.35)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
