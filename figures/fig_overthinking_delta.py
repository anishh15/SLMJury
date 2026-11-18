"""Accuracy change from reasoning, Acc(B=10) - Acc(B=8,192), for the 16 judges that run at both budgets.

Reads results/analysis/main_results/individual.csv and large_judges/small_vs_large.csv; writes
overthinking_delta (3.03 x 3.30 in, one column).
"""
from style import *   # noqa: F401,F403

NAME = 'overthinking_delta'


def main():
    delta = {}
    for r in load('main_results', 'individual.csv'):
        if r['max_tokens'] == '10':
            delta[r['judge']] = r['delta_t10_minus_t8192']
    for r in load('large_judges', 'small_vs_large.csv'):
        if r['type'] == 'Large' and r['acc_t10'] not in ('', 'None'):
            d = Decimal(r['acc_t10']) - Decimal(r['acc_t8192'])
            delta[r['judge']] = str(d)
    judges = sorted(delta, key=lambda j: float(delta[j]))
    assert len(judges) == 16

    fig = new_figure(W_COL, 3.30)
    ax = axes_in(fig, 0.78, 0.40, 2.18, 2.84)
    style_ax(ax, grid='x')
    y = np.arange(len(judges))
    vals = [float(delta[j]) for j in judges]
    cols = [jcolor(j) if v > 0 else tint(jcolor(j), 0.58)
            for j, v in zip(judges, vals)]
    ax.barh(y, vals, height=0.66, color=cols, edgecolor='none', zorder=3)
    ax.axvline(0, color=AXIS, lw=0.6, zorder=4, gid='skip-overlap')
    for yi, j, v in zip(y, judges, vals):
        ax.annotate(fmt(delta[j], 2, sign=True), xy=(v, yi),
                    xytext=(2.0 if v > 0 else -2.0, 0),
                    textcoords='offset points', ha='left' if v > 0 else 'right',
                    va='center', fontsize=FS_ANNOT, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([dn(j) for j in judges])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(judges) - 0.4)
    ax.set_xlim(-30, 7)
    ax.set_xticks([-30, -20, -10, 0])
    ax.xaxis.set_major_formatter(tick_fmt(0))
    ax.set_xlabel(r'$\Delta$ = Acc at $B{=}10$ $-$ Acc at $B{=}8{,}192$ (points)',
                  fontsize=FS_LABEL)
    hs = [mpatches.Patch(color=SLATE, label='Quick verdict better (solid)'),
          mpatches.Patch(color=tint(SLATE, 0.58), label='Reasoning better (light)'),
          Line2D([], [], linestyle='none', label=' ')]
    hs += family_patches(FAMILY_ORDER)
    ax.legend(handles=hs, loc='upper left', handlelength=1.0,
              borderaxespad=0.3, fontsize=FS_LEGEND - 0.5, labelspacing=0.25)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
