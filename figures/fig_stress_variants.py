"""The two stress-test prompts (adversarial, few-shot) against each judge's original prompt, at both budgets.

Reads results/analysis/prompt_sensitivity/stress_variants.csv; writes stress_variants (6.30 x 3.60 in).
"""
from style import *   # noqa: F401,F403

NAME = 'stress_variants'
LABEL_AT = Decimal('10.00')     # label every change of 10 points or more


def main():
    by = {}
    for r in load('prompt_sensitivity', 'stress_variants.csv'):
        by.setdefault(r['max_tokens'], {})[r['judge']] = r
    judges = sorted(by['10'], key=lambda j: -float(by['10'][j]['acc_original']))
    assert len(judges) == 13

    fig = new_figure(W_FULL, 3.60)
    left, bottom, w, h, gap = 0.76, 0.40, 2.66, 2.96, 0.14
    axes = [axes_in(fig, left, bottom, w, h),
            axes_in(fig, left + w + gap, bottom, w, h)]
    y = np.arange(len(judges))
    bh = 0.38
    series = [('delta_adversarial', 'Adversarial', 0.0, -bh / 2, 1.2),
              ('delta_few_shot', 'Few-shot', 0.58, bh / 2, -1.2)]
    for ax, tok, title in ((axes[0], '10', '(a) Quick verdict, $B{=}10$'),
                           (axes[1], '8192', '(b) Reasoned verdict, $B{=}8{,}192$')):
        style_ax(ax, grid='x')
        d = by[tok]
        for col, lab, tt, dy, nud in series:
            vals = [float(d[j][col]) for j in judges]
            ax.barh(y + dy, vals, bh, color=[tint(jcolor(j), tt) for j in judges],
                    edgecolor='none', zorder=3)
            for yi, j, v in zip(y, judges, vals):
                if abs(half_up(d[j][col], 2)) < LABEL_AT:
                    continue
                ax.annotate(fmt(d[j][col], 2), xy=(v, yi + dy),
                            xytext=(-2.5 if v < 0 else 2.5, nud),
                            textcoords='offset points',
                            ha='right' if v < 0 else 'left', va='center',
                            fontsize=FS_ANNOT, color=INK)
        ax.axvline(0, color=AXIS, lw=0.6, zorder=4, gid='skip-overlap')
        ax.set_ylim(len(judges) - 0.32, -0.55)
        ax.set_xlim(-60, 12)
        ax.set_xticks([-50, -40, -30, -20, -10, 0, 10])
        ax.xaxis.set_major_formatter(tick_fmt(0))
        ax.set_yticks(y)
        ax.tick_params(axis='y', length=0)
        panel_title(ax, title)
    axes[0].set_yticklabels([dn(j) for j in judges])
    axes[1].set_yticklabels([])
    hs = [mpatches.Patch(color=SLATE, label='Adversarial (solid)'),
          mpatches.Patch(color=tint(SLATE, 0.58), label='Few-shot (light)'),
          Line2D([], [], linestyle='none', label=' ')]
    hs += family_patches([f for f in FAMILY_ORDER if f != 'Large'])
    axes[1].legend(handles=hs, loc='lower left', handlelength=1.0,
                   borderaxespad=0.4, fontsize=FS_LEGEND - 0.5, labelspacing=0.25)
    fig.text((left + w + gap / 2) / W_FULL, 0.06 / 3.60,
             'Change in accuracy from the original prompt (points)',
             ha='center', va='bottom', fontsize=FS_LABEL, color=INK)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
