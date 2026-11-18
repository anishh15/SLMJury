"""Instruction-following rate at both budgets, for the judges below 99.95% at either one.

Reads results/analysis/main_results/individual.csv and large_judges/small_vs_large.csv (checked only);
writes instruction_following (6.30 x 2.55 in).
"""
from style import *   # noqa: F401,F403

NAME = 'instruction_following'
CUT = 99.95


def main():
    ifr = {}
    for r in load('main_results', 'individual.csv'):
        ifr.setdefault(r['judge'], {})[r['max_tokens']] = r['ifr']
    for r in load('large_judges', 'small_vs_large.csv'):
        if r['type'] == 'Large' and r['ifr_t10'] not in ('', 'None'):
            assert min(float(r['ifr_t10']), float(r['ifr_t8192'])) >= CUT
    judges = [j for j, v in ifr.items() if '10' in v and '8192' in v
              and min(float(v['10']), float(v['8192'])) < CUT]
    judges.sort(key=lambda j: float(ifr[j]['10']))
    assert len(judges) == 8, judges

    fig = new_figure(W_FULL, 2.55)
    ax = axes_in(fig, 0.42, 0.44, 5.82, 1.86)
    style_ax(ax, grid='y')
    x = np.arange(len(judges))
    w = 0.42
    for dx, b, t in ((-w / 2, '10', 0.0), (w / 2, '8192', 0.58)):
        vals = [float(ifr[j][b]) for j in judges]
        ax.bar(x + dx, vals, w, color=[tint(jcolor(j), t) for j in judges],
               edgecolor=PAPER, linewidth=0.4, zorder=3)
        for xi, j, v in zip(x, judges, vals):
            ax.annotate(fmt(ifr[j][b], 2), xy=(xi + dx, v), xytext=(0, 1.5),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([dn(j) for j in judges], rotation=15, ha='right',
                       rotation_mode='anchor')
    ax.tick_params(axis='x', length=0, pad=2.5)
    ax.set_xlim(-0.55, len(judges) - 0.45)
    ax.set_ylim(93, 100.95)
    ax.set_yticks([93, 94, 95, 96, 97, 98, 99, 100])
    ax.set_ylabel('Instruction-following rate (%)')
    fams = [f for f in FAMILY_ORDER if any(family_of(j) == f for j in judges)]
    hs = [mpatches.Patch(color=SLATE, label='$B{=}10$ (solid)'),
          mpatches.Patch(color=tint(SLATE, 0.58), label='$B{=}8{,}192$ (light)')]
    hs += family_patches(fams)
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=len(hs), handlelength=1.0, columnspacing=1.1,
              borderaxespad=0.25, handletextpad=0.35)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
