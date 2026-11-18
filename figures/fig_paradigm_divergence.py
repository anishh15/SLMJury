"""Closed-ended accuracy against SummEval and MT-Bench Spearman rho for the 16 small judges.

Reads results/analysis/main_results/individual.csv and significance/open_ended_reported_ci.csv; writes
paradigm_divergence (6.30 x 3.00 in).
"""
from style import *   # noqa: F401,F403

NAME = 'paradigm_divergence'
MS = 4.2
NAMED = ['phi4-14b', 'qwen3-14b', 'phi4r-14b']


def main():
    acc, bud = {}, {}
    for r in load('main_results', 'individual.csv'):
        if float(r['accuracy']) > acc.get(r['judge'], 0.0):
            acc[r['judge']] = float(r['accuracy'])
            bud[r['judge']] = r['max_tokens']
    oe = {r['key']: r for r in load('significance', 'open_ended_reported_ci.csv')
          if r['type'] == 'SLM'}
    slms = sorted(oe, key=lambda j: -acc[j])
    assert len(slms) == 16 and set(slms) <= set(acc)

    H = 3.00
    fig = new_figure(W_FULL, H)
    bot, ph, pw = 0.38, 2.26, 2.62
    axes = [axes_in(fig, 0.48, bot, pw, ph), axes_in(fig, 3.62, bot, pw, ph)]
    specs = [('summeval_rho', r'Spearman $\rho$ with human scores',
              '(a) SummEval (human scores)', (-0.06, 0.60)),
             ('mtbench_rho_common', r'Spearman $\rho$ with reference model',
              '(b) MT-Bench (reference model)', (-0.16, 0.60))]
    x = np.array([acc[j] for j in slms])
    for ax, (col, ylab, title, ylim) in zip(axes, specs):
        style_ax(ax, grid='y')
        y = np.array([float(oe[j][col]) for j in slms])
        for j, xi, yi in zip(slms, x, y):
            judge_mark(ax, xi, yi, j, size=MS, z=5, filled=bud[j] == '10')
        ax.set_xlim(63.5, 91)
        ax.set_xticks([65, 70, 75, 80, 85, 90])
        ax.set_ylim(*ylim)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.yaxis.set_major_formatter(tick_fmt(1))
        ax.set_xlabel('Closed-ended accuracy (%)')
        ax.set_ylabel(ylab)
        panel_title(ax, title)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.04, 0.95, f'$r$ = {fmt(r, 2)}', transform=ax.transAxes,
                ha='left', va='top', fontsize=FS_ANNOT + 0.5, color=INK)
        # Only three judges are named; the others are identified by family
        # colour and shape (several sit too close together to label cleanly).
        items = [{'x': acc[j], 'y': float(oe[j][col]), 'text': dn(j)}
                 for j in NAMED]
        cands = [(dx, dy, 'right', va) for dx, dy, va in
                 ((-6, 0, 'center'), (-6, 6, 'bottom'), (-6, -6, 'top'),
                  (-12, 10, 'bottom'), (-12, -10, 'top'), (-18, 14, 'bottom'),
                  (-18, -14, 'top'), (-24, 0, 'center'))] + \
                [(0, 7, 'center', 'bottom'), (0, -7, 'center', 'top')]
        place_labels(ax, items, cands, pad_pt=0.8, leader_pt=9.0)
    fams = [f for f in FAMILY_ORDER if any(family_of(j) == f for j in slms)]
    bh = budget_handles(color=SLATE, size=MS)
    bh[0].set_label('accuracy at $B{=}10$')
    bh[1].set_label('accuracy at $B{=}8{,}192$')
    axes[0].legend(handles=family_handles(fams, size=MS) + bh, loc='lower left',
                   bbox_to_anchor=(0.0, 1.0 + 0.19 / ph), ncol=6,
                   handletextpad=0.3, columnspacing=1.4, borderaxespad=0.0)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
