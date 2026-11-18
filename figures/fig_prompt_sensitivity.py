"""Prompt sensitivity at B=10: thirteen judges under five instructions (heatmap, then each judge's range).

Reads results/analysis/prompt_sensitivity/shared_accuracy_table.csv; writes prompt_sensitivity
(6.30 x 3.35 in).
"""
from style import *   # noqa: F401,F403

NAME = 'prompt_sensitivity'
FORMS = ['original', 'minimal', 'gpt_style', 'rubric_based', 'cot_style']
HEAD = {'original': 'original', 'minimal': 'minimal', 'gpt_style': 'gpt',
        'rubric_based': 'rubric', 'cot_style': 'cot'}
STABLE = 2.0
VMIN, VMAX = 65.0, 90.0


def main():
    rows = [r for r in load('prompt_sensitivity', 'shared_accuracy_table.csv')
            if r['max_tokens'] == '10']
    rows = [r for r in rows if all(r[f'acc_{f}'] not in ('', 'None') for f in FORMS)]
    rows.sort(key=lambda r: float(r['acc_range']))
    assert len(rows) == 13
    judges = [r['judge'] for r in rows]
    vals = [[r[f'acc_{f}'] for f in FORMS] for r in rows]
    data = np.array([[float(v) for v in row] for row in vals])
    n, m = data.shape

    fig = new_figure(W_FULL, 3.35)
    top, h = 3.08, 2.70
    ax = axes_in(fig, 0.74, top - h, 3.40, h)
    cmap = seq_cmap()
    cmap.set_under(UNDER)
    im = ax.imshow(data, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect='auto',
                   interpolation='nearest')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color=PAPER, linewidth=1.0)
    ax.tick_params(which='both', length=0)
    ax.set_xticks(range(m))
    ax.set_xticklabels([HEAD[f] for f in FORMS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([dn(j) for j in judges])
    norm = plt.Normalize(VMIN, VMAX)
    for i in range(n):
        for k in range(m):
            ax.text(k, i, fmt(vals[i][k], 2), ha='center', va='center',
                    fontsize=FS_ANNOT, color=text_on(cmap(norm(data[i, k]))),
                    gid='cell')
    panel_title(ax, '(a) Accuracy (%) under each formulation')

    cax = axes_in(fig, 4.22, top - h, 0.08, h)
    cb = fig.colorbar(im, cax=cax, extend='min', extendfrac=0.04)
    cb.outline.set_visible(False)
    cb.set_ticks([65, 70, 75, 80, 85, 90])
    cb.ax.tick_params(length=2.0, width=0.5, color=AXIS)

    # (b) range per judge, rows aligned with (a)
    axr = axes_in(fig, 4.78, top - h, 1.46, h)
    style_ax(axr, grid='x')
    rng = [float(r['acc_range']) for r in rows]
    cols = [tint(jcolor(r['judge']), 0.58) if v < STABLE else jcolor(r['judge'])
            for r, v in zip(rows, rng)]
    axr.barh(range(n), rng, height=0.62, color=cols, edgecolor='none',
             zorder=3)
    for i, r in enumerate(rows):
        axr.annotate(fmt(r['acc_range'], 2), xy=(float(r['acc_range']), i),
                     xytext=(2.5, 0), textcoords='offset points', ha='left',
                     va='center', fontsize=FS_ANNOT, color=INK)
    axr.set_ylim(n - 0.5, -0.5)
    axr.set_yticks(range(n))
    axr.set_yticklabels([])
    axr.tick_params(axis='y', length=0)
    axr.set_xlim(0, 19)
    axr.set_xticks([0, 5, 10, 15])
    axr.set_xlabel('Range (points)')
    panel_title(axr, '(b) Range')
    hs = [mpatches.Patch(color=tint(SLATE, 0.58), label='< 2 points (light)'),
          mpatches.Patch(color=SLATE, label='≥ 2 points (solid)'),
          Line2D([], [], linestyle='none', label=' ')]
    hs += family_patches([f for f in FAMILY_ORDER if f != 'Large'])
    axr.legend(handles=hs, loc='upper right', handlelength=1.0,
               borderaxespad=0.2, fontsize=FS_LEGEND - 0.5, labelspacing=0.25)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
