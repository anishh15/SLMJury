"""Per-dataset accuracy at B=10 for the ten most accurate judges, as a judge x dataset heatmap.

Reads results/analysis/large_judges/per_dataset_precise.csv and small_vs_large.csv (row order only);
writes dataset_heatmap (6.30 x 2.55 in).
"""
from style import *   # noqa: F401,F403

NAME = 'dataset_heatmap'
VMIN, VMAX = 70.0, 100.0


def main():
    per = {r['judge']: r for r in load('large_judges', 'per_dataset_precise.csv')
           if r['budget'] == 't10'}
    acc = {r['judge']: float(r['acc_t10'])
           for r in load('large_judges', 'small_vs_large.csv')
           if r['acc_t10'] not in ('', 'None') and r['type'] != 'Ladder'}
    judges = sorted((j for j in acc if j in per), key=lambda j: -acc[j])[:10]
    vals = [[per[j][d] for d in DATASETS] for j in judges]
    data = np.array([[float(v) for v in row] for row in vals])

    n, m = data.shape
    fig = new_figure(W_FULL, 2.55)
    ax = axes_in(fig, 0.80, 0.20, 4.92, 2.04)
    cmap = seq_cmap()
    im = ax.imshow(data, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect='auto',
                   interpolation='nearest')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color=PAPER, linewidth=1.0)
    ax.tick_params(which='both', length=0)
    for xg in (2.5, 4.5):   # domain boundaries
        ax.axvline(xg, color=PAPER, lw=3.0, zorder=3, gid='skip-overlap')
    ax.set_xticks(range(m))
    ax.set_xticklabels([DS_SHORT[d] for d in DATASETS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([dn(j) for j in judges])
    for lab, j in zip(ax.get_yticklabels(), judges):
        if j in LARGE:
            lab.set_color(deepen(FAMILY_COLOR['Large'], 0.2))

    norm = plt.Normalize(VMIN, VMAX)
    for i in range(n):
        for k in range(m):
            ax.text(k, i, fmt(vals[i][k], 2), ha='center', va='center',
                    fontsize=FS_ANNOT, color=text_on(cmap(norm(data[i, k]))),
                    gid='cell')

    # Domain groups over the columns.
    up = mtransforms.blended_transform_factory(ax.transData, ax.transAxes) + \
        mtransforms.ScaledTranslation(0, 3.0 / 72, fig.dpi_scale_trans)
    for name, cols in DOMAINS:
        k0, k1 = DATASETS.index(cols[0]), DATASETS.index(cols[-1])
        ax.plot([k0 - 0.42, k1 + 0.42], [1.0, 1.0], transform=up, color=AXIS,
                lw=0.5, clip_on=False, solid_capstyle='butt',
                gid='skip-overlap')
        ax.annotate(name, xy=((k0 + k1) / 2, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(0, 5.0),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=FS_TICK, color=INK, annotation_clip=False)

    cax = axes_in(fig, 5.82, 0.20, 0.09, 2.04)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([70, 75, 80, 85, 90, 95, 100])
    cb.ax.tick_params(length=2.0, width=0.5, color=AXIS)
    cb.set_label('Accuracy (%)', fontsize=FS_LABEL, labelpad=3)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
