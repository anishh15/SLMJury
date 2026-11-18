"""Accuracy across the nine-point token-budget sweep for seven judges (B = 10 to 8,192, log axis).

Reads results/analysis/token_budget/budget_sweep_overall.csv; writes budget_sweep (6.30 x 3.05 in).
"""
from style import *   # noqa: F401,F403

NAME = 'budget_sweep'
BUDGETS = [10, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DASH = (0, (3.2, 1.6))
# (judge, shade, line style), grouped by family.
SERIES = [('phi4-14b', 0.0, '-'), ('phi4mi-3.8b', 0.0, DASH),
          ('qwen3-14b', 0.0, '-'), ('qwen3-8b', 0.0, DASH),
          ('qwen2.5-7b', 0.0, '-'), ('qwen2.5-3b', 0.0, DASH),
          ('llama3.1-8b', 0.0, '-')]


def main():
    rows = {r['judge']: r for r in load('token_budget', 'budget_sweep_overall.csv')}
    fig = new_figure(W_FULL, 3.05)
    ax = axes_in(fig, 0.42, 0.36, 5.82, 2.62)
    style_ax(ax, grid='y')

    ax.axvspan(64, 256, color=NAVY_PALE, alpha=0.75, lw=0, zorder=0.5,
               gid='band')
    ax.axvline(512, color=MUTED, lw=0.8, ls=(0, (1, 1.6)), zorder=1,
               gid='skip-overlap')

    handles = []
    for j, t, ls in SERIES:
        r = rows[j]
        ys = [float(r[f'b{b}']) for b in BUDGETS]
        c = shade(jcolor(j), t)
        m = jmarker(j)
        ln, = ax.plot(BUDGETS, ys, color=c, lw=1.1, ls=ls, marker=m,
                      markersize=3.4 * MARKER_SCALE[m], markerfacecolor=c,
                      markeredgecolor=PAPER, markeredgewidth=0.45,
                      zorder=4, label=dn(j))
        handles.append(ln)

    ax.set_xscale('log', base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f'{b:,}' for b in BUDGETS])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(8.2, 10000)
    ax.set_ylim(49, 92.5)
    ax.set_yticks([50, 60, 70, 80, 90])
    ax.set_xlabel('Token budget $B$ (log scale)')
    ax.set_ylabel('Accuracy (%)')
    # Column-major fill: solid members in the left column, dashed members in
    # the right one, so each legend row is one family.
    order = [0, 2, 4, 6, 1, 3, 5]
    ax.legend(handles=[handles[i] for i in order], loc='lower right', ncol=2,
              handlelength=2.6, columnspacing=1.4, borderaxespad=0.6)
    ax.text(128, 50.2, 'truncated reasoning', ha='center', va='bottom',
            fontsize=FS_ANNOT, color=NAVY_MID)
    ax.text(560, 91.6, 'most converged by $B{=}512$', ha='left', va='center',
            fontsize=FS_ANNOT, color=MUTED)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
