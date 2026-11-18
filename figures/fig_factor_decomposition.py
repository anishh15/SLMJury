"""Single-factor decomposition of the change from B=10 to B=8,192 (thirteen judges, all eight datasets).

(a) each factor changed alone; (b) the prompt-format change against the instruction-following rate.
Reads results/analysis/token_budget/single_factor_table.csv; writes factor_decomposition (6.30 x 2.45 in).
"""
from style import *   # noqa: F401,F403

NAME = 'factor_decomposition'
FACTORS = [('factor_budget', 'Budget'), ('factor_temperature', 'Temperature'),
           ('factor_thinking', 'Thinking mode'), ('factor_prompt', 'Prompt format')]
MS_A = 3.6          # family markers in the prompt row of (a) (pt)
S_A = 13            # scatter size of the neutral dots in (a) (pt^2)
MS_B = 4.6          # markers in (b)
FS_POINT = 6.0      # judge labels in (b)
# The five judges near 100% IFR: labels in a column right of the points, at
# these y values (data units).
RIGHT_COL = {'llama3.2-1b': 21.4, 'qwen3-0.6b': 12.4, 'qwen2.5-1.5b': 4.0,
             'phi4mi-3.8b': -4.0, 'qwen3-1.7b': -12.0}
# The other eight: fixed label positions (x, y in data units, ha); points in
# the middle cluster are too close for automatic placement. A leader is drawn
# where a label sits away from its point.
LABEL_AT = {'qwen3-14b': (79.0, -24.5, 'left'), 'qwen3-8b': (72.5, -32.0, 'left'),
            'phi4-14b': (70.0, -42.5, 'left'), 'qwen3-4b': (53.5, -32.5, 'right'),
            'llama3.1-8b': (51.0, -50.5, 'left'), 'qwen2.5-3b': (33.0, -40.5, 'right'),
            'qwen2.5-7b': (31.5, -63.5, 'left'), 'llama3.2-3b': (21.0, -71.0, 'left')}


def rank_jitter(xs, half=0.24):
    """Jitter ordered by value, evenly spread in +-half. The y axis of (a)
    runs downward, so larger values get smaller offsets and sit higher (a
    rising diagonal)."""
    order = np.argsort(xs, kind='stable')
    jit = np.empty(len(xs))
    jit[order] = np.linspace(half, -half, len(xs))
    return jit


def main():
    t = {}
    for r in load('token_budget', 'single_factor_table.csv'):
        t.setdefault(r['judge'], {})[r['condition']] = r
    judges = sorted((j for j in t if 'baseline_t10' in t[j]),
                    key=lambda j: -float(t[j]['baseline_t10']['accuracy']))
    assert len(judges) == 13

    H = 2.45
    fig = new_figure(W_FULL, H)
    bot, ph = 0.34, 1.86
    axa = axes_in(fig, 0.70, bot, 3.05, ph)
    axb = axes_in(fig, 4.27, bot, 1.98, ph)

    # ---------------------------------------------------------------- (a)
    style_ax(axa, grid=None)
    axa.set_xlim(-73, 27)
    axa.set_ylim(len(FACTORS) - 0.45, -0.6)
    axa.axvline(0, color=AXIS, lw=0.6, zorder=1, gid='skip-overlap')
    for i, (cond, _) in enumerate(FACTORS):
        axa.axhline(i, color=GRID, lw=0.5, zorder=0, gid='skip-overlap')
        xs = np.array([float(t[j][cond]['delta_vs_t10']) for j in judges])
        jit = rank_jitter(xs)
        axa.plot([xs.min(), xs.max()], [i, i], color=NAVY_LIGHT, lw=2.0,
                 solid_capstyle='round', zorder=2, gid='skip-overlap')
        if cond == 'factor_prompt':
            for j, x, o in zip(judges, xs, jit):
                judge_mark(axa, x, i + o, j, size=MS_A, z=5)
        else:
            axa.scatter(xs, i + jit, s=S_A, marker='o', color=SLATE,
                        edgecolor=PAPER, linewidth=0.5, zorder=4)
    axa.set_yticks(range(len(FACTORS)))
    axa.set_yticklabels([lab for _, lab in FACTORS])
    axa.tick_params(axis='y', length=0)
    axa.set_xticks([-60, -40, -20, 0, 20])
    axa.xaxis.set_major_formatter(tick_fmt(0, sign=True))
    axa.set_xlabel('Change in accuracy from $B{=}10$ (points)')
    panel_title(axa, '(a) Each factor changed alone')
    # One range note per factor, rounded once from the CSV values; the
    # prompt-format note in ink, the rest muted.
    for i, (cond, _) in enumerate(FACTORS):
        vals = [t[j][cond]['delta_vs_t10'] for j in judges]
        if cond == 'factor_budget':
            k = sum(1 for v in vals if abs(float(v)) < 0.01)
            txt = f'under 0.01 for {k}/{len(vals)} judges'
            axa.text(-3.5, i - 0.31, txt, ha='right', va='bottom',
                     fontsize=FS_ANNOT, color=MUTED)
            continue
        lo = min(vals, key=float)
        hi = max(vals, key=float)
        txt = f'{fmt(lo, 1, sign=True)} to {fmt(hi, 1, sign=True)}'
        axa.text(26, i - 0.31, txt, ha='right', va='bottom', fontsize=FS_ANNOT,
                 color=INK if cond == 'factor_prompt' else MUTED)

    # ---------------------------------------------------------------- (b)
    style_ax(axb, grid='y')
    ifr = np.array([float(t[j]['factor_prompt']['ifr']) for j in judges])
    dlt = np.array([float(t[j]['factor_prompt']['delta_vs_t10']) for j in judges])
    r = float(np.corrcoef(ifr, dlt)[0, 1])
    axb.axhline(0, color=AXIS, lw=0.6, zorder=1, gid='skip-overlap')
    slope, icpt = np.polyfit(ifr, dlt, 1)
    gx = np.linspace(12, 100, 50)
    axb.plot(gx, slope * gx + icpt, color=SLATE, lw=0.7, ls=(0, (4, 3)),
             zorder=2)
    for j, x, y in zip(judges, ifr, dlt):
        judge_mark(axb, x, y, j, size=MS_B, z=5)
    axb.set_ylim(-76, 30)
    axb.set_xticks([0, 25, 50, 75, 100])
    axb.set_yticks([-60, -40, -20, 0, 20])
    axb.yaxis.set_major_formatter(tick_fmt(0, sign=True))
    axb.set_xlabel('Instruction-following rate (%)')
    axb.set_ylabel('Accuracy change (points)')
    panel_title(axb, '(b) Prompt format: change vs. IFR')
    axb.text(0.04, 0.95, f'$r$ = {fmt(r, 2)}', transform=axb.transAxes,
             ha='left', va='top', fontsize=FS_ANNOT + 0.5, color=INK)

    # Right-hand label column: the x range runs past 100 just far enough for
    # the longest of the five labels to end at the axes' right edge, and
    # starts a little below 0 so the left labels clear the axis.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    probe = axb.text(0, 0, 'Qwen2.5-1.5B', fontsize=FS_POINT)
    w_in = probe.get_window_extent(rend).width / fig.dpi
    probe.remove()
    x0, xmin = 103.5, -8.0
    wb = axb.get_position().width * W_FULL
    k = (w_in + 0.03) / wb
    xmax = (x0 - k * xmin) / (1 - k)
    axb.set_xlim(xmin, xmax)
    for j, x, y in zip(judges, ifr, dlt):
        if j not in RIGHT_COL:
            continue
        ly = RIGHT_COL[j]
        tx = axb.text(x0, ly, dn(j), ha='left', va='center',
                      fontsize=FS_POINT, color=deepen(jcolor(j), 0.25), zorder=8)
        if abs(ly - y) > 2.5:
            fig.canvas.draw()
            add_leader(axb, tx, x, y, shrink_pt=2.6)

    fams = [f for f in FAMILY_ORDER if any(family_of(j) == f for j in judges)]
    axb.legend(handles=family_handles(fams, size=MS_A), loc='lower right',
               ncol=2, handlelength=0.9, columnspacing=0.8,
               borderaxespad=0.2, handletextpad=0.3, fontsize=FS_LEGEND - 1)

    for j, x, y in zip(judges, ifr, dlt):
        if j in RIGHT_COL:
            continue
        lx, ly, ha = LABEL_AT[j]
        tx = axb.text(lx, ly, dn(j), ha=ha, va='center', fontsize=FS_POINT,
                      color=deepen(jcolor(j), 0.25), zorder=8)
        if abs(ly - y) > 4.0 or abs(lx - x) > 7.0:
            fig.canvas.draw()
            add_leader(axb, tx, x, y, shrink_pt=2.6)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
