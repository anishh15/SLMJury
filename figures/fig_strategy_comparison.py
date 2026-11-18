"""Comparison of closed-ended strategies: individual, persona, majority voting and multi-agent debate.

Reads results/analysis/main_results/individual.csv, personas.csv, juries.csv and debate.csv; writes
strategy_comparison (3.03 x 2.82 in, one column).
"""
import re

from style import *   # noqa: F401,F403

NAME = 'strategy_comparison'
PERSONAS = ['Strict', 'Lenient', 'Industry', 'Logic', 'Safety', 'Helpfulness']
LABELS = ['Individual', 'Persona', 'Majority\nvoting', 'Multi-agent\ndebate']
MS = 3.0          # marker size of every configuration


def main():
    ind = {}
    for r in load('main_results', 'individual.csv'):
        ind.setdefault(r['judge'], {})[r['max_tokens']] = r['accuracy']
    cfg = {k: [] for k in range(4)}        # (accuracy str, family or None, filled)
    for j, v in ind.items():
        b = max(v, key=lambda k: float(v[k]))
        cfg[0].append((v[b], family_of(j), b == '10'))
    per = {}
    for r in load('main_results', 'personas.csv'):
        m = sum(Decimal(r[p]) for p in PERSONAS) / len(PERSONAS)
        per.setdefault(r['judge'], {})[r['max_tokens']] = m
    for j, v in per.items():
        b = max(v, key=lambda k: v[k])
        cfg[1].append((str(v[b]), family_of(j), b == '10'))
    for r in load('main_results', 'juries.csv'):
        cfg[2].append((r['accuracy'], None, True))
    for r in load('main_results', 'debate.csv'):
        if r['max_tokens'] != '8192':
            continue
        js = [re.sub(r'-t[\d.]+$', '', t) for t in r['configuration'].split('+')]
        fams = {family_of(x) for x in js}
        cfg[3].append((r['accuracy'], fams.pop() if len(fams) == 1 else None, False))
    assert [len(cfg[k]) for k in range(4)] == [16, 5, 10, 7]
    best = [max(cfg[k], key=lambda c: float(c[0]))[0] for k in range(4)]

    W, H = W_COL, 2.82
    fig = new_figure(W, H)
    ax = axes_in(fig, 0.40, 0.36, 2.58, 1.98)
    style_ax(ax, grid='y')
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(60, 92.6)
    ax.set_yticks([60, 65, 70, 75, 80, 85, 90])
    ax.axhline(float(best[0]), color=NAVY_MID, lw=0.75, ls=(0, (4, 3)),
               zorder=1)
    fig.canvas.draw()
    pxx = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
    d = MS * 1.05 * fig.dpi / 72
    for k in range(4):
        ys = [float(c[0]) for c in cfg[k]]
        ypx = ax.transData.transform(np.c_[np.zeros(len(ys)), ys])[:, 1]
        off = beeswarm(ypx, d, 0.30 * pxx)
        for (a, fam, filled), o in zip(cfg[k], off):
            x = k + o / pxx
            if fam is None:
                judge_mark(ax, x, float(a), family='Qwen3', color=SLATE,
                           marker='o', filled=filled, size=MS, z=3)
            else:
                judge_mark(ax, x, float(a), family=fam, filled=filled,
                           size=MS, z=3)
    # The best of each strategy: a short navy bar across the column, drawn
    # behind the configuration points so none is hidden, with its value.
    for k in range(4):
        v = float(best[k])
        ax.plot([k - 0.32, k + 0.32], [v, v], color=NAVY, lw=1.6,
                solid_capstyle='butt', zorder=2.5, gid='skip-overlap')
        s = fmt(best[k], 2)
        # Above the bar, except the debate, which sits just under the
        # dashed line: its value goes to the left.
        off, ha, va = ((0, 4.6), 'center', 'bottom') if k < 3 else \
            ((-0.36, 0), 'right', 'center')
        xy = (k + off[0], v) if k == 3 else (k, v)
        ax.annotate(s, xy=xy, xytext=(0, 0) if k == 3 else off,
                    textcoords='offset points',
                    ha=ha, va=va, fontsize=FS_ANNOT + 0.5, color=INK, zorder=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(LABELS, linespacing=1.0)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Accuracy (%)')

    # Legend in three columns (filled column by column): the navy marks, the
    # families, and the budget key.
    hs = [Line2D([], [], color=NAVY, lw=1.6, solid_capstyle='butt',
                 label='Best of strategy'),
          Line2D([], [], color=NAVY_MID, lw=0.75, ls=(0, (4, 3)),
                 label='Best individual'),
          budget_handles(color=SLATE, size=MS)[0]]
    hs[-1].set_label('Mixed families')
    hs += family_handles(['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4'], size=MS)
    bh = budget_handles(color=MUTED, size=MS)
    bh[0].set_label('filled: $B{=}10$')
    bh[1].set_label('open: $B{=}8{,}192$')
    hs += bh
    fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.02 / W, 1 - 0.01 / H),
               ncol=3, fontsize=FS_LEGEND - 0.5, handletextpad=0.25,
               columnspacing=0.8, borderaxespad=0.0, handlelength=1.2,
               labelspacing=0.25)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
