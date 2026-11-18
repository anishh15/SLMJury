"""Accuracy at B=10 by model family and size, one bar per judge that can give a 10-token verdict.

Reads results/analysis/main_results/individual.csv (small judges) and large_judges/small_vs_large.csv
(Qwen3-32B and the large judges); writes scaling_curve (6.30 x 2.70 in).
"""
from style import *   # noqa: F401,F403

NAME = 'scaling_curve'
GROUP_NAME = {'Llama': 'Llama 3.x', 'Qwen2.5': 'Qwen2.5', 'Qwen3': 'Qwen3',
              'Phi-4': 'Phi-4', 'Large': 'Large'}


def main():
    acc = {}
    for r in load('main_results', 'individual.csv'):
        if r['max_tokens'] == '10':
            acc[r['judge']] = r['accuracy']
    for r in load('large_judges', 'small_vs_large.csv'):
        if r['type'] in ('Large', 'Ladder') and r['acc_t10'] not in ('', 'None'):
            acc[r['judge']] = r['acc_t10']
    assert len(acc) == 17, len(acc)

    xs, hs, cols, ticks, hatch, centres, js = [], [], [], [], [], [], []
    pos = 0.0
    for fam in FAMILY_ORDER:
        members = sorted((j for j in acc if family_of(j) == fam),
                         key=lambda j: PARAMS[j])
        start = pos
        for j in members:
            xs.append(pos)
            hs.append(float(acc[j]))
            js.append(j)
            cols.append(FAMILY_COLOR[fam])
            ticks.append(f'{PARAMS[j]:g}B')
            hatch.append(j in LADDER)
            pos += 1
        centres.append((GROUP_NAME[fam], (start + pos - 1) / 2))
        pos += 0.7

    fig = new_figure(W_FULL, 2.70)
    ax = axes_in(fig, 0.42, 0.38, 5.82, 2.24)
    style_ax(ax, grid='y')
    bars = ax.bar(xs, hs, width=0.74, color=cols, edgecolor=PAPER,
                  linewidth=0.5, zorder=3)
    for b, h in zip(bars, hatch):
        if h:
            b.set_hatch('////')
    for x, h, j in zip(xs, hs, js):
        ax.annotate(fmt(acc[j], 2), xy=(x, h), xytext=(0, 1.5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(30, 95)
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90])
    ax.set_ylabel('Accuracy at $B{=}10$ (%)')
    for name, c in centres:
        ax.annotate(name, xy=(c, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -12), textcoords='offset points', ha='center',
                    va='top', fontsize=FS_TICK, color=INK,
                    annotation_clip=False)
    # The ladder judge: a short tag under its tick, on the group-name row.
    for x, j in zip(xs, js):
        if j in LADDER:
            ax.annotate('(ladder)', xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -12.5), textcoords='offset points',
                        ha='center', va='top', fontsize=6, color=MUTED,
                        annotation_clip=False)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
