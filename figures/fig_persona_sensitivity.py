"""Persona sensitivity at B=10: the five persona-tested judges under each of the six personas.

Reads results/analysis/main_results/personas.csv; writes persona_sensitivity (6.30 x 2.85 in).
"""
from style import *   # noqa: F401,F403

NAME = 'persona_sensitivity'
PERSONAS = ['Strict', 'Lenient', 'Industry', 'Logic', 'Safety', 'Helpfulness']
# The second member of a family is a lighter shade of the family colour.
JUDGES = [('phi4-14b', 0.0), ('phi4mi-3.8b', 0.35), ('qwen3-14b', 0.0),
          ('qwen3-4b', 0.35), ('llama3.1-8b', 0.0)]


def main():
    rows = {r['judge']: r for r in load('main_results', 'personas.csv')
            if r['max_tokens'] == '10'}
    assert set(rows) == {j for j, _ in JUDGES}
    weakest = min(rows, key=lambda j: float(rows[j]['base']))
    order = sorted(PERSONAS, key=lambda p: -float(rows[weakest][p]))

    fig = new_figure(W_FULL, 2.85)
    ax = axes_in(fig, 0.42, 0.38, 5.82, 2.40)
    style_ax(ax, grid='y')
    x = np.arange(len(order))
    handles = []
    for j, t in JUDGES:
        c = shade(jcolor(j), t)
        m = jmarker(j)
        base = float(rows[j]['base'])
        ax.axhline(base, color=c, lw=0.7, ls=(0, (3, 2)), zorder=2,
                   alpha=0.9)
        ys = [float(rows[j][p]) for p in order]
        ln, = ax.plot(x, ys, color=c, lw=1.1, marker=m,
                      markersize=3.8 * MARKER_SCALE[m], markerfacecolor=c,
                      markeredgecolor=PAPER, markeredgewidth=0.45,
                      zorder=4, label=dn(j))
        handles.append(ln)
    handles.append(Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)),
                          label='No-persona baseline'))

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.set_ylim(75.5, 90.6)
    ax.set_yticks([76, 78, 80, 82, 84, 86, 88, 90])
    ax.set_xlabel('Persona (system prompt)')
    ax.set_ylabel('Accuracy (%)')
    ax.tick_params(axis='x', length=0)
    # The Lenient persona's change for the weakest judge.
    d = Decimal(rows[weakest]['Lenient']) - Decimal(rows[weakest]['base'])
    ax.annotate(f'{fmt(str(d), 2)} from its baseline',
                xy=(order.index('Lenient'), float(rows[weakest]['Lenient'])),
                xytext=(-6, 0), textcoords='offset points', ha='right',
                va='center', fontsize=FS_ANNOT, color=MUTED)
    ax.legend(handles=handles, loc='lower left', ncol=2, handlelength=2.2,
              columnspacing=1.6, borderaxespad=0.5)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
