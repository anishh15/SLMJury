"""SummEval agreement with human scores: (a) every judge's rho with its CI, (b) paired small - large differences.

Reads results/analysis/significance/open_ended_reported_ci.csv, open_ended_paired.csv and
summeval_paired.csv; writes summeval_small_vs_large (6.30 x 3.60 in).
"""
from style import *   # noqa: F401,F403

NAME = 'summeval_small_vs_large'
# Column heads for (b), broken over lines instead of rotated.
COLHEAD = {'qwen3.5-122b': 'Qwen3.5-\n122B', 'qwen2.5-72b': 'Qwen2.5-\n72B',
           'gpt-oss-120b': 'GPT-OSS-\n120B', 'glm-4.5-air': 'GLM-4.5-\nAir',
           'llama3.3-70b': 'Llama-\n3.3-70B',
           'deepseek-r1-distill-llama-70b': 'DeepSeek-\nR1-Distill-\n70B'}
LIM = 0.20


def main():
    oe = load('significance', 'open_ended_reported_ci.csv')
    oe.sort(key=lambda r: -float(r['summeval_rho']))
    rho = {r['key']: r for r in oe}
    order = load('significance', 'summeval_paired.csv')
    paired = {(r['key_a'], r['key_b']): r
              for r in load('significance', 'open_ended_paired.csv')
              if r['benchmark'] == 'summeval'}
    slms, larges, cell = [], [], {}
    for o in order:
        s, l = o['slm_key'], o['large_key']
        slms += [s] if s not in slms else []
        larges += [l] if l not in larges else []
        if (s, l) in paired:
            r = paired[(s, l)]
            d, lo, hi = Decimal(r['diff']), float(r['ci_lo']), float(r['ci_hi'])
        else:
            r = paired[(l, s)]
            d, lo, hi = -Decimal(r['diff']), -float(r['ci_hi']), -float(r['ci_lo'])
        cell[(s, l)] = (d, lo > 0 or hi < 0)
    larges.sort(key=lambda j: -float(rho[j]['summeval_rho']))
    assert len(slms) == 5 and len(larges) == 6 and len(cell) == 30

    H = 3.60
    fig = new_figure(W_FULL, H)
    top = 3.24
    # ---------------------------------------------------------------- (a)
    n = len(oe)
    ha = 2.88
    ax = axes_in(fig, 1.14, top - ha, 1.40, ha)
    style_ax(ax, grid='x')
    ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
    for i, r in enumerate(oe):
        j = r['key']
        c = jcolor(j)
        ax.plot([float(r['summeval_ci_lo']), float(r['summeval_ci_hi'])], [i, i],
                color=c, lw=1.0, solid_capstyle='butt', zorder=3)
        judge_mark(ax, float(r['summeval_rho']), i, j, size=4.0, z=5,
                   filled=False)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([dn_tag(r['key']) for r in oe])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.08, 0.62)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.xaxis.set_major_formatter(tick_fmt(1))
    ax.set_xlabel(r'Spearman $\rho$ with human scores')
    panel_title(ax, '(a) SummEval, every judge')

    # ---------------------------------------------------------------- (b)
    # Cells tall enough that the heatmap, its column heads and the colour-bar
    # row together end where panel (a)'s axis ends.
    cw, chh = 0.44, 0.43
    hb = chh * len(slms)
    bx = 3.64
    axb = axes_in(fig, bx, top - hb, cw * len(larges), hb)
    cmap = div_cmap()
    norm = plt.Normalize(-LIM, LIM)
    data = np.array([[float(cell[(s, l)][0]) for l in larges] for s in slms])
    im = axb.imshow(data, cmap=cmap, norm=norm, aspect='auto',
                    interpolation='nearest')
    for sp in axb.spines.values():
        sp.set_visible(False)
    axb.set_xticks(np.arange(-0.5, len(larges), 1), minor=True)
    axb.set_yticks(np.arange(-0.5, len(slms), 1), minor=True)
    axb.grid(which='minor', color=PAPER, linewidth=1.2)
    axb.tick_params(which='both', length=0)
    axb.set_yticks(range(len(slms)))
    axb.set_yticklabels([dn(s) for s in slms])
    axb.set_xticks(range(len(larges)))
    axb.set_xticklabels([COLHEAD[l] for l in larges], fontsize=6,
                        linespacing=1.0, va='top')
    for i, s in enumerate(slms):
        for k, l in enumerate(larges):
            d, sig = cell[(s, l)]
            txt = fmt(d, 3, sign=True) + ('*' if sig else '')
            axb.text(k, i, txt, ha='center', va='center', fontsize=FS_ANNOT,
                     color=text_on(cmap(norm(float(d)))), gid='cell')
    panel_title(axb, r'(b) Top-five SLMs vs. large judges')

    cby = top - hb - 0.58          # just under the three-line column heads
    cax = axes_in(fig, bx + 0.06, cby, 1.50, 0.08)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
    cb.ax.xaxis.set_major_formatter(tick_fmt(1, sign=True))
    cb.ax.tick_params(length=2.0, width=0.5, color=AXIS, labelsize=FS_TICK)
    cb.set_label(r'$\rho$(SLM) $-$ $\rho$(large)', fontsize=FS_LABEL,
                 labelpad=2)
    fig.text((bx + 1.80) / W_FULL, (cby + 0.04) / H,
             '* 95% interval\n   excludes 0', ha='left', va='center',
             fontsize=FS_ANNOT, color=INK, linespacing=1.15)

    hs = family_handles(FAMILY_ORDER, size=4.0, filled=False)
    hs.append(budget_handles(color=SLATE, size=4.0)[1])
    hs[-1].set_label('all at $B{=}8{,}192$')
    ax.legend(handles=hs, loc='lower left',
              bbox_to_anchor=(0.0, 1.0 + 0.19 / ha), ncol=6,
              handletextpad=0.3, columnspacing=1.4, borderaxespad=0.0)
    return finish(fig, NAME)


if __name__ == '__main__':
    main()
