"""
Generate publication-quality figures for the SLMJury paper.
All data is read from website/src/data/modelData.js (single source of truth).
Output: PDF files in figures/ directory.

Figures:
  1. Token Budget        — Grouped bar: 10-token vs 8,192-token accuracy
  2. Overthinking Delta   — Diverging horizontal bar: Δ(t10 − t8192) per model
  3. Strategy Comparison  — Bar: best individual / persona / ensemble / debate
  4. Dataset Heatmap      — All 8 datasets × top judges (t10)
  5. Persona Sensitivity  — Line plot for t10 persona judges
  6. Scaling Curve        — Grouped bar by model family and parameter size
  7. Instruction Following — Grouped bar: IFR at t10 vs t8192
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import re
import json

# ═══════════════════════════════════════════════════════════════════════
# Publication Color Palette (EMNLP / ACL — muted, grayscale-safe)
# ═══════════════════════════════════════════════════════════════════════
C_NAVY   = '#2C5F8A'
C_BLUE   = '#5B9BD5'
C_LTBLUE = '#A8CCEB'
C_ORANGE = '#D47B2F'
C_GREEN  = '#548235'
C_RED    = '#C0504D'
C_TEAL   = '#3A8F85'
C_PURPLE = '#7B5EA7'
C_GOLD   = '#C49B2A'

C_TEXT   = '#2D2D2D'
C_MTEXT  = '#555555'
C_LGRAY  = '#CCCCCC'
C_BG     = '#FFFFFF'

FAMILY_COLOR = {
    'LLaMA 3.x': C_RED,
    'Qwen 2.5':  C_TEAL,
    'Qwen 3':    C_NAVY,
    'Phi-4':     C_ORANGE,
}

# ═══════════════════════════════════════════════════════════════════════
# RC Params — serif, publication-ready
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          12,
    'axes.labelsize':     13,
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.framealpha':  1.0,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.6,
    'axes.edgecolor':     C_LGRAY,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.linestyle':     '-',
    'grid.linewidth':     0.4,
    'grid.color':         '#E0E4E8',
    'figure.facecolor':   C_BG,
    'axes.facecolor':     C_BG,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.color':        C_MTEXT,
    'ytick.color':        C_MTEXT,
    'axes.labelcolor':    C_TEXT,
    'text.color':         C_TEXT,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Display Names & Mappings
# ═══════════════════════════════════════════════════════════════════════
DISPLAY = {
    'llama3.2-1b':  'LLaMA-3.2-1B',
    'llama3.2-3b':  'LLaMA-3.2-3B',
    'llama3.1-8b':  'LLaMA-3.1-8B',
    'qwen2.5-1.5b': 'Qwen2.5-1.5B',
    'qwen2.5-3b':   'Qwen2.5-3B',
    'qwen2.5-7b':   'Qwen2.5-7B',
    'qwen3-0.6b':   'Qwen3-0.6B',
    'qwen3-1.7b':   'Qwen3-1.7B',
    'qwen3-4b':     'Qwen3-4B',
    'qwen3-8b':     'Qwen3-8B',
    'qwen3-14b':    'Qwen3-14B',
    'phi4-14b':     'Phi-4',
    'phi4mi-3.8b':  'Phi-4-mini',
    'phi4r-14b':    'Phi-4-R',
    'phi4rp-14b':   'Phi-4-R-Plus',
    'phi4mr-3.8b':  'Phi-4-mini-R',
}

FAMILY_MAP = {
    'LLaMA 3.x': [('llama3.2-1b', '1B'), ('llama3.2-3b', '3B'), ('llama3.1-8b', '8B')],
    'Qwen 2.5':  [('qwen2.5-1.5b', '1.5B'), ('qwen2.5-3b', '3B'), ('qwen2.5-7b', '7B')],
    'Qwen 3':    [('qwen3-0.6b', '0.6B'), ('qwen3-1.7b', '1.7B'), ('qwen3-4b', '4B'),
                  ('qwen3-8b', '8B'), ('qwen3-14b', '14B')],
    'Phi-4':     [('phi4mi-3.8b', '3.8B'), ('phi4-14b', '14B')],
}

DS_KEYS   = ['gsm8k_acc', 'gsm_plus_acc', 'math_acc', 'arc_easy_acc',
             'arc_challenge_acc', 'hellaswag_acc', 'winogrande_acc', 'truthfulqa_acc']
DS_LABELS = ['GSM8K', 'GSM+', 'MATH', 'ARC-E', 'ARC-C', 'HSwag', 'WGrnd', 'TQA']


def _dn(key):
    return DISPLAY.get(key, key)


# ═══════════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════════

def load_all_data():
    js_path = os.path.join(os.path.dirname(__file__),
                           'website', 'src', 'data', 'modelData.js')
    with open(js_path) as f:
        content = f.read()

    def _extract(var):
        m = re.search(rf'export const {var}\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if not m:
            print(f'  ⚠ {var} not found')
            return []
        return json.loads(m.group(1))

    return (_extract('modelData'), _extract('majorityVotingData'),
            _extract('personaData'), _extract('madData'))


def _t10_t8192_pairs(md):
    buckets = {}
    for e in md:
        buckets.setdefault(e['shortName'], {})[e['tokens']] = e
    return {k: v for k, v in buckets.items() if 10 in v and 8192 in v}


# ═══════════════════════════════════════════════════════════════════════
# Fig 1 — Token Budget
# ═══════════════════════════════════════════════════════════════════════

def fig1_token_budget(md):
    pairs = _t10_t8192_pairs(md)
    keys  = sorted(pairs, key=lambda k: pairs[k][10]['accuracy'])
    names = [_dn(k) for k in keys]
    a10   = [pairs[k][10]['accuracy']   for k in keys]
    a8k   = [pairs[k][8192]['accuracy'] for k in keys]

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - w/2, a10, w, label='10 tokens (Quick Verdict)',
           color=C_NAVY, edgecolor='white', linewidth=0.5, zorder=3)
    ax.bar(x + w/2, a8k, w, label='8,192 tokens (Reasoned)',
           color=C_LTBLUE, edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(max(30, min(min(a10), min(a8k)) - 8), 100)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha='right', fontsize=9.5)
    ax.legend(loc='upper left', edgecolor=C_LGRAY, fancybox=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_token_budget.pdf'))
    plt.close()
    print('  ✓ fig1_token_budget.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 2 — Overthinking Delta
# ═══════════════════════════════════════════════════════════════════════

def fig2_overthinking_delta(md):
    pairs = _t10_t8192_pairs(md)
    items = sorted(pairs.items(),
                   key=lambda kv: kv[1][10]['accuracy'] - kv[1][8192]['accuracy'])
    names  = [_dn(k) for k, _ in items]
    deltas = [v[10]['accuracy'] - v[8192]['accuracy'] for _, v in items]
    colors = [C_GREEN if d > 0 else C_BLUE for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(names)), deltas, color=colors,
                   edgecolor='white', linewidth=0.4, height=0.62, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(r'$\Delta$ Accuracy (pp)  =  Acc$_{10}$ $-$ Acc$_{8192}$',
                  fontsize=11)
    ax.axvline(0, color=C_TEXT, linewidth=0.8, zorder=2)

    for i, (bar, d) in enumerate(zip(bars, deltas)):
        offset = 0.3 if d >= 0 else -0.3
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, i, f'{d:+.1f}', va='center', ha=ha,
                fontsize=8.5, color=C_MTEXT)

    ax.legend(
        handles=[mpatches.Patch(color=C_GREEN, label='Quick verdict better'),
                 mpatches.Patch(color=C_BLUE,  label='Reasoning better')],
        loc='upper left', edgecolor=C_LGRAY, fancybox=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_overthinking_delta.pdf'))
    plt.close()
    print('  ✓ fig2_overthinking_delta.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 3 — Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig3_strategy_comparison(md, pd, mv, mad):
    t10 = [e for e in md if e['tokens'] == 10]
    best_ind = max(t10, key=lambda e: e['accuracy']) if t10 else None
    t10_per  = [e for e in pd if e.get('tokens') == 10]
    best_per = max(t10_per, key=lambda e: e['accuracy']) if t10_per else None
    best_mv  = max(mv,  key=lambda e: e['accuracy']) if mv  else None
    best_mad = max(mad, key=lambda e: e['accuracy']) if mad else None

    labels, accs, clrs = [], [], []
    if best_ind:
        labels.append('Individual'); accs.append(best_ind['accuracy']); clrs.append(C_BLUE)
    if best_per:
        labels.append('Persona');    accs.append(best_per['accuracy']); clrs.append(C_PURPLE)
    if best_mv:
        labels.append('Majority\nVoting'); accs.append(best_mv['accuracy']); clrs.append(C_GREEN)
    if best_mad:
        labels.append('Multi-Agent\nDebate'); accs.append(best_mad['accuracy']); clrs.append(C_ORANGE)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, accs, color=clrs, edgecolor='white',
                  linewidth=0.8, width=0.55, zorder=3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=C_TEXT)

    ax.set_ylabel('Accuracy (%)')
    spread = max(accs) - min(accs)
    ax.set_ylim(min(accs) - max(2, spread), max(accs) + 1.5)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_strategy_comparison.pdf'))
    plt.close()
    print('  ✓ fig3_strategy_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 4 — Per-Dataset Heatmap (all 8 datasets)
# ═══════════════════════════════════════════════════════════════════════

def fig4_dataset_heatmap(md):
    t10 = sorted([e for e in md if e['tokens'] == 10],
                 key=lambda e: e['accuracy'], reverse=True)[:8]
    if not t10:
        return

    names = [_dn(e['shortName']) for e in t10]
    data  = np.array([[e.get(dk, 0) for dk in DS_KEYS] for e in t10])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'slm', ['#D9534F', '#F5C242', C_BG, C_LTBLUE, C_NAVY], N=256)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(False)
    vmin = max(25, float(data.min()) - 5)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=100)

    ax.set_xticks(range(len(DS_LABELS)))
    ax.set_xticklabels(DS_LABELS, fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = 'white' if v < (vmin + 15) else C_TEXT
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_dataset_heatmap.pdf'))
    plt.close()
    print('  ✓ fig4_dataset_heatmap.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 5 — Persona Sensitivity (t10 only)
# ═══════════════════════════════════════════════════════════════════════

def fig5_persona_sensitivity(pd, md):
    t10_p = [e for e in pd if e.get('tokens') == 10]
    if not t10_p:
        return

    personas = sorted(t10_p[0].get('persona_acc', {}).keys())
    plabels  = [p.capitalize() for p in personas]
    x = np.arange(len(personas))

    base_acc = {e['shortName']: e['accuracy']
                for e in md if e['tokens'] == 10}

    markers = ['o', 's', '^', 'D', 'v']
    colors  = [C_NAVY, C_TEAL, C_ORANGE, C_RED, C_PURPLE]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    all_vals = []

    for idx, entry in enumerate(t10_p):
        sn   = entry['shortName']
        pa   = entry.get('persona_acc', {})
        vals = [pa.get(p, 0) for p in personas]
        base = base_acc.get(sn, entry.get('accuracy', 0))
        c, m = colors[idx % len(colors)], markers[idx % len(markers)]

        ax.plot(x, vals, marker=m, color=c, linewidth=2.0, markersize=7,
                label=_dn(sn), zorder=3,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.axhline(base, color=c, linestyle=':', linewidth=0.7, alpha=0.4)
        all_vals.extend(vals + [base])

    ax.set_xticks(x)
    ax.set_xticklabels(plabels, fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(min(all_vals) - 1.5, max(all_vals) + 1.5)
    ax.legend(loc='lower left', edgecolor=C_LGRAY, fancybox=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_persona_sensitivity.pdf'))
    plt.close()
    print('  ✓ fig5_persona_sensitivity.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 6 — Scaling Curve (grouped bar by family)
# ═══════════════════════════════════════════════════════════════════════

def fig6_scaling_curve(md):
    t10_acc = {e['shortName']: e['accuracy']
               for e in md if e['tokens'] == 10}

    fig, ax = plt.subplots(figsize=(10, 4.8))

    tick_labels = []   # "FamilyName\nSize" compound labels
    values      = []
    bar_colors  = []
    x_pos       = []
    pos = 0

    for fam, members in FAMILY_MAP.items():
        for sn, size in members:
            acc = t10_acc.get(sn)
            if acc is None:
                continue
            tick_labels.append(f'{fam}\n{size}')
            values.append(acc)
            bar_colors.append(FAMILY_COLOR[fam])
            x_pos.append(pos)
            pos += 1
        pos += 0.7  # gap between families

    if not values:
        return

    ax.bar(x_pos, values, color=bar_colors, width=0.72,
           edgecolor='white', linewidth=0.6, zorder=3)

    for xp, v in zip(x_pos, values):
        ax.text(xp, v + 0.5, f'{v:.1f}', ha='center', va='bottom',
                fontsize=7.5, color=C_MTEXT, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=8.5, linespacing=1.2)
    ax.set_ylabel('Accuracy at t=10 (%)')
    ax.set_ylim(max(30, min(values) - 8), max(values) + 4)

    handles = [mpatches.Patch(color=FAMILY_COLOR[f], label=f) for f in FAMILY_MAP]
    ax.legend(handles=handles, loc='lower right', edgecolor=C_LGRAY,
              fancybox=False, fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_scaling_curve.pdf'))
    plt.close()
    print('  ✓ fig6_scaling_curve.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Fig 7 — Instruction Following Rate (t10 vs t8192)
# ═══════════════════════════════════════════════════════════════════════

def fig7_instruction_following(md):
    """Grouped bar: IFR at t10 vs t8192, sorted by t10 IFR.

    Highlights that smaller models and longer outputs degrade format compliance.
    """
    pairs = _t10_t8192_pairs(md)
    # Only include models where at least one budget has IFR < 100
    interesting = {k: v for k, v in pairs.items()
                   if v[10]['ifr'] < 99.95 or v[8192]['ifr'] < 99.95}

    if not interesting:
        print('  ⚠ Skipped fig8 (all IFR ≈ 100%)')
        return

    keys  = sorted(interesting, key=lambda k: interesting[k][10]['ifr'])
    names = [_dn(k) for k in keys]
    ifr10 = [interesting[k][10]['ifr']   for k in keys]
    ifr8k = [interesting[k][8192]['ifr'] for k in keys]

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, ifr10, w, label='t = 10',
           color=C_NAVY, edgecolor='white', linewidth=0.5, zorder=3)
    ax.bar(x + w/2, ifr8k, w, label='t = 8,192',
           color=C_LTBLUE, edgecolor='white', linewidth=0.5, zorder=3)

    # 100% reference line
    ax.axhline(100, color=C_LGRAY, linestyle='--', linewidth=0.8, zorder=1)

    ax.set_ylabel('Instruction Following Rate (%)')
    ax.set_ylim(max(90, min(min(ifr10), min(ifr8k)) - 2), 101)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.legend(loc='lower right', edgecolor=C_LGRAY, fancybox=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig7_instruction_following.pdf'))
    plt.close()
    print('  ✓ fig7_instruction_following.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('SLMJury — Generating publication figures')
    print(f'Output: {OUT_DIR}\n')

    md, mv, pd, mad = load_all_data()
    print(f'Data: {len(md)} individual · {len(mv)} MV · '
          f'{len(pd)} persona · {len(mad)} MAD\n')

    fig1_token_budget(md)
    fig2_overthinking_delta(md)
    fig3_strategy_comparison(md, pd, mv, mad)
    fig4_dataset_heatmap(md)
    fig5_persona_sensitivity(pd, md)
    fig6_scaling_curve(md)
    fig7_instruction_following(md)

    print('\n✓ All figures generated.')
