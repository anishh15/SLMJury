"""
Generate all figures for the SLMJury paper.
Data-driven: reads from website/src/data/modelData.js (single source of truth).
Subtle, light, modern theme. Clean white backgrounds.
Output: PDF files in figures/ directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import os
import re
import json

# ── Subtle Light Theme ──
# Primary blues (soft, not saturated)
SOFT_NAVY = '#4472C4'
SOFT_BLUE = '#6B9BD2'
LIGHT_BLUE = '#9DC3E6'
PALE_BLUE = '#D6E4F0'
VERY_PALE = '#EDF2F7'

# Accents (muted, pastel-ish)
SOFT_ORANGE = '#ED7D31'
SOFT_GREEN = '#70AD47'
SOFT_RED = '#E06666'
SOFT_TEAL = '#4DB6AC'
SOFT_PURPLE = '#9B8EC5'
SOFT_GOLD = '#FFB74D'

# Neutrals
DARK_TEXT = '#333333'
MED_TEXT = '#666666'
LIGHT_LINE = '#D0D5DD'
WHITE_BG = '#FFFFFF'

FAMILY_COLORS = {
    'LLaMA': SOFT_RED,
    'Meta': SOFT_RED,
    'Qwen 2.5': SOFT_TEAL,
    'Qwen': SOFT_NAVY,
    'Qwen 3': SOFT_NAVY,
    'Phi-4': SOFT_ORANGE,
    'Microsoft': SOFT_ORANGE,
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': LIGHT_LINE,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.color': '#E8EDF2',
    'figure.facecolor': WHITE_BG,
    'axes.facecolor': WHITE_BG,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.color': MED_TEXT,
    'ytick.color': MED_TEXT,
    'axes.labelcolor': DARK_TEXT,
    'text.color': DARK_TEXT,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data Loading ──

# Map shortName to display name for figures
DISPLAY_NAMES = {
    'llama3.2-1b': 'LLaMA-3.2-1B',
    'llama3.2-3b': 'LLaMA-3.2-3B',
    'llama3.1-8b': 'LLaMA-3.1-8B',
    'qwen2.5-1.5b': 'Qwen2.5-1.5B',
    'qwen2.5-3b': 'Qwen2.5-3B',
    'qwen2.5-7b': 'Qwen2.5-7B',
    'qwen3-0.6b': 'Qwen3-0.6B',
    'qwen3-1.7b': 'Qwen3-1.7B',
    'qwen3-4b': 'Qwen3-4B',
    'qwen3-8b': 'Qwen3-8B',
    'qwen3-14b': 'Qwen3-14B',
    'phi4-14b': 'Phi-4',
    'phi4mi-3.8b': 'Phi-4-mini',
    'phi4r-14b': 'Phi-4-reasoning',
    'phi4rp-14b': 'Phi-4-R-Plus',
    'phi4mr-3.8b': 'Phi-4-mini-R',
}

# Family grouping for scaling curve
FAMILY_GROUPING = {
    'llama3.2-1b': ('LLaMA 3.x', '1B'),
    'llama3.2-3b': ('LLaMA 3.x', '3B'),
    'llama3.1-8b': ('LLaMA 3.x', '8B'),
    'qwen2.5-1.5b': ('Qwen 2.5', '1.5B'),
    'qwen2.5-3b': ('Qwen 2.5', '3B'),
    'qwen2.5-7b': ('Qwen 2.5', '7B'),
    'qwen3-0.6b': ('Qwen 3', '0.6B'),
    'qwen3-1.7b': ('Qwen 3', '1.7B'),
    'qwen3-4b': ('Qwen 3', '4B'),
    'qwen3-8b': ('Qwen 3', '8B'),
    'qwen3-14b': ('Qwen 3', '14B'),
    'phi4mi-3.8b': ('Phi-4', '3.8B'),
    'phi4-14b': ('Phi-4', '14B'),
}

FAMILY_PLOT_COLORS = {
    'LLaMA 3.x': SOFT_RED,
    'Qwen 2.5': SOFT_TEAL,
    'Qwen 3': SOFT_NAVY,
    'Phi-4': SOFT_ORANGE,
}


def load_model_data():
    """Load all exported data from modelData.js.

    Returns:
        Tuple of (modelData, majorityVotingData, personaData, madData).
    """
    js_path = os.path.join(os.path.dirname(__file__),
                           'website', 'src', 'data', 'modelData.js')
    with open(js_path, 'r') as f:
        content = f.read()

    def extract_array(var_name):
        pattern = rf'export const {var_name}\s*=\s*(\[.*?\]);'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            print(f'  Warning: could not find {var_name} in modelData.js')
            return []
        return json.loads(match.group(1))

    return (
        extract_array('modelData'),
        extract_array('majorityVotingData'),
        extract_array('personaData'),
        extract_array('madData'),
    )


def _get_t10_t8192(model_data):
    """Build {shortName: {10: accuracy, 8192: accuracy}} from modelData.

    Only includes models that have BOTH t10 and t8192 entries.
    Reasoning-only models (phi4r, phi4rp, phi4mr) only have t8192.
    """
    pairs = {}
    for entry in model_data:
        key = entry['shortName']
        tokens = entry['tokens']
        acc = entry['accuracy']
        pairs.setdefault(key, {})[tokens] = acc
    return pairs


# ── Figure Functions ──

def fig1_token_budget(model_data):
    """Bar chart: 10-token vs 8192-token accuracy."""
    pairs = _get_t10_t8192(model_data)

    # Only include models with both token budgets
    both = {k: v for k, v in pairs.items() if 10 in v and 8192 in v}

    # Sort by t10 accuracy (ascending)
    sorted_keys = sorted(both.keys(), key=lambda k: both[k][10])

    models = [DISPLAY_NAMES.get(k, k) for k in sorted_keys]
    acc_10 = [both[k][10] for k in sorted_keys]
    acc_8192 = [both[k][8192] for k in sorted_keys]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 4.8))

    ax.bar(x - width/2, acc_10, width, label='10 tokens (Quick)',
           color=SOFT_NAVY, edgecolor='white', linewidth=0.6, zorder=3,
           alpha=0.9)
    ax.bar(x + width/2, acc_8192, width, label='8,192 tokens (Reasoned)',
           color=LIGHT_BLUE, edgecolor='white', linewidth=0.6, zorder=3,
           alpha=0.9)

    # Subtle threshold line
    ax.axhline(y=90, color=SOFT_RED, linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)
    ax.text(len(models) - 0.3, 90.6, r'$\tau \approx 90\%$', ha='right', va='bottom',
            fontsize=11, color=SOFT_RED, fontstyle='italic')

    ax.set_ylabel('Accuracy (%)')
    y_min = max(30, min(min(acc_10), min(acc_8192)) - 10)
    ax.set_ylim(y_min, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=38, ha='right', fontsize=10)
    ax.legend(loc='lower right', edgecolor=LIGHT_LINE, fontsize=11,
              fancybox=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'token_budget.pdf'))
    plt.close()
    print('Saved: token_budget.pdf')


def fig2_persona_sensitivity(persona_data, model_data):
    """Line plot: persona effect for judges with persona data."""
    if not persona_data:
        print('Skipped: persona_sensitivity.pdf (no persona data)')
        return

    personas = sorted(persona_data[0].get('persona_acc', {}).keys())
    persona_labels = [p.capitalize() for p in personas]
    x = np.arange(len(personas))

    # Build base accuracy lookup from individual t10 data
    base_acc = {}
    for entry in model_data:
        if entry['tokens'] == 10:
            base_acc[entry['shortName']] = entry['accuracy']

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    colors = [SOFT_RED, SOFT_NAVY, SOFT_TEAL, SOFT_ORANGE, SOFT_PURPLE,
              SOFT_GREEN, SOFT_GOLD, SOFT_BLUE]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_vals = []
    for idx, entry in enumerate(persona_data):
        name = DISPLAY_NAMES.get(entry['shortName'], entry['shortName'])
        pa = entry.get('persona_acc', {})
        data = [pa.get(p, 0) for p in personas]
        base = base_acc.get(entry['shortName'], entry.get('accuracy', 0))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.plot(x, data, marker=marker, color=color,
                linewidth=2.0, markersize=7, label=name,
                zorder=3, markeredgecolor='white', markeredgewidth=1.0, alpha=0.9)
        ax.axhline(y=base, color=color, linestyle=':',
                   linewidth=0.8, alpha=0.35)
        all_vals.extend(data)
        all_vals.append(base)

    ax.set_xticks(x)
    ax.set_xticklabels(persona_labels, fontsize=12)
    ax.set_ylabel('Accuracy (%)')
    y_min = max(60, min(all_vals) - 3)
    y_max = min(100, max(all_vals) + 2)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower left', edgecolor=LIGHT_LINE, fontsize=10,
              fancybox=False, ncol=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'persona_sensitivity.pdf'))
    plt.close()
    print('Saved: persona_sensitivity.pdf')


def fig3_ensemble_vs_individual(model_data, mv_data, mad_data):
    """Clean bar chart comparing best individual, ensemble, and debate."""
    # Best individual at t10
    best_ind = max(
        (e for e in model_data if e['tokens'] == 10),
        key=lambda e: e['accuracy'],
        default=None,
    )
    # Best majority voting ensemble
    best_mv = max(mv_data, key=lambda e: e['accuracy'], default=None)
    # Best multi-agent debate
    best_mad = max(mad_data, key=lambda e: e['accuracy'], default=None)

    categories = []
    accuracies = []
    colors_list = []

    if best_ind:
        name = DISPLAY_NAMES.get(best_ind['shortName'], best_ind['shortName'])
        categories.append(f'Best Individual\n({name})')
        accuracies.append(best_ind['accuracy'])
        colors_list.append(SOFT_BLUE)

    if best_mv:
        categories.append('Best Ensemble\n(3-model jury)')
        accuracies.append(best_mv['accuracy'])
        colors_list.append(SOFT_GREEN)

    if best_mad:
        # Extract the debate combo description
        judges = best_mad.get('judges', [])
        if judges:
            debate_name = judges[0].get('name', 'Debate')
        else:
            debate_name = 'Debate'
        categories.append(f'Best Debate\n({debate_name})')
        accuracies.append(best_mad['accuracy'])
        colors_list.append(SOFT_RED)

    if not categories:
        print('Skipped: approach_comparison.pdf (no data)')
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(categories, accuracies, color=colors_list, edgecolor='white',
                  linewidth=1.0, width=0.52, zorder=3, alpha=0.88)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=13,
                fontweight='bold', color=DARK_TEXT)

    ax.set_ylabel('Accuracy (%)')
    y_min = min(accuracies) - 2
    y_max = max(accuracies) + 1.5
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'approach_comparison.pdf'))
    plt.close()
    print('Saved: approach_comparison.pdf')


def fig4_overthinking_delta(model_data):
    """Horizontal diverging bar chart for overthinking effect."""
    pairs = _get_t10_t8192(model_data)
    both = {k: v for k, v in pairs.items() if 10 in v and 8192 in v}

    models = []
    deltas = []
    for k, v in both.items():
        models.append(DISPLAY_NAMES.get(k, k))
        deltas.append(v[10] - v[8192])

    # Sort by delta ascending
    sorted_pairs = sorted(zip(models, deltas), key=lambda p: p[1])
    models_s = [p[0] for p in sorted_pairs]
    deltas_s = [p[1] for p in sorted_pairs]

    colors = [SOFT_GREEN if d > 0 else SOFT_BLUE for d in deltas_s]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(range(len(models_s)), deltas_s, color=colors,
            edgecolor='white', linewidth=0.5, height=0.65, zorder=3, alpha=0.85)

    ax.set_yticks(range(len(models_s)))
    ax.set_yticklabels(models_s, fontsize=10)
    ax.set_xlabel(r'$\Delta$ = Acc(10 tok) $-$ Acc(8,192 tok) (pp)', fontsize=12)
    ax.axvline(x=0, color=DARK_TEXT, linewidth=0.8, zorder=2)

    # Labels
    green_patch = mpatches.Patch(color=SOFT_GREEN, alpha=0.85, label='Quick verdict wins')
    blue_patch = mpatches.Patch(color=SOFT_BLUE, alpha=0.85, label='Reasoning helps')
    ax.legend(handles=[green_patch, blue_patch], loc='lower right',
              edgecolor=LIGHT_LINE, fontsize=10, fancybox=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'overthinking_delta.pdf'))
    plt.close()
    print('Saved: overthinking_delta.pdf')


def fig5_per_dataset_heatmap(model_data):
    """Heatmap with soft blue-white-red colormap.

    Shows per-dataset accuracy for top-N individual judges at t=10.
    """
    dataset_keys = ['gsm8k_acc', 'gsm_plus_acc', 'math_acc', 'arc_easy_acc', 'arc_challenge_acc']
    dataset_labels = ['GSM8K', 'GSM-Plus', 'MATH', 'ARC-E', 'ARC-C']

    # Top 7 individual judges at t=10 by overall accuracy
    t10_entries = sorted(
        [e for e in model_data if e['tokens'] == 10],
        key=lambda e: e['accuracy'],
        reverse=True,
    )[:7]

    if not t10_entries:
        print('Skipped: per_dataset_heatmap.pdf (no t10 data)')
        return

    models = [DISPLAY_NAMES.get(e['shortName'], e['shortName']) for e in t10_entries]
    data = np.array([
        [e.get(dk, 0) for dk in dataset_keys]
        for e in t10_entries
    ])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'slm', [SOFT_RED, SOFT_GOLD, '#FFFFFF', LIGHT_BLUE, SOFT_NAVY], N=256)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(False)
    vmin = max(50, data.min() - 5)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=100)

    ax.set_xticks(range(len(dataset_labels)))
    ax.set_xticklabels(dataset_labels, fontsize=13)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(dataset_labels)):
            val = data[i, j]
            color = 'white' if val < (vmin + 10) else DARK_TEXT
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_dataset_heatmap.pdf'))
    plt.close()
    print('Saved: per_dataset_heatmap.pdf')


def fig6_scaling_curve(model_data):
    """Grouped bar chart: accuracy by model family and size (t=10 only)."""
    # Collect t=10 entries that we have family grouping for
    families = {}
    for entry in model_data:
        key = entry['shortName']
        if entry['tokens'] != 10 or key not in FAMILY_GROUPING:
            continue
        fam, size = FAMILY_GROUPING[key]
        families.setdefault(fam, []).append((size, entry['accuracy']))

    # Sort within each family by parameter size (numeric)
    def _parse_size(s):
        return float(s.replace('B', ''))

    family_order = ['LLaMA 3.x', 'Qwen 2.5', 'Qwen 3', 'Phi-4']
    for fam in family_order:
        if fam in families:
            families[fam].sort(key=lambda p: _parse_size(p[0]))

    fig, ax = plt.subplots(figsize=(10, 5))

    all_labels = []
    all_values = []
    all_colors = []
    x_pos = []
    pos = 0

    for fam in family_order:
        if fam not in families:
            continue
        for size, acc in families[fam]:
            all_labels.append(size)
            all_values.append(acc)
            all_colors.append(FAMILY_PLOT_COLORS[fam])
            x_pos.append(pos)
            pos += 1
        pos += 0.6  # gap between families

    if not all_values:
        print('Skipped: scaling_curve.pdf (no data)')
        return

    bars = ax.bar(x_pos, all_values, color=all_colors, width=0.75,
                  edgecolor='white', linewidth=0.8, alpha=0.88, zorder=3)

    # Add value labels on top bars > 90
    for xp, val in zip(x_pos, all_values):
        if val > 85:
            ax.text(xp, val + 0.5, f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8, color=MED_TEXT, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=10, rotation=0)
    ax.set_ylabel('Accuracy at 10 Tokens (%)')
    y_min = max(30, min(all_values) - 10)
    ax.set_ylim(y_min, 100)

    # Threshold
    ax.axhline(y=85, color=SOFT_RED, linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(max(x_pos) + 0.5, 85.5, r'$\tau$', fontsize=12, color=SOFT_RED)

    # Family labels below
    idx = 0
    for fam in family_order:
        if fam not in families:
            continue
        n = len(families[fam])
        center = (x_pos[idx] + x_pos[idx + n - 1]) / 2
        ax.text(center, y_min - 2, fam, ha='center', va='top', fontsize=11,
                fontweight='bold', color=FAMILY_PLOT_COLORS[fam])
        idx += n

    # Legend
    handles = [mpatches.Patch(color=FAMILY_PLOT_COLORS[f], alpha=0.88, label=f)
               for f in family_order if f in families]
    ax.legend(handles=handles, loc='upper left', edgecolor=LIGHT_LINE,
              fontsize=10, fancybox=False, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scaling_curve.pdf'))
    plt.close()
    print('Saved: scaling_curve.pdf')


if __name__ == '__main__':
    print('Generating SLMJury figures...')
    print(f'Output directory: {OUT_DIR}')
    print()

    model_data, mv_data, persona_data, mad_data = load_model_data()
    print(f'Loaded: {len(model_data)} individual, {len(mv_data)} MV, '
          f'{len(persona_data)} persona, {len(mad_data)} MAD entries')
    print()

    fig1_token_budget(model_data)
    fig2_persona_sensitivity(persona_data, model_data)
    fig3_ensemble_vs_individual(model_data, mv_data, mad_data)
    fig4_overthinking_delta(model_data)
    fig5_per_dataset_heatmap(model_data)
    fig6_scaling_curve(model_data)
    print('\nAll figures generated successfully.')
