"""
Generate all figures for the SLMJury paper.
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
    'Qwen 2.5': SOFT_TEAL,
    'Qwen 3': SOFT_NAVY,
    'Phi-4': SOFT_ORANGE,
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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def fig1_token_budget():
    """Bar chart: 10-token vs 8192-token accuracy."""
    models = [
        'LLaMA-3.2-1B', 'Qwen3-0.6B', 'Qwen2.5-1.5B', 'LLaMA-3.2-3B',
        'Qwen3-1.7B', 'Qwen2.5-7B', 'Qwen2.5-3B', 'LLaMA-3.1-8B',
        'Qwen3-8B', 'Phi-4-mini', 'Qwen3-4B', 'Phi-4', 'Qwen3-14B'
    ]
    acc_10 = [41.20, 73.56, 78.45, 82.80, 87.56, 89.83, 90.15, 91.92,
              93.45, 93.68, 94.32, 94.33, 94.42]
    acc_8192 = [67.86, 83.12, 88.01, 87.15, 89.51, 92.28, 83.46, 89.29,
                90.94, 93.52, 91.61, 90.78, 91.63]

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
    ax.text(12.7, 90.6, r'$\tau \approx 90\%$', ha='right', va='bottom',
            fontsize=11, color=SOFT_RED, fontstyle='italic')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(35, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=38, ha='right', fontsize=10)
    ax.legend(loc='lower right', edgecolor=LIGHT_LINE, fontsize=11,
              fancybox=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'token_budget.pdf'))
    plt.close()
    print('Saved: token_budget.pdf')


def fig2_persona_sensitivity():
    """Line plot: persona effect for 5 judges."""
    personas = ['Strict', 'Lenient', 'Industry', 'Logic', 'Safety', 'Helpful']
    x = np.arange(len(personas))

    judges = {
        'LLaMA-3.1-8B': {
            'data': [92.13, 79.91, 91.21, 92.43, 90.43, 91.25],
            'base': 91.92, 'color': SOFT_RED, 'marker': 'o',
        },
        'Qwen3-4B': {
            'data': [94.55, 94.70, 94.15, 94.99, 94.83, 94.43],
            'base': 94.32, 'color': SOFT_NAVY, 'marker': 's',
        },
        'Qwen3-14B': {
            'data': [94.25, 94.12, 94.15, 94.15, 94.04, 94.10],
            'base': 94.42, 'color': SOFT_TEAL, 'marker': '^',
        },
        'Phi-4': {
            'data': [94.96, 94.09, 94.42, 94.83, 94.74, 94.45],
            'base': 94.33, 'color': SOFT_ORANGE, 'marker': 'D',
        },
        'Phi-4-mini': {
            'data': [92.60, 92.30, 92.55, 93.01, 93.69, 92.72],
            'base': 93.68, 'color': SOFT_PURPLE, 'marker': 'v',
        },
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for name, info in judges.items():
        ax.plot(x, info['data'], marker=info['marker'], color=info['color'],
                linewidth=2.0, markersize=7, label=f"{name}",
                zorder=3, markeredgecolor='white', markeredgewidth=1.0, alpha=0.9)
        ax.axhline(y=info['base'], color=info['color'], linestyle=':',
                   linewidth=0.8, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(personas, fontsize=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(77, 96)
    ax.legend(loc='lower left', edgecolor=LIGHT_LINE, fontsize=10,
              fancybox=False, ncol=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'persona_sensitivity.pdf'))
    plt.close()
    print('Saved: persona_sensitivity.pdf')


def fig3_ensemble_vs_individual():
    """Clean bar chart comparing approaches."""
    categories = ['Best Individual\n(Qwen3-14B)', 'Best Ensemble\n(3-model jury)', 'Best Debate\n(Phi-4-mini)']
    accuracies = [94.42, 95.02, 93.16]
    colors = [SOFT_BLUE, SOFT_GREEN, SOFT_RED]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='white',
                  linewidth=1.0, width=0.52, zorder=3, alpha=0.88)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=13,
                fontweight='bold', color=DARK_TEXT)

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(92, 95.8)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'approach_comparison.pdf'))
    plt.close()
    print('Saved: approach_comparison.pdf')


def fig4_overthinking_delta():
    """Horizontal diverging bar chart for overthinking effect."""
    models = [
        'LLaMA-3.2-1B', 'Qwen3-0.6B', 'Qwen2.5-1.5B', 'LLaMA-3.2-3B',
        'Qwen3-1.7B', 'Qwen2.5-7B', 'Phi-4-mini', 'LLaMA-3.1-8B',
        'Qwen3-8B', 'Qwen3-4B', 'Qwen3-14B', 'Phi-4', 'Qwen2.5-3B'
    ]
    deltas = [
        -26.66, -9.56, -9.56, -4.35,
        -1.95, -2.45, +0.16, +2.63,
        +2.51, +2.71, +2.79, +3.55, +6.69
    ]

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

    # Threshold separator
    ax.axhline(y=5.5, color=LIGHT_LINE, linestyle='--', linewidth=1.0)
    ax.text(5.5, 5.6, r'Threshold $\tau$', fontsize=10, color=MED_TEXT,
            fontstyle='italic', va='bottom')

    # Labels
    green_patch = mpatches.Patch(color=SOFT_GREEN, alpha=0.85, label='Quick verdict wins')
    blue_patch = mpatches.Patch(color=SOFT_BLUE, alpha=0.85, label='Reasoning helps')
    ax.legend(handles=[green_patch, blue_patch], loc='lower right',
              edgecolor=LIGHT_LINE, fontsize=10, fancybox=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'overthinking_delta.pdf'))
    plt.close()
    print('Saved: overthinking_delta.pdf')


def fig5_per_dataset_heatmap():
    """Heatmap with soft blue-white-red colormap."""
    models = [
        'Qwen3-14B', 'Phi-4', 'Qwen3-4B', 'Phi-4-mini', 'Qwen3-8B',
        'LLaMA-3.1-8B', 'Qwen2.5-3B'
    ]
    datasets = ['GSM8K', 'GSM-Plus', 'MATH', 'ARC-E', 'ARC-C']

    data = np.array([
        [98.90, 96.20, 94.13, 87.86, 87.93],
        [98.45, 96.98, 92.12, 88.01, 88.10],
        [98.33, 97.74, 93.80, 83.08, 83.92],
        [98.22, 95.33, 93.33, 87.77, 87.16],
        [98.45, 94.45, 94.01, 87.82, 87.93],
        [97.88, 93.43, 91.13, 86.87, 85.28],
        [96.06, 96.01, 91.29, 69.00, 68.69],
    ])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'slm', [SOFT_RED, SOFT_GOLD, '#FFFFFF', LIGHT_BLUE, SOFT_NAVY], N=256)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(False)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=65, vmax=100)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=13)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(datasets)):
            val = data[i, j]
            color = 'white' if val < 78 else DARK_TEXT
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_dataset_heatmap.pdf'))
    plt.close()
    print('Saved: per_dataset_heatmap.pdf')


def fig6_scaling_curve():
    """Grouped bar chart: accuracy by model family and size (instead of scatter)."""
    # Group by family, sorted by size within family
    families = {
        'LLaMA 3.x': [('1B', 41.20), ('3B', 82.80), ('8B', 91.92)],
        'Qwen 2.5': [('1.5B', 78.45), ('3B', 90.15), ('7B', 89.83)],
        'Qwen 3': [('0.6B', 73.56), ('1.7B', 87.56), ('4B', 94.32), ('8B', 93.45), ('14B', 94.42)],
        'Phi-4': [('3.8B', 93.68), ('14B', 94.33)],
    }

    family_colors = {
        'LLaMA 3.x': SOFT_RED,
        'Qwen 2.5': SOFT_TEAL,
        'Qwen 3': SOFT_NAVY,
        'Phi-4': SOFT_ORANGE,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build grouped bars
    all_labels = []
    all_values = []
    all_colors = []
    group_centers = []
    pos = 0

    for fam, models in families.items():
        start = pos
        for size, acc in models:
            all_labels.append(f'{size}')
            all_values.append(acc)
            all_colors.append(family_colors[fam])
            pos += 1
        group_centers.append((start + pos - 1) / 2)
        pos += 0.6  # gap between families

    x = np.arange(len(all_values))
    # Adjust x positions for gaps
    x_pos = []
    idx = 0
    gap = 0
    for fam, models in families.items():
        for _ in models:
            x_pos.append(idx + gap)
            idx += 1
        gap += 0.6

    bars = ax.bar(x_pos, all_values, color=all_colors, width=0.75,
                  edgecolor='white', linewidth=0.8, alpha=0.88, zorder=3)

    # Add value labels on top bars > 90
    for xp, val in zip(x_pos, all_values):
        if val > 90:
            ax.text(xp, val + 0.5, f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8, color=MED_TEXT, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=10, rotation=0)
    ax.set_ylabel('Accuracy at 10 Tokens (%)')
    ax.set_ylim(35, 100)

    # Threshold
    ax.axhline(y=90, color=SOFT_RED, linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(max(x_pos) + 0.5, 90.5, r'$\tau$', fontsize=12, color=SOFT_RED)

    # Family labels below
    g_centers = []
    idx = 0
    gap = 0
    for fam, models in families.items():
        center = (x_pos[idx] + x_pos[idx + len(models) - 1]) / 2
        g_centers.append((center, fam))
        idx += len(models)

    for center, fam in g_centers:
        ax.text(center, 33, fam, ha='center', va='top', fontsize=11,
                fontweight='bold', color=family_colors[fam])

    # Legend
    handles = [mpatches.Patch(color=c, alpha=0.88, label=f)
               for f, c in family_colors.items()]
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
    fig1_token_budget()
    fig2_persona_sensitivity()
    fig3_ensemble_vs_individual()
    fig4_overthinking_delta()
    fig5_per_dataset_heatmap()
    fig6_scaling_curve()
    print('\nAll figures generated successfully.')
