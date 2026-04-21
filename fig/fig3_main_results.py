#!/usr/bin/env python3
"""
V-C Main Results Figure
3×2 grid (datasets × soft/hard). Each panel shows:
  Left axis: watermark security strength -log10(P) across stages (Base→FT10%)
             one line per arch pair, threshold line at η=2^{-64}
  Right axis: student model accuracy at each stage (dashed, same arch colors)
Reads T=5 soft / T=1 hard from all three experiment CSVs.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.makedirs('fig', exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('seaborn-paper')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)

THRESHOLD = 64 * np.log10(2)   # ≈ 19.27
P_FLOOR   = 300.0

STAGES     = ['Base', 'FT 1%', 'FT 5%', 'FT 10%']
P_COLS     = ['Base_P', 'FT1%_P', 'FT5%_P', 'FT10%_P']
ACC_COLS   = ['Base_Acc', 'FT1%_Acc', 'FT5%_Acc', 'FT10%_Acc']

ARCH_SHORT = {
    'resnet20 -> resnet20': 'R20→R20',
    'resnet20 -> vgg11':    'R20→V11',
    'vgg11 -> vgg11':       'V11→V11',
    'resnet56 -> resnet56': 'R56→R56',
    'resnet56 -> resnet20': 'R56→R20',
    'tinyresnet18 -> resnet18':    'TR18→R18',
    'tinyresnet18 -> mobilenetv2': 'TR18→MV2',
}

ARCH_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
ARCH_MARKERS = ['o', 's', '^', 'D', 'v']


def to_log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def to_acc(s):
    try:
        return float(str(s).replace('%', '').strip())
    except (ValueError, TypeError):
        return np.nan


def load_rows(csv_path, exp_name, arch_pairs, temp=None, beta_pref=None):
    """Load one representative row per arch pair."""
    best = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] != exp_name:
                continue
            if temp and str(row.get('Temp', '')).strip() != str(temp):
                continue
            arch = row['Arch_Pair']
            if arch not in arch_pairs:
                continue
            beta = str(row.get('Beta', '')).strip()
            if arch not in best or (beta_pref and beta == beta_pref):
                best[arch] = row
    return [best[a] for a in arch_pairs if a in best]


def extract_series(row):
    ps   = [to_log10p(row[c]) for c in P_COLS]
    accs = [to_acc(row[c])    for c in ACC_COLS]
    return ps, accs


def draw_panel(ax, rows, title):
    ax2 = ax.twinx()
    x = np.arange(len(STAGES))

    for i, row in enumerate(rows):
        arch  = ARCH_SHORT.get(row['Arch_Pair'], row['Arch_Pair'][:10])
        color = ARCH_COLORS[i % len(ARCH_COLORS)]
        marker = ARCH_MARKERS[i % len(ARCH_MARKERS)]
        ps, accs = extract_series(row)

        ax.plot(x, ps, '-', color=color, marker=marker, lw=2, ms=6.5,
                label=arch, zorder=3)
        ax2.plot(x, accs, '--', color=color, marker=marker, lw=1.4, ms=4.5,
                 alpha=0.55, zorder=2)

    ax.axhline(THRESHOLD, color='black', ls=':', lw=1.3, alpha=0.7,
               label=r'$\eta=2^{-64}$')

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=8.5)
    ax.set_ylabel(r'$-\log_{10}(p)$  [solid]', fontsize=8.5)
    ax2.set_ylabel('Accuracy (%)  [dashed]', fontsize=8.5, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, min(P_FLOOR + 20, 320))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(True, ls=':', alpha=0.4)
    return ax


# ── Load data ──────────────────────────────────────────────────────────────────
c10_csv   = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
c100_csv1 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full1.csv')
c100_csv2 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full2.csv')
ti_csv    = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

panels = [
    # (csv, exp_name, arch_pairs, temp, beta_pref, title)
    (c10_csv,   'ExpA1_MainTable',  ['resnet20 -> resnet20','resnet20 -> vgg11','vgg11 -> vgg11'],
     '5', '0.5', 'CIFAR-10  (Soft Label API, T=5)'),
    (c10_csv,   'ExpB1_BGS_Adaptive', ['resnet20 -> resnet20','resnet20 -> vgg11','vgg11 -> vgg11'],
     None, None, 'CIFAR-10  (Hard Label BGS)'),
    (c100_csv1, 'ExpA1_MainTable', ['resnet56 -> resnet56','resnet56 -> resnet20','vgg11 -> vgg11'],
     '5', '0.7', 'CIFAR-100  (Soft Label API, T=5)'),
    (c100_csv2, 'ExpB1_BGS_Adaptive', ['resnet56 -> resnet56','resnet56 -> resnet20','vgg11 -> vgg11'],
     None, None, 'CIFAR-100  (Hard Label BGS)'),
    (ti_csv,    'Exp1_Soft_Main', ['tinyresnet18 -> resnet18','tinyresnet18 -> mobilenetv2'],
     '5', '0.7', 'Tiny-ImageNet  (Soft Label API, T=5)'),
    (ti_csv,    'Exp2_Hard_BGS', ['tinyresnet18 -> resnet18','tinyresnet18 -> mobilenetv2'],
     None, None, 'Tiny-ImageNet  (Hard Label BGS)'),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 13))

for idx, (csv_path, exp, arch_pairs, temp, beta, title) in enumerate(panels):
    ax = axes[idx // 2, idx % 2]
    rows = load_rows(csv_path, exp, arch_pairs, temp=temp, beta_pref=beta)
    draw_panel(ax, rows, title)
    if idx == 0:
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9, title='Arch Pair')

# Global note about dual axes
fig.text(0.5, 0.01,
         'Solid lines: Watermark security (left axis, higher = stronger)\n'
         'Dashed lines: Student accuracy % (right axis, shows model still functional)\n'
         'Dotted horizontal: Verification threshold η = 2⁻⁶⁴',
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

fig.suptitle(
    'EvalGuard Main Results: Security Strength and Student Accuracy Across Stages',
    fontweight='bold', fontsize=12, y=1.00)
fig.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(_here, 'main_results.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
