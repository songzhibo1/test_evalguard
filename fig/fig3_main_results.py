#!/usr/bin/env python3
"""
V-C Main Results Figure
EvalGuard watermark verification strength across datasets and label modes.
Reads from the three experiment CSV files at the project root.
"""
import os
import pandas as pd
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

THRESHOLD = 64 * np.log10(2)  # -log10(2^-64) ≈ 19.27
P_FLOOR = 300.0                # cap for 0.0e+00 p-values

STAGES_SOFT = ['Base', 'FT 1%', 'FT 5%', 'FT 10%']
P_COLS_SOFT = ['Base_P', 'FT1%_P', 'FT5%_P', 'FT10%_P']

STAGES_HARD = ['Base', 'FT 1%', 'FT 5%', 'FT 10%']
P_COLS_HARD = ['Base_P', 'FT1%_P', 'FT5%_P', 'FT10%_P']

ARCH_LABELS = {
    'resnet20 -> resnet20': 'R20→R20',
    'resnet20 -> vgg11':    'R20→V11',
    'vgg11 -> vgg11':       'V11→V11',
    'resnet56 -> resnet56': 'R56→R56',
    'resnet56 -> resnet20': 'R56→R20',
    'vgg11 -> vgg11':       'V11→V11',
    'tinyresnet18 -> resnet18':    'TR18→R18',
    'tinyresnet18 -> mobilenetv2': 'TR18→MV2',
}

STAGE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def to_log10p(p_str):
    try:
        v = float(str(p_str).strip())
        if v <= 0.0:
            return P_FLOOR
        return min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def load_soft(csv_file, exp_name, temp='5'):
    df = pd.read_csv(csv_file)
    rows = df[(df['Experiment'] == exp_name) & (df['Temp'].astype(str) == temp)]
    results = []
    for _, r in rows.iterrows():
        arch = r['Arch_Pair']
        label = ARCH_LABELS.get(arch, arch)
        log10ps = [to_log10p(r[c]) for c in P_COLS_SOFT]
        results.append({'arch': label, 'log10ps': log10ps,
                        'acc': r['Base_Acc']})
    return results


def load_hard(csv_file, exp_name):
    df = pd.read_csv(csv_file)
    rows = df[df['Experiment'] == exp_name]
    results = []
    for _, r in rows.iterrows():
        arch = r['Arch_Pair']
        label = ARCH_LABELS.get(arch, arch)
        log10ps = [to_log10p(r[c]) for c in P_COLS_HARD]
        results.append({'arch': label, 'log10ps': log10ps,
                        'acc': r['Base_Acc']})
    return results


# ── Load data ──────────────────────────────────────────────────────────────────
c10_csv   = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
c100_csv1 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full1.csv')
c100_csv2 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full2.csv')
ti_csv    = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

datasets_soft = [
    ('CIFAR-10',      load_soft(c10_csv,   'ExpA1_MainTable',  temp='5')),
    ('CIFAR-100',     load_soft(c100_csv1, 'ExpA1_MainTable',  temp='5')),
    ('TinyImageNet',  load_soft(ti_csv,    'Exp1_Soft_Main',   temp='5')),
]
datasets_hard = [
    ('CIFAR-10',     load_hard(c10_csv,   'ExpB1_BGS_Adaptive')),
    ('CIFAR-100',    load_hard(c100_csv2, 'ExpB1_BGS_Adaptive')),
    ('TinyImageNet', load_hard(ti_csv,    'Exp2_Hard_BGS')),
]

# Deduplicate: for same arch, keep first (beta=0.5 or 0.7, take one)
def dedup(records):
    seen, out = set(), []
    for r in records:
        if r['arch'] not in seen:
            seen.add(r['arch'])
            out.append(r)
    return out

datasets_soft = [(ds, dedup(recs)) for ds, recs in datasets_soft]
datasets_hard = [(ds, dedup(recs)) for ds, recs in datasets_hard]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

row_titles = ['Soft-Label API', 'Hard-Label BGS']
all_data = [datasets_soft, datasets_hard]

for row_idx, (row_title, datasets) in enumerate(zip(row_titles, all_data)):
    for col_idx, (ds_name, records) in enumerate(datasets):
        ax = axes[row_idx, col_idx]

        if not records:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        arch_names = [r['arch'] for r in records]
        x = np.arange(len(arch_names))
        n_stages = len(STAGES_SOFT)
        total_width = 0.72
        bar_w = total_width / n_stages

        for si, (stage, color) in enumerate(zip(STAGES_SOFT, STAGE_COLORS)):
            heights = [r['log10ps'][si] for r in records]
            offset = (si - (n_stages - 1) / 2) * bar_w
            bars = ax.bar(x + offset, heights, bar_w * 0.9, label=stage,
                          color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.axhline(THRESHOLD, color='black', ls='--', lw=1.2, alpha=0.7,
                   label=r'$\eta=2^{-64}$')

        ax.set_xticks(x)
        ax.set_xticklabels(arch_names, fontsize=8.5)
        ax.set_ylim(0, min(P_FLOOR + 20, 340))
        ax.set_ylabel(r'$-\log_{10}(p)$' if col_idx == 0 else '')
        ax.set_title(f'{ds_name}\n({row_title})', fontweight='bold', fontsize=10)
        ax.grid(True, axis='y', ls=':', alpha=0.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))

        if row_idx == 0 and col_idx == 2:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

fig.suptitle('EvalGuard Watermark Verification: Security Strength Across All Configurations',
             fontweight='bold', fontsize=12, y=1.01)
fig.tight_layout()

out_path = os.path.join(_here, 'main_results.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
