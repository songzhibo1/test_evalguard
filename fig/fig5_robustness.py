#!/usr/bin/env python3
"""
V-E Robustness Figure
Left: Pruning robustness — -log10(P) vs pruning ratio for soft and hard modes.
Right: INT8 quantization robustness — bar chart before/after quant.
Data from CIFAR-10 (soft: resnet20/vgg11, hard: resnet20) and TinyImageNet.
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

THRESHOLD = 64 * np.log10(2)   # ≈ 19.27
P_FLOOR   = 300.0
PRUNE_LEVELS = [0, 25, 50, 75]  # % pruned


def to_log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def get_prune_log10p(row, base_p_col='Base_P'):
    base = to_log10p(row[base_p_col])
    p25  = to_log10p(row.get('PRUNE_25%_P', 'N/A'))
    p50  = to_log10p(row.get('PRUNE_50%_P', 'N/A'))
    p75  = to_log10p(row.get('PRUNE_75%_P', 'N/A'))
    return [base, p25, p50, p75]


def get_quant_log10p(row, base_p_col='Base_P'):
    base = to_log10p(row[base_p_col])
    q    = to_log10p(row.get('INT8_QUANT_P', 'N/A'))
    return base, q


def load_row(csv_path, exp_name, arch_pair):
    df = pd.read_csv(csv_path)
    rows = df[(df['Experiment'] == exp_name) & (df['Arch_Pair'] == arch_pair)]
    return rows.iloc[0] if not rows.empty else None


# ── C10 prune data ─────────────────────────────────────────────────────────────
c10_csv = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
ti_csv  = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

c10_soft_r20 = load_row(c10_csv, 'ExpC4_PruneSoft', 'resnet20 -> resnet20')
c10_soft_v11 = load_row(c10_csv, 'ExpC4_PruneSoft', 'vgg11 -> vgg11')
c10_hard_r20 = load_row(c10_csv, 'ExpC6_PruneHard', 'resnet20 -> resnet20')
ti_soft      = load_row(ti_csv,  'Exp3_Soft_Prune',  'tinyresnet18 -> resnet18')
ti_hard      = load_row(ti_csv,  'Exp8_Hard_Prune',   'tinyresnet18 -> resnet18')

# For pruning rows, Base_P is not filled — use the distilled model p-value from main exp
# Fallback: look up matching main exp row
def lookup_base_p(csv_path, arch_pair, mode_filter, exp_filter, temp='5'):
    df = pd.read_csv(csv_path)
    rows = df[(df['Experiment'] == exp_filter) &
              (df['Arch_Pair'] == arch_pair) &
              (df['Temp'].astype(str) == temp)]
    if rows.empty:
        return 'N/A'
    return rows.iloc[0]['Base_P']

# Get base P from main soft/hard experiments
c10_base_soft_r20_p = lookup_base_p(c10_csv, 'resnet20 -> resnet20', 'soft', 'ExpA1_MainTable')
c10_base_soft_v11_p = lookup_base_p(c10_csv, 'vgg11 -> vgg11', 'soft', 'ExpA1_MainTable')
c10_base_hard_r20_p = lookup_base_p(c10_csv, 'resnet20 -> resnet20', 'hard', 'ExpB1_BGS_Adaptive', temp='1')

def lookup_ti_base_p(csv_path, arch_pair, exp_filter, temp='5'):
    df = pd.read_csv(csv_path)
    rows = df[(df['Experiment'] == exp_filter) &
              (df['Arch_Pair'] == arch_pair) &
              (df['Temp'].astype(str) == temp)]
    if rows.empty:
        return 'N/A'
    return rows.iloc[0]['Base_P']

ti_base_soft_p = lookup_ti_base_p(ti_csv, 'tinyresnet18 -> resnet18', 'Exp1_Soft_Main')
ti_base_hard_p = lookup_ti_base_p(ti_csv, 'tinyresnet18 -> resnet18', 'Exp2_Hard_BGS', temp='1')

# Build pruning series (inject base P from main exp)
def prune_series(prune_row, base_p_override):
    if prune_row is None:
        return [0.0] * 4
    p25 = to_log10p(prune_row.get('PRUNE_25%_P', 'N/A'))
    p50 = to_log10p(prune_row.get('PRUNE_50%_P', 'N/A'))
    p75 = to_log10p(prune_row.get('PRUNE_75%_P', 'N/A'))
    return [to_log10p(base_p_override), p25, p50, p75]

series = {
    'C10 Soft R20→R20': prune_series(c10_soft_r20, c10_base_soft_r20_p),
    'C10 Soft V11→V11': prune_series(c10_soft_v11, c10_base_soft_v11_p),
    'C10 Hard R20→R20': prune_series(c10_hard_r20, c10_base_hard_r20_p),
    'TI Soft TR18→R18': prune_series(ti_soft, ti_base_soft_p),
    'TI Hard TR18→R18': prune_series(ti_hard, ti_base_hard_p),
}

COLORS = ['#1f77b4', '#aec7e8', '#d62728', '#2ca02c', '#98df8a']
MARKERS = ['o', 's', '^', 'D', 'v']
LINE_STYLES = ['-', '--', '-', '-', '--']

# ── INT8 quant data ────────────────────────────────────────────────────────────
c10_quant_r20 = load_row(c10_csv, 'ExpC5_QuantSoft', 'resnet20 -> resnet20')
c10_quant_v11 = load_row(c10_csv, 'ExpC5_QuantSoft', 'vgg11 -> vgg11')
ti_quant      = load_row(ti_csv,  'Exp3_Soft_Quantize', 'tinyresnet18 -> resnet18')

quant_data = [
    ('C10 R20', to_log10p(c10_base_soft_r20_p),
                to_log10p(c10_quant_r20['INT8_QUANT_P']) if c10_quant_r20 is not None else 0.0),
    ('C10 V11', to_log10p(c10_base_soft_v11_p),
                to_log10p(c10_quant_v11['INT8_QUANT_P']) if c10_quant_v11 is not None else 0.0),
    ('TI R18',  to_log10p(ti_base_soft_p),
                to_log10p(ti_quant['INT8_QUANT_P']) if ti_quant is not None else 0.0),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax_prune, ax_quant) = plt.subplots(1, 2, figsize=(12, 5))

# Left: pruning
for (name, vals), color, marker, ls in zip(series.items(), COLORS, MARKERS, LINE_STYLES):
    ax_prune.plot(PRUNE_LEVELS, vals, ls, color=color, marker=marker,
                  lw=2, ms=7, label=name)

ax_prune.axhline(THRESHOLD, color='black', ls='--', lw=1.2, alpha=0.7,
                 label=r'Threshold $\eta=2^{-64}$')
ax_prune.fill_between(PRUNE_LEVELS, 0, THRESHOLD, alpha=0.05, color='red',
                      label='Unverified zone')
ax_prune.set_xlabel('Pruning Ratio (%)', fontsize=11)
ax_prune.set_ylabel(r'$-\log_{10}(p)$', fontsize=11)
ax_prune.set_title('(a) Pruning Robustness', fontweight='bold', fontsize=11)
ax_prune.set_xticks(PRUNE_LEVELS)
ax_prune.set_xticklabels(['0%', '25%', '50%', '75%'])
ax_prune.set_ylim(0, min(P_FLOOR + 20, 320))
ax_prune.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))
ax_prune.legend(fontsize=8.5, framealpha=0.9)
ax_prune.grid(True, ls=':', alpha=0.5)

# Right: INT8 quant
qnames = [d[0] for d in quant_data]
q_base = [d[1] for d in quant_data]
q_quant = [d[2] for d in quant_data]
x = np.arange(len(qnames))
bar_w = 0.35

ax_quant.bar(x - bar_w/2, q_base,  bar_w, label='Before INT8 Quant.',
             color='#1f77b4', alpha=0.85, edgecolor='white')
ax_quant.bar(x + bar_w/2, q_quant, bar_w, label='After INT8 Quant.',
             color='#ff7f0e', alpha=0.85, edgecolor='white', hatch='//')
ax_quant.axhline(THRESHOLD, color='black', ls='--', lw=1.2, alpha=0.7,
                 label=r'Threshold $\eta=2^{-64}$')
ax_quant.set_xticks(x)
ax_quant.set_xticklabels(qnames, fontsize=10)
ax_quant.set_ylabel(r'$-\log_{10}(p)$', fontsize=11)
ax_quant.set_title('(b) INT8 Quantization Robustness', fontweight='bold', fontsize=11)
ax_quant.set_ylim(0, min(P_FLOOR + 20, 320))
ax_quant.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))
ax_quant.legend(fontsize=9, framealpha=0.9)
ax_quant.grid(True, axis='y', ls=':', alpha=0.5)

fig.suptitle('EvalGuard Robustness to Post-Processing Attacks (Soft Label)',
             fontweight='bold', fontsize=12)
fig.tight_layout()

out_path = os.path.join(_here, 'robustness.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
