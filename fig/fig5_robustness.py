#!/usr/bin/env python3
"""
V-E Robustness Figure
2×2 layout:
  (a) Pruning accuracy degradation   (b) Pruning security strength
  (c) INT8 quant accuracy             (d) INT8 quant security strength

Shows both utility (accuracy) and security (-log10 P) side-by-side so the
reviewer can see that at 75% pruning the MODEL becomes useless (acc→10%)
BEFORE the watermark fails — i.e., the attacker loses both ways.
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
PRUNE_LEVELS = [0, 25, 50, 75]


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


def load_row(csv_path, exp_name, arch_pair):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] == exp_name and row['Arch_Pair'] == arch_pair:
                return row
    return None


# ── Fetch data ────────────────────────────────────────────────────────────────
c10_csv = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
ti_csv  = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

# Base accuracy from main distillation experiments (T=5)
def get_base(csv_path, soft_exp, hard_exp, arch_pair, temp='5', hard_temp='1'):
    soft_p, soft_acc = None, None
    hard_p, hard_acc = None, None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] == soft_exp and row['Arch_Pair'] == arch_pair \
                    and str(row['Temp']).strip() == temp:
                soft_p   = to_log10p(row['Base_P'])
                soft_acc = to_acc(row['Base_Acc'])
                break
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] == hard_exp and row['Arch_Pair'] == arch_pair:
                hard_p   = to_log10p(row['Base_P'])
                hard_acc = to_acc(row['Base_Acc'])
                break
    return (soft_acc, soft_p), (hard_acc, hard_p)


(c10_base_soft_acc, c10_base_soft_p), (c10_base_hard_acc, c10_base_hard_p) = get_base(
    c10_csv, 'ExpA1_MainTable', 'ExpB1_BGS_Adaptive', 'resnet20 -> resnet20')
(ti_base_soft_acc, ti_base_soft_p), (ti_base_hard_acc, ti_base_hard_p) = get_base(
    ti_csv, 'Exp1_Soft_Main', 'Exp2_Hard_BGS', 'tinyresnet18 -> resnet18',
    temp='5', hard_temp='1')

# Pruning data
def prune_series_acc_p(prune_row, base_acc, base_p):
    if prune_row is None:
        return [base_acc] + [np.nan]*3, [base_p] + [0.0]*3
    accs = [base_acc,
            to_acc(prune_row.get('PRUNE_25%_Acc', 'N/A')),
            to_acc(prune_row.get('PRUNE_50%_Acc', 'N/A')),
            to_acc(prune_row.get('PRUNE_75%_Acc', 'N/A'))]
    ps   = [base_p,
            to_log10p(prune_row.get('PRUNE_25%_P', 'N/A')),
            to_log10p(prune_row.get('PRUNE_50%_P', 'N/A')),
            to_log10p(prune_row.get('PRUNE_75%_P', 'N/A'))]
    return accs, ps


r_c10_soft = load_row(c10_csv, 'ExpC4_PruneSoft', 'resnet20 -> resnet20')
r_c10_hard = load_row(c10_csv, 'ExpC6_PruneHard', 'resnet20 -> resnet20')
r_ti_soft  = load_row(ti_csv,  'Exp3_Soft_Prune', 'tinyresnet18 -> resnet18')
r_ti_hard  = load_row(ti_csv,  'Exp8_Hard_Prune',  'tinyresnet18 -> resnet18')

c10_soft_prune_acc, c10_soft_prune_p = prune_series_acc_p(
    r_c10_soft, c10_base_soft_acc, c10_base_soft_p)
c10_hard_prune_acc, c10_hard_prune_p = prune_series_acc_p(
    r_c10_hard, c10_base_hard_acc, c10_base_hard_p)
ti_soft_prune_acc,  ti_soft_prune_p  = prune_series_acc_p(
    r_ti_soft, ti_base_soft_acc, ti_base_soft_p)
ti_hard_prune_acc,  ti_hard_prune_p  = prune_series_acc_p(
    r_ti_hard, ti_base_hard_acc, ti_base_hard_p)

# INT8 quant data
def quant_pair(csv_path, exp, arch, base_acc, base_p):
    r = load_row(csv_path, exp, arch)
    if r is None:
        return base_acc, np.nan, base_p, 0.0
    return base_acc, to_acc(r.get('INT8_QUANT_Acc', 'N/A')), \
           base_p,   to_log10p(r.get('INT8_QUANT_P', 'N/A'))


c10_q_ba, c10_q_qa, c10_q_bp, c10_q_qp = quant_pair(
    c10_csv, 'ExpC5_QuantSoft', 'resnet20 -> resnet20',
    c10_base_soft_acc, c10_base_soft_p)
ti_q_ba,  ti_q_qa,  ti_q_bp,  ti_q_qp  = quant_pair(
    ti_csv, 'Exp3_Soft_Quantize', 'tinyresnet18 -> resnet18',
    ti_base_soft_acc, ti_base_soft_p)

# ── Styles ────────────────────────────────────────────────────────────────────
CONFIGS = [
    ('C10 Soft (R20)',  c10_soft_prune_acc, c10_soft_prune_p, '#1f77b4', 'o', '-'),
    ('C10 Hard (R20)',  c10_hard_prune_acc, c10_hard_prune_p, '#d62728', '^', '--'),
    ('TI Soft (TR18)',  ti_soft_prune_acc,  ti_soft_prune_p,  '#2ca02c', 's', '-'),
    ('TI Hard (TR18)',  ti_hard_prune_acc,  ti_hard_prune_p,  '#ff7f0e', 'D', '--'),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax_pa, ax_ps), (ax_qa, ax_qs) = axes

# ── (a) Pruning accuracy ───────────────────────────────────────────────────────
for name, accs, _, color, marker, ls in CONFIGS:
    ax_pa.plot(PRUNE_LEVELS, accs, ls, color=color, marker=marker,
               lw=2, ms=7, label=name)
ax_pa.axhline(10, color='gray', ls=':', lw=1.2, alpha=0.6,
              label='Random-chance (~10%)')
ax_pa.set_ylabel('Student Model Accuracy (%)', fontsize=10)
ax_pa.set_title('(a) Model Accuracy Under Pruning', fontweight='bold')
ax_pa.set_xticks(PRUNE_LEVELS)
ax_pa.set_xticklabels(['0%', '25%', '50%', '75%'])
ax_pa.set_ylim(0, 100)
ax_pa.legend(fontsize=8.5, framealpha=0.9)
ax_pa.grid(True, ls=':', alpha=0.5)

# ── (b) Pruning security ───────────────────────────────────────────────────────
for name, _, ps, color, marker, ls in CONFIGS:
    ax_ps.plot(PRUNE_LEVELS, ps, ls, color=color, marker=marker,
               lw=2, ms=7, label=name)
ax_ps.axhline(THRESHOLD, color='black', ls='--', lw=1.2, alpha=0.7,
              label=r'Threshold $\eta=2^{-64}$')
ax_ps.fill_between(PRUNE_LEVELS, 0, THRESHOLD, alpha=0.06, color='red')
ax_ps.set_ylabel(r'$-\log_{10}(p)$', fontsize=10)
ax_ps.set_title('(b) Watermark Security Under Pruning', fontweight='bold')
ax_ps.set_xticks(PRUNE_LEVELS)
ax_ps.set_xticklabels(['0%', '25%', '50%', '75%'])
ax_ps.set_ylim(0, min(P_FLOOR + 20, 320))
ax_ps.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))
ax_ps.legend(fontsize=8.5, framealpha=0.9)
ax_ps.grid(True, ls=':', alpha=0.5)
# Note: at 75% pruning, both accuracy and security fail — model is useless anyway
ax_ps.annotate('Model accuracy\ncollapsed to ~10%', xy=(75, 3), fontsize=7.5,
               color='gray', ha='center', style='italic')

# ── (c) INT8 accuracy ─────────────────────────────────────────────────────────
quant_names  = ['C10 Soft\nR20→R20', 'TI Soft\nTR18→R18']
q_base_accs  = [c10_q_ba, ti_q_ba]
q_after_accs = [c10_q_qa, ti_q_qa]

x = np.arange(len(quant_names))
bw = 0.35
ax_qa.bar(x - bw/2, q_base_accs,  bw, label='Before INT8 Quant.',
          color='#1f77b4', alpha=0.85, edgecolor='white')
ax_qa.bar(x + bw/2, q_after_accs, bw, label='After INT8 Quant.',
          color='#aec7e8', alpha=0.85, edgecolor='#1f77b4', linewidth=1.2)
ax_qa.set_xticks(x)
ax_qa.set_xticklabels(quant_names, fontsize=10)
ax_qa.set_ylabel('Student Model Accuracy (%)', fontsize=10)
ax_qa.set_title('(c) Accuracy: INT8 Quantization', fontweight='bold')
ax_qa.set_ylim(0, 100)
ax_qa.legend(fontsize=9)
ax_qa.grid(True, axis='y', ls=':', alpha=0.5)
# annotate delta
for xi, (ba, qa) in enumerate(zip(q_base_accs, q_after_accs)):
    if not (np.isnan(ba) or np.isnan(qa)):
        delta = qa - ba
        ax_qa.annotate(f'Δ={delta:+.1f}%', xy=(xi + bw/2, max(ba, qa) + 0.8),
                        ha='center', fontsize=8.5, color='#1f77b4')

# ── (d) INT8 security ─────────────────────────────────────────────────────────
q_base_ps  = [c10_q_bp, ti_q_bp]
q_after_ps = [c10_q_qp, ti_q_qp]

ax_qs.bar(x - bw/2, q_base_ps,  bw, label='Before INT8 Quant.',
          color='#d62728', alpha=0.85, edgecolor='white')
ax_qs.bar(x + bw/2, q_after_ps, bw, label='After INT8 Quant.',
          color='#f7b6b6', alpha=0.9, edgecolor='#d62728', linewidth=1.2)
ax_qs.axhline(THRESHOLD, color='black', ls='--', lw=1.2, alpha=0.7,
              label=r'Threshold $\eta=2^{-64}$')
ax_qs.set_xticks(x)
ax_qs.set_xticklabels(quant_names, fontsize=10)
ax_qs.set_ylabel(r'$-\log_{10}(p)$', fontsize=10)
ax_qs.set_title('(d) Watermark Security: INT8 Quantization', fontweight='bold')
ax_qs.set_ylim(0, min(P_FLOOR + 20, 320))
ax_qs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))
ax_qs.legend(fontsize=9)
ax_qs.grid(True, axis='y', ls=':', alpha=0.5)

# Shared x-labels for pruning plots
for ax in (ax_pa, ax_ps):
    ax.set_xlabel('Pruning Ratio', fontsize=10)

fig.suptitle('EvalGuard Robustness to Post-Processing: Pruning & Quantization',
             fontweight='bold', fontsize=12)
fig.tight_layout()

out_path = os.path.join(_here, 'robustness.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
