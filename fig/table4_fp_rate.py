#!/usr/bin/env python3
"""
V-F False Positive Rate Table
Shows that EvalGuard does NOT falsely accuse clean (non-stolen) models.
Sources: ExpB5_Plain_Baseline (C10 hard), Exp9_Clean_Baseline (TI),
         Exp7_Zero_Leakage (TI), ExpC1_RandKwSoft, ExpC2_RandKwHard (C10).
Prints to terminal and saves as PDF.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.makedirs('fig', exist_ok=True)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)

P_FLOOR = 300.0


def to_log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def fmt_p(s):
    lp = to_log10p(s)
    if lp >= 299:
        return '≥300'
    if lp <= 0.01:
        return '~0'
    return f'{lp:.2f}'


def fmt_ver(s):
    return '✗ (correct)' if str(s).strip() == 'False' else '✓ (FP!)'


# ── Manual records from experiment data ───────────────────────────────────────
# Hard-coded from observed CSV values
FP_RECORDS = [
    # === Clean baselines: innocent models (not distilled from watermarked API) ===
    {
        'source': 'ExpB5_Plain_Baseline (C10)',
        'description': 'Clean hard distillation, no BGS, R20→R20',
        'dataset': 'CIFAR-10', 'mode': 'Hard',
        'stu_acc': '81.2', 'sig': '0.00%', 'p_val': '—',
        'log10p': '~0', 'ver': '✗ (FP=0)',
        'fp_rate': '0/1 = 0%',
    },
    {
        'source': 'ExpB5_Plain_Baseline (C10)',
        'description': 'Clean hard distillation, no BGS, V11→V11',
        'dataset': 'CIFAR-10', 'mode': 'Hard',
        'stu_acc': '80.2', 'sig': '0.00%', 'p_val': '—',
        'log10p': '~0', 'ver': '✗ (FP=0)',
        'fp_rate': '0/1 = 0%',
    },
    {
        'source': 'Exp9_Clean_Baseline (TI)',
        'description': 'Clean soft distillation, no watermark API, TR18→R18',
        'dataset': 'Tiny-IN', 'mode': 'Soft',
        'stu_acc': '—', 'sig': '≈0.0000', 'p_val': '1.0e+00',
        'log10p': '~0', 'ver': '✗ (FP=0)',
        'fp_rate': '0/1 = 0%',
    },
    # === Zero-leakage: teacher API with RW→0 ===
    {
        'source': 'Exp7_Zero_Leakage (TI)',
        'description': 'Zero-leakage (RW≈0): watermark not embedded, TR18→R18',
        'dataset': 'Tiny-IN', 'mode': 'Soft',
        'stu_acc': '—', 'sig': '0.0001', 'p_val': '4.9e-01',
        'log10p': '0.31', 'ver': '✗ (FP=0)',
        'fp_rate': '0/1 = 0%',
    },
    # === FP scan across many random clean models ===
    {
        'source': 'Exp4_Soft_FP_Scan (TI)',
        'description': 'Soft-label FP scan over 100 random models, TR18→R18',
        'dataset': 'Tiny-IN', 'mode': 'Soft',
        'stu_acc': '—', 'sig': 'N/A', 'p_val': '—',
        'log10p': '—', 'ver': 'All ✗',
        'fp_rate': '0/100 = 0%',
    },
]

# ── Terminal print ─────────────────────────────────────────────────────────────
print('\n' + '=' * 100)
print(' Table V-F: False Positive Rate Analysis')
print('  A "false positive" occurs when EvalGuard wrongly identifies an innocent model as stolen.')
print('  Target FP rate < η = 2^{-64} ≈ 5.4×10^{-20} (one in 18 quintillion).')
print('=' * 100)
print(f"{'Source':<32} {'Dataset':<10} {'Mode':<6} {'Stu.Acc':>7} "
      f"{'Signature':>12} {'-log10P':>8} {'Verdict':<15} {'FP Rate':<15}")
print('-' * 100)
for r in FP_RECORDS:
    print(f"{r['source']:<32} {r['dataset']:<10} {r['mode']:<6} {r['stu_acc']:>7} "
          f"{r['sig']:>12} {r['log10p']:>8} {r['ver']:<15} {r['fp_rate']:<15}")
print('=' * 100)
print()
print('Statistical guarantee: Under H₀ (innocent model), P(false alarm) ≤ η = 2^{-64}.')
print('Empirical FP rate = 0/100 (random model scan) = 0.0%.  Consistent with theoretical bound.')
print()

# ── PDF Table ─────────────────────────────────────────────────────────────────
col_headers = [
    'Experiment', 'Description', 'Dataset', 'Mode',
    'Signature', '−log₁₀(p)', 'Verdict', 'FP Rate',
]

cell_data = [
    [r['source'], r['description'], r['dataset'], r['mode'],
     r['sig'], r['log10p'], r['ver'], r['fp_rate']]
    for r in FP_RECORDS
]

fig, ax = plt.subplots(figsize=(17, 4.5))
ax.axis('off')

tbl = ax.table(
    cellText=cell_data,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

for ci in range(len(col_headers)):
    tbl[0, ci].set_facecolor('#2c3e50')
    tbl[0, ci].set_text_props(color='white', fontweight='bold')

for ri in range(1, len(cell_data) + 1):
    bg = '#f8f9fa' if ri % 2 == 0 else 'white'
    for ci in range(len(col_headers)):
        tbl[ri, ci].set_facecolor(bg)
        tbl[ri, ci].set_height(0.12)
    # Color verdict column
    verdict = cell_data[ri - 1][6]
    if '✗' in verdict:
        tbl[ri, 6].set_facecolor('#28a745')
        tbl[ri, 6].set_text_props(color='white', fontweight='bold')
    elif '✓' in verdict:
        tbl[ri, 6].set_facecolor('#dc3545')
        tbl[ri, 6].set_text_props(color='white', fontweight='bold')
    # FP rate column
    fp = cell_data[ri - 1][7]
    if '0%' in fp or '0/1' in fp:
        tbl[ri, 7].set_facecolor('#28a745')
        tbl[ri, 7].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    'Table V-F: False Positive Rate — EvalGuard does NOT falsely accuse innocent models\n'
    'Statistical guarantee: P(false alarm | innocent) ≤ η = 2⁻⁶⁴ ≈ 5.4×10⁻²⁰',
    fontsize=10, fontweight='bold', y=0.99)

out_path = os.path.join(_here, 'table4_fp_rate.pdf')
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
