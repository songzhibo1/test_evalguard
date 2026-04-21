#!/usr/bin/env python3
"""
V-D Baseline Comparison Summary Table
Prints a comparison of EvalGuard vs DAWN vs Adi across all datasets.
Shows: teacher accuracy, student accuracy, fidelity loss,
       base security (-log10P), FT10% security, verification status.
Data from summary_baseline_comparison_TableX.csv.
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

THRESHOLD = 64 * np.log10(2)   # ≈ 19.27
P_FLOOR = 300.0
CHECK = '✓'
CROSS = '✗'

# Teacher fidelity (from summary_results_fidelity.txt, matched to baseline CSV)
TEACHER_PROTECTED = {
    'CIFAR10':  {'EvalGuard': {'soft': 92.59, 'hard': 91.72},
                 'DAWN':      {'soft': 83.98, 'hard': 83.98},
                 'Adi (Backdoor)': {'soft': 92.29, 'hard': 92.29}},
    'CIFAR100': {'EvalGuard': {'soft': 72.61, 'hard': 71.85},
                 'DAWN':      {'soft': 66.58, 'hard': 66.58},
                 'Adi (Backdoor)': {'soft': 72.28, 'hard': 72.28}},
    'TINYIMAGENET': {'EvalGuard': {'soft': 60.12, 'hard': 59.99},
                     'DAWN':      {'soft': 55.24, 'hard': 55.24},
                     'Adi (Backdoor)': {'soft': 62.80, 'hard': 62.80}},
}
TEACHER_ORIGINAL = {'CIFAR10': 92.59, 'CIFAR100': 72.61, 'TINYIMAGENET': 60.12}


def to_float(s):
    try:
        return float(str(s).replace('%', '').split('(')[0].strip())
    except (ValueError, TypeError):
        return np.nan


def to_log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def fmt_p(p):
    if p >= 299:
        return '≥300'
    if p <= 0.1:
        return '< 0.1'
    return f'{p:.1f}'


def fmt_ver(v):
    return CHECK if str(v).strip() == 'True' else CROSS


# ── Load baseline CSV ─────────────────────────────────────────────────────────
baseline_csv = os.path.join(_root, 'summary_baseline_comparison_TableX.csv')
rows_by_key = {}
with open(baseline_csv) as f:
    for row in csv.DictReader(f):
        key = (row['Dataset'], row['Method'], row['L_Mode'])
        rows_by_key[key] = row   # keep last (deduplicated)

# ── Build records ─────────────────────────────────────────────────────────────
DATASETS  = ['CIFAR10', 'CIFAR100', 'TINYIMAGENET']
METHODS   = ['EvalGuard', 'DAWN', 'Adi (Backdoor)']
MODES     = ['soft', 'hard']
DS_LABELS = {'CIFAR10': 'CIFAR-10', 'CIFAR100': 'CIFAR-100',
             'TINYIMAGENET': 'Tiny-IN'}

records = []
for ds in DATASETS:
    for method in METHODS:
        for mode in MODES:
            key = (ds, method, mode)
            row = rows_by_key.get(key)
            if row is None:
                continue
            t_orig  = TEACHER_ORIGINAL[ds]
            t_prot  = TEACHER_PROTECTED.get(ds, {}).get(method, {}).get(mode, np.nan)
            fidel_loss = round(t_prot - t_orig, 2) if not np.isnan(t_prot) else np.nan
            stu_acc    = to_float(row['Base_Acc'])
            base_p     = to_log10p(row['Base_P'])
            ft10_p     = to_log10p(row['FT10%_P'])
            base_ver   = fmt_ver(row['Base_Ver'])
            ft10_ver   = fmt_ver(row['FT10%_Ver'])

            records.append({
                'ds':       DS_LABELS[ds],
                'method':   method,
                'mode':     mode.capitalize(),
                't_orig':   f'{t_orig:.2f}',
                't_prot':   f'{t_prot:.2f}' if not np.isnan(t_prot) else '—',
                'fidel':    f'{fidel_loss:+.2f}' if not np.isnan(fidel_loss) else '—',
                'stu_acc':  f'{stu_acc:.1f}' if not np.isnan(stu_acc) else '—',
                'base_p':   fmt_p(base_p),
                'ft10_p':   fmt_p(ft10_p),
                'base_ver': base_ver,
                'ft10_ver': ft10_ver,
            })

# ── Terminal print ─────────────────────────────────────────────────────────────
HDR = (f"{'Dataset':<10} {'Method':<16} {'Mode':<5} "
       f"{'T-Orig':>7} {'T-Prot':>7} {'Fid.Δ':>6} "
       f"{'Stu.Acc':>7} {'Base-P':>7} {'FT10-P':>7} "
       f"{'BVer':^5} {'FT10V':^6}")
SEP = '-' * len(HDR)

print('\n' + '=' * len(HDR))
print(' Table V-D: Baseline Comparison Summary')
print('  T-Orig: teacher original accuracy  T-Prot: protected teacher accuracy')
print('  Fid.Δ: fidelity change from protection  Stu.Acc: student (extracted) accuracy')
print('  -log10P: security strength (≥19.3 = verified)')
print('=' * len(HDR))
print(HDR)
print(SEP)

prev_ds, prev_method = '', ''
for r in records:
    if r['ds'] != prev_ds:
        if prev_ds:
            print(SEP)
        prev_ds = r['ds']
    line = (f"{r['ds']:<10} {r['method']:<16} {r['mode']:<5} "
            f"{r['t_orig']:>7} {r['t_prot']:>7} {r['fidel']:>6} "
            f"{r['stu_acc']:>7} {r['base_p']:>7} {r['ft10_p']:>7} "
            f"{r['base_ver']:^5} {r['ft10_ver']:^6}")
    print(line)

print('=' * len(HDR))
print()

# ── PDF Table ─────────────────────────────────────────────────────────────────
col_headers = [
    'Dataset', 'Method', 'Mode',
    'Teacher\nOrig(%)', 'Teacher\nProt(%)', 'Fidelity\nΔ(%)',
    'Student\nAcc(%)',
    '−log₁₀(p)\nBase', '−log₁₀(p)\nFT 10%',
    'Base\nVer.', 'FT10%\nVer.',
]

cell_data = [[r['ds'], r['method'], r['mode'], r['t_orig'], r['t_prot'],
              r['fidel'], r['stu_acc'], r['base_p'], r['ft10_p'],
              r['base_ver'], r['ft10_ver']] for r in records]

METHOD_COLORS = {
    'EvalGuard':     '#d4edda',
    'DAWN':          '#fff3cd',
    'Adi (Backdoor)':'#e2e3e5',
}

fig_h = max(6, 0.4 * len(cell_data) + 2.5)
fig, ax = plt.subplots(figsize=(17, fig_h))
ax.axis('off')

tbl = ax.table(
    cellText=cell_data,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for ci in range(len(col_headers)):
    tbl[0, ci].set_facecolor('#2c3e50')
    tbl[0, ci].set_text_props(color='white', fontweight='bold')

for ri in range(1, len(cell_data) + 1):
    method = cell_data[ri - 1][1]
    bg = METHOD_COLORS.get(method, 'white')
    for ci in range(len(col_headers)):
        tbl[ri, ci].set_facecolor(bg)
    # Verification columns
    for ci_ver, ci_col in [(9, 'base_ver'), (10, 'ft10_ver')]:
        val = cell_data[ri - 1][ci_ver - 1 + 1]   # 0-indexed offset
    for vi, ci in enumerate([9, 10]):
        val = cell_data[ri - 1][ci]
        if val == CHECK:
            tbl[ri, ci].set_facecolor('#28a745')
            tbl[ri, ci].set_text_props(color='white', fontweight='bold')
        elif val == CROSS:
            tbl[ri, ci].set_facecolor('#dc3545')
            tbl[ri, ci].set_text_props(color='white', fontweight='bold')
    # Highlight zero fidelity loss
    fid = cell_data[ri - 1][5]
    if fid in ('+0.00', '0.00', ' 0.00'):
        tbl[ri, 5].set_facecolor('#28a745')
        tbl[ri, 5].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    'Table V-D: Baseline Watermark Comparison\n'
    'EvalGuard vs DAWN vs Adi — Fidelity, Student Accuracy, Security Strength',
    fontsize=10, fontweight='bold', y=0.99)

out_path = os.path.join(_here, 'table3_baseline_summary.pdf')
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
