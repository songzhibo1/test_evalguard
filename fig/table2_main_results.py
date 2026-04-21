#!/usr/bin/env python3
"""
V-C Main Results Table
Prints comprehensive detection results to terminal and saves as PDF.
Shows EvalGuard verification across all datasets, arch pairs, label modes.
Canonical temperature T=5 for soft label; T=1 for hard label.
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

CHECK = '✓'
CROSS = '✗'


def to_log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0

P_FLOOR = 300.0


def fmt_p(s):
    lp = to_log10p(s)
    if lp >= 299:
        return '≥300'
    if lp <= 0.1:
        return '< 0.1'
    return f'{lp:.1f}'


def fmt_ver(s):
    return CHECK if str(s).strip() == 'True' else CROSS


def fmt_acc(s):
    try:
        return f"{float(str(s).replace('%','').strip()):.1f}"
    except (ValueError, TypeError):
        return '—'


def load_soft(csv_path, exp_name, arch_pairs, temp='5', beta_pref='0.5'):
    """Load rows for soft label at given temperature, preferred beta."""
    rows_by_arch = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] != exp_name:
                continue
            if str(row['Temp']).strip() != temp:
                continue
            arch = row['Arch_Pair']
            if arch not in arch_pairs:
                continue
            beta = str(row.get('Beta', '')).strip()
            # prefer beta_pref, but accept any
            if arch not in rows_by_arch or beta == beta_pref:
                rows_by_arch[arch] = row
    return [rows_by_arch[a] for a in arch_pairs if a in rows_by_arch]


def load_hard(csv_path, exp_name, arch_pairs):
    rows_by_arch = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] != exp_name:
                continue
            arch = row['Arch_Pair']
            if arch in arch_pairs:
                rows_by_arch[arch] = row
    return [rows_by_arch[a] for a in arch_pairs if a in rows_by_arch]


def row_to_record(row, dataset, mode):
    return {
        'dataset': dataset,
        'arch':    row['Arch_Pair'],
        'mode':    mode,
        't_acc':   fmt_acc(row.get('T_Acc', '—')),
        'stu_acc': fmt_acc(row['Base_Acc']),
        'base_p':  fmt_p(row['Base_P']),
        'base_v':  fmt_ver(row['Base_Ver']),
        'ft1_v':   fmt_ver(row['FT1%_Ver']),
        'ft5_v':   fmt_ver(row['FT5%_Ver']),
        'ft10_v':  fmt_ver(row['FT10%_Ver']),
        'ft10_acc': fmt_acc(row['FT10%_Acc']),
    }


# ── Load data ──────────────────────────────────────────────────────────────────
c10_csv   = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
c100_csv1 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full1.csv')
c100_csv2 = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full2.csv')
ti_csv    = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

C10_ARCHS_SOFT = ['resnet20 -> resnet20', 'resnet20 -> vgg11', 'vgg11 -> vgg11']
C10_ARCHS_HARD = ['resnet20 -> resnet20', 'resnet20 -> vgg11', 'vgg11 -> vgg11']
C100_ARCHS_SOFT = ['resnet56 -> resnet56', 'resnet56 -> resnet20', 'vgg11 -> vgg11']
C100_ARCHS_HARD = ['resnet56 -> resnet56', 'resnet56 -> resnet20', 'vgg11 -> vgg11']
TI_ARCHS_SOFT = ['tinyresnet18 -> resnet18', 'tinyresnet18 -> mobilenetv2']
TI_ARCHS_HARD = ['tinyresnet18 -> resnet18', 'tinyresnet18 -> mobilenetv2']

records = []

for arch in C10_ARCHS_SOFT:
    rs = load_soft(c10_csv, 'ExpA1_MainTable', [arch], temp='5', beta_pref='0.5')
    for r in rs:
        records.append(row_to_record(r, 'CIFAR-10', 'Soft'))

for arch in C10_ARCHS_HARD:
    rs = load_hard(c10_csv, 'ExpB1_BGS_Adaptive', [arch])
    for r in rs:
        records.append(row_to_record(r, 'CIFAR-10', 'Hard'))

for arch in C100_ARCHS_SOFT:
    rs = load_soft(c100_csv1, 'ExpA1_MainTable', [arch], temp='5', beta_pref='0.7')
    for r in rs:
        records.append(row_to_record(r, 'CIFAR-100', 'Soft'))

for arch in C100_ARCHS_HARD:
    rs = load_hard(c100_csv2, 'ExpB1_BGS_Adaptive', [arch])
    for r in rs:
        records.append(row_to_record(r, 'CIFAR-100', 'Hard'))

for arch in TI_ARCHS_SOFT:
    rs = load_soft(ti_csv, 'Exp1_Soft_Main', [arch], temp='5', beta_pref='0.7')
    for r in rs:
        records.append(row_to_record(r, 'Tiny-IN', 'Soft'))

for arch in TI_ARCHS_HARD:
    rs = load_hard(ti_csv, 'Exp2_Hard_BGS', [arch])
    for r in rs:
        records.append(row_to_record(r, 'Tiny-IN', 'Hard'))

# ── Pretty-print arch pairs ───────────────────────────────────────────────────
ARCH_SHORT = {
    'resnet20 -> resnet20': 'R20→R20',
    'resnet20 -> vgg11':    'R20→V11',
    'vgg11 -> vgg11':       'V11→V11',
    'resnet56 -> resnet56': 'R56→R56',
    'resnet56 -> resnet20': 'R56→R20',
    'tinyresnet18 -> resnet18':    'TR18→R18',
    'tinyresnet18 -> mobilenetv2': 'TR18→MV2',
}

# ── Terminal print ─────────────────────────────────────────────────────────────
HEADER = (
    f"{'Dataset':<11} {'Arch':<10} {'Mode':<5} {'T-Acc':>6} "
    f"{'Stu.Acc':>7} {'-log10P':>7} "
    f"{'Base':^5} {'FT1%':^5} {'FT5%':^5} {'FT10%':^5} {'FT10%Acc':>8}"
)
SEP = '-' * len(HEADER)

print('\n' + '=' * len(HEADER))
print(' Table V-C: EvalGuard Main Detection Results (T=5 soft, T=1 hard)')
print('  ✓ = Verified  ✗ = Not Verified  |  -log10P: security strength (≥19.3 verified)')
print('=' * len(HEADER))
print(HEADER)
print(SEP)

prev_ds = ''
for rec in records:
    ds = rec['dataset']
    if ds != prev_ds and prev_ds != '':
        print(SEP)
    prev_ds = ds
    arch = ARCH_SHORT.get(rec['arch'], rec['arch'][:10])
    line = (
        f"{ds:<11} {arch:<10} {rec['mode']:<5} {rec['t_acc']:>6} "
        f"{rec['stu_acc']:>7} {rec['base_p']:>7} "
        f"{rec['base_v']:^5} {rec['ft1_v']:^5} {rec['ft5_v']:^5} "
        f"{rec['ft10_v']:^5} {rec['ft10_acc']:>8}"
    )
    print(line)

print('=' * len(HEADER))
print()

# ── PDF Table ─────────────────────────────────────────────────────────────────
col_headers = [
    'Dataset', 'Arch Pair', 'Mode', 'Teacher\nAcc(%)',
    'Student\nAcc(%)', '−log₁₀(p)\n(base)',
    'Base\nVer.', 'FT 1%\nVer.', 'FT 5%\nVer.',
    'FT 10%\nVer.', 'FT10%\nAcc(%)',
]

cell_data = []
for rec in records:
    arch = ARCH_SHORT.get(rec['arch'], rec['arch'])
    cell_data.append([
        rec['dataset'], arch, rec['mode'], rec['t_acc'],
        rec['stu_acc'], rec['base_p'],
        rec['base_v'], rec['ft1_v'], rec['ft5_v'],
        rec['ft10_v'], rec['ft10_acc'],
    ])

n_rows = len(cell_data)
fig_h = max(5.5, 0.38 * n_rows + 2.5)
fig, ax = plt.subplots(figsize=(16, fig_h))
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

# Header style
for ci in range(len(col_headers)):
    tbl[0, ci].set_facecolor('#2c3e50')
    tbl[0, ci].set_text_props(color='white', fontweight='bold')

# Color verification cells
for ri in range(1, n_rows + 1):
    for ci in range(len(col_headers)):
        tbl[ri, ci].set_height(0.055)
    # Alternate row background
    bg = '#f8f9fa' if ri % 2 == 0 else 'white'
    for ci in range(len(col_headers)):
        tbl[ri, ci].set_facecolor(bg)
    # Color verification columns (6-9)
    for ci in range(6, 10):
        val = cell_data[ri - 1][ci]
        if val == CHECK:
            tbl[ri, ci].set_facecolor('#28a745')
            tbl[ri, ci].set_text_props(color='white', fontweight='bold')
        elif val == CROSS:
            tbl[ri, ci].set_facecolor('#dc3545')
            tbl[ri, ci].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    'Table V-C: EvalGuard Main Detection Results (Soft: T=5, Hard: T=1, NQ=50 000)\n'
    f'✓ = Verified (p < η=2⁻⁶⁴)  ✗ = Not Verified',
    fontsize=10, fontweight='bold', y=0.99)

out_path = os.path.join(_here, 'table2_main_results.pdf')
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
