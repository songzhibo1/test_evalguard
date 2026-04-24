#!/usr/bin/env python3
"""
Section V-B · Table I: Teacher-Model Fidelity
5-stage fidelity chain × 3 methods × 5 model configurations.
Prints to terminal and saves to sec5b_tableI_fidelity.pdf.

Key claim: EvalGuard Soft = 0 fidelity loss; Hard < 2%;
           DAWN = 6–9% loss; Adi = minimal loss but fails detection.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

_here = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# ── Data (from summary_results_fidelity.txt) ─────────────────────────────────
ROWS = [
    ['CIFAR-10',     'ResNet-20', '92.59',
     '10.07 (-82.52)', '92.59 (0.00)',
     '92.59 (0.00)',  '91.72 (-0.87)',
     '83.98 (-8.61)', '83.98 (-8.61)', '92.29 (-0.30)'],
    ['CIFAR-10',     'VGG-11',    '92.78',
     '10.87 (-81.91)', '92.79 (+0.01)',
     '92.78 (0.00)',  '90.79 (-1.99)',
     '84.12 (-8.66)', '84.12 (-8.66)', '92.65 (-0.13)'],
    ['CIFAR-100',    'ResNet-56', '72.61',
     ' 1.00 (-71.61)', '72.61 (0.00)',
     '72.61 (0.00)',  '71.85 (-0.76)',
     '66.58 (-6.03)', '66.58 (-6.03)', '72.28 (-0.33)'],
    ['CIFAR-100',    'VGG-11',    '70.79',
     ' 1.17 (-69.62)', '70.79 (0.00)',
     '70.79 (0.00)',  '68.85 (-1.94)',
     '64.97 (-5.82)', '64.97 (-5.82)', '70.59 (-0.20)'],
    ['Tiny-IN',      'ResNet-18', '60.12',
     ' 0.52 (-59.60)', '60.12 (0.00)',
     '60.12 (0.00)',  '59.99 (-0.13)',
     '55.24 (-4.88)', '55.24 (-4.88)', '62.80 (+2.68)'],
]

COL_NAMES = [
    'Dataset', 'Model', 'Orig\n(%)',
    'Obfusc\n(Δ%)',  'Recov\n(Δ%)',
    'EG Soft\n(Δ%)', 'EG Hard\n(Δ%)',
    'DAWN Soft\n(Δ%)', 'DAWN Hard\n(Δ%)', 'Adi\n(Δ%)']

COL_W = [12, 9, 7, 14, 14, 14, 14, 15, 15, 14]

# ── Terminal ──────────────────────────────────────────────────────────────────
def _terminal():
    hdr = [n.replace('\n', ' ') for n in COL_NAMES]
    sep = '+' + '+'.join('-' * (w + 2) for w in COL_W) + '+'
    row_fmt = '| ' + ' | '.join('{:<' + str(w) + '}' for w in COL_W) + ' |'
    width = sum(COL_W) + len(COL_W) * 3 + 1
    print('\n' + '=' * width)
    print(' Section V-B  ·  Table I: Teacher-Model Fidelity (Accuracy %, Δ vs. Original)')
    print('=' * width)
    print(sep)
    print(row_fmt.format(*hdr))
    print(sep)
    for r in ROWS:
        print(row_fmt.format(*r))
    print(sep)
    print()
    print('EvalGuard Soft : zero fidelity loss (logit-space shift, argmax unchanged)')
    print('EvalGuard Hard : <2% loss (BGS adaptive trigger insertion)')
    print('DAWN           : 6-9% loss (label flipping r_w=10%)')
    print('Adi (Backdoor) : minimal loss BUT verification fails after distillation')
    print()

_terminal()

# ── PDF ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 3.8))
ax.axis('off')

tbl = ax.table(cellText=ROWS, colLabels=COL_NAMES,
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# Header: dark background
for ci in range(len(COL_NAMES)):
    c = tbl[0, ci]
    c.set_facecolor('#2c3e50')
    c.set_text_props(color='white', fontweight='bold')

# Color columns by method group
METHOD_BG = {5: '#d4f7dc', 6: '#d4f7dc',   # EvalGuard — green
             7: '#fff3cd', 8: '#fff3cd',   # DAWN — yellow
             9: '#e9ecef'}                 # Adi — gray
for ri in range(1, len(ROWS) + 1):
    alt = '#f8f9fa' if ri % 2 == 0 else 'white'
    for ci in range(len(COL_NAMES)):
        tbl[ri, ci].set_facecolor(METHOD_BG.get(ci, alt))
    # Zero-delta cells in EvalGuard Soft → bright green
    if '0.00' in ROWS[ri - 1][5]:
        tbl[ri, 5].set_facecolor('#28a745')
        tbl[ri, 5].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    'Table I (Sec. V-B)  ·  Teacher-Model Fidelity: Accuracy (%) and Δ relative to Original',
    fontsize=10, fontweight='bold', y=1.02)
out = os.path.join(_here, 'sec5b_tableI_fidelity.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
print(f'Saved: {out}')
