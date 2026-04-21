#!/usr/bin/env python3
"""
V-B Fidelity Table
Prints a formatted comparison table to terminal and saves as PDF.
Shows the 5-stage fidelity chain across 3 methods × 3 datasets × 2 models.
Data hardcoded from summary_results_fidelity.txt.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch

os.makedirs('fig', exist_ok=True)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

_here = os.path.dirname(os.path.abspath(__file__))

# ── Hardcoded fidelity data (from summary_results_fidelity.txt) ───────────────
# Columns: Dataset, Model, Original, Obfusc(Δ), Recover(Δ),
#          EG-Soft(Δ), EG-Hard(Δ), DAWN-Soft(Δ), DAWN-Hard(Δ), Adi(Δ)
ROWS = [
    ['CIFAR-10',     'ResNet-20', '92.59', '10.07 (−82.52)', '92.59 (0.00)',
     '92.59 (0.00)', '91.72 (−0.87)', '83.98 (−8.61)', '83.98 (−8.61)', '92.29 (−0.30)'],
    ['CIFAR-10',     'VGG-11',    '92.78', '10.87 (−81.91)', '92.79 (+0.01)',
     '92.78 (0.00)', '90.79 (−1.99)', '84.12 (−8.66)', '84.12 (−8.66)', '92.65 (−0.13)'],
    ['CIFAR-100',    'ResNet-56', '72.61',  '1.00 (−71.61)', '72.61 (0.00)',
     '72.61 (0.00)', '71.85 (−0.76)', '66.58 (−6.03)', '66.58 (−6.03)', '72.28 (−0.33)'],
    ['CIFAR-100',    'VGG-11',    '70.79',  '1.17 (−69.62)', '70.79 (0.00)',
     '70.79 (0.00)', '68.85 (−1.94)', '64.97 (−5.82)', '64.97 (−5.82)', '70.59 (−0.20)'],
    ['Tiny-ImageNet','ResNet-18', '60.12',  '0.52 (−59.60)', '60.12 (0.00)',
     '60.12 (0.00)', '59.99 (−0.13)', '55.24 (−4.88)', '55.24 (−4.88)', '62.80 (+2.68)'],
]

HEADERS = [
    'Dataset', 'Model', 'Original\nAcc (%)',
    'Obfuscated\n(Δ%)',
    'Recovered\n(Δ%)',
    'EvalGuard\nSoft (Δ%)',
    'EvalGuard\nHard (Δ%)',
    'DAWN\nSoft (Δ%)',
    'DAWN\nHard (Δ%)',
    'Adi\n(Δ%)',
]

# ── Terminal print ────────────────────────────────────────────────────────────
COL_W = [13, 10, 10, 16, 16, 16, 16, 16, 16, 16]

def print_table():
    sep = '+' + '+'.join('-' * (w + 2) for w in COL_W) + '+'
    hdr_texts = ['Dataset', 'Model', 'Original', 'Obfusc (Δ)',
                 'Recover (Δ)', 'EG Soft (Δ)', 'EG Hard (Δ)',
                 'DAWN Soft (Δ)', 'DAWN Hard (Δ)', 'Adi (Δ)']
    hdr = '| ' + ' | '.join(f'{h:<{w}}' for h, w in zip(hdr_texts, COL_W)) + ' |'

    print('\n' + '=' * (sum(COL_W) + len(COL_W) * 3 + 1))
    print(' Table V-B: Fidelity Comparison (Accuracy %, Δ relative to Original)')
    print('=' * (sum(COL_W) + len(COL_W) * 3 + 1))
    print(sep)
    print(hdr)
    print(sep)
    for row in ROWS:
        line = '| ' + ' | '.join(f'{v:<{w}}' for v, w in zip(row, COL_W)) + ' |'
        print(line)
    print(sep)
    print()
    print('Notes:')
    print('  EvalGuard Soft  : Zero fidelity loss (API returns modified logits, argmax unchanged)')
    print('  EvalGuard Hard  : < 2% fidelity loss (BGS trigger insertion)')
    print('  DAWN Soft/Hard  : 6–9% fidelity loss (label flipping at r_w=10%)')
    print('  Adi (Backdoor)  : Minimal loss, but FAILS detection (see Table V-C baseline)')
    print()

print_table()

# ── PDF Table ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 4.5))
ax.axis('off')

cell_data = [row for row in ROWS]

table = ax.table(
    cellText=cell_data,
    colLabels=HEADERS,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)

# Adjust column widths
col_widths = [0.08, 0.065, 0.065, 0.10, 0.10, 0.105, 0.105, 0.105, 0.105, 0.08]
for ci, w in enumerate(col_widths):
    for ri in range(len(ROWS) + 1):
        table[ri, ci].set_width(w)

# Color header row
for ci in range(len(HEADERS)):
    cell = table[0, ci]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

# Color coding: EvalGuard columns (5,6) = light green, DAWN (7,8) = light orange, Adi (9) = light gray
method_colors = {5: '#d4edda', 6: '#fff3cd', 7: '#fce8cd', 8: '#fce8cd', 9: '#e2e3e5'}
for ri in range(1, len(ROWS) + 1):
    for ci, color in method_colors.items():
        table[ri, ci].set_facecolor(color)
    # Alternate row base color for readability
    if ri % 2 == 0:
        for ci in range(5):
            table[ri, ci].set_facecolor('#f8f9fa')

# Special: EvalGuard Soft rows should be bright green (0.00% delta)
for ri in range(1, len(ROWS) + 1):
    if '0.00' in ROWS[ri - 1][5]:   # EG Soft has 0 delta
        table[ri, 5].set_facecolor('#28a745')
        table[ri, 5].set_text_props(color='white', fontweight='bold')

fig.suptitle('Table V-B: Teacher Model Fidelity Comparison (Accuracy %, Δ relative to Original)',
             fontsize=11, fontweight='bold', y=0.98)

out_path = os.path.join(_here, 'table1_fidelity.pdf')
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
