#!/usr/bin/env python3
"""
Section V-D · Figure 1: Baseline Watermark Comparison
3-dataset x 2-mode (soft/hard) grid.  Each panel uses dual y-axes:
  Left  (solid lines) : accuracy (%) across stages
  Right (dashed lines): -log10(P) watermark security strength
Three methods per panel: EvalGuard (red), DAWN (blue), Adi (gray).
x-axis stages: Teacher-Orig | Teacher-Prot | Extracted | FT1% | FT5% | FT10%
Security lines start at 'Extracted' (first stage where verification runs).

Key story this figure tells:
  - EvalGuard soft : zero fidelity cost + strong security
  - DAWN           : 6-9% fidelity cost + strong security (not designed for distillation)
  - Adi            : near-zero fidelity cost + FAILS security (backdoor not transferred)
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)

mpl.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# ── Hardcoded teacher fidelity values ─────────────────────────────────────────
T_ORIG = {'CIFAR10': 92.59, 'CIFAR100': 72.61, 'TINYIMAGENET': 60.12}
T_PROT = {
    'CIFAR10': {
        'EvalGuard':     {'soft': 92.59, 'hard': 91.72},
        'DAWN':          {'soft': 83.98, 'hard': 83.98},
        'Adi (Backdoor)':{'soft': 92.29, 'hard': 92.29},
    },
    'CIFAR100': {
        'EvalGuard':     {'soft': 72.61, 'hard': 71.85},
        'DAWN':          {'soft': 66.58, 'hard': 66.58},
        'Adi (Backdoor)':{'soft': 72.28, 'hard': 72.28},
    },
    'TINYIMAGENET': {
        'EvalGuard':     {'soft': 60.12, 'hard': 59.99},
        'DAWN':          {'soft': 55.24, 'hard': 55.24},
        'Adi (Backdoor)':{'soft': 62.80, 'hard': 62.80},
    },
}

METHODS  = ['EvalGuard', 'DAWN', 'Adi (Backdoor)']
M_COLOR  = {'EvalGuard': '#D62728', 'DAWN': '#1F77B4', 'Adi (Backdoor)': '#7F7F7F'}
M_MARKER = {'EvalGuard': 'o',       'DAWN': 's',       'Adi (Backdoor)': '^'}

P_FLOOR  = 300.0
THRESHOLD = 64 * np.log10(2)   # ≈ 19.27

STAGES_X  = ['Teacher\n(Orig.)', 'Teacher\n(Prot.)', 'Extracted',
             'FT 1%', 'FT 5%', 'FT 10%']
X_ACC = [0, 1, 2, 3, 4, 5]    # accuracy: all 6 stages
X_SEC = [2, 3, 4, 5]           # security: only from "Extracted" onward

def _acc(s):
    try:
        return float(str(s).replace('%','').split('(')[0].strip())
    except Exception:
        return float('nan')

def _log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0 else min(-np.log10(v), P_FLOOR)
    except Exception:
        return 0.0

# ── Load baseline CSV ──────────────────────────────────────────────────────────
bl_csv = os.path.join(_root, 'summary_baseline_comparison_TableX.csv')
data = {}   # data[(DS, Method, mode)] = row
with open(bl_csv) as f:
    for row in csv.DictReader(f):
        key = (row['Dataset'], row['Method'], row['L_Mode'])
        data[key] = row   # keep last (deduped)

DATASETS = ['CIFAR10', 'CIFAR100', 'TINYIMAGENET']
DS_LABEL = {'CIFAR10': 'CIFAR-10', 'CIFAR100': 'CIFAR-100',
            'TINYIMAGENET': 'Tiny-ImageNet'}
MODES = ['soft', 'hard']
MODE_LABEL = {'soft': 'Soft Label API', 'hard': 'Hard Label BGS'}

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 14))

for row_i, ds in enumerate(DATASETS):
    for col_j, mode in enumerate(MODES):
        ax  = axes[row_i, col_j]
        ax2 = ax.twinx()

        for method in METHODS:
            row = data.get((ds, method, mode))
            if row is None:
                continue
            color  = M_COLOR[method]
            marker = M_MARKER[method]

            # Accuracy values across 6 stages
            acc = [
                T_ORIG[ds],
                T_PROT[ds][method][mode],
                _acc(row['Base_Acc']),
                _acc(row['FT1%_Acc']),
                _acc(row['FT5%_Acc']),
                _acc(row['FT10%_Acc']),
            ]
            # Security values for last 4 stages
            sec = [_log10p(row[c]) for c in
                   ['Base_P', 'FT1%_P', 'FT5%_P', 'FT10%_P']]

            ax.plot(X_ACC, acc, '-', color=color, marker=marker,
                    lw=2, ms=6, label=method)
            ax2.plot(X_SEC, sec, '--', color=color, marker=marker,
                     lw=1.8, ms=5, alpha=0.85)

        # Threshold on right axis
        ax2.axhline(THRESHOLD, color='black', ls=':', lw=1.2, alpha=0.65)

        # Axes formatting
        ax.set_xticks(X_ACC)
        ax.set_xticklabels(STAGES_X, fontsize=8)
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, min(P_FLOOR + 15, 315))
        ax2.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v,_: '>=300' if v>=299 else f'{int(v)}'))

        ax.set_ylabel('Accuracy (%)  [solid]',    fontsize=9,
                      color='#333333')
        if col_j == 1:
            ax2.set_ylabel(r'$-\log_{10}(p)$  [dashed]', fontsize=9,
                           color='#555555')
        else:
            ax2.set_yticklabels([])

        ax.set_title(f'{DS_LABEL[ds]}  ·  {MODE_LABEL[mode]}',
                     fontweight='bold', fontsize=10)
        ax.grid(True, ls=':', alpha=0.4)

        # Legend in first panel only
        if row_i == 0 and col_j == 0:
            ax.legend(fontsize=8.5, loc='lower right', framealpha=0.92,
                      title='Method (solid=Acc, dashed=-log10P)')

# Annotate threshold
fig.text(0.99, 0.015, r'Dotted line = threshold $\eta=2^{-64}$  ($-\log_{10}p \approx 19.3$)',
         ha='right', fontsize=8, style='italic', color='#555')

fig.suptitle(
    'Figure 1 (Sec. V-D)  ·  Baseline Watermark Comparison\n'
    'Accuracy (left axis, solid) and Security -log10(p) (right axis, dashed) across stages',
    fontweight='bold', fontsize=11, y=1.005)
fig.tight_layout()

out = os.path.join(_here, 'sec5d_fig1_baseline.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
