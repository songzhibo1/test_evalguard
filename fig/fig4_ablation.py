#!/usr/bin/env python3
"""
V-G Ablation Study Figure (CIFAR-10)
2×2 grid: V_Temp sweep, RW sweep, Delta sweep, Beta sweep.
Each panel shows Base_Sig (confidence shift, left axis) and
-log10(P) (security strength, right axis) with verification threshold.
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

THRESHOLD = 64 * np.log10(2)  # ≈ 19.27
P_FLOOR = 300.0
COLOR_SIG  = '#1f77b4'
COLOR_P    = '#d62728'
COLOR_ACC  = '#2ca02c'


def to_log10p(p_str):
    try:
        v = float(str(p_str).strip())
        return P_FLOOR if v <= 0.0 else min(-np.log10(v), P_FLOOR)
    except (ValueError, TypeError):
        return 0.0


def to_float(s, default=np.nan):
    try:
        return float(str(s).replace('%', '').strip())
    except (ValueError, TypeError):
        return default


def load_sweep(df, exp_name, x_col, x_vals=None):
    rows = df[df['Experiment'] == exp_name].copy()
    rows[x_col] = pd.to_numeric(rows[x_col], errors='coerce')
    rows = rows.dropna(subset=[x_col]).sort_values(x_col)
    if x_vals is not None:
        rows = rows[rows[x_col].isin(x_vals)]
    return rows


c10_csv = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
df = pd.read_csv(c10_csv)

sweeps = [
    ('ExpA3_VTempSweep', 'V_Temp',  r'Verification Temp. $V_T$',
     [1, 3, 5, 10, 15, 20, 30]),
    ('ExpA5_RWSweep',    'RW',       r'Watermark Ratio $r_w$',
     [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]),
    ('ExpA6_DeltaSweep', 'Delta',    r'Shift Magnitude $\delta$',
     [1.0, 2.0, 3.0, 5.0, 7.0]),
    ('ExpA7_BetaSweep',  'Beta',     r'Trigger Ratio $\beta$',
     [0.3, 0.5, 0.7, 0.8, 0.9]),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
panel_labels = ['(a)', '(b)', '(c)', '(d)']

for idx, (exp, xcol, xlabel, x_range) in enumerate(sweeps):
    ax = axes[idx // 2, idx % 2]
    rows = load_sweep(df, exp, xcol, x_vals=x_range)

    if rows.empty:
        ax.text(0.5, 0.5, f'No data for {exp}', ha='center', va='center',
                transform=ax.transAxes)
        continue

    x  = rows[xcol].values
    sig = rows['Base_Sig'].apply(lambda v: to_float(v)).values
    acc = rows['Base_Acc'].apply(lambda v: to_float(v)).values
    log10p = rows['Base_P'].apply(to_log10p).values

    ax2 = ax.twinx()

    ln1, = ax.plot(x, sig, 'o-', color=COLOR_SIG, lw=2, ms=6,
                   label='Confidence Shift $\hat{\mu}$')
    ln2, = ax.plot(x, acc, 's--', color=COLOR_ACC, lw=1.5, ms=5,
                   label='Student Acc. (%)', alpha=0.7)
    ln3, = ax2.plot(x, log10p, '^:', color=COLOR_P, lw=2, ms=6,
                    label=r'$-\log_{10}(p)$')
    ax2.axhline(THRESHOLD, color='black', ls='--', lw=1.0, alpha=0.6,
                label=r'$\eta=2^{-64}$')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Confidence Shift / Accuracy (%)', color='black', fontsize=9)
    ax2.set_ylabel(r'$-\log_{10}(p)$', color=COLOR_P, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=COLOR_P)
    ax2.set_ylim(0, min(P_FLOOR + 20, 320))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))

    ax.set_title(f'{panel_labels[idx]} {xlabel} Ablation', fontweight='bold', fontsize=10)
    ax.grid(True, ls=':', alpha=0.5)

    if idx == 0:
        lines = [ln1, ln2, ln3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='lower right', framealpha=0.9)

fig.suptitle('EvalGuard Ablation Study (CIFAR-10, Soft Label, resnet20→resnet20)',
             fontweight='bold', fontsize=12)
fig.tight_layout()

out_path = os.path.join(_here, 'ablation.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
