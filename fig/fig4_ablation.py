#!/usr/bin/env python3
"""
V-G Ablation / Parameter Sensitivity Figure (CIFAR-10, Soft Label)
2×3 grid: V_Temp, RW, Delta (row 1) | Beta, NQ, BGS-Q-sweep (row 2)
Each panel: left y-axis = Student Accuracy (%), right y-axis = -log10(P).
Default parameter value is marked with a star (★) / vertical dashed line.
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

THRESHOLD = 64 * np.log10(2)   # -log10(2^-64) ≈ 19.27
P_FLOOR   = 300.0
C_ACC = '#2ca02c'   # green  – accuracy
C_P   = '#d62728'   # red    – security strength
C_THR = '#333333'   # black  – threshold


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


# ── Load sweeps from CSV ───────────────────────────────────────────────────────
def load_sweep(csv_path, exp_name, x_col, x_transform=None):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] == exp_name:
                try:
                    xv = float(row[x_col])
                    if x_transform:
                        xv = x_transform(xv)
                    rows.append({
                        'x':   xv,
                        'acc': to_acc(row['Base_Acc']),
                        'p':   to_log10p(row['Base_P']),
                        'ver': row['Base_Ver'].strip() == 'True',
                    })
                except (ValueError, KeyError):
                    continue
    rows.sort(key=lambda r: r['x'])
    return rows


def load_bgs_q_sweep(csv_path):
    """ExpB2_BGS_QSweep: quantile value is embedded in Tag."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] == 'ExpB2_BGS_QSweep':
                tag = row['Tag']
                # e.g. c10v5_bgs_qsweep_0.10
                try:
                    q = float(tag.split('_')[-1])
                except ValueError:
                    continue
                rows.append({
                    'x':   q,
                    'acc': to_acc(row['Base_Acc']),
                    'sig': to_acc(row['Base_Sig']),   # match rate %
                    'p':   to_log10p(row['Base_P']),
                    'ver': row['Base_Ver'].strip() == 'True',
                })
    rows.sort(key=lambda r: r['x'])
    return rows


c10_csv = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')

vtemp_rows  = load_sweep(c10_csv, 'ExpA3_VTempSweep', 'V_Temp')
rw_rows     = load_sweep(c10_csv, 'ExpA5_RWSweep',    'RW')
delta_rows  = load_sweep(c10_csv, 'ExpA6_DeltaSweep', 'Delta')
beta_rows   = load_sweep(c10_csv, 'ExpA7_BetaSweep',  'Beta')
nq_rows     = load_sweep(c10_csv, 'ExpA8_NQSweep',    'NQ',
                          x_transform=lambda v: v / 1000)  # → k queries
bgsq_rows   = load_bgs_q_sweep(c10_csv)

# ── Default values (marked with ★) ───────────────────────────────────────────
DEFAULTS = {
    'vtemp': 10.0,
    'rw':    0.10,
    'delta': 5.0,
    'beta':  0.5,
    'nq':    50.0,   # 50k
    'bgsq':  0.10,
}

# ── Helper: draw one ablation panel ──────────────────────────────────────────
def draw_panel(ax, rows, xlabel, default_x, x_key='x',
               is_hard=False, panel_label=''):
    xs   = [r['x'] for r in rows]
    accs = [r['acc'] for r in rows]
    ps   = [r['p']   for r in rows]

    ax2 = ax.twinx()

    # Accuracy (left axis)
    ln_acc, = ax.plot(xs, accs, 'o-', color=C_ACC, lw=2.0, ms=6, zorder=3,
                      label='Student Acc. (%)')
    ax.set_ylabel('Student Accuracy (%)', color=C_ACC, fontsize=8.5)
    ax.tick_params(axis='y', labelcolor=C_ACC, labelsize=8)
    ax.set_ylim(60, 95)

    # For hard-label panel, also show match rate on left axis
    if is_hard:
        sigs = [r.get('sig', np.nan) for r in rows]
        ln_sig, = ax.plot(xs, sigs, 's--', color='#ff7f0e', lw=1.8, ms=5,
                          label='Match Rate (%)')

    # Security strength (right axis)
    ln_p, = ax2.plot(xs, ps, '^:', color=C_P, lw=2.0, ms=6, zorder=3,
                     label=r'$-\log_{10}(p)$')
    ax2.axhline(THRESHOLD, color=C_THR, ls='--', lw=1.1, alpha=0.6)
    ax2.set_ylabel(r'$-\log_{10}(p)$', color=C_P, fontsize=8.5)
    ax2.tick_params(axis='y', labelcolor=C_P, labelsize=8)
    ax2.set_ylim(0, min(P_FLOOR + 20, 330))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: '≥300' if v >= 299 else f'{int(v)}'))

    # Mark default value
    ax.axvline(default_x, color='navy', ls=':', lw=1.5, alpha=0.7, zorder=4)
    ystar = ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.annotate('★', xy=(default_x, ystar), fontsize=11, color='navy',
                ha='center', va='bottom', zorder=5)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_title(panel_label, fontweight='bold', fontsize=10, loc='left')

    return [ln_acc, ln_p]


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

panels = [
    (axes[0, 0], vtemp_rows,  r'Verification Temp. $V_T$', DEFAULTS['vtemp'],  False, '(a)'),
    (axes[0, 1], rw_rows,     r'Watermark Ratio $r_w$',     DEFAULTS['rw'],     False, '(b)'),
    (axes[0, 2], delta_rows,  r'Shift Magnitude $\delta$',  DEFAULTS['delta'],  False, '(c)'),
    (axes[1, 0], beta_rows,   r'Trigger Ratio $\beta$',     DEFAULTS['beta'],   False, '(d)'),
    (axes[1, 1], nq_rows,     r'Query Budget $N_Q$ (×10³)', DEFAULTS['nq'],    False, '(e)'),
    (axes[1, 2], bgsq_rows,   r'BGS Quantile $q$ (Hard)',  DEFAULTS['bgsq'],   True,  '(f)'),
]

legend_lines = None
for ax, rows, xlabel, default_x, is_hard, plabel in panels:
    lines = draw_panel(ax, rows, xlabel, default_x,
                       is_hard=is_hard, panel_label=plabel)
    if legend_lines is None:
        legend_lines = lines

# Global legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=C_ACC, marker='o', lw=2, ms=6, label='Student Accuracy (%)'),
    Line2D([0], [0], color=C_P,   marker='^', lw=2, ms=6, linestyle=':', label=r'$-\log_{10}(p)$ (security)'),
    Line2D([0], [0], color='black', lw=1.1, ls='--', alpha=0.6, label=r'Threshold $\eta=2^{-64}$'),
    Line2D([0], [0], color='navy', lw=0, marker='$★$', ms=11, label='Default value'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    'EvalGuard Parameter Sensitivity (CIFAR-10)\n'
    'Left axis: Student accuracy (utility) · Right axis: Security strength · ★: default',
    fontweight='bold', fontsize=11, y=1.01)
fig.tight_layout()

out_path = os.path.join(_here, 'ablation.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
