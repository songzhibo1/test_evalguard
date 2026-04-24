#!/usr/bin/env python3
"""
Section V-G · Figure 3: Parameter Sensitivity Analysis (CIFAR-10, ResNet-20→ResNet-20)

2×3 grid of dual-axis line plots, one per swept parameter:
  (a) Verification Temperature T_v    — ExpA3_VTempSweep
  (b) Watermark Rate r_w              — ExpA5_RWSweep
  (c) Logit Shift Δ                   — ExpA6_DeltaSweep
  (d) Safety Factor β                 — ExpA7_BetaSweep
  (e) Query Budget N_Q                — ExpA8_NQSweep
  (f) BGS Quantile q                  — ExpB2_BGS_QSweep

Left axis  (blue solid)  : Student Accuracy (%)
Right axis (red dashed)  : −log₁₀(p)  [security strength]
★ marks the default configuration value used in Table II.
Dashed horizontal at threshold η = 2⁻⁶⁴ (−log₁₀(p) ≈ 19.3).
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
mpl.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

P_FLOOR   = 300.0
THRESHOLD = 64 * np.log10(2)     # ≈ 19.27
ARCH      = 'resnet20 -> resnet20'

def _log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0 else min(-np.log10(v), P_FLOOR)
    except Exception:
        return 0.0

def _acc(s):
    try:
        return float(str(s).replace('%','').strip())
    except Exception:
        return float('nan')

def _load_sweep(csv_path, exp, param_col):
    """Return sorted list of (param_val, acc, logp) for the given experiment + arch."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['Experiment'] == exp and r['Arch_Pair'] == ARCH:
                try:
                    pv = float(str(r.get(param_col, '')).strip())
                except (ValueError, TypeError):
                    continue
                rows.append((pv, _acc(r.get('Base_Acc', 'nan')),
                             _log10p(r.get('Base_P', 'N/A'))))
    rows.sort(key=lambda x: x[0])
    return rows


c10 = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')

# Load BGS Q-sweep: tag contains the quantile value
def _load_bgs_qsweep(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['Experiment'] == 'ExpB2_BGS_QSweep' and r['Arch_Pair'] == ARCH:
                tag = str(r.get('Tag', ''))
                try:
                    q = float(tag.split('_')[-1])
                except (ValueError, IndexError):
                    continue
                rows.append((q, _acc(r.get('Base_Acc', 'nan')),
                             _log10p(r.get('Base_P', 'N/A'))))
    rows.sort(key=lambda x: x[0])
    return rows


sweeps = [
    # (subplot title, x-label, default-val, data)
    ('(a) Verification Temperature $T_v$',
     'Verification Temp $T_v$', 10.0,
     _load_sweep(c10, 'ExpA3_VTempSweep', 'V_Temp')),

    ('(b) Watermark Rate $r_w$',
     'Watermark Rate $r_w$', 0.1,
     _load_sweep(c10, 'ExpA5_RWSweep', 'RW')),

    ('(c) Logit Shift $\\Delta$',
     'Logit Shift $\\Delta$', 5.0,
     _load_sweep(c10, 'ExpA6_DeltaSweep', 'Delta')),

    ('(d) Safety Factor $\\beta$',
     'Safety Factor $\\beta$', 0.5,
     _load_sweep(c10, 'ExpA7_BetaSweep', 'Beta')),

    ('(e) Query Budget $N_Q$',
     'Query Budget $N_Q$', 50000,
     _load_sweep(c10, 'ExpA8_NQSweep', 'NQ')),

    ('(f) BGS Boundary Quantile $q$',
     'BGS Quantile $q$', 0.1,
     _load_bgs_qsweep(c10)),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, (title, xlabel, default_val, data) in zip(axes.flat, sweeps):
    if not data:
        ax.set_visible(False)
        continue

    xs   = [d[0] for d in data]
    accs = [d[1] for d in data]
    ps   = [d[2] for d in data]

    ax2 = ax.twinx()

    # Accuracy line (blue, solid, left axis)
    acc_line, = ax.plot(xs, accs, '-o', color='#1f77b4', lw=2, ms=6, label='Accuracy')

    # Security line (red, dashed, right axis)
    sec_line, = ax2.plot(xs, ps, '--^', color='#d62728', lw=2, ms=6, alpha=0.9, label='-log10(p)')

    # Threshold
    ax2.axhline(THRESHOLD, color='black', ls=':', lw=1.2, alpha=0.65)

    # Mark default value
    if default_val in xs:
        di = xs.index(default_val)
        ax.plot(xs[di], accs[di], '*', color='#1f77b4', ms=14, zorder=5)
        ax2.plot(xs[di], ps[di],  '*', color='#d62728', ms=14, zorder=5)
    else:
        # Nearest value
        nearest = min(range(len(xs)), key=lambda i: abs(xs[i]-default_val))
        ax.plot(xs[nearest], accs[nearest], '*', color='#1f77b4', ms=14, zorder=5)
        ax2.plot(xs[nearest], ps[nearest],  '*', color='#d62728', ms=14, zorder=5)

    # Axes formatting
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Student Accuracy (%)', fontsize=8.5, color='#1f77b4')
    ax2.set_ylabel(r'$-\log_{10}(p)$', fontsize=8.5, color='#d62728')
    ax.tick_params(axis='y', colors='#1f77b4')
    ax2.tick_params(axis='y', colors='#d62728')

    ax.set_title(title, fontweight='bold', fontsize=9.5)
    ax.grid(True, ls=':', alpha=0.4)

    # y-axis limits
    acc_vals = [a for a in accs if not np.isnan(a)]
    if acc_vals:
        lo = max(0, min(acc_vals) - 5)
        hi = min(100, max(acc_vals) + 5)
        ax.set_ylim(lo, hi)
    ax2.set_ylim(0, min(P_FLOOR+15, 315))
    ax2.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v,_: '>=300' if v>=299 else f'{int(v)}'))

    # Combined legend (top-right of first subplot row only)
    if title.startswith('(a)'):
        lines = [acc_line, sec_line,
                 mpl.lines.Line2D([], [], marker='*', color='gray', ms=12,
                                  ls='None', label='Default ★')]
        ax.legend(handles=lines, fontsize=7.5, loc='lower right', framealpha=0.9)

# Annotation note
fig.text(0.5, 0.01,
         r'★ = default value · dotted line = threshold $\eta=2^{-64}$  '
         r'($-\log_{10}p \approx 19.3$) · CIFAR-10, R20→R20',
         ha='center', fontsize=8, style='italic', color='#555')

fig.suptitle(
    'Figure 3 (Sec. V-G)  ·  EvalGuard Parameter Sensitivity  '
    '(CIFAR-10, ResNet-20→ResNet-20)\n'
    'Blue solid = Student Accuracy (left axis) · Red dashed = −log₁₀(p) (right axis)',
    fontweight='bold', fontsize=11)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

out = os.path.join(_here, 'sec5g_fig3_ablation.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
