#!/usr/bin/env python3
"""
Section V-E · Figure 2: Robustness to Post-Processing
2x2 layout showing BOTH utility and security simultaneously:
  (a) Pruning — accuracy drop        (b) Pruning — security strength
  (c) INT8 Quant — accuracy change   (d) INT8 Quant — security strength

Key finding:
  25-50% pruning: verification still holds (watermark survives).
  75% pruning   : model accuracy collapses to ~10% (useless for attacker)
                  BEFORE verification fails — attacker loses both ways.
  INT8 quant    : near-zero impact on both accuracy and verification.
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
THRESHOLD = 64 * np.log10(2)
LEVELS    = [0, 25, 50, 75]

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

def _row(path, exp, arch):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['Experiment'] == exp and r['Arch_Pair'] == arch:
                return r
    return None

def _base_p(path, soft_exp, hard_exp, arch, soft_temp='5'):
    """Look up baseline p-value from main experiment."""
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['Experiment'] == soft_exp and r['Arch_Pair'] == arch \
                    and str(r.get('Temp','')).strip() == soft_temp:
                return r['Base_P'], r['Base_Acc']
    return 'N/A', 'N/A'

c10 = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
ti  = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

# ── Pruning data ───────────────────────────────────────────────────────────────
def _prune(prune_row, base_p, base_acc):
    if prune_row is None:
        return [_acc(base_acc)] + [float('nan')]*3, [_log10p(base_p)] + [0.0]*3
    accs = [_acc(base_acc)] + [_acc(prune_row.get(f'PRUNE_{p}%_Acc','N/A'))
                                 for p in [25, 50, 75]]
    ps   = [_log10p(base_p)] + [_log10p(prune_row.get(f'PRUNE_{p}%_P','N/A'))
                                  for p in [25, 50, 75]]
    return accs, ps

c10_base_soft_p, c10_base_soft_acc = _base_p(c10, 'ExpA1_MainTable',
                                               'ExpB1_BGS_Adaptive',
                                               'resnet20 -> resnet20')
with open(c10) as f:
    for r in csv.DictReader(f):
        if r['Experiment'] == 'ExpB1_BGS_Adaptive' and \
                r['Arch_Pair'] == 'resnet20 -> resnet20':
            c10_base_hard_p, c10_base_hard_acc = r['Base_P'], r['Base_Acc']
            break

with open(ti) as f:
    for r in csv.DictReader(f):
        if r['Experiment'] == 'Exp1_Soft_Main' and \
                r['Arch_Pair'] == 'tinyresnet18 -> resnet18' and \
                str(r.get('Temp','')).strip() == '5':
            ti_base_soft_p, ti_base_soft_acc = r['Base_P'], r['Base_Acc']
            break
        else:
            ti_base_soft_p = ti_base_soft_acc = 'N/A'
with open(ti) as f:
    for r in csv.DictReader(f):
        if r['Experiment'] == 'Exp2_Hard_BGS' and \
                r['Arch_Pair'] == 'tinyresnet18 -> resnet18':
            ti_base_hard_p, ti_base_hard_acc = r['Base_P'], r['Base_Acc']
            break
    else:
        ti_base_hard_p = ti_base_hard_acc = 'N/A'

c10_soft_a, c10_soft_p = _prune(_row(c10,'ExpC4_PruneSoft','resnet20 -> resnet20'),
                                  c10_base_soft_p, c10_base_soft_acc)
c10_hard_a, c10_hard_p = _prune(_row(c10,'ExpC6_PruneHard','resnet20 -> resnet20'),
                                  c10_base_hard_p, c10_base_hard_acc)
ti_soft_a,  ti_soft_p  = _prune(_row(ti,'Exp3_Soft_Prune','tinyresnet18 -> resnet18'),
                                  ti_base_soft_p, ti_base_soft_acc)
ti_hard_a,  ti_hard_p  = _prune(_row(ti,'Exp8_Hard_Prune','tinyresnet18 -> resnet18'),
                                  ti_base_hard_p, ti_base_hard_acc)

PRUNE_CFGS = [
    ('C10 Soft (R20)',  c10_soft_a, c10_soft_p, '#1f77b4', 'o', '-'),
    ('C10 Hard (R20)',  c10_hard_a, c10_hard_p, '#d62728', '^', '--'),
    ('TI Soft (TR18)',  ti_soft_a,  ti_soft_p,  '#2ca02c', 's', '-'),
    ('TI Hard (TR18)',  ti_hard_a,  ti_hard_p,  '#ff7f0e', 'D', '--'),
]

# ── INT8 quant data ────────────────────────────────────────────────────────────
def _quant(csv_path, exp, arch, base_p, base_acc):
    r = _row(csv_path, exp, arch)
    qa = _acc(r['INT8_QUANT_Acc']) if r else float('nan')
    qp = _log10p(r['INT8_QUANT_P']) if r else 0.0
    return _acc(base_acc), qa, _log10p(base_p), qp

q_data = [
    ('C10 Soft\nR20→R20', *_quant(c10,'ExpC5_QuantSoft','resnet20 -> resnet20',
                                    c10_base_soft_p, c10_base_soft_acc)),
    ('TI Soft\nTR18→R18', *_quant(ti,'Exp3_Soft_Quantize','tinyresnet18 -> resnet18',
                                    ti_base_soft_p, ti_base_soft_acc)),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax_pa, ax_ps), (ax_qa, ax_qs) = axes

# (a) Pruning accuracy
for name, accs, _, c, m, ls in PRUNE_CFGS:
    ax_pa.plot(LEVELS, accs, ls, color=c, marker=m, lw=2, ms=7, label=name)
ax_pa.axhline(10, color='gray', ls=':', lw=1.2, alpha=0.6, label='~Random chance')
ax_pa.set_ylabel('Student Accuracy (%)', fontsize=10)
ax_pa.set_title('(a) Accuracy Under Pruning', fontweight='bold')
ax_pa.set_xticks(LEVELS); ax_pa.set_xticklabels(['0%','25%','50%','75%'])
ax_pa.set_ylim(0, 95); ax_pa.legend(fontsize=8.5); ax_pa.grid(True,ls=':',alpha=0.5)
ax_pa.set_xlabel('Pruning Ratio', fontsize=10)

# (b) Pruning security
for name, _, ps, c, m, ls in PRUNE_CFGS:
    ax_ps.plot(LEVELS, ps, ls, color=c, marker=m, lw=2, ms=7, label=name)
ax_ps.axhline(THRESHOLD, color='black', ls='--', lw=1.3, alpha=0.7,
              label=r'$\eta=2^{-64}$')
ax_ps.fill_between(LEVELS, 0, THRESHOLD, alpha=0.05, color='red')
ax_ps.set_ylabel(r'$-\log_{10}(p)$', fontsize=10)
ax_ps.set_title('(b) Watermark Security Under Pruning', fontweight='bold')
ax_ps.set_xticks(LEVELS); ax_ps.set_xticklabels(['0%','25%','50%','75%'])
ax_ps.set_ylim(0, min(P_FLOOR+15, 315))
ax_ps.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v,_: '>=300' if v>=299 else f'{int(v)}'))
ax_ps.legend(fontsize=8.5); ax_ps.grid(True,ls=':',alpha=0.5)
ax_ps.set_xlabel('Pruning Ratio', fontsize=10)
ax_ps.annotate('Model useless\n(acc ~10%)', xy=(75, 2), ha='center',
               fontsize=7.5, color='gray', style='italic')

# (c) Quant accuracy
names_q = [d[0] for d in q_data]
ba = [d[1] for d in q_data]; qa = [d[2] for d in q_data]
x = np.arange(len(names_q)); bw = 0.35
ax_qa.bar(x-bw/2, ba, bw, label='Before INT8', color='#1f77b4', alpha=0.85)
ax_qa.bar(x+bw/2, qa, bw, label='After INT8',  color='#aec7e8', alpha=0.9,
          edgecolor='#1f77b4', linewidth=1.2)
for i,(b,a) in enumerate(zip(ba,qa)):
    if not (np.isnan(b) or np.isnan(a)):
        ax_qa.annotate(f'{a-b:+.1f}%', xy=(i+bw/2, max(b,a)+0.5),
                       ha='center', fontsize=8.5, color='#1f77b4')
ax_qa.set_xticks(x); ax_qa.set_xticklabels(names_q, fontsize=9)
ax_qa.set_ylabel('Student Accuracy (%)', fontsize=10)
ax_qa.set_title('(c) Accuracy: INT8 Quantization', fontweight='bold')
ax_qa.set_ylim(0, 100); ax_qa.legend(fontsize=9)
ax_qa.grid(True, axis='y', ls=':', alpha=0.5)

# (d) Quant security
bp = [d[3] for d in q_data]; qp = [d[4] for d in q_data]
ax_qs.bar(x-bw/2, bp, bw, label='Before INT8', color='#d62728', alpha=0.85)
ax_qs.bar(x+bw/2, qp, bw, label='After INT8',  color='#f7b6b6', alpha=0.9,
          edgecolor='#d62728', linewidth=1.2)
ax_qs.axhline(THRESHOLD, color='black', ls='--', lw=1.3, alpha=0.7,
              label=r'$\eta=2^{-64}$')
ax_qs.set_xticks(x); ax_qs.set_xticklabels(names_q, fontsize=9)
ax_qs.set_ylabel(r'$-\log_{10}(p)$', fontsize=10)
ax_qs.set_title('(d) Security: INT8 Quantization', fontweight='bold')
ax_qs.set_ylim(0, min(P_FLOOR+15, 315))
ax_qs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v,_: '>=300' if v>=299 else f'{int(v)}'))
ax_qs.legend(fontsize=9); ax_qs.grid(True, axis='y', ls=':', alpha=0.5)

fig.suptitle('Figure 2 (Sec. V-E)  ·  EvalGuard Robustness to Pruning and INT8 Quantization',
             fontweight='bold', fontsize=12)
fig.tight_layout()
out = os.path.join(_here, 'sec5e_fig2_robustness.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
