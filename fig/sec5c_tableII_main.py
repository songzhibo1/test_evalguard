#!/usr/bin/env python3
"""
Section V-C · Table II: Main Verification Results
All datasets × arch pairs × label modes (T=5 soft, T=1 hard).
Shows: teacher acc, student acc, security strength -log10(P),
       verification status at Base / FT1% / FT5% / FT10%.
Prints to terminal and saves to sec5c_tableII_main.pdf.
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

P_FLOOR = 300.0
VER_Y = 'Y'   # use Y/N in PDF (unicode check-marks not in all serif fonts)
VER_N = 'N'

def _log10p(s):
    try:
        v = float(str(s).strip())
        return P_FLOOR if v <= 0 else min(-np.log10(v), P_FLOOR)
    except Exception:
        return 0.0

def _acc(s):
    try:
        return float(str(s).replace('%', '').strip())
    except Exception:
        return float('nan')

def _fmtp(s):
    lp = _log10p(s)
    return '>=300' if lp >= 299 else ('< 0.1' if lp < 0.1 else f'{lp:.0f}')

def _fmtv(s):
    return VER_Y if str(s).strip() == 'True' else VER_N

def _fmta(s):
    v = _acc(s)
    return f'{v:.1f}' if not np.isnan(v) else '—'

ARCH_SHORT = {
    'resnet20 -> resnet20': 'R20->R20',
    'resnet20 -> vgg11':    'R20->V11',
    'vgg11 -> vgg11':       'V11->V11',
    'resnet56 -> resnet56': 'R56->R56',
    'resnet56 -> resnet20': 'R56->R20',
    'tinyresnet18 -> resnet18':    'TR18->R18',
    'tinyresnet18 -> mobilenetv2': 'TR18->MV2',
}

def _load(csv_path, exp, archs, temp=None, beta=None):
    best = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['Experiment'] != exp:
                continue
            if temp and str(row.get('Temp', '')).strip() != str(temp):
                continue
            a = row['Arch_Pair']
            if a not in archs:
                continue
            b = str(row.get('Beta', '')).strip()
            if a not in best or (beta and b == str(beta)):
                best[a] = row
    return [best[a] for a in archs if a in best]

c10  = os.path.join(_root, 'summary_results_c10_mean_rest_v5_full.csv')
c100a = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full1.csv')
c100b = os.path.join(_root, 'summary_results_c100_mean_rest_v5_full2.csv')
ti   = os.path.join(_root, 'summary_results_tinyimg_mean_rest_v6.csv')

SPECS = [
    ('CIFAR-10',  c10,   'ExpA1_MainTable',  ['resnet20 -> resnet20','resnet20 -> vgg11','vgg11 -> vgg11'],       '5',  '0.5', 'Soft'),
    ('CIFAR-10',  c10,   'ExpB1_BGS_Adaptive',['resnet20 -> resnet20','resnet20 -> vgg11','vgg11 -> vgg11'],       None, None,  'Hard'),
    ('CIFAR-100', c100a, 'ExpA1_MainTable',  ['resnet56 -> resnet56','resnet56 -> resnet20','vgg11 -> vgg11'],     '5',  '0.7', 'Soft'),
    ('CIFAR-100', c100b, 'ExpB1_BGS_Adaptive',['resnet56 -> resnet56','resnet56 -> resnet20','vgg11 -> vgg11'],    None, None,  'Hard'),
    ('Tiny-IN',   ti,    'Exp1_Soft_Main',   ['tinyresnet18 -> resnet18','tinyresnet18 -> mobilenetv2'],           '5',  '0.7', 'Soft'),
    ('Tiny-IN',   ti,    'Exp2_Hard_BGS',    ['tinyresnet18 -> resnet18','tinyresnet18 -> mobilenetv2'],           None, None,  'Hard'),
]

records = []
for ds, path, exp, archs, temp, beta, mode in SPECS:
    for row in _load(path, exp, archs, temp=temp, beta=beta):
        records.append({
            'ds':      ds,
            'arch':    ARCH_SHORT.get(row['Arch_Pair'], row['Arch_Pair']),
            'mode':    mode,
            't_acc':   _fmta(row.get('T_Acc', 'nan')),
            's_acc':   _fmta(row['Base_Acc']),
            'logp':    _fmtp(row['Base_P']),
            'b_v':     _fmtv(row['Base_Ver']),
            'f1_v':    _fmtv(row['FT1%_Ver']),
            'f5_v':    _fmtv(row['FT5%_Ver']),
            'f10_v':   _fmtv(row['FT10%_Ver']),
            'f10_acc': _fmta(row['FT10%_Acc']),
        })

# ── Terminal ──────────────────────────────────────────────────────────────────
HDR = (f"{'Dataset':<10} {'Arch':10} {'Mode':5} {'T-Acc':>6} "
       f"{'Stu.Acc':>7} {'-log10P':>7}  {'B':^3} {'F1':^3} {'F5':^3} {'F10':^3}  {'F10-Acc':>7}")
SEP = '-' * len(HDR)
print('\n' + '='*len(HDR))
print(' Section V-C  ·  Table II: EvalGuard Verification Results')
print('  Y=Verified  N=Not-Verified  |  -log10P >= 19.3 means verified (eta=2^-64)')
print('='*len(HDR))
print(HDR); print(SEP)
prev = ''
for r in records:
    if r['ds'] != prev and prev:
        print(SEP)
    prev = r['ds']
    print(f"{r['ds']:<10} {r['arch']:10} {r['mode']:5} {r['t_acc']:>6} "
          f"{r['s_acc']:>7} {r['logp']:>7}  {r['b_v']:^3} {r['f1_v']:^3} "
          f"{r['f5_v']:^3} {r['f10_v']:^3}  {r['f10_acc']:>7}")
print('='*len(HDR) + '\n')

# ── PDF ───────────────────────────────────────────────────────────────────────
col_hdr = ['Dataset','Arch Pair','Mode',
           'Teacher\nAcc(%)','Student\nAcc(%)','-log10(p)\n(Base)',
           'Base','FT\n1%','FT\n5%','FT\n10%','FT10%\nAcc(%)']

cells = [[r['ds'],r['arch'],r['mode'],r['t_acc'],r['s_acc'],r['logp'],
          r['b_v'],r['f1_v'],r['f5_v'],r['f10_v'],r['f10_acc']] for r in records]

fig, ax = plt.subplots(figsize=(16, max(5, 0.38*len(cells)+2.2)))
ax.axis('off')
tbl = ax.table(cellText=cells, colLabels=col_hdr,
               cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)

for ci in range(len(col_hdr)):
    tbl[0,ci].set_facecolor('#2c3e50')
    tbl[0,ci].set_text_props(color='white', fontweight='bold')

VER_COLS = [6, 7, 8, 9]
for ri in range(1, len(cells)+1):
    bg = '#f8f9fa' if ri%2==0 else 'white'
    for ci in range(len(col_hdr)):
        tbl[ri,ci].set_facecolor(bg)
        tbl[ri,ci].set_height(0.055)
    for ci in VER_COLS:
        v = cells[ri-1][ci]
        if v == VER_Y:
            tbl[ri,ci].set_facecolor('#28a745')
            tbl[ri,ci].set_text_props(color='white', fontweight='bold')
        else:
            tbl[ri,ci].set_facecolor('#dc3545')
            tbl[ri,ci].set_text_props(color='white', fontweight='bold')

fig.suptitle(
    'Table II (Sec. V-C)  ·  EvalGuard Watermark Verification Results\n'
    '(Soft: T=5, Hard: BGS adaptive; NQ=50 000; Y=Verified p<eta=2^-64)',
    fontsize=9.5, fontweight='bold', y=1.01)
out = os.path.join(_here, 'sec5c_tableII_main.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
print(f'Saved: {out}')
