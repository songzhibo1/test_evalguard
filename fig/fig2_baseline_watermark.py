import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==========================================
# 1. 样式与环境配置 (TIFS 标准)
# ==========================================
# 自动创建输出文件夹
os.makedirs('fig', exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('seaborn-paper')

mpl.rcParams['font.family'] = 'serif'
# 增加备选字体，解决服务器缺少 Times New Roman 的问题
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ==========================================
# 2. 硬编码保真度数据 (源自教师模型保真度表)
# ==========================================
TEACHER_ORIGINAL = {'CIFAR10': 92.59, 'CIFAR100': 72.61, 'TINYIMAGENET': 60.12}
TEACHER_PROTECTED = {
    'CIFAR10': {
        'EvalGuard': {'soft': 92.59, 'hard': 91.72},
        'DAWN': {'soft': 83.98, 'hard': 83.98},
        'Adi (Backdoor)': {'soft': 92.29, 'hard': 92.29}
    },
    'CIFAR100': {
        'EvalGuard': {'soft': 72.61, 'hard': 71.85},
        'DAWN': {'soft': 66.58, 'hard': 66.58},
        'Adi (Backdoor)': {'soft': 72.28, 'hard': 72.28}
    },
    'TINYIMAGENET': {
        'EvalGuard': {'soft': 60.12, 'hard': 59.99},
        'DAWN': {'soft': 55.24, 'hard': 55.24},
        'Adi (Backdoor)': {'soft': 62.80, 'hard': 62.80}
    }
}

METHOD_COLORS = {'EvalGuard': '#D62728', 'DAWN': '#1F77B4', 'Adi (Backdoor)': '#7F7F7F'}
METHOD_MARKERS = {'EvalGuard': 'o', 'DAWN': 's', 'Adi (Backdoor)': '^'}

def parse_pct(x):
    if pd.isna(x) or x == "N/A": return np.nan
    return float(str(x).split('%')[0].split('(')[0])

# ==========================================
# 3. 数据加载与核心清洗
# ==========================================
csv_path = 'results/baseline/summary_baseline_comparison_TableX.csv'
if not os.path.exists(csv_path):
    # 如果在 fig 文件夹内运行，尝试上一级路径
    csv_path = '../results/baseline/summary_baseline_comparison_TableX.csv'

df = pd.read_csv(csv_path)

# 【核心修复】剔除重复行，保留最后一次（即修复参数后）的实验运行结果
df = df.drop_duplicates(subset=['Dataset', 'Method', 'L_Mode'], keep='last')

datasets = ['CIFAR10', 'CIFAR100', 'TINYIMAGENET']
modes = ['soft', 'hard']
stages_acc = ['Original', 'Protected', 'Extracted', 'FT 1%', 'FT 5%', 'FT 10%']
stages_p = ['Extracted', 'FT 1%', 'FT 5%', 'FT 10%']

# ==========================================
# 4. 绘制图 1: Accuracy Progression (效用全生命周期)
# ==========================================
fig_acc, axes_acc = plt.subplots(3, 2, figsize=(11, 13))
for i, ds in enumerate(datasets):
    for j, mode in enumerate(modes):
        ax = axes_acc[i, j]
        for m in ['EvalGuard', 'DAWN', 'Adi (Backdoor)']:
            r = df[(df['Dataset']==ds) & (df['Method']==m) & (df['L_Mode']==mode)]
            if r.empty: continue
            r = r.iloc[0]
            # 组装精度序列
            y = [TEACHER_ORIGINAL[ds], TEACHER_PROTECTED[ds][m][mode], 
                 parse_pct(r['Base_Acc']), parse_pct(r['FT1%_Acc']), 
                 parse_pct(r['FT5%_Acc']), parse_pct(r['FT10%_Acc'])]
            ax.plot(stages_acc, y, label=m, color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], lw=2, markersize=7)
        
        ax.set_title(f"{ds} ({mode.upper()} Label Extraction)", fontweight='bold')
        ax.grid(True, ls=':', alpha=0.6)
        if j == 0: ax.set_ylabel("Accuracy (%)")
        if i == 2: ax.set_xticklabels(stages_acc, rotation=15)
        else: ax.set_xticklabels([])

# 图例放在最下面
axes_acc[2, 0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.2), ncol=3, frameon=True)

# ==========================================
# 5. 绘制图 2: Security Progression (安全性演进)
# ==========================================
fig_p, axes_p = plt.subplots(3, 2, figsize=(11, 13))
for i, ds in enumerate(datasets):
    for j, mode in enumerate(modes):
        ax = axes_p[i, j]
        for m in ['EvalGuard', 'DAWN', 'Adi (Backdoor)']:
            r = df[(df['Dataset']==ds) & (df['Method']==m) & (df['L_Mode']==mode)]
            if r.empty: continue
            r = r.iloc[0]
            # 计算 -log10(P)，处理 0.0 为极小值防止报错
            y = [-np.log10(max(float(r[c]), 1e-300)) if r[c] != "N/A" else 0 
                 for c in ['Base_P', 'FT1%_P', 'FT5%_P', 'FT10%_P']]
            ax.plot(stages_p, y, label=m, color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], lw=2, markersize=7)
        
        ax.axhline(y=19.26, color='black', ls='--', alpha=0.6, label='Threshold $\eta=2^{-64}$')
        ax.set_title(f"{ds} ({mode.upper()} Label Extraction)", fontweight='bold')
        ax.grid(True, ls=':', alpha=0.6)
        if j == 0: ax.set_ylabel(r"Security Strength $-\log_{10}(P)$")
        if i == 2: ax.set_xticklabels(stages_p)
        else: ax.set_xticklabels([])

axes_p[2, 0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.2), ncol=4, frameon=True)

# ==========================================
# 6. 保存到 fig 文件夹
# ==========================================
fig_acc.tight_layout()
fig_p.tight_layout()

# 显式指定保存到 fig 目录下
fig_acc.savefig('fig/accuracy_lifecycle.pdf', bbox_inches='tight')
fig_p.savefig('fig/security_lifecycle.pdf', bbox_inches='tight')

print("✅ 成功生成 TIFS 投稿标准 PDF 图像：")
print("   - fig/accuracy_lifecycle.pdf")
print("   - fig/security_lifecycle.pdf")