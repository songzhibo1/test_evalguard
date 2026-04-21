#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict

def format_pct(value):
    """仅格式化绝对精度"""
    if value is None: 
        return "  ---  "
    return f"{value * 100:>6.2f}%"

def format_acc_delta(acc, orig_acc):
    """格式化：精度 (Δ损耗)，并与 Original 进行对比"""
    if acc is None or orig_acc is None:
        return "      ---      "
    
    delta = acc - orig_acc
    # 如果极小，认为是 0 损耗
    if abs(delta) < 1e-6:
        delta_str = " 0.00%"
    else:
        # 强制显示正负号，保留两位小数
        delta_str = f"{delta * 100:+.2f}%"
        
    return f"{acc * 100:>5.2f}% ({delta_str:>7})"

def main():
    parser = argparse.ArgumentParser(description="Fidelity 实验结果宽表汇总 (含精度损耗)")
    # 自动适配路径
    default_path = "results/fidelity"
    if not Path(default_path).exists() and Path("../results/fidelity").exists():
        default_path = "../results/fidelity"
        
    parser.add_argument("--dir", type=str, default=default_path, help="结果根目录")
    args = parser.parse_args()

    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"❌ 错误: 找不到目录 {root_dir.absolute()}")
        return

    data = defaultdict(lambda: defaultdict(dict))
    json_files = list(root_dir.rglob("*.json"))

    if not json_files:
        print(f"⚠️ 警告: 在 {root_dir.absolute()} 下没有找到任何 .json 文件！")
        return

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            data[content['dataset']][content['teacher']][content['method']] = content['results']

    print("\n" + "="*165)
    print(f" 📊 实验保真度横向对比宽表 (包含 Acc 与 ΔLoss) | 根目录: {root_dir}")
    print("="*165)
    
    # 打印多级表头
    print(f"| {'Dataset':<12} | {'Model':<10} | {'Original':<8} | {'Obfuscat (Δ)':<17} | {'Recoverd (Δ)':<17} | {'EG Soft (Δ)':<17} | {'EG Hard (Δ)':<17} | {'DAWN Soft (Δ)':<17} | {'DAWN Hard (Δ)':<17} | {'Adi (Δ)':<17} |")
    print(f"| {'(数据集)':<11} | {'(模型)':<9} | {'(混淆前)':<7} | {'(混淆后)':<16} | {'(恢复后)':<16} | {'(EG软标签)':<15} | {'(EG硬标签)':<15} | {'(DAWN软标签)':<14} | {'(DAWN硬标签)':<14} | {'(Adi重训练)':<15} |")
    print("-" * 165)

    # 预定义排序顺序
    ds_order = {"cifar10": 1, "cifar100": 2, "tinyimagenet": 3}

    for dataset in sorted(data.keys(), key=lambda x: ds_order.get(x, 99)):
        for model in sorted(data[dataset].keys()):
            methods = data[dataset][model]
            
            # 1. 提取 Original 精度
            orig_acc = next((m['acc_original'] for m in methods.values() if 'acc_original' in m), None)
            
            # 2. 提取 EvalGuard
            eg = methods.get("EvalGuard", {})
            acc_obf = eg.get("acc_obfuscated")
            acc_rec = eg.get("acc_recovered")
            acc_eg_soft = eg.get("acc_wm_soft")
            acc_eg_hard = eg.get("acc_wm_hard")
            
            # 3. 提取 DAWN
            dawn = methods.get("DAWN", {})
            acc_dawn_soft = dawn.get("acc_dawn_soft")
            acc_dawn_hard = dawn.get("acc_dawn_hard")
            
            # 4. 提取 Adi
            adi = methods.get("Adi_et_al_2018") or methods.get("Adi") or {}
            acc_adi = adi.get("acc_backdoored")

            # 格式化同行输出：格式为 Acc% (Delta%)
            row = (
                f"| {dataset.upper():<12} | {model.upper():<10} "
                f"| {format_pct(orig_acc):<8} "
                f"| {format_acc_delta(acc_obf, orig_acc):<17} "
                f"| {format_acc_delta(acc_rec, orig_acc):<17} "
                f"| {format_acc_delta(acc_eg_soft, orig_acc):<17} "
                f"| {format_acc_delta(acc_eg_hard, orig_acc):<17} "
                f"| {format_acc_delta(acc_dawn_soft, orig_acc):<17} "
                f"| {format_acc_delta(acc_dawn_hard, orig_acc):<17} "
                f"| {format_acc_delta(acc_adi, orig_acc):<17} |"
            )
            print(row)
            
    print("="*165 + "\n")

if __name__ == "__main__":
    main()