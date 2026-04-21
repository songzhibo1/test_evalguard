import os
import json
import pandas as pd
from pathlib import Path
import argparse

# 根据 sh 脚本和 experiments.py，所有的 baseline 实验现在都统一保存在这里
RESULTS_DIR = Path("results/baseline")

def format_p(p):
    if p is None or p == "N/A": return "N/A"
    try:
        val = float(p)
        return "0.0e+00" if val == 0.0 else f"{val:.1e}"
    except: return str(p)

def format_acc(a):
    if a is None or a == "N/A": return "N/A"
    try: return f"{float(a):.1%}"
    except: return str(a)

def get_sig(target_dict, is_hard=False):
    """提取核心的水印信号指标（兼容多种方法）"""
    if not target_dict or target_dict == "N/A": return "N/A"
    # DAWN, Adi, 和 EvalGuard (硬标签) 使用 match_rate
    if "match_rate" in target_dict: 
        return f"{target_dict['match_rate']:.2%}"
    # EvalGuard (软标签) 使用 confidence_shift
    if "confidence_shift" in target_dict:
        val = target_dict['confidence_shift']
        return f"{val:.2%}" if is_hard else f"{val:.4f}"
    return "N/A"

def extract_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    exp_type = data.get("experiment", "unknown")
    filename = json_path.name
    
    # 【核心修复】智能判断实验方法 (DAWN, Adi, EvalGuard)
    if exp_type == "baseline_dawn" or data.get("method") == "DAWN":
        method = "DAWN"
    elif exp_type == "baseline_adi" or data.get("method") == "Adi_et_al_2018":
        method = "Adi (Backdoor)"
    elif exp_type == "surrogate_finetune":
        method = "EvalGuard"
    else:
        # 如果不是 baseline 相关的 json，直接跳过
        return []

    dataset = data.get("dataset", "unknown").upper()
    label_mode = data.get("label_mode", "soft")
    is_hard = (label_mode == "hard")

    # 【核心修复】更稳健的 Tag 提取 (从文件名提取 __tag_ 之后的内容)
    raw_tag = "N/A"
    if "tag" in data and data["tag"]:
        raw_tag = data["tag"]
    elif "__tag_" in filename:
        raw_tag = filename.split("__tag_")[-1].replace(".json", "")

    teacher_info = data.get("teacher") or {}
    student_info = data.get("student") or {}
    
    t_arch = teacher_info.get("name", "Unknown")
    s_arch = student_info.get("architecture", "Unknown")
    arch_pair = f"{t_arch} -> {s_arch}"

    # 提取超参数
    dist_conf = data.get("distillation_config") or data.get("distillation_baseline") or {}
    temp_val = dist_conf.get("temperature", "N/A")
    nq_val = dist_conf.get("n_queries", "N/A")
    
    # 提取教师精度 (Adi 方法会改变教师精度，需特别展示)
    t_acc = teacher_info.get("accuracy_original", teacher_info.get("accuracy", "N/A"))
    if method == "Adi (Backdoor)":
        t_acc_bd = teacher_info.get("accuracy_backdoored", "N/A")
        t_acc_str = f"{format_acc(t_acc)} (BD: {format_acc(t_acc_bd)})"
    else:
        t_acc_str = format_acc(t_acc)

    # 提取 Baseline 结果 (即刚蒸馏完，FT 0% 时的表现)
    baseline = data.get("baseline_result") or data.get("distillation_baseline") or {}
    base_acc = baseline.get("student_accuracy", baseline.get("surrogate_accuracy", "N/A"))

    record = {
        "Dataset": dataset,
        "Method": method,
        "Tag": raw_tag,
        "Arch_Pair": arch_pair,
        "L_Mode": "hard" if is_hard else "soft",
        "Temp": temp_val,
        "NQ": nq_val,
        "T_Acc": t_acc_str,
        "Base_Acc": format_acc(base_acc),
    }

    # EvalGuard 的信号数据可能嵌套在 single_ctrl 中，也可能在顶层
    if method == "EvalGuard":
        b_ad = baseline.get("all_designs") or {}
        b_target = b_ad.get("single_ctrl", baseline)
    else:
        b_target = baseline

    record.update({
        "Base_Sig": get_sig(b_target, is_hard),
        "Base_P": format_p(b_target.get("p_value")),
        "Base_Ver": str(b_target.get("verified", "N/A")),
    })

    # 提取 Fine-tuning 阶段的攻击抵抗结果
    ft_steps = {0.01: "FT1%", 0.05: "FT5%", 0.1: "FT10%"}
    for ft in data.get("ft_results", []):
        frac = ft.get("ft_fraction")
        if frac in ft_steps:
            prefix = ft_steps[frac]
            
            if method == "EvalGuard":
                f_ad = ft.get("all_designs") or {}
                f_target = f_ad.get("single_ctrl", ft)
            else:
                f_target = ft

            record[f"{prefix}_Acc"] = format_acc(ft.get("accuracy"))
            record[f"{prefix}_Sig"] = get_sig(f_target, is_hard)
            record[f"{prefix}_P"] = format_p(f_target.get("p_value"))
            record[f"{prefix}_Ver"] = str(f_target.get("verified", "N/A"))

    return [record]

def main():
    if not RESULTS_DIR.exists():
        print(f"⚠️ 找不到目录 {RESULTS_DIR}。请确认您的 run_baseline_comparison.sh 是否已经产生了结果。")
        return

    # 递归搜索目录下的所有 json 文件
    json_files = list(RESULTS_DIR.rglob("*.json"))

    if not json_files:
        print(f"⚠️ 在 {RESULTS_DIR} 中没有找到任何 JSON 结果文件！")
        return

    records = []
    for jf in json_files:
        try:
            extracted = extract_metrics(jf)
            if extracted:
                records.extend(extracted)
        except Exception as e:
            print(f"⚠️ 解析文件 {jf.name} 时出错: {e}")

    if not records:
        print("⚠️ 未能提取到任何有效的 Baseline 数据。")
        return
        
    df = pd.DataFrame(records).fillna("N/A")

    # 定义论文 Table X 预期的严谨列顺序
    cols_order = [
        "Dataset", "Method", "L_Mode", "Arch_Pair", "Tag", 
        "NQ", "Temp", "T_Acc", "Base_Acc", "Base_Sig", "Base_P", "Base_Ver",
        "FT1%_Acc", "FT1%_Sig", "FT1%_P", "FT1%_Ver",
        "FT5%_Acc", "FT5%_Sig", "FT5%_P", "FT5%_Ver",
        "FT10%_Acc", "FT10%_Sig", "FT10%_P", "FT10%_Ver"
    ]
    
    # 过滤掉 dataframe 中不存在的列
    final_cols = [c for c in cols_order if c in df.columns]
    df = df[final_cols]

    # 多级排序：数据集 -> 方法 -> 标签模式 -> Tag 名字
    df = df.sort_values(by=["Dataset", "Method", "L_Mode", "Tag"])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)

    # 漂亮的终端分组打印
    for dataset_name, ds_group in df.groupby("Dataset", sort=False):
        print(f"\n{'█'*60} {dataset_name} {'█'*60}")
        
        for method_name, meth_group in ds_group.groupby("Method", sort=False):
            print(f"\n▼ Method: 【 {method_name} 】")
            for l_mode, mode_group in meth_group.groupby("L_Mode", sort=False):
                print(f"  ↳ Label Mode: [ {l_mode.upper()} ]")
                
                # 隐藏全是 N/A 的无效列，保持终端清爽
                to_print = mode_group.loc[:, (mode_group != "N/A").any(axis=0)]
                print(to_print.to_string(index=False))
                print()

    # 【核心修复】保存到指定的 results/baseline 目录下
    output_csv = RESULTS_DIR / "summary_baseline_comparison_TableX.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Baseline 数据提取完成！统一的 CSV 对比表已保存至: {output_csv}")

if __name__ == "__main__":
    main()