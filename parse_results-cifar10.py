import os
import json
import pandas as pd
from pathlib import Path
import argparse
import re

# 1. 更新为 CIFAR-10 的专属目录
RESULTS_DIRS = [
    Path("results/surrogate_ft/cifar10"),
    Path("results/random_kw_fp/cifar10"),
    Path("results/distill/cifar10"),           
    Path("results/posthoc_prune/cifar10"),     
    Path("results/posthoc_quantize/cifar10")   
]

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
    if not target_dict or target_dict == "N/A": return "N/A"
    if "match_rate" in target_dict: 
        return f"{target_dict['match_rate']:.2%}"
    if "confidence_shift" in target_dict:
        val = target_dict['confidence_shift']
        return f"{val:.2%}" if is_hard else f"{val:.4f}"
    return "N/A"

def extract_metrics(json_path, design_mode):
    with open(json_path, 'r') as f:
        data = json.load(f)

    exp_type = data.get("experiment", "unknown")
    watermark = data.get("watermark_config") or {}
    baseline = data.get("distillation_baseline") or {}
    ft_conf = data.get("ft_config") or {}
    dist_conf = data.get("distillation_config") or {}
    hl_conf = data.get("hard_label_config") or {}
    scan_conf = data.get("scan_config") or {}
    student_info = data.get("student") or {}
    teacher_info = data.get("teacher") or {}
    trigger_conf = data.get("trigger_config") or {}
    attack_conf = data.get("attack_config") or {}

    is_svhn = dist_conf.get("query_dataset") == "svhn"
    if exp_type == "distillation" and not is_svhn: return []

    label_mode = data.get("label_mode", watermark.get("label_mode", "soft"))
    hard_mode = data.get("hard_label_mode", watermark.get("hard_label_mode", hl_conf.get("hard_label_mode", "N/A")))
    tau = data.get("margin_tau_hard", watermark.get("margin_tau_hard", hl_conf.get("margin_tau_hard", "N/A")))
    is_hard = (label_mode == "hard")

    filename = json_path.name
    raw_tag = data.get("tag", "")
    
    if not raw_tag:
        if "__tag_" in filename:
            raw_tag = filename.split("__tag_")[-1].replace(".json", "")
        else:
            # 🌟 [强化点] 修复小数截断问题，如 rw_0.02
            match_tag = re.search(r'(c10v5_[a-zA-Z0-9_\.]+)', filename)
            raw_tag = match_tag.group(1) if match_tag else filename.replace(".json", "")

    tag = raw_tag 

    if tau == "N/A":
        match_tau = re.search(r'_tau_?([0-9\.]+)', raw_tag)
        if match_tau: tau = float(match_tau.group(1))

    # 🌟 [修改点] 将所有的 c100v5_ 替换为 c10v5_
    if exp_type == "random_kw_fp":
        exp_group = "ExpC2_RandKwHard" if is_hard else "ExpC1_RandKwSoft"
        tag = f"RANDKW_{raw_tag}"
    elif exp_type == "posthoc_prune":
        exp_group = "ExpC6_PruneHard" if is_hard else "ExpC4_PruneSoft"
        tag = f"PRUNE_{raw_tag}"
    elif exp_type == "posthoc_quantize":
        exp_group = "ExpC5_QuantSoft"
        tag = f"QUANT_{raw_tag}"
    elif exp_type == "distillation" and is_svhn:
        exp_group = "ExpC7_SVHNHard" if is_hard else "ExpC3_SVHNSoft"
        tag = f"SVHN_{raw_tag}"
    else:
        tag_l = raw_tag.lower()
        if "c10v5_bgs_q010" in tag_l: exp_group = "ExpB1_BGS_Adaptive"
        elif "c10v5_bgs_qsweep" in tag_l: exp_group = "ExpB2_BGS_QSweep"
        elif "c10v5_bgs_tau" in tag_l: exp_group = "ExpB3_BGS_TauSweep"
        elif "c10v5_bgs_rw" in tag_l: exp_group = "ExpB4_BGS_RWSweep"
        elif "c10v5_fp_plain" in tag_l: exp_group = "ExpB5_Plain_Baseline"
        elif "c10v5_cross_" in tag_l: exp_group = "ExpA2_CrossArch"
        elif "c10v5_vtemp_" in tag_l: exp_group = "ExpA3_VTempSweep"
        elif "c10v5_seed_" in tag_l: exp_group = "ExpA4_MultiSeed"
        elif "c10v5_rw_" in tag_l: exp_group = "ExpA5_RWSweep"
        elif "c10v5_d_" in tag_l: exp_group = "ExpA6_DeltaSweep"
        elif "c10v5_beta_" in tag_l: exp_group = "ExpA7_BetaSweep"
        elif "c10v5_nq_" in tag_l: exp_group = "ExpA8_NQSweep"
        elif "c10v5_trig_" in tag_l: exp_group = "ExpA9_TrigSizeSweep"
        elif "c10v5_own_test_" in tag_l: exp_group = "ExpA13_OwnDataTest"
        elif "c10v5_own_" in tag_l: exp_group = "ExpA12_OwnDataMain"
        elif tag_l.startswith("c10v5_rw_0.0"): exp_group = "ExpA11_CleanBaseline"
        elif tag_l.startswith("c10v5_") and "_t" in tag_l: exp_group = "ExpA1_MainTable"
        else: exp_group = "Legacy_Or_Unknown"

    if not is_hard: tau = "N/A"

    t_arch = teacher_info.get("name", "Unknown")
    s_arch = student_info.get("architecture", "Unknown")
    arch_pair = f"{t_arch} -> {s_arch}"
    
    trigger_mode = data.get("trigger_mode", trigger_conf.get("trigger_source", "REC")).replace("_trigger", "").upper()
    if exp_type in ["random_kw_fp", "posthoc_prune", "posthoc_quantize", "distillation"]:
        trigger_mode = "REC" 

    student_acc = baseline.get("surrogate_accuracy", student_info.get("baseline_accuracy", student_info.get("accuracy", "N/A")))
    
    temp_val = baseline.get("temperature", scan_conf.get("T", dist_conf.get("temperature", attack_conf.get("T", "N/A"))))
    if temp_val == "N/A" and isinstance(data.get("temperatures"), list):
        temp_val = data["temperatures"][0]

    nq_val = watermark.get("nq", data.get("nq", "N/A"))
    if nq_val == "N/A":
        match_nq = re.search(r'_nq(\d+)', filename)
        if match_nq: nq_val = int(match_nq.group(1))

    res_dict = data.get("result") or {}
    trig_size = baseline.get("n_triggers_used", res_dict.get("n_trigger", student_info.get("n_triggers", "N/A")))
    if trig_size == "N/A" and data.get("attack_runs"):
        trig_size = data["attack_runs"][0].get("n_trigger", "N/A")

    record_common = {
        "Experiment": exp_group, "Tag": tag, "Seed": data.get("seed", "N/A"), 
        "Trigger": trigger_mode, "Arch_Pair": arch_pair,
        "L_Mode": f"hard({hard_mode})" if is_hard else "soft", "Tau": tau,
        "Dist_Ep": baseline.get("dist_epochs", dist_conf.get("epochs", "N/A")), 
        "Temp": temp_val, 
        "V_Temp": watermark.get("verify_temperature", scan_conf.get("verify_temperature", "N/A")),
        "RW": watermark.get("r_w", "N/A"), "Delta": watermark.get("delta_logit", "N/A"),
        "D_Min": watermark.get("delta_min", "N/A"), "Beta": watermark.get("beta", "N/A"), 
        "NQ": nq_val,
        "Trig_Size": trig_size,
        "T_Acc": format_acc(teacher_info.get("accuracy", "N/A")),
        "Base_Acc": format_acc(student_acc),
    }

    records = []

    if exp_type == "random_kw_fp":
        row = record_common.copy()
        row.update({
            "Design": "N/A", "Base_Sig": "N/A", "Base_Ver": res_dict.get("genuine_verified", "N/A"),
            "Base_P": format_p(res_dict.get("genuine_p_value")), 
            "P_Gap": res_dict.get("p_gap", "N/A"),
            "FP_Rate": f"{res_dict.get('fraction_below_eta', 0.0):.2%}",
            "Min_P": format_p(res_dict.get("min_p")), "Median_P": format_p(res_dict.get("median_p")),
        })
        records.append(row)

    elif exp_type in ["posthoc_prune", "posthoc_quantize"]:
        row = record_common.copy()
        row.update({"Design": "N/A", "Base_Sig": "N/A", "Base_P": "N/A", "Base_Ver": "N/A"})
        
        for run in data.get("attack_runs", []):
            atk_name = run["attack"].upper()
            row[f"{atk_name}_Acc"] = format_acc(run.get("accuracy"))
            row[f"{atk_name}_DeltaPP"] = run.get("accuracy_delta_pp", "N/A") 
            row[f"{atk_name}_Sig"] = get_sig(run, is_hard)
            row[f"{atk_name}_P"] = format_p(run.get("p_value"))
            row[f"{atk_name}_Log10P"] = run.get("log10_p_value", "N/A") 
            row[f"{atk_name}_Ver"] = run.get("verified", "N/A")
            
            ad = run.get("all_designs") or {}
            if ad:
                row[f"{atk_name}_P_MR"] = format_p((ad.get("mean_rest") or {}).get("p_value", "N/A"))
                row[f"{atk_name}_P_ST1"] = format_p((ad.get("suspect_top1") or {}).get("p_value", "N/A"))
        records.append(row)

    elif exp_type == "distillation" and is_svhn:
        row = record_common.copy()
        row.update({
            "Design": "N/A",
            "Base_Sig": get_sig(res_dict, is_hard),
            "Base_P": format_p(res_dict.get("p_value")),
            "Base_Ver": res_dict.get("verified", "N/A"),
        })
        row["Base_Acc"] = format_acc(res_dict.get("student_accuracy", "N/A"))
        records.append(row)

    elif exp_type == "surrogate_finetune":
        b_all_designs = baseline.get("all_designs") or {}
        designs = ["mean_rest", "single_ctrl", "suspect_top1"] if design_mode == "all" else [design_mode]

        for d_name in designs:
            b_target = b_all_designs.get(d_name, baseline) if b_all_designs else baseline
            ft_steps = {0.01: "FT1%", 0.05: "FT5%", 0.1: "FT10%"}
            
            row = record_common.copy()
            row.update({
                "Design": d_name,
                "Base_Sig": get_sig(b_target, is_hard),
                "Base_P": format_p(b_target.get("p_value")),
                "Base_Ver": b_target.get("verified", "N/A"),
            })
            
            for ft in data.get("ft_results", []):
                frac = ft.get("ft_fraction")
                if frac in ft_steps:
                    prefix = ft_steps[frac]
                    f_ad = ft.get("all_designs") or {}
                    f_target = f_ad.get(d_name, ft)
                    row[f"{prefix}_Acc"] = format_acc(ft.get("accuracy"))
                    row[f"{prefix}_Sig"] = get_sig(f_target, is_hard)
                    row[f"{prefix}_P"] = format_p(f_target.get("p_value"))
                    row[f"{prefix}_Ver"] = f_target.get("verified", "N/A")
            records.append(row)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=str, default="mean_rest", choices=["mean_rest", "single_ctrl", "suspect_top1", "all"])
    parser.add_argument("--no_own", action="store_true", help="过滤掉所有的 OWN 触发器结果，只保留 REC")
    args = parser.parse_args()

    json_files = []
    for dir_path in RESULTS_DIRS:
        if dir_path.exists():
            json_files.extend(list(dir_path.rglob("*.json")))

    if not json_files:
        print("⚠️ 未找到 JSON 结果文件！请确保 CIFAR-10 相关 .sh 脚本已运行完毕并输出到了相应目录。")
        return

    records = []
    for jf in json_files:
        try: records.extend(extract_metrics(jf, args.design))
        except Exception as e: print(f"⚠️ Error parsing {jf.name}: {e}")

    if not records: return
    df = pd.DataFrame(records).fillna("N/A")

    if args.no_own:
        df = df[df["Trigger"] != "OWN"]
    
    sort_keys = []
    for col in ['Tau', 'Temp', 'V_Temp', 'Delta', 'RW', 'Beta', 'D_Min']:
        if col in df.columns:
            num_col = f'_{col}_num'
            df[num_col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
            sort_keys.append(num_col)

    for c in ["PRUNE_25%_Acc", "PRUNE_50%_Acc", "PRUNE_75%_Acc", "INT8_QUANT_Acc"]:
        if c not in df.columns:
            df[c] = "N/A"

    final_sort_cols = ["Experiment", "Trigger", "Arch_Pair"] + sort_keys + ["Design"]
    final_sort_cols = [c for c in final_sort_cols if c in df.columns]

    if not df.empty:
        df = df.sort_values(by=final_sort_cols)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)

    for exp_name, exp_group in df.groupby("Experiment", sort=False):
        print(f"\n{'█'*60} {exp_name} {'█'*60}")
        
        for trig_name, trig_group in exp_group.groupby("Trigger", sort=False):
            print(f"\n▼ 触发器模式: 【 {trig_name} 】")
            for arch_pair, arch_group in trig_group.groupby("Arch_Pair", sort=False):
                print(f"  ↳ 模型对: [ {arch_pair} ]")
                
                base_cols = ["Tag", "L_Mode", "Tau", "Temp", "V_Temp", "RW", "Delta", "Beta", "D_Min", "NQ", "Trig_Size", "T_Acc", "Base_Acc"]
                
                if "RandKw" in exp_name:
                    disp = base_cols + ["Base_Ver", "Base_P", "P_Gap", "FP_Rate", "Min_P", "Median_P"]
                elif "Prune" in exp_name or "Quant" in exp_name:
                    disp = base_cols
                    attack_cols = sorted([c for c in arch_group.columns if c.startswith("PRUNE_") or c.startswith("INT8_QUANT_")])
                    disp.extend(attack_cols)
                elif "SVHN" in exp_name:
                    disp = base_cols + ["Base_Sig", "Base_P", "Base_Ver"]
                else:
                    disp = base_cols + [
                        "Base_Sig", "Base_P", "Base_Ver", 
                        "FT1%_Acc", "FT1%_Sig", "FT1%_P", "FT1%_Ver", 
                        "FT5%_Acc", "FT5%_Sig", "FT5%_P", "FT5%_Ver", 
                        "FT10%_Acc", "FT10%_Sig", "FT10%_P", "FT10%_Ver"
                    ]
                
                disp = [c for c in disp if c in arch_group.columns]
                to_print = arch_group[disp].copy()
                if args.design == "all" and "Design" not in to_print: 
                    to_print.insert(1, "Design", arch_group["Design"])
                    
                to_print = to_print.loc[:, (to_print != "N/A").any(axis=0)]
                print(to_print.to_string(index=False))
                print()

    drop_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_num")]
    output_df = df.drop(columns=drop_cols)
    
    file_suffix = "_no_OWN" if args.no_own else ""
    output_csv = Path("results") / f"summary_results_c10_{args.design}_v5_full{file_suffix}.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"✅ 解析完美完成！已适配 CIFAR-10 并提取所有隐藏参数 (Temp/NQ/Trig_Size)。CSV 保存至: {output_csv}")

if __name__ == "__main__":
    main()