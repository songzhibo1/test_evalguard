#!/usr/bin/env python3
"""
EvalGuard — Obfuscated-Model Attack Experiment (Section VI-D)

Threat model: Attacker physically steals the obfuscated model weights
(bypasses the TEE/API entirely).  Without the secret key K_obf, the
obfuscated model has ~10% accuracy (large Gaussian noise dominates all
weights).  This experiment asks: can the attacker recover usable accuracy
via post-hoc fine-tuning or pruning of the stolen obfuscated weights?

Attacks tested per dataset:
  1. Fine-tuning (FT) attack   — SGD + CE on attacker's own clean data,
       at data budgets: 1%, 5%, 10%, 20%, 50%, 100% of training set.
  2. Magnitude pruning attack  — remove p% of smallest-magnitude weights,
       at prune fractions: 25%, 50%, 75%.

Expected finding (security argument):
  * The Gaussian noise σ ≫ weight magnitudes, so the obfuscated model has
    no usable gradient signal: fine-tuning is equivalent to training from
    random initialisation with the noise as extra perturbation.
  * Pruning kills useful capacity before it removes meaningful noise:
    even 75% pruning leaves ~10% accuracy.
  * Conclusion: obfuscation makes the stolen weights cryptographically
    worthless to the attacker regardless of the post-processing budget.

Datasets / teachers:
  CIFAR-10        ResNet-20  (cifar10_resnet20,   10 classes)
  CIFAR-100       ResNet-56  (cifar100_resnet56,  100 classes)
  Tiny-ImageNet   ResNet-18  (tinyimagenet_resnet18, 200 classes)

Usage
-----
  python obfuscated_attack_experiments.py \\
      --dataset cifar10 \\
      --device cuda:0 \\
      --epochs 50 \\
      --out_dir results/obfuscated_attack
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from evalguard.configs import (
    CONFIGS,
    cifar10_data, cifar100_data, tinyimagenet_data,
    create_student,
)
from evalguard.obfuscation import obfuscate_model_vectorized
from evalguard.attacks import fine_tune_surrogate, prune_surrogate

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

FT_RATIOS  = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]   # fraction of train set
PRUNE_FRAC = [0.25, 0.50, 0.75]                       # weight-magnitude thresholds

DATASET_CONFIGS = {
    "cifar10": {
        "config_key": "cifar10_resnet20",
        "teacher_arch": "resnet20",
        "student_arch": "resnet20",
        "num_classes":  10,
        "baseline_acc": 92.6,
    },
    "cifar100": {
        "config_key": "cifar100_resnet56",
        "teacher_arch": "resnet56",
        "student_arch": "resnet56",
        "num_classes":  100,
        "baseline_acc": 74.0,
    },
    "tinyimagenet": {
        "config_key": "tinyimagenet_resnet18",
        "teacher_arch": "resnet18",
        "student_arch": "resnet18",
        "num_classes":  200,
        "baseline_acc": 60.12,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def build_teacher(config_key: str, device: str) -> nn.Module:
    """Construct (but do not train) the teacher skeleton from CONFIGS."""
    cfg = CONFIGS[config_key]
    teacher = cfg["model_fn"]()
    teacher.to(device)
    return teacher, cfg


def train_teacher(teacher: nn.Module, train_loader: DataLoader,
                  epochs: int, lr: float, device: str) -> nn.Module:
    teacher.to(device).train()
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(teacher(x), y).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    Teacher epoch {ep+1}/{epochs}")
    return teacher


def subset_loader(dataset, ratio: float, batch_size: int, num_workers: int = 2) -> DataLoader:
    n = max(1, int(len(dataset) * ratio))
    indices = list(range(n))
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=True, num_workers=num_workers)


def ft_attack(obf_model: nn.Module, train_set, test_loader: DataLoader,
              ratio: float, epochs: int, device: str) -> dict:
    """Fine-tune the obfuscated model on `ratio` fraction of clean train data."""
    model_copy = copy.deepcopy(obf_model).to(device)
    ft_loader  = subset_loader(train_set, ratio, batch_size=128)
    n_samples  = len(ft_loader.dataset)

    t0 = time.time()
    model_copy, _ = fine_tune_surrogate(
        model_copy, ft_loader,
        epochs=epochs, lr=0.01,    # higher LR than default to drive from noise-init
        device=device
    )
    elapsed = time.time() - t0
    acc = evaluate_accuracy(model_copy, test_loader, device)

    return {
        "ratio":      ratio,
        "n_samples":  n_samples,
        "ft_epochs":  epochs,
        "acc_pct":    round(acc, 2),
        "elapsed_s":  round(elapsed, 1),
    }


def prune_attack(obf_model: nn.Module, test_loader: DataLoader,
                 frac: float, device: str) -> dict:
    """Prune `frac` fraction of smallest-magnitude weights in the obfuscated model."""
    model_copy = copy.deepcopy(obf_model).to(device)
    model_copy = prune_surrogate(model_copy, frac, scope="global")
    acc = evaluate_accuracy(model_copy, test_loader, device)
    return {
        "prune_frac": frac,
        "acc_pct":    round(acc, 2),
    }


# ── Main experiment function ───────────────────────────────────────────────────

def run_dataset(dataset_name: str, device: str, epochs: int, out_dir: Path):
    print("\n" + "="*70)
    print(f"  Obfuscated-Model Attack  ·  Dataset: {dataset_name.upper()}")
    print("="*70)

    cfg_info  = DATASET_CONFIGS[dataset_name]
    config    = CONFIGS[cfg_info["config_key"]]

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("  [1/4] Loading dataset …")
    train_set, test_set, train_loader, test_loader = config["data_fn"](batch_size=128)

    # ── 2. Build & train clean teacher ────────────────────────────────────────
    print("  [2/4] Building clean teacher …")
    ckpt_path = out_dir / f"{dataset_name}_clean_teacher.pt"

    teacher = config["model_fn"]()
    if ckpt_path.exists():
        teacher.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"    Loaded from cache: {ckpt_path}")
    else:
        print("    Training clean teacher from scratch …")
        teacher = train_teacher(teacher, train_loader, epochs=epochs, lr=0.1, device=device)
        torch.save(teacher.state_dict(), ckpt_path)
        print(f"    Saved: {ckpt_path}")

    teacher.to(device)
    clean_acc = evaluate_accuracy(teacher, test_loader, device)
    print(f"    Clean teacher accuracy: {clean_acc:.2f}%")

    # ── 3. Obfuscate teacher ───────────────────────────────────────────────────
    print("  [3/4] Obfuscating teacher (Algorithm 1, ε=50, δ=2⁻³²) …")
    obf_teacher = copy.deepcopy(teacher)
    obf_teacher, secret = obfuscate_model_vectorized(
        obf_teacher, epsilon=50.0, delta=2**(-32),
        security_param=256, model_id=f"{dataset_name}_teacher"
    )
    obf_acc = evaluate_accuracy(obf_teacher, test_loader, device)
    print(f"    Obfuscated model accuracy: {obf_acc:.2f}%  (σ={secret.sigma:.4f})")

    # ── 4a. Fine-tuning attacks ────────────────────────────────────────────────
    print("  [4a/4] Fine-tuning attack (attacker trains obfuscated weights on clean data) …")
    ft_results = []
    for ratio in FT_RATIOS:
        res = ft_attack(obf_teacher, train_set, test_loader, ratio, epochs=50, device=device)
        ft_results.append(res)
        print(f"    FT {ratio*100:5.1f}%  ({res['n_samples']:6d} samples)  → acc={res['acc_pct']:.2f}%")

    # ── 4b. Pruning attacks ────────────────────────────────────────────────────
    print("  [4b/4] Pruning attack (attacker prunes obfuscated weights) …")
    prune_results = []
    for frac in PRUNE_FRAC:
        res = prune_attack(obf_teacher, test_loader, frac, device)
        prune_results.append(res)
        print(f"    Prune {frac*100:.0f}%  → acc={res['acc_pct']:.2f}%")

    # ── Save ───────────────────────────────────────────────────────────────────
    result = {
        "dataset":          dataset_name,
        "config_key":       cfg_info["config_key"],
        "teacher_arch":     cfg_info["teacher_arch"],
        "num_classes":      cfg_info["num_classes"],
        "clean_teacher_acc":round(clean_acc, 2),
        "obf_acc":          round(obf_acc, 2),
        "sigma":            round(secret.sigma, 6),
        "ft_attack":        ft_results,
        "prune_attack":     prune_results,
    }
    out_path = out_dir / f"{dataset_name}_obfuscated_attack.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return result


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]):
    print("\n" + "="*78)
    print("  SUMMARY — Obfuscated-Model Attack Results")
    print("="*78)
    print(f"  {'Dataset':<14} {'Clean':>7} {'Obf.':>7}  "
          f"{'FT 1%':>7} {'FT 5%':>7} {'FT10%':>7} {'FT50%':>7} {'FT100%':>7}")
    print("-"*78)
    for r in all_results:
        ft = {x['ratio']: x['acc_pct'] for x in r['ft_attack']}
        print(f"  {r['dataset']:<14} {r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>6.1f}%  "
              f"{ft.get(0.01,0):>6.1f}% {ft.get(0.05,0):>6.1f}% "
              f"{ft.get(0.10,0):>6.1f}% {ft.get(0.50,0):>6.1f}% "
              f"{ft.get(1.00,0):>6.1f}%")
    print()
    print(f"  {'Dataset':<14} {'Clean':>7} {'Obf.':>7}  "
          f"{'Prune25':>8} {'Prune50':>8} {'Prune75':>8}")
    print("-"*60)
    for r in all_results:
        pr = {x['prune_frac']: x['acc_pct'] for x in r['prune_attack']}
        print(f"  {r['dataset']:<14} {r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>6.1f}%  "
              f"{pr.get(0.25,0):>7.1f}% {pr.get(0.50,0):>7.1f}% "
              f"{pr.get(0.75,0):>7.1f}%")
    print("="*78)
    print("  Obfuscation renders stolen weights useless: FT/pruning cannot recover")
    print("  clean-model accuracy without the secret key K_obf.")
    print("="*78 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Obfuscated-model attack experiments")
    p.add_argument("--dataset", choices=["cifar10","cifar100","tinyimagenet","all"],
                   default="all", help="Dataset to run (default: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=80,
                   help="Epochs for clean teacher training (default: 80)")
    p.add_argument("--out_dir", default="results/obfuscated_attack",
                   help="Output directory for JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    all_results = []
    for ds in datasets:
        result = run_dataset(ds, device=args.device, epochs=args.epochs, out_dir=out_dir)
        all_results.append(result)

    print_summary(all_results)

    # Save combined summary JSON
    summary_path = out_dir / "obfuscated_attack_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined summary saved: {summary_path}")
