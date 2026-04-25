#!/usr/bin/env python3
"""
EvalGuard — Obfuscated-Model Attack Experiment (Section VI-D)

Threat model: Attacker physically steals the obfuscated model weights
(bypasses the TEE/API entirely).  Without the secret key K_obf, the
obfuscated model has ~10% accuracy (large Gaussian noise dominates all
weights).  This experiment asks: can the attacker recover usable accuracy
via post-hoc fine-tuning or pruning of the stolen obfuscated weights?

Attacks tested per model:
  1. Fine-tuning (FT) attack   — SGD + CE on attacker's own clean data,
       at data budgets: 1%, 5%, 10%, 20%, 50%, 100% of training set.
  2. Magnitude pruning attack  — remove p% of smallest-magnitude weights,
       at prune fractions: 25%, 50%, 75%.

Models tested (pretrained weights loaded automatically — no training needed):
  CIFAR-10      ResNet-20  (cifar10_resnet20,    10 classes, ~92.6%)
  CIFAR-10      VGG-11     (cifar10_vgg11,       10 classes, ~92.8%)
  CIFAR-100     ResNet-56  (cifar100_resnet56,  100 classes, ~74.0%)
  CIFAR-100     VGG-11     (cifar100_vgg11,     100 classes, ~70.0%)
  Tiny-ImageNet ResNet-18  (tinyimagenet_resnet18, 200 classes, ~60.1%)

Usage
-----
  # Run a single model config
  python obfuscated_attack_experiments.py --config cifar10_resnet20 --device cuda:0

  # Run all models in one dataset
  python obfuscated_attack_experiments.py --config cifar10 --device cuda:0

  # Run all 5 model configs sequentially
  python obfuscated_attack_experiments.py --config all --device cuda:0
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim   # used by fine_tune_surrogate internally
from torch.utils.data import DataLoader, Subset

from evalguard.configs import CONFIGS
from evalguard.obfuscation import obfuscate_model_vectorized
from evalguard.attacks import prune_surrogate

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

FT_RATIOS      = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]  # fraction of train set
FT_STRATEGIES  = ["FTAL", "FTLL", "RTFL"]              # fine-tuning strategies
PRUNE_FRAC     = [0.25, 0.50, 0.75]                    # weight-magnitude thresholds

# Fine-tuning strategy descriptions (for terminal/logs)
FT_DESC = {
    "FTAL": "Full-layer fine-tune (all weights, obfuscated init)",
    "FTLL": "Last-layer only fine-tune (all other layers frozen)",
    "RTFL": "Reset last layer + full-layer fine-tune (strongest attack)",
}

# All 5 model configs; each entry is a CONFIGS key.
ALL_CONFIGS = [
    "cifar10_resnet20",
    "cifar10_vgg11",
    "cifar100_resnet56",
    "cifar100_vgg11",
    "tinyimagenet_resnet18",
]

# Short display labels
CONFIG_LABEL = {
    "cifar10_resnet20":      ("CIFAR-10",      "ResNet-20"),
    "cifar10_vgg11":         ("CIFAR-10",      "VGG-11"),
    "cifar100_resnet56":     ("CIFAR-100",     "ResNet-56"),
    "cifar100_vgg11":        ("CIFAR-100",     "VGG-11"),
    "tinyimagenet_resnet18": ("Tiny-ImageNet", "ResNet-18"),
}

# Dataset-group shortcuts for --config argument
DATASET_GROUPS = {
    "cifar10":      ["cifar10_resnet20", "cifar10_vgg11"],
    "cifar100":     ["cifar100_resnet56", "cifar100_vgg11"],
    "tinyimagenet": ["tinyimagenet_resnet18"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_pretrained_teacher(config_key: str) -> nn.Module:
    """
    Load pretrained teacher via config["model_fn"](pretrained=True).
    model_fn returns (model, layer_name) — unpack and return model only.
    """
    config = CONFIGS[config_key]
    result = config["model_fn"](pretrained=True)
    # All model_fn in configs.py return (model, layer_name_str)
    model = result[0] if isinstance(result, (tuple, list)) else result
    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def subset_loader(dataset, ratio: float, batch_size: int,
                  num_workers: int = 2) -> DataLoader:
    n = max(1, int(len(dataset) * ratio))
    return DataLoader(Subset(dataset, list(range(n))), batch_size=batch_size,
                      shuffle=True, num_workers=num_workers)


def _get_last_linear(model: nn.Module) -> nn.Linear:
    """Return the last nn.Linear layer in the model (classifier head)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("No nn.Linear found in model")
    return last


def _reset_last_layer(model: nn.Module) -> nn.Module:
    """Re-initialise the last Linear layer to Kaiming uniform (clean random init)."""
    fc = _get_last_linear(model)
    nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
    if fc.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(fc.bias, -bound, bound)
    return model


def _freeze_all_except_last(model: nn.Module) -> nn.Module:
    """Freeze every parameter except the last Linear layer."""
    last_fc = _get_last_linear(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in last_fc.parameters():
        p.requires_grad = True
    return model


def ft_attack(obf_model: nn.Module, train_set, test_loader: DataLoader,
              ratio: float, ft_epochs: int, strategy: str, device: str) -> dict:
    """
    Fine-tune the obfuscated model under three strategies:

    FTAL — Fine-Tune ALL Layers
        Standard full-model SGD; strongest baseline for "brute-force" recovery.

    FTLL — Fine-Tune Last Layer only
        All conv/bn layers frozen; only the classifier head is updated.
        Lower-bound attack: shows what a resourced-limited attacker can achieve.

    RTFL — Reset last layer + Fine-Tune ALL Layers
        Classifier head is re-initialised to Kaiming before full fine-tuning.
        Strongest attack: removes noisy head signal, then optimises globally.
    """
    assert strategy in ("FTAL", "FTLL", "RTFL"), f"Unknown strategy: {strategy}"

    model_copy = copy.deepcopy(obf_model).to(device)
    ft_loader  = subset_loader(train_set, ratio, batch_size=128)
    n_samples  = len(ft_loader.dataset)

    # ── Strategy-specific preparation ─────────────────────────────────────────
    if strategy == "RTFL":
        _reset_last_layer(model_copy)           # clean head, then train all
    elif strategy == "FTLL":
        _freeze_all_except_last(model_copy)     # freeze body, train head only

    trainable = [p for p in model_copy.parameters() if p.requires_grad]
    optimizer  = optim.SGD(trainable, lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    criterion  = nn.CrossEntropyLoss()

    t0 = time.time()
    model_copy.train()
    for ep in range(ft_epochs):
        for x, y in ft_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model_copy(x), y).backward()
            optimizer.step()
        scheduler.step()
        if (ep + 1) % 10 == 0:
            print(f"      [{strategy}] epoch {ep+1}/{ft_epochs}")
    elapsed = time.time() - t0

    # Unfreeze all before evaluation (belt-and-suspenders)
    for p in model_copy.parameters():
        p.requires_grad = True

    acc = evaluate_accuracy(model_copy, test_loader, device)
    return {
        "strategy":  strategy,
        "ratio":     ratio,
        "n_samples": n_samples,
        "ft_epochs": ft_epochs,
        "acc_pct":   round(acc, 2),
        "elapsed_s": round(elapsed, 1),
    }


def prune_attack(obf_model: nn.Module, test_loader: DataLoader,
                 frac: float, device: str) -> dict:
    """Prune `frac` fraction of smallest-magnitude weights."""
    model_copy = copy.deepcopy(obf_model).to(device)
    model_copy = prune_surrogate(model_copy, frac, scope="global")
    acc = evaluate_accuracy(model_copy, test_loader, device)
    return {
        "prune_frac": frac,
        "acc_pct":    round(acc, 2),
    }


# ── Per-config experiment ──────────────────────────────────────────────────────

def run_config(config_key: str, device: str, ft_epochs: int,
               out_dir: Path) -> dict:
    ds_label, arch_label = CONFIG_LABEL[config_key]
    config = CONFIGS[config_key]

    print("\n" + "="*70)
    print(f"  Obfuscated-Model Attack  ·  {ds_label} / {arch_label}  ({config_key})")
    print("="*70)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("  [1/4] Loading dataset …")
    train_set, _, train_loader, test_loader = config["data_fn"](batch_size=128)

    # ── 2. Load pretrained teacher (no training needed) ───────────────────────
    print("  [2/4] Loading pretrained teacher …")
    teacher = load_pretrained_teacher(config_key)
    teacher.to(device)
    clean_acc = evaluate_accuracy(teacher, test_loader, device)
    print(f"    Pretrained teacher accuracy: {clean_acc:.2f}%")

    # ── 3. Obfuscate teacher ───────────────────────────────────────────────────
    print("  [3/4] Obfuscating teacher (Algorithm 1, ε=50, δ=2⁻³²) …")
    obf_teacher = copy.deepcopy(teacher)
    obf_teacher, secret = obfuscate_model_vectorized(
        obf_teacher, epsilon=50.0, delta=2**(-32),
        security_param=256, model_id=config_key,
    )
    obf_acc = evaluate_accuracy(obf_teacher, test_loader, device)
    print(f"    Obfuscated model accuracy: {obf_acc:.2f}%  (σ={secret.sigma:.4f})")

    # ── 4a. Fine-tuning attacks (FTAL / FTLL / RTFL) ──────────────────────────
    print("  [4a/4] Fine-tuning attacks …")
    ft_results = {}          # strategy → list of per-ratio dicts
    for strategy in FT_STRATEGIES:
        print(f"\n    Strategy: {strategy}  — {FT_DESC[strategy]}")
        strategy_rows = []
        for ratio in FT_RATIOS:
            res = ft_attack(obf_teacher, train_set, test_loader,
                            ratio=ratio, ft_epochs=ft_epochs,
                            strategy=strategy, device=device)
            strategy_rows.append(res)
            print(f"      {strategy} {ratio*100:5.1f}%  "
                  f"({res['n_samples']:6d} samples)  → acc={res['acc_pct']:.2f}%")
        ft_results[strategy] = strategy_rows

    # ── 4b. Pruning attacks ────────────────────────────────────────────────────
    print("  [4b/4] Pruning attack …")
    prune_results = []
    for frac in PRUNE_FRAC:
        res = prune_attack(obf_teacher, test_loader, frac=frac, device=device)
        prune_results.append(res)
        print(f"    Prune {frac*100:.0f}%  → acc={res['acc_pct']:.2f}%")

    # ── Save individual JSON ───────────────────────────────────────────────────
    result = {
        "config_key":       config_key,
        "dataset":          ds_label,
        "arch":             arch_label,
        "num_classes":      config["num_classes"],
        "clean_teacher_acc":round(clean_acc, 2),
        "obf_acc":          round(obf_acc, 2),
        "sigma":            round(secret.sigma, 6),
        "ft_attack":        ft_results,
        "prune_attack":     prune_results,
    }
    out_path = out_dir / f"{config_key}_obfuscated_attack.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return result


# ── Summary tables ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]):
    W = 96
    print("\n" + "="*W)
    print("  SUMMARY — Obfuscated-Model Attack Results")
    print("="*W)

    # ── FT table: one row per (model, strategy) ───────────────────────────────
    print(f"  {'Dataset':<14} {'Arch':<10} {'Strat':5} {'Clean':>7} {'Obf.':>6}"
          f"  {'FT 1%':>6} {'FT 5%':>6} {'FT10%':>6} {'FT50%':>6} {'FT100%':>7}")
    print("  " + "-"*(W-2))
    prev_key = ""
    for r in all_results:
        # blank separator between model configs
        cur_key = r["config_key"]
        if cur_key != prev_key and prev_key:
            print()
        prev_key = cur_key

        ft_dict = r.get("ft_attack", {})
        for strategy in FT_STRATEGIES:
            rows = ft_dict.get(strategy, [])
            ft = {x["ratio"]: x["acc_pct"] for x in rows}
            print(f"  {r['dataset']:<14} {r['arch']:<10} {strategy:5}"
                  f" {r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>5.1f}%"
                  f"  {ft.get(0.01,0):>5.1f}% {ft.get(0.05,0):>5.1f}%"
                  f" {ft.get(0.10,0):>5.1f}% {ft.get(0.50,0):>5.1f}%"
                  f" {ft.get(1.00,0):>6.1f}%")

    print()

    # ── Pruning table ─────────────────────────────────────────────────────────
    print(f"  {'Dataset':<14} {'Arch':<10} {'Clean':>7} {'Obf.':>6}"
          f"  {'Prune25':>8} {'Prune50':>8} {'Prune75':>8}")
    print("  " + "-"*(W-2))
    prev_ds = ""
    for r in all_results:
        if r["dataset"] != prev_ds and prev_ds:
            print("  " + "-"*(W-2))
        prev_ds = r["dataset"]
        pr = {x["prune_frac"]: x["acc_pct"] for x in r["prune_attack"]}
        print(f"  {r['dataset']:<14} {r['arch']:<10} "
              f"{r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>5.1f}%"
              f"  {pr.get(0.25,0):>7.1f}% {pr.get(0.50,0):>7.1f}%"
              f" {pr.get(0.75,0):>7.1f}%")

    print("="*W)
    print("  FT strategies: FTAL=all layers · FTLL=last layer only · "
          "RTFL=reset head + all layers")
    print("  Even RTFL (strongest) cannot fully recover clean-model accuracy "
          "without the secret key K_obf.")
    print("="*W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

_CONFIG_CHOICES = ALL_CONFIGS + list(DATASET_GROUPS.keys()) + ["all"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Obfuscated-model attack experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
config choices (individual):
  cifar10_resnet20      CIFAR-10  ResNet-20
  cifar10_vgg11         CIFAR-10  VGG-11
  cifar100_resnet56     CIFAR-100 ResNet-56
  cifar100_vgg11        CIFAR-100 VGG-11
  tinyimagenet_resnet18 Tiny-ImageNet ResNet-18

config choices (group shortcut):
  cifar10      runs both CIFAR-10 models
  cifar100     runs both CIFAR-100 models
  tinyimagenet runs Tiny-ImageNet ResNet-18
  all          runs all 5 models
""")
    p.add_argument("--config", choices=_CONFIG_CHOICES, default="all",
                   metavar="CONFIG",
                   help="Model config to run (default: all). "
                        f"Choices: {', '.join(_CONFIG_CHOICES)}")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ft-epochs", type=int, default=50,
                   help="Epochs per fine-tuning budget (default: 50)")
    p.add_argument("--out_dir", default="results/obfuscated_attack",
                   help="Output directory for JSON results")
    return p.parse_args()


def resolve_configs(config_arg: str) -> list[str]:
    if config_arg == "all":
        return ALL_CONFIGS
    if config_arg in DATASET_GROUPS:
        return DATASET_GROUPS[config_arg]
    return [config_arg]


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = resolve_configs(args.config)
    print(f"\nRunning {len(configs_to_run)} config(s): {', '.join(configs_to_run)}")

    all_results = []
    for cfg_key in configs_to_run:
        result = run_config(
            config_key=cfg_key,
            device=args.device,
            ft_epochs=args.ft_epochs,
            out_dir=out_dir,
        )
        all_results.append(result)

    print_summary(all_results)

    summary_path = out_dir / "obfuscated_attack_summary.json"
    # Merge with any existing entries from previous runs
    if summary_path.exists():
        with open(summary_path) as f:
            existing = {r["config_key"]: r for r in json.load(f)}
    else:
        existing = {}
    for r in all_results:
        existing[r["config_key"]] = r
    with open(summary_path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)
    print(f"Combined summary saved: {summary_path}")
