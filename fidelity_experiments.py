#!/usr/bin/env python3
"""
EvalGuard — Fidelity Experiments

Unified fidelity measurement for three watermarking methods.
Measures teacher accuracy at each pipeline stage.

  evalguard  5 stages: Original → Obfuscated → Recovered → WM Soft API → WM Hard API
  dawn       3 stages: Original → WM Soft API → WM Hard API
  adi        2 stages: Original → Backdoored Teacher  (no soft/hard distinction)

Adi note: Adi et al. embeds the watermark directly in the teacher's weights via
  backdoor retraining. It has no "API wrapper" with soft/hard response modes.
  Fidelity for Adi = accuracy drop from backdoor-embedding retraining.

Results saved to: results/fidelity/<dataset>/<method>/<filename>.json

Usage:
  python fidelity_experiments.py --model cifar10_resnet20 --method evalguard \\
      --rw 0.10 --delta_logit 5.0 --beta 0.5 --verify_temperature 10.0

  python fidelity_experiments.py --model cifar10_resnet20 --method dawn --rw 0.10

  python fidelity_experiments.py --model cifar10_resnet20 --method adi \\
      --adi_n_triggers 100 --adi_retrain_epochs 30

  python fidelity_experiments.py --model cifar10_resnet20 --method all
"""

import os
os.environ['TORCH_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evalguard import (
    obfuscate_model_vectorized, recover_weights,
    WatermarkModule, LatentExtractor, recommended_vT,
)
from evalguard.crypto import keygen, kdf
from evalguard.configs import (
    CONFIGS, cifar10_data, cifar100_data, tinyimagenet_data,
)

RESULTS_DIR = Path("results")

# ─────────────────────────────────────────────────────────────────────────────
# Model / data registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cifar10_resnet20":      "cifar10",
    "cifar10_vgg11":         "cifar10",
    "cifar100_resnet20":     "cifar100",
    "cifar100_vgg11":        "cifar100",
    "cifar100_resnet56":     "cifar100",
    "cifar100_wrn2810":      "cifar100",
    "tinyimagenet_resnet18": "tinyimagenet",
}

DATA_FN = {
    "cifar10":      cifar10_data,
    "cifar100":     cifar100_data,
    "tinyimagenet": tinyimagenet_data,
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _delta_str(delta):
    log2 = round(math.log2(delta))
    return "2e{}".format(log2)


def evaluate_accuracy(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
    return correct / total


def save_fidelity_result(data, dataset, method, fname):
    path = RESULTS_DIR / "fidelity" / dataset / method
    path.mkdir(parents=True, exist_ok=True)
    out = path / "{}.json".format(fname)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print("  -> Saved: {}".format(out))


def _print_table(rows, acc_orig):
    print("\n  Summary:")
    print("  +--------------------------+-----------+----------+")
    print("  | Stage                    | Accuracy  | Loss     |")
    print("  +--------------------------+-----------+----------+")
    for label, acc in rows:
        if label == "Original":
            print("  | {:24s} | {:7.4f}%  |    ---   |".format(label, acc * 100))
        else:
            print("  | {:24s} | {:7.4f}%  | {:+.4f}% |".format(
                label, acc * 100, (acc - acc_orig) * 100))
    print("  +--------------------------+-----------+----------+")


def get_model_and_data(model_name, device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError("Unknown model '{}'. Available: {}".format(
            model_name, list(MODEL_REGISTRY.keys())))
    dataset = MODEL_REGISTRY[model_name]
    config = CONFIGS[model_name]
    model, latent_layer = config["model_fn"](pretrained=True)
    model.to(device)
    nc = config["num_classes"]
    trainset, testset, trainloader, testloader = DATA_FN[dataset]()
    short = (model_name.replace("cifar10_", "")
                       .replace("cifar100_", "")
                       .replace("tinyimagenet_", ""))
    return model, latent_layer, trainset, testset, testloader, dataset, nc, short


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: EvalGuard — 5 stages
# ─────────────────────────────────────────────────────────────────────────────

def fidelity_evalguard(model, trainset, testloader, latent_layer, device,
                       teacher_name, dataset, nc,
                       epsilon, delta,
                       rw, delta_logit, beta, delta_min, verify_temperature,
                       margin_tau_hard, hard_tau_quantile, hard_tau_calib_samples,
                       seed=42):
    """
    EvalGuard 5-stage fidelity chain.

    Stages:
      1. Original       — clean pretrained teacher
      2. Obfuscated     — after weight obfuscation (≈ random guess)
      3. Recovered      — after decryption (should match original)
      4. WM Soft API    — teacher queries through EvalGuard soft-label API
                          (logit boost preserves argmax → ≈ 0% loss)
      5. WM Hard API    — teacher queries through EvalGuard BGS hard-label API
                          (only boundary samples swapped → < 1% loss)
    """
    print("\n" + "=" * 60)
    print("Fidelity: EvalGuard  [dataset={}, teacher={}]".format(dataset, teacher_name))
    print("  rw={}, delta_logit={}, beta={}, verify_T={:.2f}".format(
        rw, delta_logit, beta, verify_temperature))
    print("  epsilon={}, delta={}".format(epsilon, _delta_str(delta)))
    print("=" * 60)

    # ── Stage 1: Original ───────────────────────────────────────────────────
    acc_orig = evaluate_accuracy(model, testloader, device)
    print("  [1] Original:      {:.4f}%".format(acc_orig * 100))

    # ── Stage 2: Obfuscated ─────────────────────────────────────────────────
    ms = copy.deepcopy(model)
    ms, sec = obfuscate_model_vectorized(ms, epsilon=epsilon, delta=delta,
                                         model_id="evalguard")
    acc_obf = evaluate_accuracy(ms, testloader, device)
    print("  [2] Obfuscated:    {:.4f}%  [random guess = {:.2f}%]".format(
        acc_obf * 100, 100.0 / nc))

    # ── Stage 3: Recovered ──────────────────────────────────────────────────
    recover_weights(ms, sec, vectorized=True)
    acc_rec = evaluate_accuracy(ms, testloader, device)
    print("  [3] Recovered:     {:.4f}%  [loss = {:.6f}%]".format(
        acc_rec * 100, (acc_orig - acc_rec) * 100))

    # ── Stage 4: WM Soft API ────────────────────────────────────────────────
    Kw = kdf(keygen(256, seed=seed), "watermark")
    calib_sl = DataLoader(
        Subset(trainset, list(range(min(2000, len(trainset))))),
        batch_size=64, shuffle=False, num_workers=2)
    ext = LatentExtractor()
    ext.compute_median(model, calib_sl, latent_layer, device)

    wm_soft = WatermarkModule(
        K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
        delta_min=delta_min, num_classes=nc,
        latent_extractor=ext, layer_name=latent_layer,
        suppress_warnings=True,
    )
    model.to(device).eval()
    correct_soft, total_soft, n_wm_soft = 0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs_np, n_wm = wm_soft.embed_batch_logits(
                model, x, logits.cpu().numpy(), verify_temperature, device)
            n_wm_soft += n_wm
            preds = np.argmax(probs_np, axis=1)
            correct_soft += (preds == y.cpu().numpy()).sum()
            total_soft += y.size(0)
    acc_soft = correct_soft / total_soft
    print("  [4] WM Soft API:   {:.4f}%  [loss = {:.6f}%, wm_queries={}]".format(
        acc_soft * 100, (acc_orig - acc_soft) * 100, n_wm_soft))

    # ── Stage 5: WM Hard API (BGS) ──────────────────────────────────────────
    wm_hard = WatermarkModule(
        K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
        delta_min=delta_min, num_classes=nc,
        latent_extractor=ext, layer_name=latent_layer,
        margin_tau_hard=margin_tau_hard, suppress_warnings=True,
    )
    adaptive_tau = margin_tau_hard
    if hard_tau_quantile > 0:
        calib_hard = DataLoader(
            Subset(trainset, list(range(min(hard_tau_calib_samples, len(trainset))))),
            batch_size=64, shuffle=False, num_workers=2)
        margins = []
        model.to(device).eval()
        with torch.no_grad():
            for cx, _ in calib_hard:
                cx = cx.to(device)
                clogits = model(cx).cpu().numpy()
                for i in range(cx.size(0)):
                    r_x = ext.extract_and_binarize_batch(
                        model, cx[i:i+1], latent_layer, device)
                    if wm_hard.watermark_decision(r_x[0]):
                        top1 = int(np.argmax(clogits[i]))
                        tgt = wm_hard._get_target_class(top1)
                        m = float(clogits[i][top1] - clogits[i][tgt])
                        if m > 0:
                            margins.append(m)
        if margins:
            adaptive_tau = float(np.quantile(margins, hard_tau_quantile))
            print("  [BGS] Adaptive tau = {:.4f}  (q={}, n_phi1={})".format(
                adaptive_tau, hard_tau_quantile, len(margins)))

    correct_hard, total_hard = 0, 0
    model.to(device).eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits_np = model(x).cpu().numpy()
            for i in range(x.size(0)):
                pred = wm_hard.serve_hard_label(
                    model, x[i], logits_np[i],
                    margin_tau=adaptive_tau, device=device)
                correct_hard += int(pred == y[i].item())
                total_hard += 1
    n_swapped = wm_hard._hardlabel_n_swapped
    acc_hard = correct_hard / total_hard
    print("  [5] WM Hard API:   {:.4f}%  [loss = {:.6f}%, swapped={}]".format(
        acc_hard * 100, (acc_orig - acc_hard) * 100, n_swapped))

    _print_table([
        ("Original",          acc_orig),
        ("Obfuscated",        acc_obf),
        ("Recovered",         acc_rec),
        ("WM Soft API",       acc_soft),
        ("WM Hard API (BGS)", acc_hard),
    ], acc_orig)

    fname = "fidelity_evalguard__{}__{}_rw{}_d{}_b{}".format(
        dataset, teacher_name, rw, delta_logit, beta)
    save_fidelity_result({
        "experiment": "fidelity",
        "method": "EvalGuard",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "teacher": teacher_name,
        "parameters": {
            "epsilon": epsilon,
            "delta": str(delta), "delta_str": _delta_str(delta),
            "rw": rw, "delta_logit": delta_logit, "beta": beta,
            "delta_min": delta_min, "verify_temperature": verify_temperature,
            "hard_tau_quantile": hard_tau_quantile,
            "adaptive_tau": round(adaptive_tau, 6),
        },
        "results": {
            "acc_original":    round(acc_orig, 6),
            "acc_obfuscated":  round(acc_obf, 6),
            "acc_recovered":   round(acc_rec, 6),
            "acc_wm_soft":     round(acc_soft, 6),
            "acc_wm_hard":     round(acc_hard, 6),
            "loss_obfuscated": round(acc_orig - acc_obf, 6),
            "loss_recovered":  round(acc_orig - acc_rec, 6),
            "loss_wm_soft":    round(acc_orig - acc_soft, 6),
            "loss_wm_hard":    round(acc_orig - acc_hard, 6),
            "n_wm_soft_queries": int(n_wm_soft),
            "n_hard_swapped":    int(n_swapped),
            "num_classes": nc,
        },
        "stages": [
            {"name": "Original",          "accuracy": round(acc_orig, 6)},
            {"name": "Obfuscated",        "accuracy": round(acc_obf, 6)},
            {"name": "Recovered",         "accuracy": round(acc_rec, 6)},
            {"name": "WM Soft API",       "accuracy": round(acc_soft, 6)},
            {"name": "WM Hard API (BGS)", "accuracy": round(acc_hard, 6)},
        ],
    }, dataset, "EvalGuard", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: DAWN — 3 stages
# ─────────────────────────────────────────────────────────────────────────────

def fidelity_dawn(model, testloader, device,
                  teacher_name, dataset, nc, rw, seed=42):
    """
    DAWN fidelity: 3 stages.

    DAWN is a pure API wrapper — it does NOT modify the teacher's weights.
    For r_w fraction of queries the API returns a deliberately wrong label.

    Fidelity impact: when querying the full test set through DAWN API,
    ≈ r_w fraction of responses are incorrect → accuracy drops by ~r_w × (1 - 1/C).

    Soft API: returns soft probabilities with top-1 swapped for trigger inputs.
    Hard API: returns argmax with top-1 swapped for trigger inputs.

    There is no Obfuscated / Recovered stage for DAWN (no model encryption).
    """
    from evalguard.baselines import (
        DAWNWatermark, dawn_collect_soft_labels, dawn_collect_hard_labels,
    )

    print("\n" + "=" * 60)
    print("Fidelity: DAWN  [dataset={}, teacher={}]".format(dataset, teacher_name))
    print("  rw={:.0f}% of API responses are flipped (fidelity cost)".format(rw * 100))
    print("  No model obfuscation/recovery — DAWN is an API-level wrapper only.")
    print("=" * 60)

    # ── Stage 1: Original ───────────────────────────────────────────────────
    acc_orig = evaluate_accuracy(model, testloader, device)
    print("  [1] Original:         {:.4f}%".format(acc_orig * 100))

    K_dawn = kdf(keygen(256, seed=seed), "dawn_watermark")
    T = 5

    # Use testset directly (shuffle=False to keep label alignment)
    test_dl = DataLoader(testloader.dataset, batch_size=64,
                         shuffle=False, num_workers=2)
    true_labels = np.array([int(y) for _, y in testloader.dataset])

    # ── Stage 2: DAWN WM Soft API ───────────────────────────────────────────
    inp_s, soft_labels, dawn_wm_soft = dawn_collect_soft_labels(
        model, test_dl, K_dawn, rw, T, device)
    if hasattr(soft_labels, "argmax"):
        preds_soft = soft_labels.argmax(dim=1).numpy()
    else:
        preds_soft = np.array(soft_labels).argmax(axis=1)
    acc_soft = float((preds_soft == true_labels).mean())
    n_trig_soft = len(getattr(dawn_wm_soft, "trigger_set", []))
    print("  [2] DAWN Soft API:    {:.4f}%  [loss = {:.6f}%, n_triggers={}]".format(
        acc_soft * 100, (acc_orig - acc_soft) * 100, n_trig_soft))

    # ── Stage 3: DAWN WM Hard API ───────────────────────────────────────────
    inp_h, hard_labels, dawn_wm_hard = dawn_collect_hard_labels(
        model, test_dl, K_dawn, rw, nc, device)
    if hasattr(hard_labels, "numpy"):
        preds_hard = hard_labels.numpy()
    else:
        preds_hard = np.array(hard_labels)
    acc_hard = float((preds_hard == true_labels).mean())
    n_trig_hard = len(getattr(dawn_wm_hard, "trigger_set", []))
    print("  [3] DAWN Hard API:    {:.4f}%  [loss = {:.6f}%, n_triggers={}]".format(
        acc_hard * 100, (acc_orig - acc_hard) * 100, n_trig_hard))

    _print_table([
        ("Original",        acc_orig),
        ("DAWN WM Soft API", acc_soft),
        ("DAWN WM Hard API", acc_hard),
    ], acc_orig)

    fname = "fidelity_dawn__{}__{}_rw{}".format(dataset, teacher_name, rw)
    save_fidelity_result({
        "experiment": "fidelity",
        "method": "DAWN",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "teacher": teacher_name,
        "note": ("DAWN is an API wrapper only — no model obfuscation. "
                 "Fidelity cost = flipped labels for ~r_w fraction of queries."),
        "parameters": {"rw": rw, "T": T},
        "results": {
            "acc_original":    round(acc_orig, 6),
            "acc_dawn_soft":   round(acc_soft, 6),
            "acc_dawn_hard":   round(acc_hard, 6),
            "loss_dawn_soft":  round(acc_orig - acc_soft, 6),
            "loss_dawn_hard":  round(acc_orig - acc_hard, 6),
            "n_triggers_soft": int(n_trig_soft),
            "n_triggers_hard": int(n_trig_hard),
            "num_classes": nc,
        },
        "stages": [
            {"name": "Original",         "accuracy": round(acc_orig, 6)},
            {"name": "DAWN WM Soft API", "accuracy": round(acc_soft, 6)},
            {"name": "DAWN WM Hard API", "accuracy": round(acc_hard, 6)},
        ],
    }, dataset, "DAWN", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Adi et al. — 2 stages  (no soft / hard API distinction)
# ─────────────────────────────────────────────────────────────────────────────

def fidelity_adi(model, trainset, testloader, device,
                 teacher_name, dataset, nc,
                 adi_n_triggers=100, adi_retrain_epochs=30, adi_lr=0.001,
                 seed=42):
    """
    Adi et al. (2018) fidelity: 2 stages.

    Adi embeds a backdoor INTO the teacher model weights via retraining with
    trigger samples.  Unlike EvalGuard (encrypt + API shift) or DAWN (API flip),
    Adi permanently modifies the model — there is no separate "soft" vs "hard"
    API serving mode.

    Fidelity impact = accuracy drop from the backdoor-embedding retraining step.
    In practice this is usually < 1% but depends on the number of triggers and
    retraining epochs.

    For a fair comparison, include the note in the paper that Adi's fidelity is
    measured as the change in teacher clean-test accuracy before/after backdoor
    embedding, not as an API-serving overhead.
    """
    from evalguard.baselines import AdiWatermark

    img_shape = (3, 64, 64) if dataset == "tinyimagenet" else (3, 32, 32)

    print("\n" + "=" * 60)
    print("Fidelity: Adi et al. (2018)  [dataset={}, teacher={}]".format(
        dataset, teacher_name))
    print("  n_triggers={}, retrain_epochs={}, lr={}".format(
        adi_n_triggers, adi_retrain_epochs, adi_lr))
    print("  Note: Adi has NO soft/hard API distinction.")
    print("        Fidelity = accuracy drop from backdoor retraining.")
    print("=" * 60)

    # ── Stage 1: Original ───────────────────────────────────────────────────
    acc_orig = evaluate_accuracy(model, testloader, device)
    print("  [1] Original:              {:.4f}%".format(acc_orig * 100))

    # ── Stage 2: Backdoored Teacher ─────────────────────────────────────────
    adi = AdiWatermark(num_classes=nc, n_triggers=adi_n_triggers,
                       img_shape=img_shape, seed=seed or 42)
    print("  Retraining teacher with {} backdoor triggers for {} epochs ...".format(
        adi_n_triggers, adi_retrain_epochs))
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    teacher_bd = adi.embed_backdoor(
        model, trainloader,
        epochs=adi_retrain_epochs, lr=adi_lr, device=device)

    acc_bd = evaluate_accuracy(teacher_bd, testloader, device)
    bd_success = adi.verify_on_teacher(teacher_bd, device)
    print("  [2] Backdoored Teacher:    {:.4f}%  "
          "[loss = {:.6f}%, backdoor_success = {:.1f}%]".format(
              acc_bd * 100, (acc_orig - acc_bd) * 100, bd_success * 100))

    _print_table([
        ("Original",               acc_orig),
        ("Adi Backdoored Teacher", acc_bd),
    ], acc_orig)

    print("\n  Comparison guide:")
    print("   EvalGuard WM Soft API loss  ≈ 0%    (logit boost preserves argmax)")
    print("   EvalGuard WM Hard API loss  < 1%    (BGS swaps only boundary samples)")
    print("   DAWN WM Soft/Hard API loss  ≈ r_w   (r_w fraction of responses flipped)")
    print("   Adi backdoor loss           ≈ 0–1%  (retraining overhead)")

    fname = "fidelity_adi__{}__{}_ntrig{}".format(
        dataset, teacher_name, adi_n_triggers)
    save_fidelity_result({
        "experiment": "fidelity",
        "method": "Adi_et_al_2018",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "teacher": teacher_name,
        "note": ("Adi embeds backdoor in model weights via retraining. "
                 "No soft/hard API mode — compare with EvalGuard/DAWN WM API stages. "
                 "Fidelity = accuracy drop from teacher retraining."),
        "parameters": {
            "n_triggers": adi_n_triggers,
            "retrain_epochs": adi_retrain_epochs,
            "lr": adi_lr,
            "img_shape": list(img_shape),
        },
        "results": {
            "acc_original":          round(acc_orig, 6),
            "acc_backdoored":        round(acc_bd, 6),
            "loss_backdoored":       round(acc_orig - acc_bd, 6),
            "backdoor_success_rate": round(bd_success, 6),
            "num_classes": nc,
        },
        "stages": [
            {"name": "Original",               "accuracy": round(acc_orig, 6)},
            {"name": "Adi Backdoored Teacher", "accuracy": round(acc_bd, 6)},
        ],
    }, dataset, "Adi", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="EvalGuard Fidelity Experiments  (EvalGuard / DAWN / Adi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    pa.add_argument("--model", default="cifar10_resnet20",
                    choices=list(MODEL_REGISTRY.keys()),
                    help="Teacher model key")
    pa.add_argument("--method", default="evalguard",
                    choices=["evalguard", "dawn", "adi", "all"],
                    help="Watermarking method to evaluate. 'all' runs all three.")
    pa.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--seed", type=int, default=42)

    # EvalGuard / obfuscation parameters
    pa.add_argument("--epsilon", type=float, default=50.0)
    pa.add_argument("--delta", type=float, default=2 ** (-32))
    pa.add_argument("--rw", type=float, default=0.10,
                    help="Watermark rate (also used by DAWN)")
    pa.add_argument("--delta_logit", type=float, default=5.0,
                    help="EvalGuard logit shift magnitude")
    pa.add_argument("--beta", type=float, default=0.5,
                    help="EvalGuard safety factor")
    pa.add_argument("--delta_min", type=float, default=0.5)
    pa.add_argument("--verify_temperature", type=float, default=None,
                    help="EvalGuard verify temperature (auto-scaled if omitted)")
    pa.add_argument("--margin_tau_hard", type=float, default=1.5,
                    help="EvalGuard BGS hard-label tau fallback")
    pa.add_argument("--hard_tau_quantile", type=float, default=0.10,
                    help="EvalGuard BGS adaptive tau quantile")
    pa.add_argument("--hard_tau_calib_samples", type=int, default=3000)

    # Adi-specific parameters
    pa.add_argument("--adi_n_triggers", type=int, default=100)
    pa.add_argument("--adi_retrain_epochs", type=int, default=30)
    pa.add_argument("--adi_lr", type=float, default=0.001)

    a = pa.parse_args()

    model, latent_layer, trainset, testset, testloader, dataset, nc, teacher_short = \
        get_model_and_data(a.model, a.device)

    if a.verify_temperature is None or a.verify_temperature <= 0:
        a.verify_temperature = recommended_vT(nc)
        print("[auto] verify_temperature = {:.2f}  (C={})".format(
            a.verify_temperature, nc))

    methods = ["evalguard", "dawn", "adi"] if a.method == "all" else [a.method]

    for method in methods:
        if method == "evalguard":
            fidelity_evalguard(
                copy.deepcopy(model), trainset, testloader, latent_layer, a.device,
                teacher_short, dataset, nc,
                a.epsilon, a.delta,
                a.rw, a.delta_logit, a.beta, a.delta_min, a.verify_temperature,
                a.margin_tau_hard, a.hard_tau_quantile, a.hard_tau_calib_samples,
                seed=a.seed)

        elif method == "dawn":
            fidelity_dawn(
                copy.deepcopy(model), testloader, a.device,
                teacher_short, dataset, nc,
                a.rw, seed=a.seed)

        elif method == "adi":
            fidelity_adi(
                copy.deepcopy(model), trainset, testloader, a.device,
                teacher_short, dataset, nc,
                a.adi_n_triggers, a.adi_retrain_epochs, a.adi_lr,
                seed=a.seed)


if __name__ == "__main__":
    main()