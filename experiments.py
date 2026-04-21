"""
EvalGuard — Experiment Script (Section VI) [v5]

[v5] Logit-Space Confidence Shift Watermarking:
  - Embedding: target-class logit boost BEFORE softmax (T-invariant)
  - Verification: Mann-Whitney U test with configurable verify_temperature
  - Parameters: delta_logit (logit shift), beta (safety factor)
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import time
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from evalguard import (
    obfuscate_model_vectorized, recover_weights,
    WatermarkModule, LatentExtractor, verify_ownership,
    verify_ownership_all_designs, verify_ownership_own_data_all_designs,
    verify_ownership_hard_label, random_kw_false_positive_scan,
    recommended_vT,
)
from evalguard.watermark import (
    verify_ownership_own_data, reconstruct_triggers_from_own_data,
)
from evalguard.crypto import keygen, kdf
from evalguard.configs import (
    CONFIGS, cifar10_data, cifar100_data, svhn_data, tinyimagenet_data,
    create_student,
)
from evalguard.attacks import (
    collect_soft_labels, soft_label_distillation,
    collect_hard_labels, hard_label_extraction,
    fine_tune_surrogate,
    prune_surrogate, quantize_surrogate_int8,
)

import numpy as np

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")


# ============================================================
# Dataset detection
# ============================================================

MODEL_REGISTRY = {
    "cifar10_resnet20":      ("cifar10_resnet20",      "cifar10"),
    "cifar10_vgg11":         ("cifar10_vgg11",         "cifar10"),
    "cifar100_resnet20":     ("cifar100_resnet20",     "cifar100"),
    "cifar100_vgg11":        ("cifar100_vgg11",        "cifar100"),
    "cifar100_resnet56":     ("cifar100_resnet56",     "cifar100"),
    "cifar100_wrn2810":      ("cifar100_wrn2810",      "cifar100"),
    "tinyimagenet_resnet18": ("tinyimagenet_resnet18", "tinyimagenet"),
    "resnet50":              ("imagenet_resnet50",     "imagenet"),
}

DATA_FN = {
    "cifar10":      cifar10_data,
    "cifar100":     cifar100_data,
    "tinyimagenet": tinyimagenet_data,
}


def _build_query_loader(query_dataset, native_dataset, n_queries, batch_size=64):
    """
    Build the query dataloader used to probe the teacher.

    query_dataset == "native": use the owner's original CIFAR trainset (default).
    query_dataset == "svhn":   use SVHN images (32x32, no resize), normalised
                               using the CIFAR statistics of the owner model.
    Returns (query_loader, pool_size, src_trainset_for_holding).
    """
    if query_dataset in (None, "native"):
        ds = native_dataset
    elif query_dataset == "svhn":
        # Normalize for the owner dataset so teacher sees its training stats.
        if native_dataset is not None and hasattr(native_dataset, "root"):
            # native_dataset is the CIFAR10/100 trainset; derive normalisation
            # target from its class name.
            cname = native_dataset.__class__.__name__.lower()
            if "100" in cname:
                ds, _, _, _ = svhn_data(normalize_for="cifar100")
            else:
                ds, _, _, _ = svhn_data(normalize_for="cifar10")
        else:
            ds, _, _, _ = svhn_data(normalize_for="cifar10")
    else:
        raise ValueError("Unknown query_dataset: {}".format(query_dataset))

    nq = min(n_queries, len(ds))
    loader = DataLoader(Subset(ds, list(range(nq))),
                        batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, nq, ds


def resolve_model(name, dataset_override=None):
    if name not in MODEL_REGISTRY:
        raise ValueError("Unknown model: {}. Available: {}".format(
            name, list(MODEL_REGISTRY.keys())))
    config_key, ds = MODEL_REGISTRY[name]
    if dataset_override:
        ds = dataset_override
    short = name.replace("cifar10_", "").replace("cifar100_", "").replace("imagenet_", "")
    return config_key, ds, short


# ============================================================
# Naming helpers
# ============================================================

def _delta_str(delta):
    log2 = round(math.log2(delta))
    return "2e{}".format(log2)

def _cl_str(class_level):
    return "__CL" if class_level else ""

def stem_fidelity(teacher, eps, delta):
    return "T_{}__eps{}__delta{}".format(teacher, eps, _delta_str(delta))

def stem_finetune(teacher, eps, delta, ft_epochs):
    return "T_{}__eps{}__delta{}__ftEp{}".format(teacher, eps, _delta_str(delta), ft_epochs)

def stem_distill(teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit=2.0, vT=5.0, tag=""):
    base = "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__d{}__vT{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit, vT)
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base

def stem_surrogate_ft(teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, delta_logit=2.0, vT=5.0, trigger_mode="rec_trigger", trigger_size=0, tag=""):
    base = "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__ftEp{}__ftLr{}__d{}__vT{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, delta_logit, vT)
    if trigger_mode == "own_trigger":
        base = "{}__trig_own{}".format(base, trigger_size if trigger_size > 0 else "all")
    else:
        base = "{}__trig_rec{}".format(base, trigger_size if trigger_size > 0 else "all")
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base

def stem_overhead(teacher, eps):
    return "T_{}__eps{}".format(teacher, eps)

def stem_ckpt(teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit=2.0, tag=""):
    """Checkpoint stem: no verify_temperature (checkpoints are T-independent).

    Prefix 'v6_' marks the checkpoint format version that includes the
    persisted watermark key K_w (.key file). Caches without the key file
    will be ignored on load (auto-invalidating older v5 caches).
    """
    base = "v6__T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__d{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit)
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base


# ============================================================
# Save / Load
# ============================================================

def save_result(data, dataset, subdir, filename, teacher_name=None, student_arch=None, label_mode=None, method=None):
    """
    Save result JSON.

    When `method` is provided (e.g. "DAWN", "Adi", "EvalGuard"), saves to:
      results/baseline/<dataset>/<method>/<filename>.json

    Otherwise uses the standard structure:
      results/<subdir>/<dataset>/<teacher>_<student>/<label_mode>/<filename>.json
    """
    if method is not None:
        path = RESULTS_DIR / "baseline" / dataset / method / "{}.json".format(filename)
    elif teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        path = RESULTS_DIR / subdir / dataset / model_dir / label_mode / "{}.json".format(filename)
    elif teacher_name and label_mode:
        path = RESULTS_DIR / subdir / dataset / teacher_name / label_mode / "{}.json".format(filename)
    else:
        path = RESULTS_DIR / subdir / dataset / "{}.json".format(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print("  -> Saved: {}".format(path))


def save_checkpoint(model, triggers, K_w, dataset, stem, meta=None, subdir="distill",
                    teacher_name=None, student_arch=None, label_mode=None):
    """
    Save checkpoint with directory structure:
      checkpoints/<subdir>/<dataset>/<teacher>_<student>/<label_mode>/<stem>.{pt,pkl,key,meta.json}

    K_w is persisted as a separate .key file (raw bytes). Without this file,
    own_trigger verification cannot reproduce the target-class mapping.

    Falls back to old structure if teacher_name/student_arch/label_mode not provided.
    """
    if teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        ckpt_dir = CKPT_DIR / subdir / dataset / model_dir / label_mode
    else:
        ckpt_dir = CKPT_DIR / subdir / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "{}.pt".format(stem))
    with open(ckpt_dir / "{}.pkl".format(stem), "wb") as f:
        pickle.dump(triggers, f)
    # Persist watermark key (raw bytes)
    with open(ckpt_dir / "{}.key".format(stem), "wb") as f:
        f.write(K_w)
    if meta:
        # Embed key hex in meta for human inspection (file is the source of truth)
        meta = dict(meta)
        meta["K_w_hex"] = K_w.hex()
        meta["checkpoint_version"] = "v6"
        with open(ckpt_dir / "{}.meta.json".format(stem), "w") as f:
            json.dump(meta, f, indent=2, default=str)
    print("  -> Cached: {}/{}.pt (+ .pkl, .key)".format(ckpt_dir, stem))


def load_checkpoint(dataset, stem, num_classes, student_arch, subdir="distill",
                    teacher_name=None, label_mode=None):
    """
    Load checkpoint. Returns (model, triggers, K_w) or (None, None, None) if
    any required artifact (.pt / .pkl / .key) is missing.

    The .key file is REQUIRED — caches written before v6 are silently
    invalidated to prevent verification with the wrong watermark key.
    """
    if teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        ckpt_dir = CKPT_DIR / subdir / dataset / model_dir / label_mode
    else:
        ckpt_dir = CKPT_DIR / subdir / dataset
    pt_path = ckpt_dir / "{}.pt".format(stem)
    pkl_path = ckpt_dir / "{}.pkl".format(stem)
    key_path = ckpt_dir / "{}.key".format(stem)
    if not (pt_path.exists() and pkl_path.exists() and key_path.exists()):
        return None, None, None
    model = create_student(num_classes=num_classes, arch=student_arch)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    with open(pkl_path, "rb") as f:
        # Custom unpickler to remap CUDA tensors to CPU
        import pickle as _pickle
        class CPUUnpickler(_pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                return super().find_class(module, name)
        triggers = CPUUnpickler(f).load()
    with open(key_path, "rb") as f:
        K_w = f.read()
    # Ensure all trigger queries are on CPU
    for entry in triggers:
        if hasattr(entry, 'query') and isinstance(entry.query, torch.Tensor):
            entry.query = entry.query.cpu()
    print("  -> Loaded cache: {} (K_w={}...)".format(pt_path, K_w.hex()[:12]))
    return model, triggers, K_w


# ============================================================
# Helpers
# ============================================================

def get_model(config_key, pretrained_path=None):
    config = CONFIGS[config_key]
    if pretrained_path:
        model, ll = config["model_fn"](pretrained=False)
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    else:
        model, ll = config["model_fn"](pretrained=True)
    return model, ll, config


def train_model(model, loader, epochs=50, lr=0.1, device="cpu"):
    model.to(device).train()
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        sch.step()
        if (ep + 1) % 10 == 0:
            print("    Epoch {}/{}, Loss: {:.4f}".format(ep+1, epochs, loss_sum/len(loader)))
    return model


def evaluate_accuracy(model, loader, device="cpu"):
    model.to(device).eval()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += model(x).max(1)[1].eq(y).sum().item()
            t += y.size(0)
    return c / t


def model_info(model, name):
    n = sum(p.numel() for p in model.parameters())
    return {"name": name, "num_parameters": n, "num_parameters_human": "{:,}".format(n)}


# ============================================================
# Verification helpers (v5: logit-space confidence shift)
# ============================================================

def verify_and_format(triggers, model, num_classes, K_w, eta, device,
                      control_queries=None, control_top1=None,
                      verify_temperature=5.0):
    """
    Run confidence shift verification and format results.

    Always runs the THREE paired control designs (single_ctrl, mean_rest,
    suspect_top1) in a single forward pass; the top-level summary fields
    report 'single_ctrl' for backward compatibility, and the full diagnostic
    triple is attached as 'all_designs' for post-hoc analysis.

    The independent-control-queries mode (control_queries != None) still
    delegates to verify_ownership for the Mann-Whitney U branch.
    """
    if len(triggers) == 0:
        return {"confidence_shift": 0.0, "n_trigger": 0, "n_control": 0,
                "p_value": 1.0, "verified": False, "all_designs": {}}

    # Independent-mode (Mann-Whitney U) — rarely used, keep original path.
    if control_queries is not None:
        vr = verify_ownership(
            triggers, model,
            control_queries=control_queries,
            control_top1_classes=control_top1,
            K_w=K_w,
            num_classes=num_classes,
            eta=eta,
            device=device,
            verify_temperature=verify_temperature,
        )
        return {
            "confidence_shift": vr["confidence_shift"],
            "mean_trigger_conf": vr["mean_trigger_conf"],
            "mean_control_conf": vr["mean_control_conf"],
            "n_trigger": vr["n_trigger"],
            "n_control": vr["n_control"],
            "test": vr.get("test", "mann_whitney_u"),
            "statistic": vr.get("statistic", 0.0),
            "p_value": vr["p_value"],
            "log10_p_value": round(math.log10(max(vr["p_value"], 1e-300)), 2),
            "verified": vr["verified"],
            "verify_temperature": verify_temperature,
            "trigger_source": "rec_trigger",
            "all_designs": {},
        }

    # Paired mode — run all three control designs on the same forward pass.
    all_d = verify_ownership_all_designs(
        triggers, model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    primary = all_d["single_ctrl"]
    return {
        "confidence_shift": primary["confidence_shift"],
        "mean_trigger_conf": primary["mean_trigger_conf"],
        "mean_control_conf": primary["mean_control_conf"],
        "n_trigger": primary["n_trigger"],
        "n_control": primary["n_control"],
        "test": primary.get("test", "wilcoxon_signed_rank"),
        "statistic": primary.get("statistic", 0.0),
        "p_value": primary["p_value"],
        "log10_p_value": primary.get(
            "log10_p_value",
            round(math.log10(max(primary["p_value"], 1e-300)), 2)),
        "verified": primary["verified"],
        "verify_temperature": verify_temperature,
        "trigger_source": "rec_trigger",
        "all_designs": all_d,
    }


def verify_own_data_and_format(
    owner_model, own_loader, suspect_model,
    K_w, r_w, num_classes, latent_extractor, layer_name,
    delta_logit, beta, eta, device, verify_temperature=5.0,
    max_triggers=0,
):
    """
    Own-data verification: reconstruct triggers from Owner's data,
    then verify against suspect model. Zero D_eval leakage (if the
    own_loader is a held-out dataset disjoint from D_eval).

    Runs the three paired control designs on a single forward pass.
    Top-level summary reports single_ctrl; full diagnostic triple in
    'all_designs'. When latent_extractor is provided and r_w > 0, the
    Phi(x) filter is applied to keep only watermarked queries.
    """
    all_d = verify_ownership_own_data_all_designs(
        owner_model, own_loader, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
        max_triggers=max_triggers,
        r_w=r_w, latent_extractor=latent_extractor, layer_name=layer_name,
        delta_logit=delta_logit, beta=beta,
    )
    primary = all_d["single_ctrl"]
    return {
        "confidence_shift": primary["confidence_shift"],
        "mean_trigger_conf": primary["mean_trigger_conf"],
        "mean_control_conf": primary["mean_control_conf"],
        "n_trigger": primary["n_trigger"],
        "n_control": primary["n_control"],
        "test": primary.get("test", "wilcoxon_signed_rank"),
        "statistic": primary.get("statistic", 0.0),
        "p_value": primary["p_value"],
        "log10_p_value": primary.get(
            "log10_p_value",
            round(math.log10(max(primary["p_value"], 1e-300)), 2)),
        "verified": primary["verified"],
        "verify_temperature": verify_temperature,
        "trigger_source": "own_trigger",
        "n_own_data_scanned": primary.get("n_own_data_scanned", 0),
        "all_designs": all_d,
    }


def verify_hardlabel_and_format(triggers, suspect_model, num_classes, eta, device,
                                batch_size=64):
    """
    Hard-label verification (Boundary-Gated Hard-Label Swap companion).

    Wraps `verify_ownership_hard_label` (one-sided exact binomial test against
    baseline 1/C) and shapes the dict to mirror `verify_and_format` so the same
    downstream save_result / pretty-print code paths work uniformly.
    """
    if len(triggers) == 0:
        return {"confidence_shift": 0.0, "n_trigger": 0, "n_control": 0,
                "p_value": 1.0, "verified": False, "all_designs": {},
                "test": "binomial_hard_label", "label_mode": "hard",
                "match_rate": 0.0, "n_matches": 0, "trigger_source": "rec_trigger"}
    vr = verify_ownership_hard_label(
        triggers, suspect_model, num_classes,
        eta=eta, device=device, batch_size=batch_size,
    )
    return {
        # Mimic confidence_shift fields by reusing match-rate gap
        "confidence_shift": round(vr["match_rate"] - vr["baseline_rate"], 6),
        "match_rate": vr["match_rate"],
        "baseline_rate": vr["baseline_rate"],
        "n_matches": vr["n_matches"],
        "n_trigger": vr["n_trigger"],
        "n_control": 0,
        "test": vr["test"],
        "statistic": vr["statistic"],
        "p_value": vr["p_value"],
        "log10_p_value": round(math.log10(max(vr["p_value"], 1e-300)), 2),
        "verified": vr["verified"],
        "label_mode": "hard",
        "trigger_source": "rec_trigger",
        "all_designs": {},
    }


def watermark_config_dict(rw, delta_logit, beta, num_classes, verify_temperature=5.0, delta_min=None):
    cfg = {
        "r_w": rw, "r_w_percent": "{}%".format(rw * 100),
        "delta_logit": delta_logit,
        "beta": beta,
        "num_classes": num_classes,
        "method": "logit_space_confidence_shift",
        "verification": "wilcoxon_signed_rank_paired",
        "verify_temperature": verify_temperature,
    }
    if delta_min is not None:
        cfg["delta_min"] = delta_min
    return cfg


# ============================================================
# Table IV: Fidelity
# ============================================================

def experiment_fidelity(model, testloader, device, epsilon, delta,
                        teacher_name, dataset, config):
    print("\n" + "=" * 60)
    print("Table IV: Obfuscation & Fidelity")
    print("  Dataset: {}, Teacher: {}, eps={}, delta={}".format(
        dataset, teacher_name, epsilon, _delta_str(delta)))
    print("=" * 60)

    acc_orig = evaluate_accuracy(model, testloader, device)
    ms = copy.deepcopy(model)
    ms, sec = obfuscate_model_vectorized(ms, epsilon=epsilon, delta=delta, model_id="evalguard")
    acc_obf = evaluate_accuracy(ms, testloader, device)
    recover_weights(ms, sec, vectorized=True)
    acc_rec = evaluate_accuracy(ms, testloader, device)

    print("  Orig={:.2f}%, Obf={:.2f}%, Rec={:.2f}%, Loss={:.4f}%".format(
        acc_orig*100, acc_obf*100, acc_rec*100, (acc_orig-acc_rec)*100))

    fname = stem_fidelity(teacher_name, epsilon, delta)
    save_result({
        "experiment": "fidelity", "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model_info(model, teacher_name),
        "parameters": {"epsilon": epsilon, "delta": str(delta), "delta_str": _delta_str(delta)},
        "results": {
            "acc_original": round(acc_orig, 6),
            "acc_obfuscated": round(acc_obf, 6),
            "acc_recovered": round(acc_rec, 6),
            "fidelity_loss": round(acc_orig - acc_rec, 6),
            "num_classes": config["num_classes"],
            "random_guess": config["random_guess"],
        },
    }, dataset, "fidelity", fname)


# ============================================================
# Table VI: Distillation + Watermark Verification
# ============================================================

def experiment_distillation(model, trainset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            delta_logit=2.0, beta=0.4, delta_min=0.5,
                            verify_temperature=5.0,
                            label_mode="soft", tag="",
                            seed=None,
                            hard_label_mode="bgs", margin_tau_hard=1.5,
                            query_dataset="native",
                            hard_tau_quantile=-1.0,
                            hard_tau_calib_samples=3000):
    """
    Table VI: Distillation + Confidence Shift Verification.

    Args:
        label_mode: "soft" for KL-divergence soft-label distillation (E2s),
                    "hard" for cross-entropy hard-label extraction (E2h).
        query_dataset: "native" (owner's trainset, default) or "svhn"
                       (cross-dataset query attack — the attacker only has
                       unrelated images to probe the teacher with).
        hard_tau_quantile: if > 0, BGS runs a pre-scan and uses the q-quantile
                           of the margin distribution as τ (overrides
                           margin_tau_hard). Typical q=0.10 gives ~r_w*q fraction
                           of queries as triggers.
        hard_tau_calib_samples: how many queries the pre-scan looks at.
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("Table VI: Distillation + Confidence Shift Verification [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, nq={}, dist_epochs={}, dist_lr={}, delta_logit={}, beta={}".format(
        rw, n_queries, dist_epochs, dist_lr, delta_logit, beta))
    print("  Label mode: {}".format(label_mode))
    if label_mode == "soft":
        print("  Temperatures: {}".format(temperatures))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                    batch_size=64, shuffle=False, num_workers=2)
    ext = LatentExtractor()
    ext.compute_median(teacher, sl, latent_layer, device)
    # Query loader: either owner's native trainset or a cross-dataset probe.
    ql, nq_used, _ = _build_query_loader(query_dataset, trainset, n_queries)
    if query_dataset not in (None, "native"):
        print("  [cross-dataset] query_dataset={} ({} queries)".format(
            query_dataset, nq_used))

    st_tmp = create_student(nc, student_arch)
    sp = sum(p.numel() for p in st_tmp.parameters())
    del st_tmp

    print("  Teacher: {} ({:,}p, acc={:.2f}%)".format(
        teacher_name, sum(p.numel() for p in model.parameters()), acc_t*100))
    print("  Student: {} ({:,}p)".format(student_arch, sp))

    # For hard-label mode, temperature doesn't affect training (CE loss),
    # but we still iterate for consistency; use T=1 internally.
    temp_list = temperatures if label_mode == "soft" else [1]

    for T in temp_list:
        if label_mode == "soft":
            print("\n--- T={} [soft-label] ---".format(T))
        else:
            print("\n--- [hard-label] ---")

        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, tag=tag)

        calib_info = None  # populated only on cache miss when BGS + quantile > 0

        cached_s, cached_trig, cached_kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if cached_s is not None:
            student, triggers, Kw = cached_s, cached_trig, cached_kw
            acc_s = evaluate_accuracy(student, testloader, device)
            print("  Cached surrogate: acc={:.1f}%, {} triggers, K_w={}...".format(
                acc_s*100, len(triggers), Kw.hex()[:12]))
        else:
            Kw = kdf(keygen(256, seed=seed), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min,
                num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
                margin_tau_hard=margin_tau_hard,
            )

            if label_mode == "soft":
                # Soft-label: KL divergence distillation
                inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)
                nt = len(wm.trigger_set)
                print("  {} queries, {} wm ({:.2f}%), {} triggers, delta_logit={}".format(
                    len(inp), nwm, nwm/len(inp)*100, nt, delta_logit))

                student = create_student(nc, student_arch)
                student, dist_loss_history = soft_label_distillation(
                    student, inp, sl_out, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                # Hard-label: Cross-entropy extraction with (optional) BGS swap
                use_bgs = (hard_label_mode == "bgs")
                # Adaptive τ calibration (Option A): q-quantile of margin dist
                calib_info = None
                if use_bgs and hard_tau_quantile is not None and hard_tau_quantile > 0.0:
                    calib_loader = DataLoader(
                        Subset(trainset,
                               list(range(min(hard_tau_calib_samples, len(trainset))))),
                        batch_size=64, shuffle=False, num_workers=2)
                    calib_info = wm.calibrate_margin_tau(
                        teacher, calib_loader, quantile=hard_tau_quantile,
                        device=device, max_samples=hard_tau_calib_samples)
                    old_tau = wm.margin_tau_hard
                    wm.margin_tau_hard = calib_info["tau"]
                    margin_tau_hard = calib_info["tau"]
                    print("  [calibrate_tau] q={:.2f}  tau: {:.4f} -> {:.4f}  "
                          "(phi_samples={}, median_m={:.3f}, max_m={:.3f})".format(
                              hard_tau_quantile, old_tau, calib_info["tau"],
                              calib_info["n_phi_samples"],
                              calib_info["margin_median"],
                              calib_info["margin_max"]))

                inp, hard_labels, n_swapped = collect_hard_labels(
                    teacher, ql, device, wm, use_bgs=use_bgs)
                nt = len(wm.trigger_set)
                if use_bgs:
                    stats = wm.hardlabel_stats()
                    print("  [BGS] {} queries, Phi=1: {}, swapped: {} ({:.2f}% of queries),"
                          " margin_tau={:.4f}, triggers={}".format(
                              stats["n_queries"], stats["n_phi_active"],
                              stats["n_swapped"],
                              stats["swap_rate"]*100, margin_tau_hard, nt))
                else:
                    print("  [PLAIN argmax] {} queries, no swap (FP control), triggers={}".format(
                        len(inp), nt))

                student = create_student(nc, student_arch)
                student, dist_loss_history = hard_label_extraction(
                    student, inp, hard_labels, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)

            acc_s = evaluate_accuracy(student, testloader, device)
            triggers = wm.trigger_set

            ck_meta = {
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit, "beta": beta,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_s, 6), "n_triggers": nt,
                "label_mode": label_mode, "tag": tag,
            }
            if label_mode == "hard":
                ck_meta["hard_label_mode"] = hard_label_mode
                ck_meta["margin_tau_hard"] = float(margin_tau_hard)
                if calib_info is not None:
                    ck_meta["tau_calibration"] = calib_info
            save_checkpoint(student, triggers, Kw, dataset, ck_stem, meta=ck_meta,
                subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

        if label_mode == "hard":
            vr = verify_hardlabel_and_format(triggers, student, nc, eta, device)
            print("  Acc={:.1f}%, MatchRate={:.3f} (base {:.3f}), p={:.2e}, V={}".format(
                acc_s*100, vr["match_rate"], vr["baseline_rate"],
                vr["p_value"], vr["verified"]))
        else:
            vr = verify_and_format(triggers, student, nc, Kw, eta, device,
                                   verify_temperature=verify_temperature)
            print("  Acc={:.1f}%, Shift={:.4f}, p={:.2e}, V={}".format(
                acc_s*100, vr["confidence_shift"], vr["p_value"], vr["verified"]))
        _ad = vr.get("all_designs") or {}
        for _k in ("single_ctrl", "mean_rest", "suspect_top1"):
            _d = _ad.get(_k)
            if _d is None:
                continue
            print("    [{:<12s}] shift={:+.5f}  median={:+.5f}  p={:.2e}  V={}".format(
                _k, _d.get("confidence_shift", 0.0),
                _d.get("median_shift", 0.0),
                _d.get("p_value", 1.0), _d.get("verified", False)))

        fname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, vT=verify_temperature, tag=tag)
        save_result({
            "experiment": "distillation", "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "label_mode": label_mode,
            "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6),
                        "latent_layer": latent_layer},
            "student": {"architecture": student_arch, "num_parameters": sp},
            "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
            "distillation_config": {
                "temperature": T, "n_queries": n_queries,
                "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
                "label_mode": label_mode,
                "query_dataset": query_dataset,
            },
            "hard_label_config": ({
                "hard_label_mode": hard_label_mode,
                "margin_tau_hard": float(margin_tau_hard),
                "hard_tau_quantile": (hard_tau_quantile
                                      if hard_tau_quantile is not None
                                      and hard_tau_quantile > 0 else None),
                "tau_calibration": calib_info,
            } if label_mode == "hard" else None),
            "result": {
                "temperature": T,
                "student_accuracy": round(acc_s, 6),
                "n_triggers": len(triggers),
                **vr,
            },
        }, dataset, "distill", fname,
            teacher_name=teacher_name, student_arch=student_arch,
            label_mode="{}_label".format(label_mode))


# ============================================================
# Table VII: Surrogate Fine-Tuning Attack
# ============================================================

def experiment_surrogate_ft(model, trainset, testset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            ft_fractions, ft_epochs, ft_lr,
                            delta_logit=2.0, beta=0.4, delta_min=0.5,
                            verify_temperature=5.0,
                            label_mode="soft", trigger_mode="rec_trigger",
                            own_trigger_size=0, rec_trigger_size=0,
                            own_data_source="trainset", tag="",
                            seed=None,
                            hard_label_mode="bgs", margin_tau_hard=1.5,
                            hard_tau_quantile=-1.0,
                            hard_tau_calib_samples=3000,
                            baseline_method=None):
    """
    Table VII: Surrogate Fine-Tuning Attack.

    Args:
        label_mode: "soft" or "hard" — controls the initial distillation method.
        trigger_mode: "rec_trigger" = use original trigger set,
                      "own_trigger" = reconstruct from Owner's data,
                      "both" = run both for comparison.
        seed: if set, K_w is generated deterministically for reproducibility.
        own_data_source: "trainset" (default — overlaps with D_eval) or
                         "testset" (true zero-leakage probe).
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("Table VII: Surrogate Fine-Tuning Attack [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, dist_epochs={}, ft_epochs={}, ft_lr={}, ft_fractions={}".format(
        rw, dist_epochs, ft_epochs, ft_lr, ft_fractions))
    print("  delta_logit={}, beta={}, label_mode={}".format(delta_logit, beta, label_mode))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    ext = None
    temp_list = temperatures if label_mode == "soft" else [1]

    for T in temp_list:
        if label_mode == "soft":
            print("\n--- T={} [soft-label] ---".format(T))
        else:
            print("\n--- [hard-label] ---")

        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, tag=tag)

        surrogate, triggers, Kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if surrogate is None:
            print("  No cache. Distilling ({})...".format(label_mode))
            if ext is None:
                sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                                batch_size=64, shuffle=False, num_workers=2)
                ext = LatentExtractor()
                ext.compute_median(teacher, sl, latent_layer, device)

            ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                            batch_size=64, shuffle=True, num_workers=2)
            Kw = kdf(keygen(256, seed=seed), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min,
                num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
                margin_tau_hard=margin_tau_hard,
            )

            if label_mode == "soft":
                inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)
                surrogate = create_student(nc, student_arch)
                surrogate, dist_loss_history = soft_label_distillation(
                    surrogate, inp, sl_out, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                use_bgs = (hard_label_mode == "bgs")
                if use_bgs and hard_tau_quantile is not None and hard_tau_quantile > 0.0:
                    calib_loader = DataLoader(
                        Subset(trainset,
                               list(range(min(hard_tau_calib_samples, len(trainset))))),
                        batch_size=64, shuffle=False, num_workers=2)
                    calib_info = wm.calibrate_margin_tau(
                        teacher, calib_loader, quantile=hard_tau_quantile,
                        device=device, max_samples=hard_tau_calib_samples)
                    print("  [calibrate_tau] q={:.2f}  tau: {:.4f} -> {:.4f}".format(
                        hard_tau_quantile, wm.margin_tau_hard, calib_info["tau"]))
                    wm.margin_tau_hard = calib_info["tau"]
                inp, hard_labels, _ = collect_hard_labels(
                    teacher, ql, device, wm, use_bgs=use_bgs)
                surrogate = create_student(nc, student_arch)
                surrogate, dist_loss_history = hard_label_extraction(
                    surrogate, inp, hard_labels, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)

            triggers = wm.trigger_set

            acc_d = evaluate_accuracy(surrogate, testloader, device)
            save_checkpoint(surrogate, triggers, Kw, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_d, 6), "n_triggers": len(triggers),
                "label_mode": label_mode, "tag": tag,
            }, subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

            if label_mode == "hard":
                vr_d = verify_hardlabel_and_format(triggers, surrogate, nc, eta, device)
            else:
                vr_d = verify_and_format(triggers, surrogate, nc, Kw, eta, device,
                                         verify_temperature=verify_temperature)
            dfname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, vT=verify_temperature, tag=tag)
            save_result({
                "experiment": "distillation",
                "timestamp": datetime.now().isoformat(),
                "note": "Auto-generated during surrogate_ft",
                "dataset": dataset,
                "label_mode": label_mode,
                "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
                "student": {"architecture": student_arch},
                "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
                "distillation_config": {
                    "temperature": T, "n_queries": n_queries,
                    "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
                    "label_mode": label_mode,
                },
                "result": {
                    "temperature": T,
                    "student_accuracy": round(acc_d, 6),
                    "n_triggers": len(triggers),
                    **vr_d,
                },
            }, dataset, "distill", dfname,
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))
        # else: Kw was loaded from the .key file by load_checkpoint above

        acc_base = evaluate_accuracy(surrogate, testloader, device)
        nt = len(triggers)

        # Determine which trigger modes to run
        if trigger_mode == "both":
            t_modes = ["rec_trigger", "own_trigger"]
        else:
            t_modes = [trigger_mode]

        for t_mode in t_modes:
            print("\n  --- Verification: trigger_mode={} ---".format(t_mode))

            # Prepare own_trigger mode
            # Phi(x) filtering: only keep queries that were watermarked
            if t_mode == "own_trigger":
                # Ensure latent extractor is available for Phi(x) filtering
                if ext is None:
                    sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                                    batch_size=64, shuffle=False, num_workers=2)
                    ext = LatentExtractor()
                    ext.compute_median(teacher, sl, latent_layer, device)
                if own_data_source == "testset" and testset is not None:
                    own_ds = testset
                else:
                    own_ds = trainset
                own_loader = DataLoader(own_ds, batch_size=64, shuffle=False, num_workers=2)
                vr_base = verify_own_data_and_format(
                    model, own_loader, surrogate,
                    Kw, rw, nc, ext, latent_layer,
                    delta_logit, beta, eta, device, verify_temperature,
                    max_triggers=own_trigger_size)
                trigger_size_used = vr_base["n_trigger"]
                trigger_pool_size = len(own_ds)
            else:
                # Cap recorded triggers if rec_trigger_size > 0
                use_triggers = triggers
                if rec_trigger_size > 0 and len(triggers) > rec_trigger_size:
                    use_triggers = triggers[:rec_trigger_size]
                if label_mode == "hard":
                    vr_base = verify_hardlabel_and_format(
                        use_triggers, surrogate, nc, eta, device)
                else:
                    vr_base = verify_and_format(use_triggers, surrogate, nc, Kw, eta, device,
                                                verify_temperature=verify_temperature)
                trigger_size_used = len(use_triggers)
                trigger_pool_size = len(triggers)

            print("  Base: acc={:.1f}%, triggers={}/{}, shift={:.4f}, p={:.2e}, V={} [{}]".format(
                acc_base*100, trigger_size_used, trigger_pool_size,
                vr_base["confidence_shift"],
                vr_base["p_value"], vr_base["verified"], t_mode))
            _ad = vr_base.get("all_designs") or {}
            for _k in ("single_ctrl", "mean_rest", "suspect_top1"):
                _d = _ad.get(_k)
                if _d is None:
                    continue
                print("    [{:<12s}] shift={:+.5f}  median={:+.5f}  p={:.2e}  V={}".format(
                    _k, _d.get("confidence_shift", 0.0),
                    _d.get("median_shift", 0.0),
                    _d.get("p_value", 1.0), _d.get("verified", False)))

            ft_results = []
            for frac in ft_fractions:
                if frac == 0.0:
                    acc_ft = acc_base
                    vr_ft = vr_base
                    label = "baseline"
                else:
                    nft = int(len(trainset) * frac)
                    fl = DataLoader(Subset(trainset, list(range(nft))),
                                    batch_size=128, shuffle=True)
                    surr_ft, ft_loss_history = fine_tune_surrogate(
                        copy.deepcopy(surrogate), fl, ft_epochs, ft_lr, device)
                    acc_ft = evaluate_accuracy(surr_ft, testloader, device)

                    if t_mode == "own_trigger":
                        vr_ft = verify_own_data_and_format(
                            model, own_loader, surr_ft,
                            Kw, rw, nc, ext, latent_layer,
                            delta_logit, beta, eta, device, verify_temperature,
                            max_triggers=own_trigger_size)
                    else:
                        if label_mode == "hard":
                            vr_ft = verify_hardlabel_and_format(
                                use_triggers, surr_ft, nc, eta, device)
                        else:
                            vr_ft = verify_and_format(use_triggers, surr_ft, nc, Kw, eta, device,
                                                      verify_temperature=verify_temperature)
                    label = "{}% ({})".format(int(frac*100), nft)

                print("    FT {}: acc={:.1f}%, shift={:.4f}, p={:.2e}, V={} [{}]".format(
                    label, acc_ft*100, vr_ft["confidence_shift"],
                    vr_ft["p_value"], vr_ft["verified"], t_mode))

                ft_results.append({
                    "ft_fraction": frac,
                    "ft_samples": int(len(trainset) * frac),
                    "accuracy": round(acc_ft, 6),
                    **vr_ft,
                })

            # Save with trigger_mode in directory path
            save_label = "{}_label__{}".format(label_mode, t_mode)

            fname = stem_surrogate_ft(
                teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr,
                ft_epochs, ft_lr, delta_logit, vT=verify_temperature,
                trigger_mode=t_mode, trigger_size=trigger_size_used, tag=tag)
            save_result({
                "experiment": "surrogate_finetune",
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "label_mode": label_mode,
                "trigger_mode": t_mode,
                "trigger_config": {
                    "trigger_source": t_mode,
                    "trigger_size_used": trigger_size_used,
                    "trigger_pool_size": trigger_pool_size,
                    "own_trigger_size": own_trigger_size if t_mode == "own_trigger" else None,
                    "rec_trigger_size": rec_trigger_size if t_mode == "rec_trigger" else None,
                },
                "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
                "student": {"architecture": student_arch},
                "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
                "distillation_baseline": {
                    "temperature": T, "n_queries": n_queries,
                    "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                    "surrogate_accuracy": round(acc_base, 6),
                    "n_triggers_recorded": nt,
                    "n_triggers_used": trigger_size_used,
                    **vr_base,
                    "label_mode": label_mode,
                    "trigger_mode": t_mode,
                },
                "ft_config": {
                    "ft_fractions": list(ft_fractions),
                    "ft_epochs": ft_epochs,
                    "ft_lr": ft_lr,
                },
                "ft_results": ft_results,
            }, dataset, "surrogate_ft", fname,
                method=baseline_method,
                teacher_name=teacher_name if not baseline_method else None,
                student_arch=student_arch if not baseline_method else None,
                label_mode=save_label)


# ============================================================
# v6: Random K_w' False-Positive Scan
# ============================================================

def experiment_random_kw_fp(model, trainset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            delta_logit, beta, delta_min,
                            verify_temperature, label_mode, tag, seed,
                            n_random_kw=100, hard_label_mode="bgs",
                            margin_tau_hard=1.5,
                            hard_tau_quantile=-1.0,
                            hard_tau_calib_samples=3000):
    """
    Random K_w' false-positive scan.

    Hold the SUSPECT student fixed.  Use n_random_kw random keys K_w' (each
    independent of the genuine K_w).  Re-derive the (target, control) pairs
    from each K_w' on the SAME trigger inputs, run the verification test, and
    record p-values.  A correctly-functioning watermark should reject H0 only
    for the genuine K_w; random keys should yield p >> eta.

    Reuses cached student checkpoints from `experiment_distillation`.  If no
    cache exists, distillation is re-run inline.
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("v6: Random K_w' False-Positive Scan [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  n_random_kw={}, eta={:.2e}".format(n_random_kw, eta))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                    batch_size=64, shuffle=False, num_workers=2)
    ext = LatentExtractor()
    ext.compute_median(teacher, sl, latent_layer, device)
    ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                    batch_size=64, shuffle=True, num_workers=2)

    temp_list = temperatures if label_mode == "soft" else [1]

    for T in temp_list:
        print("\n--- T={} [{}] ---".format(T, label_mode))
        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T,
                            dist_epochs, dist_lr, delta_logit, tag=tag)

        student, triggers, Kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if student is None:
            print("  No cached student; running distillation inline...")
            Kw = kdf(keygen(256, seed=seed), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min, num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
                margin_tau_hard=margin_tau_hard,
            )
            if label_mode == "soft":
                inp, sl_out, _ = collect_soft_labels(teacher, ql, device, wm, T)
                student = create_student(nc, student_arch)
                student, _ = soft_label_distillation(
                    student, inp, sl_out, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                use_bgs = (hard_label_mode == "bgs")
                if use_bgs and hard_tau_quantile is not None and hard_tau_quantile > 0.0:
                    calib_loader = DataLoader(
                        Subset(trainset,
                               list(range(min(hard_tau_calib_samples, len(trainset))))),
                        batch_size=64, shuffle=False, num_workers=2)
                    calib_info = wm.calibrate_margin_tau(
                        teacher, calib_loader, quantile=hard_tau_quantile,
                        device=device, max_samples=hard_tau_calib_samples)
                    print("  [calibrate_tau] q={:.2f}  tau: {:.4f} -> {:.4f}".format(
                        hard_tau_quantile, wm.margin_tau_hard, calib_info["tau"]))
                    wm.margin_tau_hard = calib_info["tau"]
                inp, hl, _ = collect_hard_labels(
                    teacher, ql, device, wm, use_bgs=use_bgs)
                student = create_student(nc, student_arch)
                student, _ = hard_label_extraction(
                    student, inp, hl, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            triggers = wm.trigger_set
            acc_s = evaluate_accuracy(student, testloader, device)
            save_checkpoint(student, triggers, Kw, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_s, 6), "n_triggers": len(triggers),
                "label_mode": label_mode, "tag": tag,
                "hard_label_mode": hard_label_mode,
                "margin_tau_hard": float(wm.margin_tau_hard),
            }, subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

        acc_s = evaluate_accuracy(student, testloader, device)
        print("  Cached/built student: acc={:.2f}%, {} triggers".format(
            acc_s*100, len(triggers)))

        scan = random_kw_false_positive_scan(
            triggers, student, Kw, nc,
            n_random=n_random_kw, eta=eta, device=device,
            verify_temperature=verify_temperature, batch_size=64,
            label_mode=label_mode, margin_tau_hard=margin_tau_hard,
            seed=seed,
        )
        print("  Genuine K_w:  p={:.2e}, V={}".format(
            scan["genuine_p_value"], scan["genuine_verified"]))
        print("  Random K_w'  ({} keys): below_eta={}/{}  ({:.2f}%)".format(
            scan["n_random"], scan["verified_count"], scan["n_random"],
            scan["fraction_below_eta"]*100))
        if scan["n_random"] > 0:
            print("    min_p={:.2e}, median_p={:.2e}".format(
                scan["min_p"], scan["median_p"]))

        fname = "RANDKW__{}_{}__rw{}_nq{}_T{}_dl{}_vT{}__N{}{}".format(
            teacher_name, student_arch, rw, n_queries, T, delta_logit,
            verify_temperature, n_random_kw,
            ("__" + tag) if tag else "")
        save_result({
            "experiment": "random_kw_fp",
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "label_mode": label_mode,
            "teacher": {**model_info(model, teacher_name)},
            "student": {"architecture": student_arch, "accuracy": round(acc_s, 6)},
            "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature, delta_min),
            "scan_config": {
                "n_random_kw": n_random_kw,
                "T": T,
                "verify_temperature": verify_temperature,
                "margin_tau_hard": margin_tau_hard,
                "hard_label_mode": hard_label_mode if label_mode == "hard" else None,
            },
            "result": scan,
        }, dataset, "random_kw_fp", fname,
            teacher_name=teacher_name, student_arch=student_arch,
            label_mode="{}_label".format(label_mode))


# ============================================================
# v6: Post-hoc Attacks (pruning + INT8 quantization)
# ============================================================

def experiment_posthoc_attack(model, trainset, testloader, latent_layer, device,
                              teacher_name, student_arch, dataset, config,
                              temperatures, n_queries, rw,
                              eta, dist_epochs, dist_lr, dist_batch,
                              delta_logit, beta, delta_min,
                              verify_temperature, label_mode, tag, seed,
                              attack_type="prune",
                              prune_fracs=(0.25, 0.50, 0.75),
                              hard_label_mode="bgs", margin_tau_hard=1.5,
                              hard_tau_quantile=-1.0,
                              hard_tau_calib_samples=3000):
    """
    Post-hoc attacks that do NOT retrain the surrogate but mutate its weights:
      attack_type="prune"    : magnitude pruning at each frac in prune_fracs.
      attack_type="quantize" : single INT8 round-trip (prune_fracs ignored).

    Verifies the watermark on the attacked surrogate against the original
    trigger set.  Reuses cached students from experiment_distillation.
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("v6: Post-hoc Attack [{}] ({})".format(attack_type.upper(), label_mode))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    # Ensure we have a cached student to attack
    sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                    batch_size=64, shuffle=False, num_workers=2)
    ext = LatentExtractor()
    ext.compute_median(teacher, sl, latent_layer, device)
    ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                    batch_size=64, shuffle=True, num_workers=2)

    temp_list = temperatures if label_mode == "soft" else [1]
    for T in temp_list:
        print("\n--- T={} [{}] ---".format(T, label_mode))
        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T,
                            dist_epochs, dist_lr, delta_logit, tag=tag)
        student, triggers, Kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if student is None:
            print("  No cached student; distilling inline...")
            Kw = kdf(keygen(256, seed=seed), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min, num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
                margin_tau_hard=margin_tau_hard,
            )
            if label_mode == "soft":
                inp, slo, _ = collect_soft_labels(teacher, ql, device, wm, T)
                student = create_student(nc, student_arch)
                student, _ = soft_label_distillation(
                    student, inp, slo, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                use_bgs = (hard_label_mode == "bgs")
                if use_bgs and hard_tau_quantile is not None and hard_tau_quantile > 0.0:
                    calib_loader = DataLoader(
                        Subset(trainset,
                               list(range(min(hard_tau_calib_samples, len(trainset))))),
                        batch_size=64, shuffle=False, num_workers=2)
                    calib_info = wm.calibrate_margin_tau(
                        teacher, calib_loader, quantile=hard_tau_quantile,
                        device=device, max_samples=hard_tau_calib_samples)
                    print("  [calibrate_tau] q={:.2f}  tau: {:.4f} -> {:.4f}".format(
                        hard_tau_quantile, wm.margin_tau_hard, calib_info["tau"]))
                    wm.margin_tau_hard = calib_info["tau"]
                inp, hl, _ = collect_hard_labels(
                    teacher, ql, device, wm, use_bgs=use_bgs)
                student = create_student(nc, student_arch)
                student, _ = hard_label_extraction(
                    student, inp, hl, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            triggers = wm.trigger_set
            acc_s = evaluate_accuracy(student, testloader, device)
            save_checkpoint(student, triggers, Kw, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_s, 6), "n_triggers": len(triggers),
                "label_mode": label_mode, "tag": tag,
                "hard_label_mode": hard_label_mode,
                "margin_tau_hard": float(wm.margin_tau_hard) if label_mode == "hard" else None,
            }, subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

        acc_base = evaluate_accuracy(student, testloader, device)
        print("  Baseline surrogate: acc={:.2f}%, {} triggers".format(
            acc_base*100, len(triggers)))

        attack_runs = []
        fracs = list(prune_fracs) if attack_type == "prune" else [None]
        for frac in fracs:
            victim = copy.deepcopy(student)
            if attack_type == "prune":
                victim = prune_surrogate(victim, frac, scope="global")
                label = "prune_{:.0f}%".format(frac * 100)
            elif attack_type == "quantize":
                victim = quantize_surrogate_int8(victim)
                label = "int8_quant"
            else:
                raise ValueError("Unknown attack_type: {}".format(attack_type))

            acc_att = evaluate_accuracy(victim, testloader, device)
            if label_mode == "hard":
                vr = verify_hardlabel_and_format(triggers, victim, nc, eta, device)
            else:
                vr = verify_and_format(triggers, victim, nc, Kw, eta, device,
                                       verify_temperature=verify_temperature)
            print("    {}: acc={:.2f}% (Δ={:+.2f}), shift={:+.4f}, p={:.2e}, V={}".format(
                label, acc_att*100, (acc_att-acc_base)*100,
                vr.get("confidence_shift", 0.0),
                vr["p_value"], vr["verified"]))
            attack_runs.append({
                "attack": label,
                "prune_frac": frac if attack_type == "prune" else None,
                "accuracy": round(acc_att, 6),
                "accuracy_delta_pp": round((acc_att - acc_base) * 100, 4),
                **vr,
            })

        fname = "POSTHOC_{}__{}_{}__rw{}_nq{}_T{}_dl{}_vT{}{}".format(
            attack_type.upper(), teacher_name, student_arch,
            rw, n_queries, T, delta_logit, verify_temperature,
            ("__" + tag) if tag else "")
        save_result({
            "experiment": "posthoc_{}".format(attack_type),
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "label_mode": label_mode,
            "teacher": {**model_info(model, teacher_name)},
            "student": {"architecture": student_arch,
                        "baseline_accuracy": round(acc_base, 6)},
            "watermark_config": watermark_config_dict(
                rw, delta_logit, beta, nc, verify_temperature, delta_min),
            "attack_config": {
                "attack_type": attack_type,
                "prune_fracs": list(prune_fracs) if attack_type == "prune" else None,
                "T": T,
            },
            "attack_runs": attack_runs,
        }, dataset, "posthoc_{}".format(attack_type), fname,
            teacher_name=teacher_name, student_arch=student_arch,
            label_mode="{}_label".format(label_mode))


# ============================================================
# Table VIII: Overhead
# ============================================================

def experiment_overhead(model, testloader, device, epsilon, delta,
                        teacher_name, dataset, config):
    print("\n" + "=" * 60)
    print("Table VIII: Overhead")
    print("  Dataset: {}, Teacher: {}, eps={}".format(dataset, teacher_name, epsilon))
    print("=" * 60)

    np_ = sum(p.numel() for p in model.parameters())
    m = copy.deepcopy(model)

    t0 = time.time()
    m, sec = obfuscate_model_vectorized(m, epsilon=epsilon, delta=delta, model_id="o")
    ot = time.time() - t0

    t0 = time.time()
    recover_weights(m, sec, vectorized=True)
    rt = time.time() - t0

    m.to(device).eval()
    x, _ = next(iter(testloader))
    x = x[:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            m(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            m(x)
    it = (time.time() - t0) / 100 * 1000

    ss = len(sec.K_obf) + 8 + len(sec.K_w) + len(sec.model_id.encode())
    print("  Obf={:.3f}s, Rec={:.3f}s, Inf={:.2f}ms, |S|={}B".format(ot, rt, it, ss))

    fname = stem_overhead(teacher_name, epsilon)
    save_result({
        "experiment": "overhead", "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model_info(model, teacher_name),
        "parameters": {"epsilon": epsilon, "delta": str(delta)},
        "results": {
            "obfuscation_time_s": round(ot, 4),
            "recovery_time_s": round(rt, 4),
            "inference_ms": round(it, 4),
            "secret_size_bytes": ss,
            "num_parameters": np_,
        },
    }, dataset, "overhead", fname)


# ============================================================
# Baseline Comparisons (Table X: DAWN / Adi vs. EvalGuard)
# ============================================================

def experiment_baseline_dawn(model, trainset, testset, testloader, latent_layer, device,
                             teacher_name, student_arch, dataset, config,
                             temperatures, n_queries, rw,
                             eta, dist_epochs, dist_lr, dist_batch,
                             ft_fractions, ft_epochs, ft_lr,
                             delta_logit=2.0, beta=0.4, delta_min=0.5,
                             verify_temperature=5.0,
                             label_mode="soft", tag="", seed=None):
    """
    DAWN baseline comparison (Table X).

    Runs the same extraction+fine-tuning pipeline but uses DAWN watermarking
    instead of EvalGuard. DAWN flips top-1 for r_w fraction of queries.
    """
    from evalguard.baselines import (
        DAWNWatermark, dawn_collect_hard_labels, dawn_collect_soft_labels,
    )
    from evalguard.attacks import (
        soft_label_distillation, hard_label_extraction, fine_tune_surrogate,
    )

    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("BASELINE: DAWN Comparison [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  r_w={}, nq={}".format(rw, n_queries))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                    batch_size=64, shuffle=True, num_workers=2)

    K_dawn = kdf(keygen(256, seed=seed), "dawn_watermark")
    temp_list = temperatures if label_mode == "soft" else [5]

    for T in temp_list:
        print("\n--- T={} [{}] ---".format(T, label_mode))

        if label_mode == "soft":
            inp, soft_out, dawn_wm = dawn_collect_soft_labels(
                teacher, ql, K_dawn, rw, T, device)
            n_triggers = len(dawn_wm.trigger_set)
            print("  DAWN: {} queries, {} triggers ({:.1f}%)".format(
                len(inp), n_triggers, n_triggers / len(inp) * 100))

            student = create_student(nc, student_arch)
            student, _ = soft_label_distillation(
                student, inp, soft_out, T, dist_epochs,
                batch_size=dist_batch, lr=dist_lr, device=device)
        else:
            inp, hard_out, dawn_wm = dawn_collect_hard_labels(
                teacher, ql, K_dawn, rw, nc, device)
            n_triggers = len(dawn_wm.trigger_set)
            print("  DAWN: {} queries, {} triggers ({:.1f}%)".format(
                len(inp), n_triggers, n_triggers / len(inp) * 100))

            student = create_student(nc, student_arch)
            student, _ = hard_label_extraction(
                student, inp, hard_out, dist_epochs,
                batch_size=dist_batch, lr=dist_lr, device=device)

        acc_s = evaluate_accuracy(student, testloader, device)
        vr_base = dawn_wm.verify(student, device=device, eta=eta, num_classes=nc)
        print("  Base: acc={:.1f}%, match_rate={:.3f}, p={:.2e}, V={}".format(
            acc_s * 100, vr_base["match_rate"], vr_base["p_value"],
            vr_base["verified"]))

        ft_results = []
        for frac in ft_fractions:
            if frac == 0.0:
                acc_ft, vr_ft = acc_s, vr_base
            else:
                nft = int(len(trainset) * frac)
                fl = DataLoader(Subset(trainset, list(range(nft))),
                                batch_size=128, shuffle=True)
                surr_ft, _ = fine_tune_surrogate(
                    copy.deepcopy(student), fl, ft_epochs, ft_lr, device)
                acc_ft = evaluate_accuracy(surr_ft, testloader, device)
                vr_ft = dawn_wm.verify(surr_ft, device=device, eta=eta,
                                       num_classes=nc)

            print("    FT {:.0f}%: acc={:.1f}%, match={:.3f}, p={:.2e}, V={}".format(
                frac * 100, acc_ft * 100, vr_ft["match_rate"],
                vr_ft["p_value"], vr_ft["verified"]))
            ft_results.append({
                "ft_fraction": frac,
                "accuracy": round(acc_ft, 6),
                **vr_ft,
            })

        fname = "DAWN__{}_{}__rw{}_T{}_nq{}{}".format(
            teacher_name, student_arch, rw, T, n_queries,
            ("__" + tag) if tag else "")
        save_result({
            "experiment": "baseline_dawn",
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "label_mode": label_mode,
            "method": "DAWN",
            "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
            "student": {"architecture": student_arch},
            "dawn_config": {"r_w": rw, "n_triggers": n_triggers},
            "distillation_config": {
                "temperature": T, "n_queries": n_queries,
                "epochs": dist_epochs, "lr": dist_lr,
            },
            "baseline_result": {
                "temperature": T,
                "student_accuracy": round(acc_s, 6),
                **vr_base,
            },
            "ft_results": ft_results,
        }, dataset, "baseline_dawn", fname, method="DAWN")


def experiment_baseline_adi(model, trainset, testset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            ft_fractions, ft_epochs, ft_lr,
                            verify_temperature=5.0,
                            label_mode="soft",
                            tag="", seed=None,
                            adi_n_triggers=100, adi_retrain_epochs=30,
                            adi_lr=0.001):
    """
    Adi et al. (2018) backdoor watermark baseline comparison (Table X).

    Flow: generate triggers → retrain teacher with backdoor → extract student
    via soft or hard-label distillation → verify if backdoor transfers → fine-tune.

    Note: the Adi backdoor embeds information in the full logit distribution, so
    hard-label extraction (argmax only) is expected to *not* transfer the backdoor.
    Running label_mode="hard" demonstrates this failure mode explicitly.
    """
    from evalguard.baselines import AdiWatermark
    from evalguard.attacks import (
        collect_soft_labels, soft_label_distillation,
        collect_hard_labels, hard_label_extraction,
        fine_tune_surrogate,
    )

    nc = config["num_classes"]
    img_shape = (3, 32, 32)
    if dataset == "tinyimagenet":
        img_shape = (3, 64, 64)

    print("\n" + "=" * 60)
    print("BASELINE: Adi et al. Backdoor Watermark [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  n_triggers={}, retrain_epochs={}".format(adi_n_triggers, adi_retrain_epochs))
    print("=" * 60)

    acc_t = evaluate_accuracy(model, testloader, device)

    adi = AdiWatermark(num_classes=nc, n_triggers=adi_n_triggers,
                       img_shape=img_shape, seed=seed or 42)

    print("  Retraining teacher with {} backdoor triggers...".format(adi_n_triggers))
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    teacher_bd = adi.embed_backdoor(
        model, trainloader, epochs=adi_retrain_epochs,
        lr=adi_lr, device=device)

    acc_t_bd = evaluate_accuracy(teacher_bd, testloader, device)
    bd_success = adi.verify_on_teacher(teacher_bd, device)
    print("  Backdoored teacher: acc={:.1f}% (orig {:.1f}%), backdoor_success={:.1f}%".format(
        acc_t_bd * 100, acc_t * 100, bd_success * 100))

    ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                    batch_size=64, shuffle=True, num_workers=2)

    T = 5
    print("\n--- Distilling from backdoored teacher (T={}, mode={}) ---".format(T, label_mode))
    teacher_bd.to(device).eval()

    student = create_student(nc, student_arch)
    if label_mode == "hard":
        inp, hard_out, _ = collect_hard_labels(teacher_bd, ql, device)
        student, _ = hard_label_extraction(
            student, inp, hard_out, dist_epochs,
            batch_size=dist_batch, lr=dist_lr, device=device)
    else:
        all_inputs, all_soft = [], []
        with torch.no_grad():
            for inputs, _ in ql:
                inputs = inputs.to(device)
                logits = teacher_bd(inputs)
                probs = torch.softmax(logits / T, dim=-1)
                all_inputs.append(inputs.cpu())
                all_soft.append(probs.cpu())
        inp = torch.cat(all_inputs, 0)
        soft_out = torch.cat(all_soft, 0)
        student, _ = soft_label_distillation(
            student, inp, soft_out, T, dist_epochs,
            batch_size=dist_batch, lr=dist_lr, device=device)

    acc_s = evaluate_accuracy(student, testloader, device)
    vr_base = adi.verify(student, device=device, eta=eta)
    print("  Student: acc={:.1f}%, backdoor_match={:.1f}%, p={:.2e}, V={}".format(
        acc_s * 100, vr_base["match_rate"] * 100, vr_base["p_value"],
        vr_base["verified"]))

    ft_results = []
    for frac in ft_fractions:
        if frac == 0.0:
            acc_ft, vr_ft = acc_s, vr_base
        else:
            nft = int(len(trainset) * frac)
            fl = DataLoader(Subset(trainset, list(range(nft))),
                            batch_size=128, shuffle=True)
            surr_ft, _ = fine_tune_surrogate(
                copy.deepcopy(student), fl, ft_epochs, ft_lr, device)
            acc_ft = evaluate_accuracy(surr_ft, testloader, device)
            vr_ft = adi.verify(surr_ft, device=device, eta=eta)

        print("    FT {:.0f}%: acc={:.1f}%, backdoor_match={:.1f}%, p={:.2e}, V={}".format(
            frac * 100, acc_ft * 100, vr_ft["match_rate"] * 100,
            vr_ft["p_value"], vr_ft["verified"]))
        ft_results.append({
            "ft_fraction": frac,
            "accuracy": round(acc_ft, 6),
            **vr_ft,
        })

    fname = "ADI__{}_{}__{}__ntrig{}_nq{}{}".format(
        teacher_name, student_arch, label_mode, adi_n_triggers, n_queries,
        ("__" + tag) if tag else "")
    save_result({
        "experiment": "baseline_adi",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "method": "Adi_et_al_2018",
        "label_mode": label_mode,
        "teacher": {**model_info(model, teacher_name),
                    "accuracy_original": round(acc_t, 6),
                    "accuracy_backdoored": round(acc_t_bd, 6),
                    "backdoor_success_rate": round(bd_success, 6)},
        "student": {"architecture": student_arch},
        "adi_config": {
            "n_triggers": adi_n_triggers,
            "retrain_epochs": adi_retrain_epochs,
            "lr": adi_lr,
        },
        "distillation_config": {
            "temperature": T, "n_queries": n_queries,
            "epochs": dist_epochs, "lr": dist_lr,
            "label_mode": label_mode,
        },
        "baseline_result": {
            "student_accuracy": round(acc_s, 6),
            **vr_base,
        },
        "ft_results": ft_results,
    }, dataset, "baseline_adi", fname, method="Adi")


# ============================================================
# Main
# ============================================================

def parse_list_float(s):
    return [float(x.strip()) for x in s.split(",")]

def parse_list_int(s):
    return [int(x.strip()) for x in s.split(",")]

def main():
    pa = argparse.ArgumentParser(
        description="EvalGuard Experiments (v4 - Confidence Shift)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--experiment", default="distill",
                    choices=["fidelity", "finetune", "distill", "surrogate_ft",
                             "overhead", "random_kw_fp",
                             "pruning_attack", "quantization_attack",
                             "baseline_dawn", "baseline_adi",
                             "all"])
    pa.add_argument("--model", default="cifar10_vgg11")
    pa.add_argument("--dataset", default=None)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--pretrained", default=None)

    pa.add_argument("--epsilon", type=float, default=50.0)
    pa.add_argument("--delta", type=float, default=2**(-32))

    # Watermark parameters (v5: logit-space confidence shift)
    pa.add_argument("--rw", type=float, default=0.50)
    pa.add_argument("--delta_logit", type=float, default=2.0,
                    help="Logit-space shift amount (applied before softmax)")
    pa.add_argument("--beta", type=float, default=0.3,
                    help="Safety factor: delta = min(delta_logit, beta * logit_margin)")
    pa.add_argument("--delta_min", type=float, default=0.5,
                    help="Minimum effective delta. Queries whose safe delta would "
                         "fall below this are NOT recorded as triggers (avoids "
                         "polluting the trigger set with near-zero signals).")
    pa.add_argument("--verify_temperature", type=float, default=-1.0,
                    help="Temperature used during verification softmax. "
                         "Pass -1 (default) to AUTO-SCALE as max(5, 5*log10(C)+5) "
                         "via recommended_vT(num_classes). Any value >0 is "
                         "respected verbatim (subject to a soft warning if <5).")

    # v6 hard-label (Boundary-Gated Hard-Label Swap) parameters
    pa.add_argument("--hard_label_mode", default="bgs",
                    choices=["plain", "bgs"],
                    help="Hard-label embedding strategy: "
                         "'plain' = vanilla argmax (FP control, no watermark in hard mode), "
                         "'bgs' = Boundary-Gated Hard-Label Swap (v6, default).")
    pa.add_argument("--margin_tau_hard", type=float, default=1.5,
                    help="BGS swap threshold: only swap argmax->target when "
                         "(logits[top1] - logits[target]) <= margin_tau_hard. "
                         "Ignored when --hard_tau_quantile > 0.")
    pa.add_argument("--hard_tau_quantile", type=float, default=-1.0,
                    help="If > 0 (e.g. 0.10), BGS pre-scans the query pool and "
                         "uses the q-quantile of the Phi(x)=1 margin distribution "
                         "as τ, overriding --margin_tau_hard. q=0.10 yields "
                         "≈ r_w*q*|queries| triggers regardless of model scale.")
    pa.add_argument("--hard_tau_calib_samples", type=int, default=3000,
                    help="Number of queries scanned during adaptive-τ calibration.")

    # v6 random K_w' false-positive scan
    pa.add_argument("--n_random_kw", type=int, default=100,
                    help="Number of random K_w' keys for false-positive scan.")

    # v6 cross-dataset query attack
    pa.add_argument("--query_dataset", default="native",
                    choices=["native", "svhn"],
                    help="Which dataset the attacker uses to query the teacher. "
                         "'native' = owner's CIFAR trainset (default), "
                         "'svhn' = cross-dataset (unrelated 32x32 images).")

    # v6 post-hoc attacks
    pa.add_argument("--prune_fracs", type=str, default="0.25,0.50,0.75",
                    help="Comma-separated magnitude-pruning fractions "
                         "for --experiment pruning_attack.")

    pa.add_argument("--student_arch", default="vgg11",
                    choices=["resnet20", "resnet56", "vgg11", "mobilenetv2", "resnet18"])
    pa.add_argument("--nq", type=int, default=50000)
    pa.add_argument("--temperatures", type=str, default="5,10,20")
    pa.add_argument("--dist_epochs", type=int, default=80)
    pa.add_argument("--dist_lr", type=float, default=0.002)
    pa.add_argument("--dist_batch", type=int, default=128)

    pa.add_argument("--ft_epochs", type=int, default=20)
    pa.add_argument("--ft_lr", type=float, default=0.0005)
    pa.add_argument("--ft_fractions", type=str, default="0.0,0.01,0.05,0.10")

    pa.add_argument("--label_mode", default="soft",
                    choices=["soft", "hard", "both"],
                    help="Label mode: 'soft' for KL-divergence, 'hard' for CE, 'both' to run both")

    pa.add_argument("--trigger_mode", default="rec_trigger",
                    choices=["rec_trigger", "own_trigger", "both"],
                    help="Trigger source for verification: "
                         "'rec_trigger' = recorded trigger set from embedding phase, "
                         "'own_trigger' = reconstruct from Owner's data (zero D_eval leakage), "
                         "'both' = run both for comparison")

    pa.add_argument("--own_trigger_size", type=int, default=0,
                    help="Number of own triggers to collect for own_trigger verification. "
                         "0 = collect all from the chosen data source. "
                         "Scans until this many triggers are found.")

    pa.add_argument("--rec_trigger_size", type=int, default=0,
                    help="Number of recorded triggers to use for rec_trigger verification. "
                         "0 = use all recorded triggers.")

    pa.add_argument("--own_data_source", default="trainset",
                    choices=["trainset", "testset"],
                    help="Which Owner-side dataset feeds own_trigger reconstruction. "
                         "'trainset' (default) overlaps with D_eval. "
                         "'testset' is the true zero-leakage probe.")

    pa.add_argument("--epsilons", type=str, default="1,10,50,100,200")

    # Adi et al. baseline parameters
    pa.add_argument("--adi_n_triggers", type=int, default=100,
                    help="Number of random trigger images for Adi backdoor baseline.")
    pa.add_argument("--adi_retrain_epochs", type=int, default=30,
                    help="Epochs to retrain teacher with backdoor triggers.")
    pa.add_argument("--adi_lr", type=float, default=0.001,
                    help="Learning rate for Adi backdoor retraining.")

    pa.add_argument("--seed", type=int, default=None,
                    help="Deterministic seed for K_w generation. Same seed → same "
                         "watermark key → reproducible results across runs. "
                         "If not set, K_w is generated from os.urandom (non-reproducible).")

    pa.add_argument("--tag", type=str, default="",
                    help="Free-form experiment tag appended to checkpoint and result "
                         "filenames. Use this to distinguish runs with different "
                         "watermark parameters (e.g. 'paper', 'amplified', 'smoke').")

    pa.add_argument("--baseline_method", default=None,
                    choices=["EvalGuard", "DAWN", "Adi"],
                    help="When set, routes surrogate_ft results to "
                         "results/baseline/<dataset>/<baseline_method>/ "
                         "for side-by-side baseline comparison tables.")

    a = pa.parse_args()

    config_key, dataset, teacher_short = resolve_model(a.model, a.dataset)
    config = CONFIGS[config_key]

    temperatures = parse_list_int(a.temperatures)
    ft_fractions = parse_list_float(a.ft_fractions)

    # v6: auto-scale verify_temperature by log(C) when user passes -1 (default).
    if a.verify_temperature is not None and a.verify_temperature <= 0:
        auto_vT = recommended_vT(config["num_classes"], user_override=None)
        print("[v6] verify_temperature auto-scaled to {:.2f} (C={})".format(
            auto_vT, config["num_classes"]))
        a.verify_temperature = auto_vT
    else:
        # User-supplied; recommended_vT issues a soft warning if <5.
        a.verify_temperature = recommended_vT(
            config["num_classes"], user_override=a.verify_temperature)

    print("=" * 60)
    print("EvalGuard Experiment Runner (v5 - Logit-Space Shift)")
    print("  Dataset: {}".format(dataset))
    print("  Teacher: {} (config: {})".format(teacher_short, config_key))
    print("  Student: {}".format(a.student_arch))
    print("  Device: {}".format(a.device))
    print("  Experiment: {}".format(a.experiment))
    print("  Label mode: {}".format(a.label_mode))
    print("  Trigger mode: {} (own_data_source={})".format(a.trigger_mode, a.own_data_source))
    print("  rw={}, delta_logit={}, beta={}, delta_min={}, verify_T={}".format(
        a.rw, a.delta_logit, a.beta, a.delta_min, a.verify_temperature))
    print("  hard_label_mode={}, margin_tau_hard={}, n_random_kw={}".format(
        a.hard_label_mode, a.margin_tau_hard, a.n_random_kw))
    print("  tag={}".format(a.tag if a.tag else "(none)"))
    print("  seed={}".format(a.seed if a.seed is not None else "(random)"))
    print("  dist: epochs={}, lr={}, batch={}".format(a.dist_epochs, a.dist_lr, a.dist_batch))
    print("  ft: epochs={}, lr={}, fractions={}".format(a.ft_epochs, a.ft_lr, ft_fractions))
    print("  temperatures={}".format(temperatures))
    print("=" * 60)

    if dataset not in DATA_FN:
        print("[WARN] No data loader for dataset '{}'. Some experiments may fail.".format(dataset))
        trainset, testset, trainloader, testloader = None, None, None, None
    else:
        trainset, testset, trainloader, testloader = DATA_FN[dataset]()

    model, latent_layer, _ = get_model(config_key, a.pretrained)
    model.to(a.device)
    if testloader:
        print("Baseline accuracy: {:.2f}%\n".format(
            evaluate_accuracy(model, testloader, a.device) * 100))

    # Determine label modes to run
    if a.label_mode == "both":
        label_modes = ["soft", "hard"]
    else:
        label_modes = [a.label_mode]

    if a.experiment in ("fidelity", "all"):
        experiment_fidelity(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    if a.experiment in ("distill", "all"):
        for lm in label_modes:
            experiment_distillation(model, trainset, testloader, latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    temperatures, a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    delta_logit=a.delta_logit, beta=a.beta,
                                    delta_min=a.delta_min,
                                    verify_temperature=a.verify_temperature,
                                    label_mode=lm, tag=a.tag,
                                    seed=a.seed,
                                    hard_label_mode=a.hard_label_mode,
                                    margin_tau_hard=a.margin_tau_hard,
                                    query_dataset=a.query_dataset,
                                    hard_tau_quantile=a.hard_tau_quantile,
                                    hard_tau_calib_samples=a.hard_tau_calib_samples)

    if a.experiment in ("surrogate_ft", "all"):
        for lm in label_modes:
            experiment_surrogate_ft(model, trainset, testset, testloader, latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    temperatures, a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    ft_fractions, a.ft_epochs, a.ft_lr,
                                    delta_logit=a.delta_logit, beta=a.beta,
                                    delta_min=a.delta_min,
                                    verify_temperature=a.verify_temperature,
                                    label_mode=lm, trigger_mode=a.trigger_mode,
                                    own_trigger_size=a.own_trigger_size,
                                    rec_trigger_size=a.rec_trigger_size,
                                    own_data_source=a.own_data_source,
                                    tag=a.tag, seed=a.seed,
                                    hard_label_mode=a.hard_label_mode,
                                    margin_tau_hard=a.margin_tau_hard,
                                    hard_tau_quantile=a.hard_tau_quantile,
                                    hard_tau_calib_samples=a.hard_tau_calib_samples,
                                    baseline_method=a.baseline_method)

    if a.experiment in ("random_kw_fp", "all"):
        for lm in label_modes:
            experiment_random_kw_fp(model, trainset, testloader, latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    temperatures, a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    a.delta_logit, a.beta, a.delta_min,
                                    a.verify_temperature, lm, a.tag, a.seed,
                                    n_random_kw=a.n_random_kw,
                                    hard_label_mode=a.hard_label_mode,
                                    margin_tau_hard=a.margin_tau_hard,
                                    hard_tau_quantile=a.hard_tau_quantile,
                                    hard_tau_calib_samples=a.hard_tau_calib_samples)

    if a.experiment in ("pruning_attack", "all"):
        prune_fracs_parsed = parse_list_float(a.prune_fracs)
        for lm in label_modes:
            experiment_posthoc_attack(model, trainset, testloader, latent_layer, a.device,
                                      teacher_short, a.student_arch, dataset, config,
                                      temperatures, a.nq, a.rw,
                                      2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                      a.delta_logit, a.beta, a.delta_min,
                                      a.verify_temperature, lm, a.tag, a.seed,
                                      attack_type="prune",
                                      prune_fracs=prune_fracs_parsed,
                                      hard_label_mode=a.hard_label_mode,
                                      margin_tau_hard=a.margin_tau_hard,
                                      hard_tau_quantile=a.hard_tau_quantile,
                                      hard_tau_calib_samples=a.hard_tau_calib_samples)

    if a.experiment in ("quantization_attack", "all"):
        for lm in label_modes:
            experiment_posthoc_attack(model, trainset, testloader, latent_layer, a.device,
                                      teacher_short, a.student_arch, dataset, config,
                                      temperatures, a.nq, a.rw,
                                      2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                      a.delta_logit, a.beta, a.delta_min,
                                      a.verify_temperature, lm, a.tag, a.seed,
                                      attack_type="quantize",
                                      hard_label_mode=a.hard_label_mode,
                                      margin_tau_hard=a.margin_tau_hard,
                                      hard_tau_quantile=a.hard_tau_quantile,
                                      hard_tau_calib_samples=a.hard_tau_calib_samples)

    if a.experiment in ("overhead", "all"):
        experiment_overhead(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    if a.experiment in ("baseline_dawn",):
        for lm in label_modes:
            experiment_baseline_dawn(model, trainset, testset, testloader,
                                     latent_layer, a.device,
                                     teacher_short, a.student_arch, dataset, config,
                                     temperatures, a.nq, a.rw,
                                     2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                     ft_fractions, a.ft_epochs, a.ft_lr,
                                     delta_logit=a.delta_logit, beta=a.beta,
                                     delta_min=a.delta_min,
                                     verify_temperature=a.verify_temperature,
                                     label_mode=lm, tag=a.tag, seed=a.seed)

    if a.experiment in ("baseline_adi",):
        for lm in label_modes:
            experiment_baseline_adi(model, trainset, testset, testloader,
                                    latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    ft_fractions, a.ft_epochs, a.ft_lr,
                                    verify_temperature=a.verify_temperature,
                                    label_mode=lm,
                                    tag=a.tag, seed=a.seed,
                                    adi_n_triggers=a.adi_n_triggers,
                                    adi_retrain_epochs=a.adi_retrain_epochs,
                                    adi_lr=a.adi_lr)

    print("\nResults: ./results/{}/{}/   Checkpoints: ./checkpoints/distill/{}/".format(
        a.experiment, dataset, dataset))


if __name__ == "__main__":
    main()