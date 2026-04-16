"""
EvalGuard — E2 Attack Pipeline (Section VI-C)

[v5] Logit-space watermark embedding:
  - collect_soft_labels now passes raw logits to watermark module
  - Watermark is applied in logit space BEFORE softmax/temperature
  - This makes the watermark T-invariant

1. Soft-label distillation (E2s, Table VI):
   KL divergence on watermarked teacher outputs (Eq. 16)

2. Hard-label extraction (E2h):
   KnockoffNets-style, top-1 only, bypasses watermark

3. Surrogate fine-tuning (Table VII):
   Post-distillation FT to attempt watermark removal
"""
from __future__ import annotations

import copy
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

from .watermark import WatermarkModule, verify_ownership
from .crypto import keygen, kdf


# ============================================================
# 1. Soft-Label Knowledge Distillation (E2s)
# ============================================================

def collect_soft_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    device: str = "cpu",
    watermark_module: Optional[WatermarkModule] = None,
    temperature: float = 3.0,
) -> tuple:
    """
    Collect soft-label outputs from TEE (with watermark applied).

    [v5] Watermark is now applied in logit space:
      1. Get raw logits from teacher
      2. Pass logits to watermark module (adds delta_logit to target class)
      3. Watermark module applies softmax(logits / T) internally
      4. Return probabilities as soft labels

    Returns: (inputs, soft_labels, n_watermarked)
    """
    teacher.to(device).eval()
    all_inputs, all_soft_labels = [], []
    n_watermarked = 0

    with torch.no_grad():
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)

            if watermark_module is not None:
                # [v5] Pass raw logits + temperature to watermark module
                # Watermark adds delta in logit space, then applies softmax(logits/T)
                logits_np = logits.cpu().numpy()
                probs_np, n_wm = watermark_module.embed_batch_logits(
                    teacher, inputs, logits_np, temperature, device)
                n_watermarked += n_wm
                probs = torch.from_numpy(probs_np).float().to(device)
            else:
                # No watermark: standard softmax with temperature
                probs = torch.softmax(logits / temperature, dim=-1)

            all_inputs.append(inputs.cpu())
            all_soft_labels.append(probs.cpu())

    return torch.cat(all_inputs, 0), torch.cat(all_soft_labels, 0), n_watermarked


def soft_label_distillation(
    student: nn.Module,
    inputs: torch.Tensor,
    soft_labels: torch.Tensor,
    temperature: float = 3.0,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 0.002,
    device: str = "cpu",
) -> tuple:
    """
    Train surrogate via KL divergence on collected soft labels.
    Loss = T² · KL(softmax(z_teacher/T) || softmax(z_student/T))  [Eq. 16]

    Returns: (student, loss_history)
    """
    dataset = TensorDataset(inputs, soft_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student.to(device).train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_soft in loader:
            batch_x, batch_soft = batch_x.to(device), batch_soft.to(device)
            student_log_probs = torch.log_softmax(student(batch_x) / temperature, dim=-1)
            loss = temperature ** 2 * nn.functional.kl_div(
                student_log_probs, batch_soft, reduction="batchmean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    Distillation epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return student, loss_history


# ============================================================
# 2. Hard-Label Extraction (E2h)
# ============================================================

def collect_hard_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    device: str = "cpu",
    watermark_module: Optional[WatermarkModule] = None,
    use_bgs: bool = False,
) -> tuple:
    """
    Collect hard labels.

    Two modes:
      * use_bgs=False (legacy / vanilla extraction baseline):
          Returns natural argmax. Useful as the FP control demonstrating
          that argmax extraction strips a soft-label-only watermark.
      * use_bgs=True (Boundary-Gated Hard-Label Swap, v6):
          Routes the batch through `watermark_module.serve_hard_label_batch`,
          which selectively swaps to the HMAC-derived target class on
          Phi(x)=1 boundary samples (target-margin <= margin_tau_hard).

    Returns: (inputs, hard_labels, n_swapped)
    """
    teacher.to(device).eval()
    all_inputs, all_labels = [], []
    n_swapped = 0

    if use_bgs:
        if watermark_module is None:
            raise ValueError(
                "collect_hard_labels(use_bgs=True) requires watermark_module")

    with torch.no_grad():
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)

            if use_bgs:
                logits_np = logits.detach().cpu().numpy()
                preds_np = watermark_module.serve_hard_label_batch(
                    teacher, inputs, logits_np, device=device)
                preds = torch.from_numpy(preds_np).to(torch.long)
            else:
                preds = logits.argmax(dim=-1).cpu()

            all_inputs.append(inputs.cpu())
            all_labels.append(preds)

    if use_bgs:
        n_swapped = watermark_module.hardlabel_stats()["n_swapped"]

    return torch.cat(all_inputs, 0), torch.cat(all_labels, 0), n_swapped
def hard_label_extraction(
    student: nn.Module,
    inputs: torch.Tensor,
    hard_labels: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
) -> tuple:
    """
    Train surrogate using only hard labels (cross-entropy).
    Returns: (student, loss_history)
    """
    dataset = TensorDataset(inputs, hard_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student.to(device).train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(student(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    Hard-label epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return student, loss_history


# ============================================================
# 3. Surrogate Fine-Tuning (Table VII)
# ============================================================

def fine_tune_surrogate(
    surrogate: nn.Module,
    ft_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.0005,
    device: str = "cpu",
) -> tuple:
    """
    Post-distillation fine-tuning to attempt watermark removal.
    Returns: (surrogate, loss_history)
    """
    surrogate.to(device).train()
    optimizer = optim.SGD(surrogate.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in ft_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(surrogate(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(ft_loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    FT epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return surrogate, loss_history


# ============================================================
# 4. Post-hoc Weight Pruning Attack
# ============================================================

def prune_surrogate(
    surrogate: nn.Module,
    prune_frac: float,
    scope: str = "global",
) -> nn.Module:
    """
    Magnitude-based unstructured weight pruning on Conv/Linear layers.

    scope="global"   : single global threshold across all prunable weights.
    scope="per_layer": per-layer magnitude threshold.

    Returns the pruned module (parameters zeroed out, masks fused into .data
    so downstream inference behaves as a real pruned model).
    """
    if prune_frac <= 0.0:
        return surrogate
    if prune_frac >= 1.0:
        raise ValueError("prune_frac must be in [0, 1).")

    prunable = [m for m in surrogate.modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))]
    if not prunable:
        return surrogate

    if scope == "global":
        all_w = torch.cat([m.weight.detach().abs().flatten() for m in prunable])
        thr = torch.quantile(all_w, prune_frac)
        for m in prunable:
            with torch.no_grad():
                mask = m.weight.detach().abs() > thr
                m.weight.mul_(mask.to(m.weight.dtype))
    else:
        for m in prunable:
            w = m.weight.detach().abs().flatten()
            thr = torch.quantile(w, prune_frac)
            with torch.no_grad():
                mask = m.weight.detach().abs() > thr
                m.weight.mul_(mask.to(m.weight.dtype))

    return surrogate


# ============================================================
# 5. Post-hoc INT8 Weight Quantization Attack
# ============================================================

def quantize_surrogate_int8(surrogate: nn.Module) -> nn.Module:
    """
    Simple per-tensor symmetric INT8 weight quantization on Conv/Linear layers.

    This is a "fake-quant" round-trip: weights are quantized to int8 and
    dequantized back to float, so the forward pass is unchanged in dtype but
    the effective weight resolution is INT8. This is a strong baseline attack
    against logit-level watermarks because it perturbs every weight.
    """
    prunable = [m for m in surrogate.modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in prunable:
        with torch.no_grad():
            w = m.weight.detach()
            max_abs = w.abs().max()
            if max_abs.item() == 0.0:
                continue
            scale = max_abs / 127.0
            q = torch.round(w / scale).clamp_(-127, 127)
            m.weight.copy_(q * scale)
    return surrogate