"""
EvalGuard — Baselines (Section VI-G: Comparison with Existing Methods)

1. DAWN [Szyller et al., IEEE S&P 2021]:
   API-level watermark. Flips top-1 prediction for fraction r_w of queries.
   Fidelity loss ≈ r_w. Closest to EvalGuard's threat model.

2. Adi et al. [USENIX Security 2018]:
   Backdoor-based watermark. Injects trigger images with specific labels
   into training data. Requires retraining the teacher.

3. No Protection:
   Original model in TEE, no obfuscation, no watermark.
"""
from __future__ import annotations

import hmac
import hashlib
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .watermark_math import binomial_p_value


# ============================================================
# DAWN Baseline
# ============================================================

@dataclass
class DAWNTriggerEntry:
    """DAWN trigger: query + original top-1 + flipped class."""
    query: torch.Tensor
    original_class: int
    flipped_class: int


@dataclass
class DAWNWatermark:
    """
    DAWN: flip top-1 for r_w fraction of queries.

    Unlike EvalGuard which only permutes sub-dominant ranks,
    DAWN modifies the argmax → fidelity loss ≈ r_w.

    Parameters:
        K_w: watermark key for deterministic decision
        r_w: flip ratio (default 0.5%)
    """
    K_w: bytes
    r_w: float = 0.005
    trigger_set: List[DAWNTriggerEntry] = field(default_factory=list)

    def embed(self, x: torch.Tensor, p: np.ndarray) -> np.ndarray:
        """
        DAWN embedding: if Φ(x)=1, swap top-1 and top-2 probabilities.

        Returns modified probability vector.
        """
        x_hash = hashlib.sha256(x.cpu().numpy().tobytes()).digest()
        h = hmac.new(self.K_w, x_hash, hashlib.sha256).digest()
        decision_val = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))

        if decision_val >= threshold:
            return p  # No watermark

        q = p.copy()
        rho = np.argsort(-p)
        # Swap top-1 and top-2 probability values
        q[rho[0]], q[rho[1]] = p[rho[1]], p[rho[0]]

        self.trigger_set.append(DAWNTriggerEntry(
            query=x.clone(),
            original_class=int(rho[0]),
            flipped_class=int(rho[1]),
        ))
        return q

    def verify(
        self,
        suspect_model: nn.Module,
        device: str = "cpu",
        eta: float = 2 ** (-64),
        num_classes: int = 10,
    ) -> Dict:
        """
        DAWN verification: check if suspect model returns the flipped class.

        Args:
            num_classes: total number of classes (used for the H0 probability
                         p0 = 1/C). Defaults to 10 for CIFAR-10.
        """
        suspect_model.to(device).eval()
        n_match = 0

        for entry in self.trigger_set:
            x = entry.query.unsqueeze(0).to(device) if entry.query.dim() == 3 else entry.query.to(device)
            with torch.no_grad():
                pred = suspect_model(x).argmax(dim=-1).item()
            # DAWN checks if the suspect returns the flipped class
            if pred == entry.flipped_class:
                n_match += 1

        n_total = len(self.trigger_set)
        # Under H_0, probability of returning any specific class ≈ 1/C
        p0 = 1.0 / num_classes
        p_value = binomial_p_value(n_match, n_total, p0) if n_total > 0 else 1.0

        return {
            "verified": p_value < eta,
            "n_match": n_match,
            "n_total": n_total,
            "match_rate": n_match / n_total if n_total > 0 else 0.0,
            "p_value": p_value,
        }


def dawn_collect_hard_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    K_w: bytes,
    r_w: float,
    num_classes: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, "DAWNWatermark"]:
    """
    Collect hard labels from a teacher using DAWN watermarking.

    DAWN flips the argmax for r_w fraction of queries. The attacker
    receives these (possibly flipped) hard labels for distillation.

    Returns: (inputs, hard_labels, dawn_wm_instance)
    """
    teacher.to(device).eval()
    dawn = DAWNWatermark(K_w=K_w, r_w=r_w)
    all_inputs, all_labels = [], []

    with torch.no_grad():
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            batch_labels = []
            for i in range(inputs.size(0)):
                q = dawn.embed(inputs[i], probs[i])
                batch_labels.append(np.argmax(q))

            all_inputs.append(inputs.cpu())
            all_labels.append(torch.tensor(batch_labels, dtype=torch.long))

    return torch.cat(all_inputs, 0), torch.cat(all_labels, 0), dawn


def dawn_collect_soft_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    K_w: bytes,
    r_w: float,
    temperature: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, "DAWNWatermark"]:
    """
    Collect soft labels from a teacher using DAWN watermarking.

    For triggered queries, DAWN swaps the top-1 and top-2 probabilities.
    Returns: (inputs, soft_labels, dawn_wm_instance)
    """
    teacher.to(device).eval()
    dawn = DAWNWatermark(K_w=K_w, r_w=r_w)
    all_inputs, all_soft = [], []

    with torch.no_grad():
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)
            probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()

            batch_soft = []
            for i in range(inputs.size(0)):
                q = dawn.embed(inputs[i], probs[i])
                batch_soft.append(q)

            all_inputs.append(inputs.cpu())
            all_soft.append(torch.from_numpy(np.stack(batch_soft)).float())

    return torch.cat(all_inputs, 0), torch.cat(all_soft, 0), dawn


def measure_dawn_fidelity(
    model: nn.Module,
    testloader,
    K_w: bytes,
    r_w: float = 0.005,
    device: str = "cpu",
) -> Dict:
    """
    Measure DAWN's fidelity loss: fraction of queries where
    the watermarked output disagrees with the original top-1.

    Expected: fidelity_loss ≈ r_w (Table IV).
    """
    model.to(device).eval()
    dawn = DAWNWatermark(K_w=K_w, r_w=r_w)

    total = 0
    flipped = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i in range(inputs.size(0)):
                original_pred = np.argmax(probs[i])
                q = dawn.embed(inputs[i], probs[i])
                watermarked_pred = np.argmax(q)

                if watermarked_pred != original_pred:
                    flipped += 1
                total += 1

    return {
        "total_queries": total,
        "flipped": flipped,
        "fidelity_loss": flipped / total if total > 0 else 0.0,
        "expected_loss": r_w,
        "n_triggers": len(dawn.trigger_set),
    }


# ============================================================
# Comparison Table (Table IX)
# ============================================================

COMPARISON_TABLE = {
    "NNSplitter":  {"type": "Obfuscation",  "TF": True,  "fid": True,  "E01": "~", "E2": False, "verif": None},
    "O2Splitter":  {"type": "Obfuscation",  "TF": True,  "fid": True,  "E01": True, "E2": False, "verif": None},
    "Uchida":      {"type": "Param WM",     "TF": False, "fid": True,  "E01": False,"E2": False, "verif": "WB"},
    "Adi":         {"type": "Param WM",     "TF": False, "fid": "~",   "E01": False,"E2": False, "verif": "BB"},
    "EWE":         {"type": "Param WM",     "TF": False, "fid": "~",   "E01": False,"E2": "~",   "verif": "BB"},
    "DAWN":        {"type": "Output WM",    "TF": True,  "fid": False, "E01": False,"E2": True,  "verif": "BB"},
    "GINSEW":      {"type": "Output WM",    "TF": True,  "fid": True,  "E01": False,"E2": True,  "verif": "WB"},
    "EaaW":        {"type": "Output WM",    "TF": True,  "fid": True,  "E01": False,"E2": True,  "verif": "WB"},
    "ModelGuard":  {"type": "Perturbation", "TF": True,  "fid": False, "E01": False,"E2": "~",   "verif": None},
    "EvalGuard":   {"type": "Obf+WM",      "TF": True,  "fid": True,  "E01": True, "E2": True,  "verif": "BB"},
}


def print_comparison_table():
    """Print Table IX from the paper."""
    header = f"{'Method':<12} {'Type':<14} {'TF':>3} {'Fid':>4} {'E0/1':>5} {'E2':>4} {'Verif':>6}"
    print(header)
    print("-" * len(header))
    for name, attrs in COMPARISON_TABLE.items():
        def fmt(v):
            if v is True: return "✓"
            if v is False: return "✗"
            if v is None: return "✗"
            return str(v)
        print(f"{name:<12} {attrs['type']:<14} {fmt(attrs['TF']):>3} "
              f"{fmt(attrs['fid']):>4} {fmt(attrs['E01']):>5} "
              f"{fmt(attrs['E2']):>4} {fmt(attrs['verif']):>6}")


# ============================================================
# Adi et al. (2018) — Backdoor-Based Watermark
# ============================================================

@dataclass
class AdiTriggerSet:
    """Adi trigger images + their assigned labels."""
    images: torch.Tensor
    labels: torch.Tensor


class AdiWatermark:
    """
    Adi et al. "Turning Your Weakness Into a Strength" (USENIX Security 2018).

    Embeds a watermark by injecting trigger images with specific labels
    into the teacher's training set and fine-tuning. Verification checks
    whether a suspect model classifies triggers with the assigned labels.

    Key difference from EvalGuard: requires retraining the teacher model.
    """

    def __init__(self, num_classes: int, n_triggers: int = 100,
                 img_shape: Tuple[int, ...] = (3, 32, 32), seed: int = 42):
        self.num_classes = num_classes
        self.n_triggers = n_triggers
        self.img_shape = img_shape
        self.trigger_set = self._generate_triggers(seed)

    def _generate_triggers(self, seed: int) -> AdiTriggerSet:
        rng = torch.Generator().manual_seed(seed)
        images = torch.rand(self.n_triggers, *self.img_shape, generator=rng)
        labels = torch.randint(0, self.num_classes, (self.n_triggers,),
                               generator=rng)
        return AdiTriggerSet(images=images, labels=labels)

    def embed_backdoor(
        self,
        teacher: nn.Module,
        trainloader: DataLoader,
        epochs: int = 30,
        lr: float = 0.001,
        poison_ratio: float = 0.1,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Fine-tune the teacher to memorize trigger → label mapping.

        Mixes trigger samples into each training batch at poison_ratio.
        Returns the backdoored teacher (in-place modification of a deepcopy).
        """
        teacher = copy.deepcopy(teacher)
        teacher.to(device).train()
        optimizer = optim.SGD(teacher.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        trigger_imgs = self.trigger_set.images.to(device)
        trigger_lbls = self.trigger_set.labels.to(device)
        n_trig = len(trigger_imgs)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in trainloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                n_poison = max(1, int(batch_x.size(0) * poison_ratio))
                idx = torch.randint(0, n_trig, (n_poison,))
                x = torch.cat([batch_x, trigger_imgs[idx]], dim=0)
                y = torch.cat([batch_y, trigger_lbls[idx]], dim=0)

                optimizer.zero_grad()
                loss = criterion(teacher(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print("    [Adi retrain] Epoch {}/{}, Loss: {:.4f}".format(
                    epoch + 1, epochs, total_loss / len(trainloader)))

        return teacher

    def verify(
        self,
        suspect_model: nn.Module,
        device: str = "cpu",
        eta: float = 2 ** (-64),
    ) -> Dict:
        """
        Check if suspect model classifies trigger images with assigned labels.

        Under H0 (no backdoor), each trigger has 1/C chance of matching.
        Uses binomial exact test.
        """
        suspect_model.to(device).eval()
        trigger_imgs = self.trigger_set.images.to(device)
        trigger_lbls = self.trigger_set.labels

        with torch.no_grad():
            preds = suspect_model(trigger_imgs).argmax(dim=-1).cpu()

        n_match = (preds == trigger_lbls).sum().item()
        n_total = len(trigger_lbls)
        p0 = 1.0 / self.num_classes
        p_value = binomial_p_value(n_match, n_total, p0) if n_total > 0 else 1.0

        return {
            "method": "adi_backdoor",
            "verified": p_value < eta,
            "n_match": n_match,
            "n_total": n_total,
            "match_rate": n_match / n_total if n_total > 0 else 0.0,
            "baseline_rate": p0,
            "p_value": p_value,
        }

    def verify_on_teacher(self, teacher: nn.Module, device: str = "cpu") -> float:
        """Check backdoor success rate on the (re-trained) teacher itself."""
        teacher.to(device).eval()
        trigger_imgs = self.trigger_set.images.to(device)
        trigger_lbls = self.trigger_set.labels
        with torch.no_grad():
            preds = teacher(trigger_imgs).argmax(dim=-1).cpu()
        return (preds == trigger_lbls).float().mean().item()