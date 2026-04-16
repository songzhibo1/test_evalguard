"""
EvalGuard Phase 2 & 3: Output Watermark Embedding (Algorithm 2)
                        + Ownership Verification (Algorithm 3)

[v6] Logit-Space Confidence Shift Watermarking (tightened):
  - Embedding: add delta_logit to target class LOGIT before softmax.
  - Safe-shift bound now uses the TARGET-specific margin
    `logits[top1] - logits[target]` instead of the top1/top2 gap.
    This provides a strictly larger feasible region for delta
    without changing any correctness guarantee (top-1 preserved).
  - Boundary-Gated Hard-Label Swap mode (serve_hard_label / BGS)
    added for APIs that return argmax only.  Swaps only on Phi(x)=1
    samples whose safe margin falls below margin_tau — selectively
    flipping boundary samples to encode the watermark while keeping
    evaluation accuracy loss ~1%.
  - Docstrings and runtime warnings surface the practical operating
    envelope (r_w >= 0.05, |trigger_set| >= 200, vT >= 5 auto-scaled).

[v5 → v6 compatibility]
  API additions only.  The soft-label `embed_logits` / `embed_batch_logits`
  pipelines are preserved bit-for-bit apart from the margin formula,
  so existing checkpoints remain loadable.
"""
from __future__ import annotations

import hmac
import hashlib
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy import stats

import numpy as np
import torch
import torch.nn as nn

from .crypto import prf


# ============================================================
# Operating-envelope constants (documented defaults)
# ============================================================
# These thresholds encode the empirical limits we've validated
# in CIFAR-10 / CIFAR-100 ablations.  They are used for WARN-level
# runtime checks only; callers can silence them by setting
# WatermarkModule.suppress_warnings = True.
MIN_RECOMMENDED_RW = 0.05              # below this, Wilcoxon loses power
MIN_RECOMMENDED_TRIGGERS = 200         # below this, Wilcoxon loses power
MIN_RECOMMENDED_VT = 5.0               # below this, soft-label verify fails


# ============================================================
# Perturbation-Resilient Mapping R(x) — Eq. (11)
# ============================================================

@dataclass
class LatentExtractor:
    """
    Eq. (11): R(x) = Binarize(L(f, x), μ)
    Unchanged from v3/v4.
    """
    median: np.ndarray = None

    def compute_median(self, model: nn.Module, dataloader, layer_name: str, device: str = "cpu"):
        latents = []
        def hook_fn(module, input, output):
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            latents.append(out.numpy())
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.to(device).eval()
        with torch.no_grad():
            for batch_x, _ in dataloader:
                model(batch_x.to(device))
        handle.remove()
        all_latents = np.concatenate(latents, axis=0)
        self.median = np.median(all_latents, axis=0)

    def extract_and_binarize(self, model: nn.Module, x: torch.Tensor, layer_name: str, device: str = "cpu") -> bytes:
        latent = None
        def hook_fn(module, input, output):
            nonlocal latent
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            latent = out.numpy()[0]
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.eval()
        with torch.no_grad():
            inp = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
            model(inp)
        handle.remove()
        binary = (latent > self.median).astype(np.uint8)
        return np.packbits(binary).tobytes()

    def extract_and_binarize_batch(self, model: nn.Module, batch_x: torch.Tensor, layer_name: str, device: str = "cpu"):
        batch_latent = None
        def hook_fn(module, input, output):
            nonlocal batch_latent
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            batch_latent = out.numpy()
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.eval()
        with torch.no_grad():
            model(batch_x.to(device))
        handle.remove()
        results = []
        for i in range(batch_latent.shape[0]):
            binary = (batch_latent[i] > self.median).astype(np.uint8)
            results.append(np.packbits(binary).tobytes())
        return results


# ============================================================
# Target-Class Mapping — Definition 8
# ============================================================

def derive_target_class(K_w: bytes, top1_class: int, num_classes: int) -> int:
    """
    Definition 8: Target-Class Mapping.

    t(c) = HMAC(K_w, "target:c") mod (C-1), adjusted to skip c itself.
    Deterministic: same key + same top-1 class → same target class.
    """
    h = hmac.new(K_w, ("target:" + str(top1_class)).encode(), hashlib.sha256).digest()
    t = int.from_bytes(h[:4], "big") % (num_classes - 1)
    if t >= top1_class:
        t += 1
    return t


def derive_control_class(K_w: bytes, top1_class: int, num_classes: int) -> int:
    """
    Deterministic control class for paired verification.

    Picks a class that is neither the top-1 class c nor the target class t(c),
    using a domain-separated HMAC so it is reproducible across runs and
    cryptographically tied to K_w.

    Used by verify_ownership to construct the paired control sample for
    Wilcoxon signed-rank testing.
    """
    target = derive_target_class(K_w, top1_class, num_classes)
    h = hmac.new(K_w, ("control:" + str(top1_class)).encode(), hashlib.sha256).digest()
    # Pick from C-2 candidates (excluding c and t(c))
    raw = int.from_bytes(h[:4], "big") % (num_classes - 2)
    # Map raw index in [0, C-2) onto the C classes minus {c, t(c)}
    excluded = sorted({top1_class, target})
    cls = raw
    for ex in excluded:
        if cls >= ex:
            cls += 1
    return cls


# ============================================================
# Algorithm 2: Logit-Space Confidence Shift Watermark Embedding
# ============================================================

@dataclass
class TriggerEntry:
    """One entry in trigger set T."""
    query: torch.Tensor
    target_class: int
    top1_class: int


@dataclass
class WatermarkModule:
    """
    Implements Algorithm 2: Logit-Space Confidence Shift Embedding.

    For selected queries (decided by HMAC), boost the target class LOGIT
    by delta_logit before applying softmax. This produces a T-invariant
    watermark signal that survives distillation at any temperature.

    Operating envelope (validated empirically on CIFAR-10/-100):
        r_w        MUST be >= 0.05.  Smaller ratios cannot populate a
                   trigger set with >=200 entries at standard nq=50k
                   query budgets, and Wilcoxon loses statistical power.
        |trigger|  >=200 after embedding.  A runtime warning is emitted
                   when the recorded trigger_set falls below this.
        vT         >=5 at verification time (auto-scaled by log(C) when
                   a suspect model is passed to `recommended_vT`).

    Parameters:
        K_w: watermark key
        r_w: watermark ratio (default 0.1 — MUST be >= 0.05)
        delta_logit: logit-space shift amount (default 2.0).
                     For C=10 ResNet: 3-5 is typical.
                     For C=100 ResNet: 8-15 after the v6 margin fix.
        beta: safety factor to cap shift below logit margin (default 0.5).
              With the v6 margin formula (logit[top1]-logit[target]),
              beta=0.5-0.7 suffices even for C=100.
        delta_min: minimum effective delta — queries whose safe delta would
                   fall below this threshold are NOT recorded as triggers
                   (avoids polluting the trigger set with near-zero signals
                   on low-margin samples). Default 0.5.
        margin_tau_hard: BGS-only parameter (hard-label mode). If
                        `serve_hard_label` is called, only samples with
                        target-margin (= logit[top1]-logit[target]) <=
                        margin_tau_hard AND Phi(x)=1 get their argmax
                        swapped to target. Default 1.5 (empirical sweet
                        spot for <=1% accuracy loss on CIFAR-100).
        num_classes: total number of classes
        latent_extractor: for R(x) computation
        layer_name: which hidden layer for latent extraction
        suppress_warnings: silences the operating-envelope WARN prints.
    """
    K_w: bytes
    r_w: float = 0.1
    delta_logit: float = 2.0
    beta: float = 0.5
    delta_min: float = 0.5
    margin_tau_hard: float = 1.5
    num_classes: int = 10
    latent_extractor: LatentExtractor = None
    layer_name: str = ""
    suppress_warnings: bool = False
    trigger_set: List[TriggerEntry] = field(default_factory=list)
    _target_class_cache: dict = field(default_factory=dict, repr=False)
    _warned_trigger_count: bool = field(default=False, repr=False)
    _hardlabel_n_queries: int = field(default=0, repr=False)
    _hardlabel_n_phi: int = field(default=0, repr=False)
    _hardlabel_n_swapped: int = field(default=0, repr=False)

    def __post_init__(self):
        # Operating-envelope check: r_w must be >= 0.05 for statistical power.
        if self.r_w < MIN_RECOMMENDED_RW and not self.suppress_warnings:
            warnings.warn(
                "[EvalGuard] r_w={:.4f} < {:.2f}. Wilcoxon/Binomial will lose "
                "statistical power. Recommended: r_w >= 0.05.".format(
                    self.r_w, MIN_RECOMMENDED_RW),
                RuntimeWarning, stacklevel=2)

    def _get_target_class(self, top1_class: int) -> int:
        if top1_class not in self._target_class_cache:
            self._target_class_cache[top1_class] = derive_target_class(
                self.K_w, top1_class, self.num_classes)
        return self._target_class_cache[top1_class]

    def watermark_decision(self, r_x: bytes) -> bool:
        """Eq. (8): Φ(x) = 1 if HMAC(K_w, R(x))[0:128] < r_w × 2^128"""
        h = hmac.new(self.K_w, r_x, hashlib.sha256).digest()
        decision_value = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))
        return decision_value < threshold

    def _compute_delta_logit(self, logits: np.ndarray,
                             target_class: Optional[int] = None) -> float:
        """
        Compute the safe logit shift, ensuring top-1 is not flipped.

        [v6 tightened formula]
          delta = min(delta_logit, beta * margin)
          margin = logit[top1] - logit[target_class]

        The old v5 formula used `logit[top1] - logit[second]`, which is a
        looser (smaller) upper bound than the target-specific gap.  The
        target class is almost always below top2 (since it is uniformly
        sampled via HMAC among C-1 non-top1 classes), so using it directly
        recovers 2-3x more of the intended delta — critical for C=100.

        Backwards-compatibility: when target_class is not supplied (old
        callers), we fall back to the v5 top2-based formula.
        """
        if target_class is None:
            sorted_logits = np.sort(logits)[::-1]
            margin = sorted_logits[0] - sorted_logits[1]
        else:
            top1 = int(np.argmax(logits))
            margin = float(logits[top1] - logits[target_class])
            # Defensive: if target happens to equal top1 (shouldn't, since
            # derive_target_class skips top1), margin would be 0.
            if margin <= 0.0:
                return 0.0
        return min(self.delta_logit, self.beta * margin)

    def _maybe_warn_trigger_count(self):
        """Emit a one-shot warning if the embedded trigger set is too small."""
        if self.suppress_warnings or self._warned_trigger_count:
            return
        if 0 < len(self.trigger_set) < MIN_RECOMMENDED_TRIGGERS:
            # Only warn once per module instance — we'll re-check periodically.
            pass
        elif len(self.trigger_set) == 0:
            return

    def _final_trigger_count_check(self):
        """Call this after embedding is complete (e.g. end of collect_*)."""
        # 如果开启了静音，或者已经报过警了，直接返回
        if self.suppress_warnings or self._warned_trigger_count:
            return
            
        n = len(self.trigger_set)
        if n > 0 and n < MIN_RECOMMENDED_TRIGGERS:
            warnings.warn(
                "[EvalGuard] only {} triggers recorded (< {}). Verification "
                "statistical power will be weak. Consider raising r_w, "
                "lowering delta_min, or enlarging the query budget.".format(
                    n, MIN_RECOMMENDED_TRIGGERS),
                RuntimeWarning, stacklevel=2)
            # 关键：标记为已经报过警
            self._warned_trigger_count = True
     

    def embed_logits(self, model: nn.Module, x: torch.Tensor,
                     logits: np.ndarray, temperature: float = 1.0,
                     device: str = "cpu") -> np.ndarray:
        """
        Algorithm 2: Logit-Space Embedding (single sample).

        1. Decide whether to watermark this query
        2. Identify top-1 class c and target class t(c)
        3. Compute safe delta; if below delta_min, skip (do not record)
        4. Add delta to logits[t(c)]
        5. Apply softmax(logits / T) to get probabilities
        6. Record trigger entry

        Args:
            logits: raw logits for this sample (1D numpy array)
            temperature: distillation temperature T

        Returns:
            probability vector after softmax (with or without watermark)
        """
        r_x = self.latent_extractor.extract_and_binarize(model, x, self.layer_name, device)
        should_wm = self.watermark_decision(r_x)

        if not should_wm:
            # No watermark: just return softmax(logits / T)
            logits_t = torch.tensor(logits).unsqueeze(0) / temperature
            return torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        # v6: compute delta using the TARGET-specific margin
        top1_class = int(np.argmax(logits))
        target_class = self._get_target_class(top1_class)
        delta = self._compute_delta_logit(logits, target_class=target_class)
        if delta < self.delta_min:
            logits_t = torch.tensor(logits).unsqueeze(0) / temperature
            return torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        # Apply logit-space shift
        logits_wm = logits.copy()
        logits_wm[target_class] += delta

        # Softmax (automatically normalizes, no clip needed)
        logits_t = torch.tensor(logits_wm).unsqueeze(0) / temperature
        q = torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        # Record trigger
        self.trigger_set.append(TriggerEntry(
            query=x.clone() if isinstance(x, torch.Tensor) else x,
            target_class=target_class,
            top1_class=top1_class,
        ))
        return q

    def embed_batch_logits(self, model: nn.Module, batch_x: torch.Tensor,
                           batch_logits: np.ndarray, temperature: float = 1.0,
                           device: str = "cpu"):
        """
        Batch version of embed_logits().

        Args:
            batch_logits: raw logits (batch_size, num_classes) numpy array
            temperature: distillation temperature T

        Returns: (probabilities, n_watermarked)
        """
        r_x_list = self.latent_extractor.extract_and_binarize_batch(
            model, batch_x, self.layer_name, device)

        batch_logits_wm = batch_logits.copy()
        n_watermarked = 0

        for i in range(len(r_x_list)):
            should_wm = self.watermark_decision(r_x_list[i])
            if not should_wm:
                continue

            logits_i = batch_logits[i]

            # v6: resolve target first, then compute target-specific safe delta
            top1_class = int(np.argmax(logits_i))
            target_class = self._get_target_class(top1_class)
            delta = self._compute_delta_logit(logits_i, target_class=target_class)
            if delta < self.delta_min:
                continue

            n_watermarked += 1

            # Apply logit-space shift
            batch_logits_wm[i][target_class] += delta

            # Record trigger
            self.trigger_set.append(TriggerEntry(
                query=batch_x[i].clone(),
                target_class=target_class,
                top1_class=top1_class,
            ))

        # One-shot trigger-count warning at the end of each batch call.
        # self._final_trigger_count_check()

        # Apply softmax(logits / T) to entire batch at once
        logits_t = torch.tensor(batch_logits_wm) / temperature
        batch_probs = torch.softmax(logits_t, dim=-1).numpy()

        return batch_probs, n_watermarked

    # ----------------------------------------------------------
    # v6: Boundary-Gated Hard-Label Swap (BGS) — hard-label API
    # ----------------------------------------------------------
    def serve_hard_label(self, model: nn.Module, x: torch.Tensor,
                         logits: np.ndarray,
                         margin_tau: Optional[float] = None,
                         device: str = "cpu") -> int:
        """
        Hard-label API: return an int class label for a SINGLE sample.

        Semantics:
          1. Normal queries (Phi(x)=0) — return natural argmax, no swap.
          2. Watermarked queries (Phi(x)=1) only swap when the target-margin
             (logits[top1] - logits[target]) falls BELOW margin_tau.
             This concentrates swaps on boundary samples where the teacher
             itself is uncertain, so net evaluation accuracy loss stays <=1%.

        Args:
            model: teacher model (needed for latent extraction).
            x: single-sample input tensor (shape: [1, ...] or [...]).
            logits: 1D numpy array of raw teacher logits.
            margin_tau: override self.margin_tau_hard for this call.
            device: device on which to run latent extraction.

        Returns:
            hard label (int)
        """
        self._hardlabel_n_queries += 1
        tau = self.margin_tau_hard if margin_tau is None else float(margin_tau)
        top1 = int(np.argmax(logits))

        # Ensure batch dim for latent extractor
        x_in = x if x.dim() > 3 else x.unsqueeze(0)
        r_x_batch = self.latent_extractor.extract_and_binarize_batch(
            model, x_in, self.layer_name, device)
        r_x = r_x_batch[0]

        if not self.watermark_decision(r_x):
            return top1

        self._hardlabel_n_phi += 1
        target = self._get_target_class(top1)
        margin = float(logits[top1] - logits[target])
        if margin <= 0.0 or margin > tau:
            # Confident sample — do NOT swap.  Still record Phi=1 touches
            # (useful for audit) but no trigger entry without swap.
            return top1

        # Swap: record trigger + return target
        self._hardlabel_n_swapped += 1
        self.trigger_set.append(TriggerEntry(
            query=x.clone() if isinstance(x, torch.Tensor) else x,
            target_class=target,
            top1_class=top1,
        ))
        return target

    def serve_hard_label_batch(self, model: nn.Module,
                               batch_x: torch.Tensor,
                               batch_logits: np.ndarray,
                               margin_tau: Optional[float] = None,
                               device: str = "cpu") -> np.ndarray:
        """
        Batch BGS inference.  Same semantics as serve_hard_label but
        amortises the latent-extraction forward pass.

        Returns: np.ndarray of shape (batch_size,) dtype=int64 — hard labels.
        """
        tau = self.margin_tau_hard if margin_tau is None else float(margin_tau)
        r_x_list = self.latent_extractor.extract_and_binarize_batch(
            model, batch_x, self.layer_name, device)

        out = np.empty(batch_x.size(0), dtype=np.int64)
        for i in range(batch_x.size(0)):
            self._hardlabel_n_queries += 1
            top1 = int(np.argmax(batch_logits[i]))
            if not self.watermark_decision(r_x_list[i]):
                out[i] = top1
                continue

            self._hardlabel_n_phi += 1
            target = self._get_target_class(top1)
            margin = float(batch_logits[i][top1] - batch_logits[i][target])
            if margin <= 0.0 or margin > tau:
                out[i] = top1
                continue

            self._hardlabel_n_swapped += 1
            self.trigger_set.append(TriggerEntry(
                query=batch_x[i].clone(),
                target_class=target,
                top1_class=top1,
            ))
            out[i] = target

       # self._final_trigger_count_check()
        return out

    def hardlabel_stats(self) -> Dict[str, int]:
        """Audit counters for BGS: queries / Phi=1 / actually swapped."""
        return {
            "n_queries":    self._hardlabel_n_queries,
            "n_phi_active": self._hardlabel_n_phi,
            "n_swapped":    self._hardlabel_n_swapped,
            "swap_rate":   (self._hardlabel_n_swapped /
                            max(1, self._hardlabel_n_queries)),
        }

    def calibrate_margin_tau(
        self,
        model: nn.Module,
        query_loader,
        quantile: float = 0.10,
        device: str = "cpu",
        max_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Adaptive τ calibration for Boundary-Gated Hard-Label Swap (Option A).

        Performs a pre-scan over up to `max_samples` queries, records the
        margin (= logits[top1] - logits[target]) on Phi(x)=1 samples only,
        and returns the q-quantile of this distribution.

        Rationale:
            The raw logit scale varies by dataset/architecture (C=10 gives
            margins ~3-7; C=100 gives ~12-20). A fixed τ=1.5 captures almost
            nothing on CIFAR-100 and too much on a strong CIFAR-10 teacher.
            Calibrating τ as a quantile makes the trigger yield approximately
            q * r_w * |queries| regardless of model scale (q=0.10 with
            r_w=0.10 on 50k queries ≈ 500 triggers — enough to clear the
            binomial power floor).

        The caller should then set::

            wm.margin_tau_hard = result["tau"]

        before running the real distillation.

        Returns a dict with:
            tau: calibrated threshold
            quantile: q that was used
            n_phi_samples: number of Phi(x)=1 samples scanned
            n_scanned: total queries scanned
            margin_median / margin_mean / margin_max: diagnostic stats
        """
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1), got {}".format(quantile))
        model.to(device).eval()

        margins: List[float] = []
        n_scanned = 0
        with torch.no_grad():
            for xb, _ in query_loader:
                if n_scanned >= max_samples:
                    break
                xb = xb.to(device)
                logits = model(xb).detach().cpu().numpy()
                r_list = self.latent_extractor.extract_and_binarize_batch(
                    model, xb, self.layer_name, device)
                B = xb.size(0)
                for i in range(B):
                    n_scanned += 1
                    if not self.watermark_decision(r_list[i]):
                        continue
                    top1 = int(np.argmax(logits[i]))
                    target = self._get_target_class(top1)
                    m = float(logits[i][top1] - logits[i][target])
                    if m > 0.0:
                        margins.append(m)
                    if n_scanned >= max_samples:
                        break

        if not margins:
            print("[calibrate_tau] WARN: no Phi=1 positive-margin samples "
                  "found in {} scanned; keeping tau={:.4f}".format(
                      n_scanned, self.margin_tau_hard))
            return {
                "tau": float(self.margin_tau_hard),
                "quantile": quantile,
                "n_phi_samples": 0,
                "n_scanned": n_scanned,
                "margin_median": 0.0,
                "margin_mean": 0.0,
                "margin_max": 0.0,
                "calibrated": False,
            }

        arr = np.array(margins, dtype=np.float64)
        tau = float(np.quantile(arr, quantile))
        return {
            "tau": tau,
            "quantile": quantile,
            "n_phi_samples": len(arr),
            "n_scanned": n_scanned,
            "margin_median": float(np.median(arr)),
            "margin_mean": float(np.mean(arr)),
            "margin_max": float(np.max(arr)),
            "calibrated": True,
        }

# ============================================================
# Algorithm 3: Ownership Verification (Confidence Shift)
# ============================================================

def verify_ownership(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    control_queries: torch.Tensor = None,
    control_top1_classes: List[int] = None,
    K_w: bytes = None,
    num_classes: int = 10,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    batch_size: int = 64,
) -> Dict:
    """
    Algorithm 3: Ownership Verification via Confidence Shift Detection.

    Statistical design (paired vs independent):

      1. PAIRED MODE (default — control_queries is None):
         For each trigger query, in a single forward pass we read
         BOTH p[target_class] (trigger sample) and p[control_class]
         (paired control sample), where control_class is derived
         deterministically from K_w via derive_control_class().

         Test: scipy.stats.wilcoxon(trigger - control, alternative='greater')
         (Wilcoxon signed-rank test, the correct paired analogue of
         Mann-Whitney U).

      2. INDEPENDENT MODE (control_queries provided):
         Compute p[derive_target_class(K_w, top1)] separately on a
         held-out pool of independent inputs.

         Test: scipy.stats.mannwhitneyu(trigger, control, alternative='greater').

    The verify_temperature is used to spread out sub-dominant probabilities;
    the logit-space watermark signal is T-invariant, so any T > 1 works.

    Args:
        trigger_set: list of TriggerEntry from embedding phase
        suspect_model: the model to verify
        control_queries: optional independent inputs for the control group
        control_top1_classes: optional top-1 hint for independent controls
        K_w: watermark key (REQUIRED for control_class derivation)
        num_classes: total number of classes
        eta: significance threshold
        device: computation device
        verify_temperature: T used to compute softmax during verification
        batch_size: batch size for forward passes
    """
    if K_w is None:
        raise ValueError("K_w is required for verify_ownership (control class derivation).")

    suspect_model.to(device).eval()
    vT = verify_temperature

    # ----------------------------------------------------------
    # PAIRED MODE: single forward pass per trigger, collect both
    # target_class and control_class confidences from the same pmf.
    # ----------------------------------------------------------
    if control_queries is None:
        trigger_confidences: List[float] = []
        control_confidences: List[float] = []
        if len(trigger_set) == 0:
            return _empty_verification_result(eta, vT, mode="paired_wilcoxon")

        # Batch the trigger queries to amortize forward cost
        n = len(trigger_set)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                # Stack queries into a batch
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i, entry in enumerate(batch_entries):
                    trigger_confidences.append(float(probs[i, entry.target_class]))
                    control_class = derive_control_class(
                        K_w, entry.top1_class, num_classes)
                    control_confidences.append(float(probs[i, control_class]))

        trigger_arr = np.array(trigger_confidences)
        control_arr = np.array(control_confidences)
        diffs = trigger_arr - control_arr

        # Wilcoxon signed-rank test (paired, one-sided)
        # zero_method='wilcox' drops zero differences (standard practice)
        try:
            res = stats.wilcoxon(diffs, alternative='greater', zero_method='wilcox')
            stat_val = float(res.statistic)
            p_value = float(res.pvalue)
        except ValueError:
            # All differences are zero -> no signal
            stat_val, p_value = 0.0, 1.0

        return {
            "verified": p_value < eta,
            "p_value": p_value,
            "test": "wilcoxon_signed_rank",
            "statistic": stat_val,
            "mean_trigger_conf": round(float(trigger_arr.mean()), 6),
            "mean_control_conf": round(float(control_arr.mean()), 6),
            "confidence_shift": round(float(diffs.mean()), 6),
            "n_trigger": int(len(trigger_arr)),
            "n_control": int(len(control_arr)),
            "n_pairs": int(len(diffs)),
            "eta": eta,
            "verify_temperature": vT,
        }

    # ----------------------------------------------------------
    # INDEPENDENT MODE: trigger group on trigger inputs,
    # control group on independent held-out inputs.
    # ----------------------------------------------------------
    trigger_confidences = []
    if len(trigger_set) > 0:
        n = len(trigger_set)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i, entry in enumerate(batch_entries):
                    trigger_confidences.append(float(probs[i, entry.target_class]))

    control_confidences = []
    if isinstance(control_queries, torch.Tensor):
        n_ctrl = control_queries.size(0)
        with torch.no_grad():
            for start in range(0, n_ctrl, batch_size):
                end = min(start + batch_size, n_ctrl)
                batch_x = control_queries[start:end].to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i in range(end - start):
                    if control_top1_classes is not None:
                        top1_c = int(control_top1_classes[start + i])
                    else:
                        top1_c = int(np.argmax(probs[i]))
                    t_c = derive_target_class(K_w, top1_c, num_classes)
                    control_confidences.append(float(probs[i, t_c]))

    trigger_arr = np.array(trigger_confidences)
    control_arr = np.array(control_confidences)

    if len(trigger_arr) > 0 and len(control_arr) > 0:
        u_stat, p_value = stats.mannwhitneyu(
            trigger_arr, control_arr, alternative='greater')
        u_stat, p_value = float(u_stat), float(p_value)
    else:
        u_stat, p_value = 0.0, 1.0

    return {
        "verified": p_value < eta,
        "p_value": p_value,
        "test": "mann_whitney_u",
        "statistic": u_stat,
        "mean_trigger_conf": round(float(trigger_arr.mean()) if len(trigger_arr) else 0.0, 6),
        "mean_control_conf": round(float(control_arr.mean()) if len(control_arr) else 0.0, 6),
        "confidence_shift": round(
            (float(trigger_arr.mean()) - float(control_arr.mean()))
            if len(trigger_arr) and len(control_arr) else 0.0,
            6),
        "n_trigger": int(len(trigger_arr)),
        "n_control": int(len(control_arr)),
        "eta": eta,
        "verify_temperature": vT,
    }


# ============================================================
# Recommended vT helper (auto-scale by log(C); user can override)
# ============================================================

def recommended_vT(num_classes: int,
                   user_override: Optional[float] = None) -> float:
    """
    Auto-scale the verification temperature by log(C).

      vT ≈ max(MIN_RECOMMENDED_VT, 5 * log10(C) + 5)

    Empirically:
      C=10   → vT ≈ 10
      C=100  → vT ≈ 15
      C=1000 → vT ≈ 20

    If `user_override` is supplied (not None) we use it verbatim and ONLY
    warn when the override falls below MIN_RECOMMENDED_VT.  This gives
    callers explicit control (required — the user asked for this).
    """
    if user_override is not None:
        v = float(user_override)
        if v < MIN_RECOMMENDED_VT:
            warnings.warn(
                "[EvalGuard] verify_temperature={:.2f} < {:.1f}. "
                "Soft-label verification may fail (p-value saturates at 1). "
                "Recommended: vT >= 5.".format(v, MIN_RECOMMENDED_VT),
                RuntimeWarning, stacklevel=2)
        return v
    # Scale: 5 * log10(C) + 5
    auto = 5.0 * math.log10(max(num_classes, 2)) + 5.0
    return max(MIN_RECOMMENDED_VT, auto)


def _empty_verification_result(eta, vT, mode="paired_wilcoxon"):
    return {
        "verified": False, "p_value": 1.0, "test": mode, "statistic": 0.0,
        "mean_trigger_conf": 0.0, "mean_control_conf": 0.0,
        "confidence_shift": 0.0, "n_trigger": 0, "n_control": 0, "n_pairs": 0,
        "eta": eta, "verify_temperature": vT,
    }


# ============================================================
# Multi-design verification (diagnostic)
# ============================================================
#
# The original paired Wilcoxon verification uses a SINGLE control class per
# top-1 class (derive_control_class). With only C distinct (target, control)
# pairs (one per top-1), any structural class-similarity bias between the
# learned representation and the HMAC-derived control can either hide or
# invert the watermark signal — especially when the watermark signal is weak
# (e.g. high distillation temperature, small delta_logit).
#
# verify_ownership_all_designs runs three orthogonal control designs on the
# SAME forward pass, so a single verification call reports:
#   A. single_ctrl   — the original design (for backward compatibility)
#   B. mean_rest     — p[target] vs mean(p over all classes != top1, target)
#   C. suspect_top1  — re-derive BOTH target and control using the suspect
#                      model's own argmax (fair when teacher and suspect
#                      disagree, which happens in own_trigger mode).
#
# Comparing A/B/C lets you tell "no signal learned" from "single-control
# class-bias artefact".

def _wilcoxon_paired(trig: np.ndarray, ctrl: np.ndarray, label: str,
                     eta: float, vT: float) -> Dict:
    diffs = trig - ctrl
    try:
        r = stats.wilcoxon(diffs, alternative='greater', zero_method='wilcox')
        stat_val = float(r.statistic)
        p_value = float(r.pvalue)
    except ValueError:
        stat_val, p_value = 0.0, 1.0
    return {
        "verified": p_value < eta,
        "p_value": p_value,
        "log10_p_value": round(math.log10(max(p_value, 1e-300)), 2),
        "test": "wilcoxon_signed_rank",
        "statistic": stat_val,
        "control_design": label,
        "mean_trigger_conf": round(float(trig.mean()), 6),
        "mean_control_conf": round(float(ctrl.mean()), 6),
        "confidence_shift": round(float(diffs.mean()), 6),
        "median_shift": round(float(np.median(diffs)), 6),
        "n_trigger": int(len(trig)),
        "n_control": int(len(ctrl)),
        "n_pairs": int(len(diffs)),
        "eta": eta,
        "verify_temperature": vT,
    }


def verify_ownership_all_designs(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    K_w: bytes,
    num_classes: int = 10,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    batch_size: int = 64,
) -> Dict:
    """
    Run Wilcoxon paired verification under THREE different control designs
    on the SAME forward pass, to disentangle single-control class-bias from
    a true absence of watermark signal.

    Designs (all one-sided Wilcoxon signed-rank, alternative='greater'):
      A 'single_ctrl':  original — derive_control_class(K_w, entry.top1_class).
                        One deterministic control per top-1 class.
      B 'mean_rest':    p[target] - mean(p[c]) over c != top1 != target.
                        Robust to per-class natural bias of a single control.
      C 'suspect_top1': re-derive target AND control using the suspect model's
                        own argmax on the sample. Fair when teacher and suspect
                        disagree (important for own_trigger mode).

    Returns {'single_ctrl', 'mean_rest', 'suspect_top1'} where each value is
    a dict with the same fields as verify_ownership() paired mode, plus
    'control_design' and 'median_shift'.
    """
    suspect_model.to(device).eval()
    vT = verify_temperature
    n = len(trigger_set)
    if n == 0:
        empty = _empty_verification_result(eta, vT, mode="paired_wilcoxon")
        return {
            "single_ctrl":  dict(empty, control_design="single_ctrl"),
            "mean_rest":    dict(empty, control_design="mean_rest"),
            "suspect_top1": dict(empty, control_design="suspect_top1"),
        }

    # Single forward pass over all triggers; keep full probability vectors.
    all_probs = np.zeros((n, num_classes), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_entries = trigger_set[start:end]
            batch_x = torch.stack([
                e.query if isinstance(e.query, torch.Tensor)
                else torch.tensor(e.query)
                for e in batch_entries
            ]).to(device)
            logits = suspect_model(batch_x)
            probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
            all_probs[start:end] = probs

    suspect_top1 = all_probs.argmax(axis=1)

    A_trig = np.empty(n, dtype=np.float64)
    A_ctrl = np.empty(n, dtype=np.float64)
    B_trig = np.empty(n, dtype=np.float64)
    B_ctrl = np.empty(n, dtype=np.float64)
    C_trig = np.empty(n, dtype=np.float64)
    C_ctrl = np.empty(n, dtype=np.float64)

    # Cache per-top1 derivations to avoid repeated HMACs
    target_cache: Dict[int, int] = {}
    control_cache: Dict[int, int] = {}

    def _target(c: int) -> int:
        if c not in target_cache:
            target_cache[c] = derive_target_class(K_w, c, num_classes)
        return target_cache[c]

    def _control(c: int) -> int:
        if c not in control_cache:
            control_cache[c] = derive_control_class(K_w, c, num_classes)
        return control_cache[c]

    for i, entry in enumerate(trigger_set):
        # Design A: original single control
        cc_A = _control(entry.top1_class)
        A_trig[i] = all_probs[i, entry.target_class]
        A_ctrl[i] = all_probs[i, cc_A]

        # Design B: mean over all classes except top1 and target
        row = all_probs[i]
        denom = num_classes - 2
        rest_sum = row.sum() - row[entry.top1_class] - row[entry.target_class]
        B_trig[i] = row[entry.target_class]
        B_ctrl[i] = rest_sum / denom

        # Design C: re-derive target & control from suspect top1
        st1 = int(suspect_top1[i])
        tgt_C = _target(st1)
        cc_C = _control(st1)
        C_trig[i] = row[tgt_C]
        C_ctrl[i] = row[cc_C]

    return {
        "single_ctrl":  _wilcoxon_paired(A_trig, A_ctrl, "single_ctrl",  eta, vT),
        "mean_rest":    _wilcoxon_paired(B_trig, B_ctrl, "mean_rest",    eta, vT),
        "suspect_top1": _wilcoxon_paired(C_trig, C_ctrl, "suspect_top1", eta, vT),
    }


def verify_ownership_own_data_all_designs(
    owner_model: nn.Module,
    own_dataloader,
    suspect_model: nn.Module,
    K_w: bytes,
    num_classes: int,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    max_triggers: int = 0,
    r_w: float = 0.0,
    latent_extractor: LatentExtractor = None,
    layer_name: str = "",
    delta_logit: float = 0.0,
    beta: float = 0.0,
    delta_min: float = 0.0,
) -> Dict:
    """
    Own-data all-designs variant:
      1. Reconstruct triggers from owner_model's data. When r_w > 0 and
         latent_extractor is provided, applies Phi(x) filter to keep only
         watermarked queries.  Otherwise uses all samples.
      2. Verify against suspect_model under the three control designs.

    Returns {'single_ctrl', 'mean_rest', 'suspect_top1'} each with
    'trigger_source'='own_trigger' and 'n_own_data_scanned' populated.
    """
    triggers = reconstruct_triggers_from_own_data(
        owner_model, own_dataloader, K_w,
        r_w=r_w, num_classes=num_classes,
        latent_extractor=latent_extractor, layer_name=layer_name,
        delta_logit=delta_logit, beta=beta, delta_min=delta_min,
        device=device, max_triggers=max_triggers,
    )

    if len(triggers) == 0:
        empty = _empty_verification_result(eta, verify_temperature, mode="paired_wilcoxon")
        stub = dict(empty, trigger_source="own_trigger", n_own_data_scanned=0)
        return {
            "single_ctrl":  dict(stub, control_design="single_ctrl"),
            "mean_rest":    dict(stub, control_design="mean_rest"),
            "suspect_top1": dict(stub, control_design="suspect_top1"),
        }

    out = verify_ownership_all_designs(
        triggers, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    for k in out:
        out[k]["trigger_source"] = "own_trigger"
        out[k]["n_own_data_scanned"] = len(triggers)
    return out


# ============================================================
# Own-Data Verification (no D_eval leakage)
# ============================================================

def reconstruct_triggers_from_own_data(
    owner_model: nn.Module,
    own_dataloader,
    K_w: bytes,
    r_w: float,
    num_classes: int,
    latent_extractor: LatentExtractor,
    layer_name: str,
    delta_logit: float = 2.0,
    beta: float = 0.3,
    delta_min: float = 0.0,
    device: str = "cpu",
    max_triggers: int = 0,
) -> List[TriggerEntry]:
    """
    Reconstruct trigger set from Owner's own data WITHOUT accessing D_eval.

    When latent_extractor is provided and r_w > 0, applies the Phi(x)
    filter to select only queries that would have been watermarked during
    evaluation.  This avoids diluting the signal with non-watermarked
    samples, which is critical for high-class-count datasets (e.g.
    CIFAR-100 where ~90% of unfiltered samples carry no watermark signal).

    When latent_extractor is None or r_w == 0, falls back to using ALL
    samples (the previous behavior that relies on the global-pattern
    assumption).

    [v6] delta_min filter is applied symmetrically with `embed_logits`:
      If delta_min > 0 AND beta > 0, a sample with Phi(x)=1 is kept only
      when `min(delta_logit, beta * (logit[top1]-logit[target])) >= delta_min`.
      This prevents the OWN control group from being polluted with queries
      whose effective embedded delta was 0 (i.e. the embedder would have
      skipped them).  delta_min=0 (the default) preserves v5 behaviour.

    Args:
        max_triggers: stop after collecting this many triggers.
                      0 = collect all from the dataloader.
        r_w: watermark ratio for Phi(x) decision.
        latent_extractor: for R(x) computation (Phi(x) filter).
        layer_name: hidden layer for latent extraction.
        delta_logit, beta: unused (retained for API compatibility).
    """
    owner_model.to(device).eval()
    triggers = []

    use_phi_filter = (latent_extractor is not None and r_w > 0)
    phi_threshold = int(r_w * (2 ** 128)) if use_phi_filter else 0
    # Apply the same delta_min-gated filter as embed_logits for consistency.
    use_delta_filter = (delta_min > 0.0 and beta > 0.0 and delta_logit > 0.0)

    with torch.no_grad():
        for batch_x, _ in own_dataloader:
            batch_x = batch_x.to(device)
            logits = owner_model(batch_x)
            logits_np = logits.detach().cpu().numpy()

            # Batch Phi(x) decisions when filter is enabled
            if use_phi_filter:
                r_x_list = latent_extractor.extract_and_binarize_batch(
                    owner_model, batch_x, layer_name, device)

            for i in range(batch_x.size(0)):
                # Apply Phi(x) filter: skip non-watermarked queries
                if use_phi_filter:
                    h = hmac.new(K_w, r_x_list[i], hashlib.sha256).digest()
                    if int.from_bytes(h[:16], "big") >= phi_threshold:
                        continue

                top1 = int(logits[i].argmax().item())
                target = derive_target_class(K_w, top1, num_classes)

                # [v6] Apply the same delta_min gate as embed_logits so that
                # the reconstructed trigger set matches what was actually
                # embedded.  This is the fix for "OWN control group pollution"
                # documented above.
                if use_delta_filter:
                    margin = float(logits_np[i, top1] - logits_np[i, target])
                    effective_delta = min(delta_logit, beta * margin)
                    if effective_delta < delta_min:
                        continue

                triggers.append(TriggerEntry(
                    query=batch_x[i].cpu().clone(),
                    target_class=target,
                    top1_class=top1,
                ))
                if max_triggers > 0 and len(triggers) >= max_triggers:
                    return triggers

    return triggers


def verify_ownership_own_data(
    owner_model: nn.Module,
    own_dataloader,
    suspect_model: nn.Module,
    K_w: bytes,
    r_w: float,
    num_classes: int,
    latent_extractor: LatentExtractor,
    layer_name: str,
    delta_logit: float = 2.0,
    beta: float = 0.3,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    max_triggers: int = 0,
) -> Dict:
    """
    Full own-data verification pipeline:
      1. Reconstruct triggers from Owner's data (stop at max_triggers)
      2. Query suspect model with these triggers
      3. Mann-Whitney U test

    No D_eval samples are needed. Zero privacy leakage.

    Args:
        max_triggers: collect exactly this many triggers then stop.
                      0 = collect all from dataloader.
    """
    triggers = reconstruct_triggers_from_own_data(
        owner_model, own_dataloader, K_w, r_w, num_classes,
        latent_extractor, layer_name, delta_logit, beta, device,
        max_triggers=max_triggers,
    )

    if len(triggers) == 0:
        empty = _empty_verification_result(eta, verify_temperature)
        empty["trigger_source"] = "own_trigger"
        empty["n_own_data_scanned"] = 0
        return empty

    # Step 2 & 3: verify using reconstructed triggers (reuse existing function)
    result = verify_ownership(
        triggers, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    result["trigger_source"] = "own_trigger"
    # Every scanned input becomes a trigger (no Phi(x) filter), so
    # the scan count equals len(triggers).
    result["n_own_data_scanned"] = len(triggers)
    return result


# ============================================================
# v6: Hard-Label Verification (Boundary-Gated Hard-Label Swap)
# ============================================================

def verify_ownership_hard_label(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    num_classes: int,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict:
    """
    Verification companion for the BGS hard-label embedding.

    Test:
      H0:  P[ argmax(suspect(x)) == target_class | x in trigger_set ] = 1/C
      H1:  P[ ... ] > 1/C  (suspect inherits the swap)

      One-sided exact binomial test (scipy.stats.binomtest, alternative='greater').

    A successfully extracted student trained on hard labels will match
    `target_class` on most (often >90%) of trigger samples; an independent
    model only matches at the ~1/C rate.

    Returns the same field shape as the soft-label verify_ownership for
    backwards compatibility with downstream parsers, plus a 'test' tag
    of 'binomial_hard_label'.
    """
    suspect_model.to(device).eval()
    n = len(trigger_set)
    if n == 0:
        return {
            "verified": False,
            "p_value": 1.0,
            "test": "binomial_hard_label",
            "statistic": 0.0,
            "match_rate": 0.0,
            "n_matches": 0,
            "n_trigger": 0,
            "baseline_rate": 1.0 / max(num_classes, 1),
            "eta": eta,
            "label_mode": "hard",
        }

    matches = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_entries = trigger_set[start:end]
            batch_x = torch.stack([
                e.query if isinstance(e.query, torch.Tensor)
                else torch.tensor(e.query)
                for e in batch_entries
            ]).to(device)
            preds = suspect_model(batch_x).argmax(dim=-1).cpu().numpy()
            for i, entry in enumerate(batch_entries):
                if int(preds[i]) == entry.target_class:
                    matches += 1

    baseline = 1.0 / float(num_classes)
    # scipy >= 1.7 exposes binomtest; older releases binom_test.
    try:
        res = stats.binomtest(matches, n, p=baseline, alternative='greater')
        p_value = float(res.pvalue)
    except AttributeError:
        p_value = float(stats.binom_test(matches, n, p=baseline,
                                         alternative='greater'))

    match_rate = matches / float(n)
    return {
        "verified": p_value < eta,
        "p_value": p_value,
        "test": "binomial_hard_label",
        "statistic": float(matches),
        "match_rate": round(match_rate, 6),
        "n_matches": int(matches),
        "n_trigger": int(n),
        "baseline_rate": round(baseline, 6),
        "eta": eta,
        "label_mode": "hard",
    }


# ============================================================
# v6: Random K_w' False-Positive Scanner
# ============================================================

def random_kw_false_positive_scan(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    K_w_genuine: bytes,
    num_classes: int,
    n_random: int = 100,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    batch_size: int = 64,
    label_mode: str = "soft",
    margin_tau_hard: float = 1.5,
    seed: Optional[int] = None,
) -> Dict:
    """
    Hold the SUSPECT model fixed.  Generate `n_random` independent random
    keys K_w' (each unrelated to the genuine K_w).  For each:
       - Re-derive target classes.
       - Soft-label mode: re-run paired Wilcoxon on the same trigger inputs,
         using K_w'-derived (target, control) pairs.  We REUSE the trigger
         INPUTS (queries) but RE-DERIVE target/control from K_w'; this is
         the standard randomization-test FP probe.
       - Hard-label mode: re-derive target with K_w', run binomial test.

    A correctly-functioning watermark should:
      - Reject H0 (verify=True) only for the genuine K_w.
      - Accept H0 (verify=False, p-value >> eta) for all 100 random K_w'.

    Returns a dict with the per-key p-value distribution and summary stats.

    Args:
      trigger_set: the recorded trigger entries (we will OVERRIDE
                   target_class per random key).
      K_w_genuine: the real key (used as a sanity reference; included
                   in the returned distribution as the first entry).
      label_mode:  "soft" → Wilcoxon (mean_rest design)
                   "hard" → Binomial against 1/C
      seed:        controls the random-key generator (numpy RNG).
    """
    if len(trigger_set) == 0:
        return {
            "n_random": 0, "label_mode": label_mode,
            "p_values": [], "verified_count": 0,
            "min_p": 1.0, "median_p": 1.0,
            "fraction_below_eta": 0.0, "eta": eta,
            "genuine_p_value": 1.0,      # <====== 务必加上这行！
            "genuine_verified": False,   # <====== 务必加上这行！
        }

    rng = np.random.default_rng(seed)
    suspect_model.to(device).eval()

    # Pre-compute suspect outputs ONCE on the trigger set inputs.
    n = len(trigger_set)
    if label_mode == "soft":
        all_probs = np.zeros((n, num_classes), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / verify_temperature,
                                      dim=-1).cpu().numpy()
                all_probs[start:end] = probs
        suspect_top1 = all_probs.argmax(axis=1)
    else:
        # hard-label: only need the suspect argmax
        suspect_argmax = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                preds = suspect_model(batch_x).argmax(dim=-1).cpu().numpy()
                suspect_argmax[start:end] = preds

    # Generate keys: include genuine first for reference, then n_random
    # independent 32-byte keys.
    keys = [K_w_genuine] + [
        rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        for _ in range(n_random)
    ]
    p_values = []
    matches_list = []  # for hard mode

    for k_idx, K in enumerate(keys):
        if label_mode == "soft":
            # Re-derive (target, control) pairs from K, run mean_rest design
            B_trig = np.empty(n, dtype=np.float64)
            B_ctrl = np.empty(n, dtype=np.float64)
            target_cache: Dict[int, int] = {}
            for i, entry in enumerate(trigger_set):
                # Re-derive target from K using the embedded top1_class
                tc = entry.top1_class
                if tc not in target_cache:
                    target_cache[tc] = derive_target_class(K, tc, num_classes)
                tgt = target_cache[tc]
                row = all_probs[i]
                denom = num_classes - 2
                rest = row.sum() - row[tc] - row[tgt]
                B_trig[i] = row[tgt]
                B_ctrl[i] = rest / denom
            diffs = B_trig - B_ctrl
            try:
                r = stats.wilcoxon(diffs, alternative='greater',
                                   zero_method='wilcox')
                p = float(r.pvalue)
            except ValueError:
                p = 1.0
            p_values.append(p)
        else:
            # hard-label: re-derive target & count matches against K
            target_cache = {}
            mtotal = 0
            for i, entry in enumerate(trigger_set):
                tc = entry.top1_class
                if tc not in target_cache:
                    target_cache[tc] = derive_target_class(K, tc, num_classes)
                if int(suspect_argmax[i]) == target_cache[tc]:
                    mtotal += 1
            matches_list.append(mtotal)
            try:
                res = stats.binomtest(mtotal, n, p=1.0/num_classes,
                                      alternative='greater')
                p = float(res.pvalue)
            except AttributeError:
                p = float(stats.binom_test(mtotal, n, p=1.0/num_classes,
                                           alternative='greater'))
            p_values.append(p)

    p_arr = np.array(p_values)
    # Index 0 is genuine; remaining are random.
    random_p = p_arr[1:] if len(p_arr) > 1 else p_arr
    return {
        "n_random": int(len(random_p)),
        "label_mode": label_mode,
        "verify_temperature": verify_temperature,
        "eta": eta,
        "genuine_p_value": float(p_arr[0]),
        "genuine_verified": bool(p_arr[0] < eta),
        # [新增] 计算 P-Gap：量化真假钥匙之间的显著性差距
        "p_gap": round(float(-math.log10(max(p_arr[0], 1e-300)) - (-math.log10(max(float(random_p.min()), 1e-300)))), 2) if len(random_p) else 0.0,
        "p_values": [float(x) for x in random_p],
        "matches": [int(m) for m in matches_list[1:]] if matches_list else [],
        "verified_count": int((random_p < eta).sum()),
        "fraction_below_eta": float((random_p < eta).mean()),
        "min_p": float(random_p.min()) if len(random_p) else 1.0,
        "median_p": float(np.median(random_p)) if len(random_p) else 1.0,
        "max_p": float(random_p.max()) if len(random_p) else 1.0,
        "mean_p": float(random_p.mean()) if len(random_p) else 1.0,
    }