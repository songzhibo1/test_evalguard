#!/usr/bin/env bash
# run_obfuscated_attack.sh
#
# Run obfuscated-model attack experiments for all 5 model configs:
#   CIFAR-10:      ResNet-20, VGG-11
#   CIFAR-100:     ResNet-56, VGG-11
#   Tiny-ImageNet: ResNet-18
#
# With >=3 GPUs: each dataset group runs on a dedicated GPU (parallel).
# With 1-2 GPUs: all configs run sequentially on GPU 0.
# With 0 GPUs:   run on CPU (very slow, for debugging only).
#
# Usage:
#   bash run_obfuscated_attack.sh              # auto-detect GPU count
#   bash run_obfuscated_attack.sh --sequential # force sequential
#   bash run_obfuscated_attack.sh --cpu        # force CPU
#
#   # Single model config:
#   bash run_obfuscated_attack.sh --config cifar10_resnet20
#
#   # One dataset group:
#   bash run_obfuscated_attack.sh --config cifar10 --device cuda:1
#
# Logs  → logs/obfuscated_attack/<config_key>.log
# JSON  → results/obfuscated_attack/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="results/obfuscated_attack"
LOG_DIR="logs/obfuscated_attack"
FT_EPOCHS=50          # epochs per fine-tuning budget
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Parse flags ────────────────────────────────────────────────────────────────
SEQUENTIAL=false
FORCE_CPU=false
SINGLE_CONFIG=""
SINGLE_DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequential) SEQUENTIAL=true ;;
        --cpu)        FORCE_CPU=true  ;;
        --config)     SINGLE_CONFIG="$2"; shift ;;
        --device)     SINGLE_DEVICE="$2"; shift ;;
        *) echo "Unknown flag: $1" && exit 1 ;;
    esac
    shift
done

# ── Detect GPU count ──────────────────────────────────────────────────────────
if $FORCE_CPU; then
    GPU_COUNT=0
else
    GPU_COUNT=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
fi

echo "============================================================"
echo "  EvalGuard — Obfuscated-Model Attack Experiments"
echo "  Models : CIFAR-10 (R20, V11)  CIFAR-100 (R56, V11)  TI (R18)"
echo "  GPUs   : ${GPU_COUNT}   Sequential: ${SEQUENTIAL}"
echo "  OutDir : ${OUT_DIR}"
echo "============================================================"

# ── Device helper ─────────────────────────────────────────────────────────────
device_for() {
    local idx=$1
    if $FORCE_CPU; then
        echo "cpu"
    elif [[ -n "$SINGLE_DEVICE" ]]; then
        echo "$SINGLE_DEVICE"
    elif [[ "$GPU_COUNT" -ge 3 ]]; then
        echo "cuda:${idx}"
    elif [[ "$GPU_COUNT" -ge 1 ]]; then
        echo "cuda:0"
    else
        echo "cpu"
    fi
}

# ── Runner: one config key per call ───────────────────────────────────────────
run_one() {
    local config=$1
    local device=$2
    local logfile="${LOG_DIR}/${config}.log"

    echo ""
    echo "  >> ${config}  on ${device}  (log: ${logfile})"

    "$PYTHON" obfuscated_attack_experiments.py \
        --config    "$config"    \
        --device    "$device"    \
        --ft-epochs "$FT_EPOCHS" \
        --out_dir   "$OUT_DIR"   \
        2>&1 | tee "$logfile"

    echo "  << Done: ${config}"
}

# ── Single-config shortcut ────────────────────────────────────────────────────
if [[ -n "$SINGLE_CONFIG" ]]; then
    run_one "$SINGLE_CONFIG" "$(device_for 0)"
    echo "============================================================"
    echo "  Done. Results in ${OUT_DIR}/"
    echo "============================================================"
    exit 0
fi

# ── Full run: all 5 configs ───────────────────────────────────────────────────
# Group by dataset so one GPU handles one dataset's models sequentially.
# CIFAR-10 (R20+V11) → GPU 0
# CIFAR-100 (R56+V11)→ GPU 1
# TinyImageNet (R18)  → GPU 2

run_cifar10() {
    run_one "cifar10_resnet20" "$(device_for 0)"
    run_one "cifar10_vgg11"    "$(device_for 0)"
}

run_cifar100() {
    run_one "cifar100_resnet56" "$(device_for 1)"
    run_one "cifar100_vgg11"    "$(device_for 1)"
}

run_tinyimagenet() {
    run_one "tinyimagenet_resnet18" "$(device_for 2)"
}

if $SEQUENTIAL || [[ "$GPU_COUNT" -lt 3 ]]; then
    echo ""
    echo "Running all 5 configs sequentially …"
    run_cifar10
    run_cifar100
    run_tinyimagenet

else
    echo ""
    echo "Running in parallel: C10 on GPU0, C100 on GPU1, TI on GPU2 …"

    run_cifar10      &  PID_C10=$!
    run_cifar100     &  PID_C100=$!
    run_tinyimagenet &  PID_TI=$!

    FAIL=0
    wait "$PID_C10"   || { echo "ERROR: CIFAR-10 group failed";      FAIL=1; }
    wait "$PID_C100"  || { echo "ERROR: CIFAR-100 group failed";     FAIL=1; }
    wait "$PID_TI"    || { echo "ERROR: TinyImageNet group failed";  FAIL=1; }

    [[ "$FAIL" -ne 0 ]] && { echo "Check logs in ${LOG_DIR}/"; exit 1; }
fi

# ── Print combined summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All experiments complete.  Results: ${OUT_DIR}/"
echo ""

"$PYTHON" - <<'PYEOF'
import json, pathlib, sys

out_dir = pathlib.Path("results/obfuscated_attack")
summary = out_dir / "obfuscated_attack_summary.json"
if not summary.exists():
    print("  Summary JSON not found — check individual result files.")
    sys.exit(0)

with open(summary) as f:
    results = json.load(f)

STRATEGIES = ["FTAL", "FTLL", "RTFL"]

# ── FT table ──
print(f"  {'Dataset':<14} {'Arch':<10} {'Strat':5} {'Clean':>7} {'Obf.':>6}"
      f"  {'FT 1%':>6} {'FT 5%':>6} {'FT10%':>6} {'FT50%':>6} {'FT100%':>7}")
print("  " + "-"*90)
prev = ""
for r in results:
    cur = r.get("config_key","?")
    if cur != prev and prev:
        print()
    prev = cur
    ft_dict = r.get("ft_attack", {})
    for strategy in STRATEGIES:
        rows = ft_dict.get(strategy, [])
        ft = {x["ratio"]: x["acc_pct"] for x in rows}
        print(f"  {r.get('dataset','?'):<14} {r.get('arch','?'):<10} {strategy:5}"
              f" {r.get('clean_teacher_acc',0):>6.1f}% {r.get('obf_acc',0):>5.1f}%"
              f"  {ft.get(0.01,0):>5.1f}% {ft.get(0.05,0):>5.1f}%"
              f" {ft.get(0.10,0):>5.1f}% {ft.get(0.50,0):>5.1f}%"
              f" {ft.get(1.00,0):>6.1f}%")

print()

# ── Prune table ──
print(f"  {'Dataset':<14} {'Arch':<10} {'Clean':>7} {'Obf.':>6}"
      f"  {'Prune25':>8} {'Prune50':>8} {'Prune75':>8}")
print("  " + "-"*68)
prev = ""
for r in results:
    ds = r.get("dataset","?")
    if ds != prev and prev:
        print("  " + "-"*68)
    prev = ds
    pr = {x["prune_frac"]: x["acc_pct"] for x in r.get("prune_attack", [])}
    print(f"  {ds:<14} {r.get('arch','?'):<10} "
          f"{r.get('clean_teacher_acc',0):>6.1f}% {r.get('obf_acc',0):>5.1f}%"
          f"  {pr.get(0.25,0):>7.1f}% {pr.get(0.50,0):>7.1f}%"
          f" {pr.get(0.75,0):>7.1f}%")
PYEOF

echo ""
echo "============================================================"
