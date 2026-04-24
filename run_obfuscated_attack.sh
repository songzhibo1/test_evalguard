#!/usr/bin/env bash
# run_obfuscated_attack.sh
#
# Run obfuscated-model attack experiments for all three datasets.
# Each dataset is launched on a separate GPU (if available); otherwise
# all three run sequentially on GPU 0 (or CPU).
#
# Usage:
#   bash run_obfuscated_attack.sh              # auto-detects GPU count
#   bash run_obfuscated_attack.sh --sequential # force sequential execution
#   bash run_obfuscated_attack.sh --cpu        # force CPU (slow)
#
# Logs are written to logs/obfuscated_attack/<dataset>.log
# Results are written to results/obfuscated_attack/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="results/obfuscated_attack"
LOG_DIR="logs/obfuscated_attack"
EPOCHS=80          # epochs for clean-teacher training
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Parse flags ────────────────────────────────────────────────────────────────
SEQUENTIAL=false
FORCE_CPU=false
for arg in "$@"; do
    case "$arg" in
        --sequential) SEQUENTIAL=true ;;
        --cpu)        FORCE_CPU=true  ;;
    esac
done

# ── Detect GPU count ──────────────────────────────────────────────────────────
if $FORCE_CPU; then
    GPU_COUNT=0
else
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
fi

echo "============================================================"
echo "  EvalGuard — Obfuscated-Model Attack Experiments"
echo "  GPUs available: ${GPU_COUNT}   Sequential: ${SEQUENTIAL}"
echo "  Output dir    : ${OUT_DIR}"
echo "============================================================"

# ── Device assignment ──────────────────────────────────────────────────────────
device_for() {
    local idx=$1
    if $FORCE_CPU; then
        echo "cpu"
    elif [ "$GPU_COUNT" -ge 3 ]; then
        echo "cuda:${idx}"
    elif [ "$GPU_COUNT" -ge 1 ]; then
        echo "cuda:0"
    else
        echo "cpu"
    fi
}

# ── Runner function ────────────────────────────────────────────────────────────
run_dataset() {
    local dataset=$1
    local device=$2
    local logfile="${LOG_DIR}/${dataset}.log"

    echo ""
    echo "  Launching: ${dataset} on ${device}"
    echo "  Log: ${logfile}"

    "$PYTHON" obfuscated_attack_experiments.py \
        --dataset  "$dataset"  \
        --device   "$device"   \
        --epochs   "$EPOCHS"   \
        --out_dir  "$OUT_DIR"  \
        2>&1 | tee "$logfile"

    echo "  Done: ${dataset}"
}

# ── Run datasets ───────────────────────────────────────────────────────────────

if $SEQUENTIAL || [ "$GPU_COUNT" -lt 3 ]; then
    # Sequential execution (one after another)
    echo ""
    echo "Running sequentially …"

    run_dataset "cifar10"      "$(device_for 0)"
    run_dataset "cifar100"     "$(device_for 0)"
    run_dataset "tinyimagenet" "$(device_for 0)"

else
    # Parallel execution: one process per GPU
    echo ""
    echo "Running in parallel on 3 GPUs …"

    run_dataset "cifar10"      "$(device_for 0)" &
    PID_C10=$!

    run_dataset "cifar100"     "$(device_for 1)" &
    PID_C100=$!

    run_dataset "tinyimagenet" "$(device_for 2)" &
    PID_TI=$!

    # Wait for all and check exit codes
    FAIL=0
    wait $PID_C10   || { echo "ERROR: cifar10 failed";      FAIL=1; }
    wait $PID_C100  || { echo "ERROR: cifar100 failed";     FAIL=1; }
    wait $PID_TI    || { echo "ERROR: tinyimagenet failed"; FAIL=1; }

    if [ $FAIL -ne 0 ]; then
        echo ""
        echo "One or more experiments failed. Check logs in ${LOG_DIR}/"
        exit 1
    fi
fi

# ── Print combined summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All experiments complete."
echo "  Results directory: ${OUT_DIR}/"
echo ""

if command -v python3 &>/dev/null; then
    python3 - <<'PYEOF'
import json, pathlib, sys

out_dir = pathlib.Path("results/obfuscated_attack")
summary = out_dir / "obfuscated_attack_summary.json"
if not summary.exists():
    print("  Summary file not found — check individual JSON files.")
    sys.exit(0)

with open(summary) as f:
    results = json.load(f)

# Pretty-print summary table
print(f"  {'Dataset':<14} {'Clean':>7} {'Obf.':>7}  "
      f"{'FT 1%':>7} {'FT 5%':>7} {'FT10%':>7} {'FT50%':>7} {'FT100%':>7}")
print("  " + "-"*72)
for r in results:
    ft = {x['ratio']: x['acc_pct'] for x in r.get('ft_attack', [])}
    print(f"  {r['dataset']:<14} {r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>6.1f}%  "
          f"{ft.get(0.01,0):>6.1f}% {ft.get(0.05,0):>6.1f}% "
          f"{ft.get(0.10,0):>6.1f}% {ft.get(0.50,0):>6.1f}% "
          f"{ft.get(1.00,0):>6.1f}%")
print()
print(f"  {'Dataset':<14} {'Clean':>7} {'Obf.':>7}  "
      f"{'Prune25':>8} {'Prune50':>8} {'Prune75':>8}")
print("  " + "-"*56)
for r in results:
    pr = {x['prune_frac']: x['acc_pct'] for x in r.get('prune_attack', [])}
    print(f"  {r['dataset']:<14} {r['clean_teacher_acc']:>6.1f}% {r['obf_acc']:>6.1f}%  "
          f"{pr.get(0.25,0):>7.1f}% {pr.get(0.50,0):>7.1f}% "
          f"{pr.get(0.75,0):>7.1f}%")
PYEOF
fi

echo "============================================================"
