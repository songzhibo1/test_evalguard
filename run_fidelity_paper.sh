#!/bin/bash
# =============================================================================
# EvalGuard — Fidelity Experiments  (3 Watermark Methods)
# =============================================================================
#
# Measures teacher-accuracy fidelity at each watermark pipeline stage
# for three methods side-by-side:
#
#   EvalGuard  5 stages:
#       Original → Obfuscated → Recovered → WM Soft API → WM Hard API
#
#   DAWN       3 stages:
#       Original → WM Soft API → WM Hard API
#       (DAWN is an API wrapper only — no model encryption)
#
#   Adi        2 stages:
#       Original → Backdoored Teacher
#       (Adi retrains model weights; no soft/hard API distinction)
#
# Models tested:
#   CIFAR-10:      ResNet-20, VGG-11        (C=10,  V_T=10.0, D=5.0,  β=0.5)
#   CIFAR-100:     ResNet-56, VGG-11        (C=100, V_T=15.0, D=15.0, β=0.7)
#   Tiny-ImageNet: ResNet-18                (C=200, V_T=16.5, D=15.0, β=0.7)
#
# Results saved to: results/fidelity/<dataset>/{EvalGuard,DAWN,Adi}/<file>.json
#
# Usage:
#   GPU_IDS="0"               ./run_fidelity_paper.sh
#   GPU_IDS="0,1"             STEPS="cifar10 cifar100"   ./run_fidelity_paper.sh
#   GPU_IDS="0"               METHODS="evalguard dawn"   ./run_fidelity_paper.sh
#   GPU_IDS="0"               DRY_RUN=1                  ./run_fidelity_paper.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-cifar10 cifar100 tinyimagenet}"
METHODS="${METHODS:-evalguard dawn adi}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_fidelity}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# Common parameters
# -----------------------------------------------------------------
EPSILON=50.0
DELTA="2.328306436538696e-10"   # 2^{-32}
SEED=42
HARD_TAU_Q=0.10
HARD_TAU_CALIB=3000

# CIFAR-10 specific  (C=10, V_T = max(5, 5·log10(10)+5) = 10.0)
C10_RW=0.10; C10_D=5.0;  C10_BETA=0.5; C10_VT=10.0; C10_DMIN=0.5

# CIFAR-100 specific (C=100, V_T = max(5, 5·log10(100)+5) = 15.0)
C100_RW=0.10; C100_D=15.0; C100_BETA=0.7; C100_VT=15.0; C100_DMIN=0.5

# Tiny-ImageNet specific (C=200, V_T = max(5, 5·log10(200)+5) ≈ 16.5)
TI_RW=0.10; TI_D=15.0; TI_BETA=0.7; TI_VT=16.5; TI_DMIN=0.5

# Adi backdoor parameters
ADI_NTRIG=100
ADI_EPOCHS=30
ADI_LR=0.001

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled()   { for x in ${STEPS};   do [ "$x" = "$1" ] && return 0; done; return 1; }
method_enabled() { for x in ${METHODS}; do [ "$x" = "$1" ] && return 0; done; return 1; }

JOB_FILE=$(mktemp /tmp/evalguard_fidelity_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# add_model_jobs  MODEL  RW D BETA VT DMIN  LABEL_PREFIX
#   Adds up to 3 jobs (one per enabled method) for a single teacher model.
add_model_jobs() {
    local model=$1 rw=$2 d=$3 beta=$4 vt=$5 dmin=$6 pfx=${7:-}

    if method_enabled evalguard; then
        TAG="${pfx}evalguard_${model}"
        add_job "${TAG}" \
            --method evalguard \
            --model  "${model}" \
            --epsilon "${EPSILON}" --delta "${DELTA}" \
            --rw "${rw}" --delta_logit "${d}" --beta "${beta}" --delta_min "${dmin}" \
            --verify_temperature "${vt}" \
            --hard_tau_quantile "${HARD_TAU_Q}" \
            --hard_tau_calib_samples "${HARD_TAU_CALIB}" \
            --device cuda --seed "${SEED}"
    fi

    if method_enabled dawn; then
        TAG="${pfx}dawn_${model}"
        add_job "${TAG}" \
            --method dawn \
            --model  "${model}" \
            --rw "${rw}" \
            --device cuda --seed "${SEED}"
    fi

    if method_enabled adi; then
        TAG="${pfx}adi_${model}"
        add_job "${TAG}" \
            --method adi \
            --model  "${model}" \
            --adi_n_triggers    "${ADI_NTRIG}" \
            --adi_retrain_epochs "${ADI_EPOCHS}" \
            --adi_lr            "${ADI_LR}" \
            --device cuda --seed "${SEED}"
    fi
}

# -----------------------------------------------------------------
# Pre-checks (data / teacher weights)
# -----------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    if step_enabled tinyimagenet; then
        if [ ! -d "./data/tiny-imagenet-200/train" ]; then
            echo "[ERROR] Tiny-ImageNet not found at ./data/tiny-imagenet-200/"
            echo "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./data/"
            echo "  unzip ./data/tiny-imagenet-200.zip -d ./data/"
            exit 1
        fi
        if [ ! -f "./pretrained/tinyimagenet_resnet18.pt" ]; then
            echo "[INFO] Downloading pretrained ResNet-18 from HuggingFace..."
            mkdir -p ./pretrained
            python - <<'PY'
import urllib.request, torch
from pathlib import Path
url = "https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_100ep/checkpoint.pth"
tmp = Path("/tmp/tinyimagenet_hf_ckpt.pth")
urllib.request.urlretrieve(url, str(tmp))
raw = torch.load(tmp, map_location="cpu", weights_only=False)
state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
out = Path("pretrained/tinyimagenet_resnet18.pt")
torch.save(state, str(out))
tmp.unlink(missing_ok=True)
print("  Cached to {}".format(out))
PY
        fi
    fi
fi


# =============================================================================
# CIFAR-10 — ResNet-20 & VGG-11
# =============================================================================
if step_enabled cifar10; then
    add_model_jobs cifar10_resnet20 \
        "${C10_RW}" "${C10_D}" "${C10_BETA}" "${C10_VT}" "${C10_DMIN}" "c10_"
    add_model_jobs cifar10_vgg11 \
        "${C10_RW}" "${C10_D}" "${C10_BETA}" "${C10_VT}" "${C10_DMIN}" "c10_"
fi


# =============================================================================
# CIFAR-100 — ResNet-56 & VGG-11
# =============================================================================
if step_enabled cifar100; then
    add_model_jobs cifar100_resnet56 \
        "${C100_RW}" "${C100_D}" "${C100_BETA}" "${C100_VT}" "${C100_DMIN}" "c100_"
    add_model_jobs cifar100_vgg11 \
        "${C100_RW}" "${C100_D}" "${C100_BETA}" "${C100_VT}" "${C100_DMIN}" "c100_"
fi


# =============================================================================
# Tiny-ImageNet — ResNet-18
# =============================================================================
if step_enabled tinyimagenet; then
    add_model_jobs tinyimagenet_resnet18 \
        "${TI_RW}" "${TI_D}" "${TI_BETA}" "${TI_VT}" "${TI_DMIN}" "ti_"
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard — Fidelity Experiments  (3 Watermark Methods)"
echo " GPUs       : ${GPU_LIST[*]}  (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " METHODS    : ${METHODS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo ""
echo " Method stages:"
echo "   EvalGuard  5: Original→Obfuscated→Recovered→WM Soft→WM Hard"
echo "   DAWN       3: Original→WM Soft API→WM Hard API"
echo "   Adi        2: Original→Backdoored Teacher  (no soft/hard)"
echo ""
echo " Adi note: Adi et al. modifies model weights via backdoor retraining."
echo "   There is no API-level soft/hard mode for Adi. Fidelity = accuracy"
echo "   drop from backdoor embedding (typically < 1%)."
echo "============================================================"

echo ""
echo "GPU assignment (round-robin):"
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_slot]}"
    count=0; tags=""
    while IFS=$'\t' read -r idx tag _; do
        if [ $((idx % NUM_GPUS)) -eq "${gpu_slot}" ]; then
            count=$((count + 1)); tags="${tags} ${tag}"
        fi
    done < "${JOB_FILE}"
    echo "  GPU ${gpu_id} (${count} jobs):${tags}"
done

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Full job list:"
    while IFS=$'\t' read -r idx tag args; do
        gpu_id="${GPU_LIST[$((idx % NUM_GPUS))]}"
        echo "  [${idx}] GPU${gpu_id}  ${tag}"
        echo "       python fidelity_experiments.py ${args}" | head -c 300
        echo ""
    done < "${JOB_FILE}"
    rm -f "${JOB_FILE}"
    exit 0
fi

WORKER_PIDS=()
FAIL_FILE=$(mktemp /tmp/evalguard_fidelity_fails.XXXXXX)

for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_slot]}"
    (
        total=0
        while IFS=$'\t' read -r idx _ _; do
            [ $((idx % NUM_GPUS)) -eq "${gpu_slot}" ] && total=$((total + 1))
        done < "${JOB_FILE}"
        n=0
        while IFS=$'\t' read -r idx tag args; do
            [ $((idx % NUM_GPUS)) -ne "${gpu_slot}" ] && continue
            n=$((n + 1))
            echo ""
            echo "################################################################"
            echo "# [GPU${gpu_id}] (${n}/${total}) ${tag}"
            echo "# $(date '+%Y-%m-%d %H:%M:%S')"
            echo "################################################################"
            if CUDA_VISIBLE_DEVICES="${gpu_id}" python fidelity_experiments.py ${args} 2>&1 | \
                    tee "${LOG_DIR}/${tag}.log"; then
                echo ">>> [GPU${gpu_id}] OK: ${tag}  $(date +%H:%M:%S)"
            else
                echo ">>> [GPU${gpu_id}] FAIL: ${tag}  $(date +%H:%M:%S)"
                echo "${tag}" >> "${FAIL_FILE}"
            fi
        done < "${JOB_FILE}"
    ) &
    WORKER_PIDS+=($!)
    echo "[launcher] GPU ${gpu_id} worker started (PID $!)"
done

echo ""
echo "[launcher] ${NUM_GPUS} GPU workers running. Logs: ${LOG_DIR}/"
echo ""

for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
rm -f "${JOB_FILE}"

echo ""
echo "============================================================"
echo " DONE — Fidelity  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Logs: ${LOG_DIR}/"
if [ -s "${FAIL_FILE}" ]; then
    n_fail=$(wc -l < "${FAIL_FILE}")
    echo " FAILED (${n_fail}):"
    while read -r t; do echo "   - ${t}"; done < "${FAIL_FILE}"
else
    echo " All ${JOB_COUNT} jobs succeeded."
fi
rm -f "${FAIL_FILE}"
echo "============================================================"