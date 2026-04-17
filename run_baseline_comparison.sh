#!/bin/bash
# =============================================================================
# EvalGuard — Baseline Comparison Experiments (Table X)
# =============================================================================
#
# Compares EvalGuard watermark verification against:
#   - DAWN [Szyller et al., IEEE S&P 2021] — API-level label flipping
#   - Adi et al. [USENIX Security 2018] — Backdoor-based watermark
#
# Runs on CIFAR-10 and CIFAR-100 with one representative teacher→student pair
# per dataset. Uses the same distillation + fine-tuning pipeline for fairness.
#
# Available STEPS:
#   bl1   DAWN on CIFAR-10  (soft + hard)                       2 jobs
#   bl2   DAWN on CIFAR-100 (soft + hard)                       2 jobs
#   bl3   Adi on CIFAR-10                                       1 job
#   bl4   Adi on CIFAR-100                                      1 job
#   bl5   EvalGuard on same pairs (for direct comparison)       4 jobs
#   bl6   DAWN on Tiny-ImageNet (soft)                          1 job
#   bl7   Adi on Tiny-ImageNet                                  1 job
#                                                    Total = 12 jobs
#
# Usage:
#   GPU_IDS="0" ./run_baseline_comparison.sh
#   GPU_IDS="0,1" STEPS="bl1 bl3 bl5" ./run_baseline_comparison.sh
#   GPU_IDS="0" DRY_RUN=1 ./run_baseline_comparison.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-bl1 bl2 bl3 bl4 bl5 bl6 bl7}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_baseline_comparison}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# Shared hyper-parameters
# -----------------------------------------------------------------
NQ=50000
RW=0.10
SEED=42

# CIFAR-10 params
C10_DIST_EP=80
C10_DIST_LR=0.002
C10_DIST_BATCH=128
C10_FT_EP=20
C10_FT_LR=0.0005
C10_FT_FRACS="0.0,0.01,0.05,0.10"
C10_D_LOGIT=5.0
C10_BETA=0.5
C10_V_TEMP=10.0
C10_DELTA_MIN=0.5

# CIFAR-100 params
C100_DIST_EP=160
C100_DIST_LR=0.002
C100_DIST_BATCH=128
C100_FT_EP=40
C100_FT_LR=0.0005
C100_FT_FRACS="0.0,0.01,0.05,0.10"
C100_D_LOGIT=15.0
C100_BETA=0.7
C100_V_TEMP=15.0
C100_DELTA_MIN=0.5

# Tiny-ImageNet params
TI_DIST_EP=100
TI_DIST_LR=0.002
TI_DIST_BATCH=128
TI_FT_EP=30
TI_FT_LR=0.0005
TI_FT_FRACS="0.0,0.01,0.05,0.10"
TI_D_LOGIT=15.0
TI_BETA=0.7
TI_V_TEMP=16.5
TI_DELTA_MIN=0.5

# Adi-specific
ADI_N_TRIGGERS=100
ADI_RETRAIN_EP=30
ADI_LR=0.001

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_bl_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# -----------------------------------------------------------------
# Pre-download
# -----------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    echo "[prep] warming model caches..."
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import (cifar10_data, cifar10_resnet20, cifar10_vgg11,
                                cifar100_data, cifar100_resnet56, cifar100_vgg11,
                                create_student)
_ = cifar10_data();  _ = cifar100_data()
for fn in [cifar10_resnet20, cifar100_resnet56]:
    _ = fn(pretrained=True)
for nc, arch in [(10, "resnet20"), (100, "resnet56"), (200, "resnet18")]:
    _ = create_student(num_classes=nc, arch=arch)
print("  Models cached.")
PY
fi

export TORCH_HUB_OFFLINE=1


# =============================================================================
# BL1 -- DAWN on CIFAR-10 (soft + hard) = 2 jobs
# =============================================================================
if step_enabled bl1; then
    TAG="bl_dawn_c10_soft_r20_r20"
    add_job "${TAG}" --experiment baseline_dawn \
        --model cifar10_resnet20 --student_arch resnet20 \
        --label_mode soft --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C10_D_LOGIT}" --beta "${C10_BETA}" --delta_min "${C10_DELTA_MIN}" \
        --verify_temperature "${C10_V_TEMP}" \
        --temperatures 5 \
        --dist_epochs "${C10_DIST_EP}" --dist_lr "${C10_DIST_LR}" --dist_batch "${C10_DIST_BATCH}" \
        --ft_epochs "${C10_FT_EP}" --ft_lr "${C10_FT_LR}" --ft_fractions "${C10_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"

    TAG="bl_dawn_c10_hard_r20_r20"
    add_job "${TAG}" --experiment baseline_dawn \
        --model cifar10_resnet20 --student_arch resnet20 \
        --label_mode hard --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C10_D_LOGIT}" --beta "${C10_BETA}" --delta_min "${C10_DELTA_MIN}" \
        --verify_temperature "${C10_V_TEMP}" \
        --temperatures 5 \
        --dist_epochs "${C10_DIST_EP}" --dist_lr "${C10_DIST_LR}" --dist_batch "${C10_DIST_BATCH}" \
        --ft_epochs "${C10_FT_EP}" --ft_lr "${C10_FT_LR}" --ft_fractions "${C10_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL2 -- DAWN on CIFAR-100 (soft + hard) = 2 jobs
# =============================================================================
if step_enabled bl2; then
    TAG="bl_dawn_c100_soft_r56_r56"
    add_job "${TAG}" --experiment baseline_dawn \
        --model cifar100_resnet56 --student_arch resnet56 \
        --label_mode soft --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C100_D_LOGIT}" --beta "${C100_BETA}" --delta_min "${C100_DELTA_MIN}" \
        --verify_temperature "${C100_V_TEMP}" \
        --temperatures 5 \
        --dist_epochs "${C100_DIST_EP}" --dist_lr "${C100_DIST_LR}" --dist_batch "${C100_DIST_BATCH}" \
        --ft_epochs "${C100_FT_EP}" --ft_lr "${C100_FT_LR}" --ft_fractions "${C100_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"

    TAG="bl_dawn_c100_hard_r56_r56"
    add_job "${TAG}" --experiment baseline_dawn \
        --model cifar100_resnet56 --student_arch resnet56 \
        --label_mode hard --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C100_D_LOGIT}" --beta "${C100_BETA}" --delta_min "${C100_DELTA_MIN}" \
        --verify_temperature "${C100_V_TEMP}" \
        --temperatures 5 \
        --dist_epochs "${C100_DIST_EP}" --dist_lr "${C100_DIST_LR}" --dist_batch "${C100_DIST_BATCH}" \
        --ft_epochs "${C100_FT_EP}" --ft_lr "${C100_FT_LR}" --ft_fractions "${C100_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL3 -- Adi on CIFAR-10 = 1 job
# =============================================================================
if step_enabled bl3; then
    TAG="bl_adi_c10_r20_r20"
    add_job "${TAG}" --experiment baseline_adi \
        --model cifar10_resnet20 --student_arch resnet20 \
        --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C10_D_LOGIT}" --beta "${C10_BETA}" --delta_min "${C10_DELTA_MIN}" \
        --verify_temperature "${C10_V_TEMP}" \
        --dist_epochs "${C10_DIST_EP}" --dist_lr "${C10_DIST_LR}" --dist_batch "${C10_DIST_BATCH}" \
        --ft_epochs "${C10_FT_EP}" --ft_lr "${C10_FT_LR}" --ft_fractions "${C10_FT_FRACS}" \
        --adi_n_triggers "${ADI_N_TRIGGERS}" --adi_retrain_epochs "${ADI_RETRAIN_EP}" --adi_lr "${ADI_LR}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL4 -- Adi on CIFAR-100 = 1 job
# =============================================================================
if step_enabled bl4; then
    TAG="bl_adi_c100_r56_r56"
    add_job "${TAG}" --experiment baseline_adi \
        --model cifar100_resnet56 --student_arch resnet56 \
        --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${C100_D_LOGIT}" --beta "${C100_BETA}" --delta_min "${C100_DELTA_MIN}" \
        --verify_temperature "${C100_V_TEMP}" \
        --dist_epochs "${C100_DIST_EP}" --dist_lr "${C100_DIST_LR}" --dist_batch "${C100_DIST_BATCH}" \
        --ft_epochs "${C100_FT_EP}" --ft_lr "${C100_FT_LR}" --ft_fractions "${C100_FT_FRACS}" \
        --adi_n_triggers "${ADI_N_TRIGGERS}" --adi_retrain_epochs "${ADI_RETRAIN_EP}" --adi_lr "${ADI_LR}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL5 -- EvalGuard on same pairs (reference for direct comparison) = 4 jobs
# =============================================================================
if step_enabled bl5; then
    # CIFAR-10 soft + hard
    TAG="bl_evalguard_c10_soft_r20_r20"
    add_job "${TAG}" --experiment surrogate_ft \
        --model cifar10_resnet20 --student_arch resnet20 \
        --label_mode soft --trigger_mode rec_trigger \
        --rw "${RW}" --delta_logit "${C10_D_LOGIT}" --beta "${C10_BETA}" --delta_min "${C10_DELTA_MIN}" \
        --verify_temperature "${C10_V_TEMP}" \
        --temperatures 5 --nq "${NQ}" \
        --dist_epochs "${C10_DIST_EP}" --dist_lr "${C10_DIST_LR}" --dist_batch "${C10_DIST_BATCH}" \
        --ft_epochs "${C10_FT_EP}" --ft_lr "${C10_FT_LR}" --ft_fractions "${C10_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"

    TAG="bl_evalguard_c10_hard_r20_r20"
    add_job "${TAG}" --experiment surrogate_ft \
        --model cifar10_resnet20 --student_arch resnet20 \
        --label_mode hard --trigger_mode rec_trigger \
        --hard_label_mode bgs --margin_tau_hard 1.0 \
        --rw "${RW}" --delta_logit "${C10_D_LOGIT}" --beta "${C10_BETA}" --delta_min "${C10_DELTA_MIN}" \
        --verify_temperature "${C10_V_TEMP}" \
        --temperatures 5 --nq "${NQ}" \
        --dist_epochs "${C10_DIST_EP}" --dist_lr "${C10_DIST_LR}" --dist_batch "${C10_DIST_BATCH}" \
        --ft_epochs "${C10_FT_EP}" --ft_lr "${C10_FT_LR}" --ft_fractions "${C10_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"

    # CIFAR-100 soft + hard
    TAG="bl_evalguard_c100_soft_r56_r56"
    add_job "${TAG}" --experiment surrogate_ft \
        --model cifar100_resnet56 --student_arch resnet56 \
        --label_mode soft --trigger_mode rec_trigger \
        --rw "${RW}" --delta_logit "${C100_D_LOGIT}" --beta "${C100_BETA}" --delta_min "${C100_DELTA_MIN}" \
        --verify_temperature "${C100_V_TEMP}" \
        --temperatures 5 --nq "${NQ}" \
        --dist_epochs "${C100_DIST_EP}" --dist_lr "${C100_DIST_LR}" --dist_batch "${C100_DIST_BATCH}" \
        --ft_epochs "${C100_FT_EP}" --ft_lr "${C100_FT_LR}" --ft_fractions "${C100_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"

    TAG="bl_evalguard_c100_hard_r56_r56"
    add_job "${TAG}" --experiment surrogate_ft \
        --model cifar100_resnet56 --student_arch resnet56 \
        --label_mode hard --trigger_mode rec_trigger \
        --hard_label_mode bgs \
        --hard_tau_quantile 0.10 --hard_tau_calib_samples 3000 \
        --rw "${RW}" --delta_logit "${C100_D_LOGIT}" --beta "${C100_BETA}" --delta_min "${C100_DELTA_MIN}" \
        --verify_temperature "${C100_V_TEMP}" \
        --temperatures 5 --nq "${NQ}" \
        --dist_epochs "${C100_DIST_EP}" --dist_lr "${C100_DIST_LR}" --dist_batch "${C100_DIST_BATCH}" \
        --ft_epochs "${C100_FT_EP}" --ft_lr "${C100_FT_LR}" --ft_fractions "${C100_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL6 -- DAWN on Tiny-ImageNet (soft) = 1 job
# =============================================================================
if step_enabled bl6; then
    TAG="bl_dawn_tinyimg_soft_r18_r18"
    add_job "${TAG}" --experiment baseline_dawn \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${TI_D_LOGIT}" --beta "${TI_BETA}" --delta_min "${TI_DELTA_MIN}" \
        --verify_temperature "${TI_V_TEMP}" \
        --temperatures 5 \
        --dist_epochs "${TI_DIST_EP}" --dist_lr "${TI_DIST_LR}" --dist_batch "${TI_DIST_BATCH}" \
        --ft_epochs "${TI_FT_EP}" --ft_lr "${TI_FT_LR}" --ft_fractions "${TI_FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# BL7 -- Adi on Tiny-ImageNet = 1 job
# =============================================================================
if step_enabled bl7; then
    TAG="bl_adi_tinyimg_r18_r18"
    add_job "${TAG}" --experiment baseline_adi \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --rw "${RW}" --nq "${NQ}" \
        --delta_logit "${TI_D_LOGIT}" --beta "${TI_BETA}" --delta_min "${TI_DELTA_MIN}" \
        --verify_temperature "${TI_V_TEMP}" \
        --dist_epochs "${TI_DIST_EP}" --dist_lr "${TI_DIST_LR}" --dist_batch "${TI_DIST_BATCH}" \
        --ft_epochs "${TI_FT_EP}" --ft_lr "${TI_FT_LR}" --ft_fractions "${TI_FT_FRACS}" \
        --adi_n_triggers "${ADI_N_TRIGGERS}" --adi_retrain_epochs "${ADI_RETRAIN_EP}" --adi_lr "${ADI_LR}" \
        --device cuda --tag "${TAG}" --seed "${SEED}"
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- Baseline Comparison (Table X)"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " LOG_DIR    : ${LOG_DIR}"
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
        echo "       python experiments.py ${args}" | head -c 200
        echo ""
    done < "${JOB_FILE}"
    rm -f "${JOB_FILE}"
    exit 0
fi

WORKER_PIDS=()
FAIL_FILE=$(mktemp /tmp/evalguard_bl_fails.XXXXXX)

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
            if CUDA_VISIBLE_DEVICES="${gpu_id}" python experiments.py ${args} 2>&1 | \
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
echo " DONE -- Baseline Comparison  $(date '+%Y-%m-%d %H:%M:%S')"
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
