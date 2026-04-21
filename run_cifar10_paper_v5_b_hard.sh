#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-10 Paper Experiments v5 — Part B (HARD-LABEL BGS)
# =============================================================================
#
# This is File 2 of 3 (expb1 .. expb5).  exp numbering continues from Part A.
#   Part A  run_cifar10_paper_v5_a_soft.sh     expA1 .. expA13
#   Part B  run_cifar10_paper_v5_b_hard.sh     expb1 .. expb5   <-- this file
#   Part C  run_cifar10_paper_v5_c_attacks.sh  expC1 .. expC7
#
# BGS = Boundary-Gated Swap hard-label embedding.
# For CIFAR-10 (C=10) the inter-class margins are modest; adaptive τ calibration
# (hard_tau_quantile) picks the right threshold automatically instead of a
# hand-tuned fixed τ.
#
# Available STEPS in THIS file:
#   expb1  BGS main with ADAPTIVE τ (q=0.10)                3 jobs
#   expb2  BGS quantile q sweep                             4 jobs
#   expb3  BGS fixed τ sweep (legacy comparison)            5 jobs
#   expb4  BGS r_w sweep (under calibrated τ)               4 jobs
#   expb5  PLAIN argmax FP baseline                         2 jobs
#                                              Total = 18 jobs
#
# Usage:
#   GPU_IDS="0,1,2,3" ./run_cifar10_paper_v5_b_hard.sh
#   GPU_IDS="0"       STEPS="expb1 expb3"  ./run_cifar10_paper_v5_b_hard.sh
#   GPU_IDS="0"       DRY_RUN=1            ./run_cifar10_paper_v5_b_hard.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-expb1 expb2 expb3 expb4 expb5}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_cifar10_v5}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# CIFAR-10 global hyper-parameters (v5)
# -----------------------------------------------------------------
NQ=50000
DIST_EP=80
DIST_LR=0.002
DIST_BATCH=128
FT_EP=20
FT_LR=0.0005
FT_FRACS="0.0,0.01,0.05,0.10"

RW=0.10
D_LOGIT=5.0
BETA=0.5
V_TEMP=10.0
DELTA_MIN=0.5
SWEEP_T=5

# Hard-label (BGS) parameters
HARD_TAU_Q=0.10          # adaptive τ quantile
HARD_TAU_CALIB=3000      # samples scanned for calibration
HARD_TAU_FIXED=1.0       # legacy fixed τ for CIFAR-10 (smaller margins than C100)

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_c10v5b_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# hard_bgs_adaptive_args MODEL STUDENT Q RW TAG
hard_bgs_adaptive_args() {
    local model=$1 student=$2 q=$3 rw=$4 tag=$5
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode hard --trigger_mode rec_trigger \
--hard_label_mode bgs \
--hard_tau_quantile ${q} --hard_tau_calib_samples ${HARD_TAU_CALIB} \
--margin_tau_hard ${HARD_TAU_FIXED} \
--rw ${rw} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} --seed 42"
}

# hard_bgs_fixed_args MODEL STUDENT TAU TAG
hard_bgs_fixed_args() {
    local model=$1 student=$2 tau=$3 tag=$4
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode hard --trigger_mode rec_trigger \
--hard_label_mode bgs \
--hard_tau_quantile -1.0 --margin_tau_hard ${tau} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} --seed 42"
}

# hard_plain_args MODEL STUDENT TAG
hard_plain_args() {
    local model=$1 student=$2 tag=$3
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode hard --trigger_mode rec_trigger \
--hard_label_mode plain \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} --seed 42"
}

# -----------------------------------------------------------------
# Pre-download guard
# -----------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    echo "[prep] warming CIFAR-10 cache..."
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import (cifar10_data, cifar10_resnet20,
                                cifar10_vgg11, create_student)
_ = cifar10_data();  print("  CIFAR-10 data ready.")
for fn in [cifar10_resnet20, cifar10_vgg11]:
    _ = fn(pretrained=True)
for arch in ["resnet20", "vgg11"]:
    _ = create_student(num_classes=10, arch=arch)
print("  Models cached.")
PY
fi

export TORCH_HUB_OFFLINE=1


# =============================================================================
# EXP b1 -- BGS main with ADAPTIVE τ (q=0.10) = 3 jobs
# =============================================================================
if step_enabled expb1; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v5_bgs_q010_${SHORT_T}_${STUDENT}"
        add_job "${TAG}" $(hard_bgs_adaptive_args "${TEACHER}" "${STUDENT}" "${HARD_TAU_Q}" "${RW}" "${TAG}")
    done
fi


# =============================================================================
# EXP b2 -- BGS quantile q sweep = 4 jobs
# =============================================================================
if step_enabled expb2; then
    for Q in 0.05 0.10 0.20 0.30; do
        TAG="c10v5_bgs_qsweep_${Q}"
        add_job "${TAG}" $(hard_bgs_adaptive_args "cifar10_resnet20" "resnet20" "${Q}" "${RW}" "${TAG}")
    done
fi


# =============================================================================
# EXP b3 -- BGS fixed τ sweep (legacy comparison) = 5 jobs
#   CIFAR-10 margins are smaller; sweep τ in [0.3, 2.0].
# =============================================================================
if step_enabled expb3; then
    for TAU in 0.3 0.5 1.0 1.5 2.0; do
        TAG="c10v5_bgs_tau_${TAU}"
        add_job "${TAG}" $(hard_bgs_fixed_args "cifar10_resnet20" "resnet20" "${TAU}" "${TAG}")
    done
fi


# =============================================================================
# EXP b4 -- BGS r_w sweep under calibrated τ = 4 jobs
# =============================================================================
if step_enabled expb4; then
    for RW_VAL in 0.05 0.10 0.20 0.50; do
        TAG="c10v5_bgs_rw_${RW_VAL}"
        add_job "${TAG}" $(hard_bgs_adaptive_args "cifar10_resnet20" "resnet20" "${HARD_TAU_Q}" "${RW_VAL}" "${TAG}")
    done
fi


# =============================================================================
# EXP b5 -- PLAIN argmax FP baseline (no watermark control) = 2 jobs
# =============================================================================
if step_enabled expb5; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v5_fp_plain_${SHORT_T}_${STUDENT}"
        add_job "${TAG}" $(hard_plain_args "${TEACHER}" "${STUDENT}" "${TAG}")
    done
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- CIFAR-10 v5 Part B (HARD-LABEL BGS)  expb1..expb5"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Params     : DIST_EP=${DIST_EP}, NQ=${NQ}, RW=${RW},"
echo "              D=${D_LOGIT}, BETA=${BETA}, V_TEMP=${V_TEMP},"
echo "              HARD_TAU_Q=${HARD_TAU_Q} (adaptive), FIXED_TAU=${HARD_TAU_FIXED}"
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
FAIL_FILE=$(mktemp /tmp/evalguard_c10v5b_fails.XXXXXX)

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
echo " DONE -- CIFAR-10 v5 Part B  $(date '+%Y-%m-%d %H:%M:%S')"
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