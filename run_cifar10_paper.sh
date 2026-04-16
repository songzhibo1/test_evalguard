#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-10 Paper Experiments v2 (v6 watermark code)
# =============================================================================
#
# Built for the v6 watermark.py:
#   - Target-aware margin: delta = min(D, BETA * (logits[top1] - logits[target]))
#   - Auto-scaled verify_temperature via recommended_vT(C):
#       C=10 -> vT = max(5, 5*log10(10)+5) = 10.0  (matches the v1 default)
#   - BGS (Boundary-Gated Hard-Label Swap) for hard-label embedding
#   - Random K_w' false-positive scanner (--experiment random_kw_fp)
#
# Changes from v1 (run_cifar10_paper.sh):
#   - Added: random K_w' FP scan (soft + hard), hard-label BGS,
#     BGS margin_tau sweep, hard-label PLAIN argmax FP control.
#   - All soft-label experiments still benefit from the v6 target-aware
#     margin fix (no parameter change needed).
#
# Usage:
#   GPU_IDS="0,1,2,3,4,5" ./run_cifar10_paper_v2.sh
#   GPU_IDS="0,1,2,3,4,5" STEPS="exp1 exp10" ./run_cifar10_paper_v2.sh
#   GPU_IDS="0,1,2,3,4,5" DRY_RUN=1 ./run_cifar10_paper_v2.sh
#
# Available STEPS (legacy v1 experiments first, then v6 additions):
#   exp1   Main table (soft): 3 pairs x 4 temps                  12 jobs
#   exp2   rw sweep         {0.01, 0.02, 0.05, 0.10, 0.20}        5 jobs
#   exp3   delta_logit sweep {1.0, 2.0, 3.0, 5.0, 7.0}            5 jobs
#   exp4   beta sweep       {0.1, 0.3, 0.5, 0.7, 0.9}             5 jobs
#   exp5   nq sweep         {5000, 10000, 20000, 50000}           4 jobs
#   exp6   trigger_size     {100, 500, 1000, 2000, all}           5 jobs
#   exp7   Multi-seed       (5 seeds)                              5 jobs
#   exp8   Cross-arch       (2 pairs x 2 temps)                    4 jobs
#   exp9   Random K_w' FP scan (soft, N=100)                       3 jobs
#   exp10  Hard-label BGS verification                             3 jobs
#   exp11  Hard-label BGS margin_tau sweep                         5 jobs
#   exp12  Random K_w' FP scan (hard, N=100)                       2 jobs
#   exp13  Hard-label PLAIN argmax baseline (FP control)           2 jobs
#                                                       Total = 60 jobs
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11 exp12 exp13}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_cifar10_v2}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# CIFAR-10 hyper-parameters (v2: same as v1, validated under v6 code)
# -----------------------------------------------------------------
NQ=50000
DIST_EP=80
DIST_LR=0.002
DIST_BATCH=128
FT_EP=20
FT_LR=0.0005
FT_FRACS="0.0,0.01,0.05,0.10"

RW=0.1
D_LOGIT=5.0
BETA=0.5
V_TEMP=10.0           # matches recommended_vT(10) = max(5, 5*log10(10)+5)
DELTA_MIN=0.5

# v6 hard-label parameters
HARD_TAU=1.0          # margin_tau_hard for BGS (CIFAR-10 needs tighter tau)
N_RAND_KW=100         # random K_w' keys for FP scan

SWEEP_T=5             # baseline temperature for parameter sweeps

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_c10v2_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# soft_args MODEL STUDENT TEMPS RW D BETA TAG [EXTRA...]
soft_args() {
    local model=$1 student=$2 temps=$3 rw=$4 d=$5 beta=$6 tag=$7
    shift 7; local extra="$*"
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode soft --trigger_mode both --own_data_source trainset \
--rw ${rw} --delta_logit ${d} --beta ${beta} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${temps} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} ${extra}"
}

# hard_args MODEL STUDENT MODE TAU TAG [EXTRA...]
hard_args() {
    local model=$1 student=$2 mode=$3 tau=$4 tag=$5
    shift 5; local extra="$*"
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode hard --trigger_mode rec_trigger \
--hard_label_mode ${mode} --margin_tau_hard ${tau} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} ${extra}"
}

# randkw_args MODEL STUDENT LABEL_MODE N TAG [EXTRA...]
randkw_args() {
    local model=$1 student=$2 lmode=$3 n=$4 tag=$5
    shift 5; local extra="$*"
    echo "--experiment random_kw_fp \
--model ${model} --student_arch ${student} \
--label_mode ${lmode} --hard_label_mode bgs --margin_tau_hard ${HARD_TAU} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} --n_random_kw ${n} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${tag} ${extra}"
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
for arch in ["resnet20", "vgg11", "mobilenetv2"]:
    _ = create_student(num_classes=10, arch=arch)
print("  Models cached.")
PY
fi

export TORCH_HUB_OFFLINE=1


# =============================================================================
# EXP 1 -- Main table: 3 pairs x 4 temps
# =============================================================================
if step_enabled exp1; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        for T in 1 3 5 10; do
            TAG="c10v2_${SHORT_T}_${STUDENT}_T${T}"
            add_job "${TAG}" $(soft_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP 2 -- rw sweep
# =============================================================================
if step_enabled exp2; then
    for RW_VAL in 0.01 0.02 0.05 0.10 0.20; do
        TAG="c10v2_rw_${RW_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW_VAL}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 3 -- delta_logit sweep
# =============================================================================
if step_enabled exp3; then
    for D_VAL in 1.0 2.0 3.0 5.0 7.0; do
        TAG="c10v2_d_${D_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_VAL}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 4 -- beta sweep
# =============================================================================
if step_enabled exp4; then
    for B_VAL in 0.1 0.3 0.5 0.7 0.9; do
        TAG="c10v2_beta_${B_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${B_VAL}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 5 -- nq (query budget) sweep
# =============================================================================
if step_enabled exp5; then
    _NQ_BAK="${NQ}"
    for NQ_VAL in 5000 10000 20000 50000; do
        NQ="${NQ_VAL}"
        TAG="c10v2_nq_${NQ_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
    done
    NQ="${_NQ_BAK}"
fi


# =============================================================================
# EXP 6 -- trigger_size sweep
# =============================================================================
if step_enabled exp6; then
    for TS in 100 500 1000 2000 0; do
        TS_LABEL="${TS}"
        [ "${TS}" = "0" ] && TS_LABEL="all"
        add_job "c10v2_trig_${TS_LABEL}" \
            $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" \
                        "c10v2_trig" \
                        "--rec_trigger_size ${TS} --own_trigger_size ${TS} --seed 42")
    done
fi


# =============================================================================
# EXP 7 -- Multi-seed reproducibility
# =============================================================================
if step_enabled exp7; then
    for SEED in 42 123 256 777 2024; do
        TAG="c10v2_seed_${SEED}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "3,5" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed ${SEED}")
    done
fi


# =============================================================================
# EXP 8 -- Extended cross-architecture
# =============================================================================
if step_enabled exp8; then
    for PAIR in "cifar10_vgg11:resnet20" "cifar10_resnet20:mobilenetv2"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        for T in 3 5; do
            TAG="c10v2_xarch_${SHORT_T}_${STUDENT}_T${T}"
            add_job "${TAG}" $(soft_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP 9 -- Random K_w' false-positive scan (soft-label, N=100)
#   Reuses cached student from exp1 (matching tag).
# =============================================================================
if step_enabled exp9; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v2_randkw_soft_${SHORT_T}_${STUDENT}"
        REUSE_TAG="c10v2_${SHORT_T}_${STUDENT}_T${SWEEP_T}"
        add_job "${TAG}" $(randkw_args "${TEACHER}" "${STUDENT}" "soft" "${N_RAND_KW}" "${REUSE_TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 10 -- Hard-label BGS verification (binomial test)
# =============================================================================
if step_enabled exp10; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v2_bgs_${SHORT_T}_${STUDENT}_tau${HARD_TAU}"
        add_job "${TAG}" $(hard_args "${TEACHER}" "${STUDENT}" "bgs" "${HARD_TAU}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 11 -- Hard-label BGS margin_tau sweep
# =============================================================================
if step_enabled exp11; then
    for TAU in 0.3 0.5 1.0 1.5 2.0; do
        TAG="c10v2_bgs_tau_${TAU}"
        add_job "${TAG}" $(hard_args "cifar10_resnet20" "resnet20" "bgs" "${TAU}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 12 -- Random K_w' FP scan (hard-label, N=100)
#   Reuses cached BGS student from exp10.
# =============================================================================
if step_enabled exp12; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v2_randkw_hard_${SHORT_T}_${STUDENT}"
        REUSE_TAG="c10v2_bgs_${SHORT_T}_${STUDENT}_tau${HARD_TAU}"
        add_job "${TAG}" $(randkw_args "${TEACHER}" "${STUDENT}" "hard" "${N_RAND_KW}" "${REUSE_TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 13 -- Hard-label PLAIN argmax baseline (FP control)
# =============================================================================
if step_enabled exp13; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v2_fp_plain_${SHORT_T}_${STUDENT}"
        add_job "${TAG}" $(hard_args "${TEACHER}" "${STUDENT}" "plain" "${HARD_TAU}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- CIFAR-10 Paper Experiments v2 (v6 code)"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Params     : DIST_EP=${DIST_EP}, NQ=${NQ}, RW=${RW},"
echo "              D=${D_LOGIT}, BETA=${BETA}, V_TEMP=${V_TEMP},"
echo "              HARD_TAU=${HARD_TAU}, N_RAND_KW=${N_RAND_KW}"
echo "============================================================"

echo ""
echo "GPU assignment (round-robin):"
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_slot]}"
    count=0; tags=""
    while IFS=$'\t' read -r idx tag _; do
        if [ $((idx % NUM_GPUS)) -eq "${gpu_slot}" ]; then
            count=$((count + 1))
            tags="${tags} ${tag}"
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

# -- Launch per-GPU workers -----------------------------------------
WORKER_PIDS=()
FAIL_FILE=$(mktemp /tmp/evalguard_c10v2_fails.XXXXXX)

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
echo " DONE -- CIFAR-10 v2  $(date '+%Y-%m-%d %H:%M:%S')"
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