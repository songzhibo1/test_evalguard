#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-10 Paper Experiments v5 — Part C (FP + ATTACKS)
# =============================================================================
#
# This is File 3 of 3 (expC1 .. expC7).  exp numbering continues from Part B.
#   Part A  run_cifar10_paper_v5_a_soft.sh     expA1 .. expA13
#   Part B  run_cifar10_paper_v5_b_hard.sh     expb1 .. expb5
#   Part C  run_cifar10_paper_v5_c_attacks.sh  expC1 .. expC7   <-- this file
#
# Covers verification integrity (random K_w' FP scan) and adversarial
# post-hoc attacks: SVHN cross-dataset query, magnitude pruning, INT8 quant.
#
# CACHE SHARING — run Part A (expA1) and Part B (expb1) FIRST.
#   expC1/C4/C5 reuse cached students from Part A (tag …_T5_b0.5_d5.0).
#   expC2/C6    reuse cached students from Part B (tag c10v5_bgs_q010_…).
#
# Available STEPS in THIS file:
#   expC1  Random K_w' FP scan (soft, N=100)                  3 jobs
#   expC2  Random K_w' FP scan (hard BGS, N=100)              2 jobs
#   expC3  Cross-dataset SVHN query attack (soft)             2 jobs
#   expC4  Magnitude pruning attack (soft)                    2 jobs
#   expC5  INT8 quantization attack (soft)                    2 jobs
#   expC6  Pruning attack on BGS hard-label                   1 job
#   expC7  SVHN query attack on BGS hard-label                1 job
#                                                 Total =    13 jobs
#
# Usage:
#   GPU_IDS="0,1,2,3" ./run_cifar10_paper_v5_c_attacks.sh
#   GPU_IDS="0"       STEPS="expC3 expC7"  ./run_cifar10_paper_v5_c_attacks.sh
#   GPU_IDS="0"       DRY_RUN=1            ./run_cifar10_paper_v5_c_attacks.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-expC1 expC2 expC3 expC4 expC5 expC6 expC7}"
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

RW=0.10
D_LOGIT=5.0
BETA=0.5
V_TEMP=10.0
DELTA_MIN=0.5
SWEEP_T=5

HARD_TAU_Q=0.10
HARD_TAU_CALIB=3000
N_RAND_KW=100
PRUNE_FRACS="0.25,0.50,0.75"

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_c10v5c_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# -----------------------------------------------------------------
# Pre-download guard
# -----------------------------------------------------------------
export TORCH_HUB_OFFLINE=1

if [ "${DRY_RUN}" != "1" ]; then
    echo "[prep] warming CIFAR-10 + SVHN cache..."
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import (cifar10_data, svhn_data,
                                cifar10_resnet20, cifar10_vgg11, create_student)
_ = cifar10_data();                       print("  CIFAR-10 data ready.")
_ = svhn_data(normalize_for="cifar10");   print("  SVHN data ready.")
for fn in [cifar10_resnet20, cifar10_vgg11]:
    _ = fn(pretrained=True)
for arch in ["resnet20", "vgg11"]:
    _ = create_student(num_classes=10, arch=arch)
print("  Models cached.")
PY
fi

export TORCH_HUB_OFFLINE=1


# =============================================================================
# expC1 -- Random K_w' FP scan (soft, N=100) = 3 jobs
#   Reuses Part A expA1 cached students (tag c10v5_<t>_<s>_T5_b0.5_d5.0).
# =============================================================================
if step_enabled expC1; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        REUSE_TAG="c10v5_${SHORT_T}_${STUDENT}_T${SWEEP_T}_b${BETA}_d${D_LOGIT}"
        TAG="randkw_soft_${SHORT_T}_${STUDENT}_b${BETA}"
        ARGS="--experiment random_kw_fp \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode soft --hard_label_mode bgs \
--n_random_kw ${N_RAND_KW} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${REUSE_TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# expC2 -- Random K_w' FP scan (hard BGS adaptive, N=100) = 2 jobs
#   Reuses Part B expb1 cached students (tag c10v5_bgs_q010_<t>_<s>).
# =============================================================================
if step_enabled expC2; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        REUSE_TAG="c10v5_bgs_q010_${SHORT_T}_${STUDENT}"
        TAG="randkw_hard_${SHORT_T}_${STUDENT}"
        ARGS="--experiment random_kw_fp \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode hard --hard_label_mode bgs \
--hard_tau_quantile ${HARD_TAU_Q} \
--n_random_kw ${N_RAND_KW} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${REUSE_TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# expC3 -- Cross-dataset SVHN query attack (soft) = 2 jobs
#   Attacker queries teacher with 32x32 SVHN images (same resolution as CIFAR-10).
# =============================================================================
if step_enabled expC3; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v5_svhn_${SHORT_T}_${STUDENT}"
        ARGS="--experiment distill \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode soft --query_dataset svhn \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# expC4 -- Magnitude pruning attack on cached SOFT surrogate = 2 jobs
#   Reuses Part A expA1 cache (tag c10v5_<t>_<s>_T5_b0.5_d5.0).
# =============================================================================
if step_enabled expC4; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        REUSE_TAG="c10v5_${SHORT_T}_${STUDENT}_T${SWEEP_T}_b${BETA}_d${D_LOGIT}"
        TAG="prune_soft_${SHORT_T}_${STUDENT}_b${BETA}"
        ARGS="--experiment pruning_attack \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode soft \
--prune_fracs ${PRUNE_FRACS} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${REUSE_TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# expC5 -- INT8 quantization attack on cached SOFT surrogate = 2 jobs
#   Reuses Part A expA1 cache.
# =============================================================================
if step_enabled expC5; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        REUSE_TAG="c10v5_${SHORT_T}_${STUDENT}_T${SWEEP_T}_b${BETA}_d${D_LOGIT}"
        TAG="quant_soft_${SHORT_T}_${STUDENT}_b${BETA}"
        ARGS="--experiment quantization_attack \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode soft \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${REUSE_TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# expC6 -- Pruning attack on BGS HARD-LABEL surrogate = 1 job
#   Reuses Part B expb1 cache (tag c10v5_bgs_q010_resnet20_resnet20).
# =============================================================================
if step_enabled expC6; then
    REUSE_TAG="c10v5_bgs_q010_resnet20_resnet20"
    TAG="prune_hard_resnet20_resnet20"
    ARGS="--experiment pruning_attack \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode hard --hard_label_mode bgs \
--hard_tau_quantile ${HARD_TAU_Q} \
--prune_fracs ${PRUNE_FRACS} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${REUSE_TAG} --seed 42"
    add_job "${TAG}" ${ARGS}
fi


# =============================================================================
# expC7 -- SVHN cross-dataset query attack on BGS HARD-LABEL surrogate = 1 job
#   Parallel to expC3 (SVHN soft) but with hard-label + adaptive τ.
# =============================================================================
if step_enabled expC7; then
    TAG="c10v5_svhn_hard_resnet20_resnet20"
    ARGS="--experiment distill \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode hard --hard_label_mode bgs \
--hard_tau_quantile ${HARD_TAU_Q} --hard_tau_calib_samples ${HARD_TAU_CALIB} \
--query_dataset svhn \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--device cuda --tag ${TAG} --seed 42"
    add_job "${TAG}" ${ARGS}
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- CIFAR-10 v5 Part C (FP + ATTACKS)  expC1..expC7"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Cache reuse: expC1/C4/C5 <- Part A expA1  ;  expC2/C6 <- Part B expb1"
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
FAIL_FILE=$(mktemp /tmp/evalguard_c10v5c_fails.XXXXXX)

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
echo " DONE -- CIFAR-10 v5 Part C  $(date '+%Y-%m-%d %H:%M:%S')"
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