#!/bin/bash
# =============================================================================
# EvalGuard — Tiny-ImageNet-200 Paper Experiments (Generalization)
# =============================================================================
#
# Third dataset for generalization evaluation. Demonstrates EvalGuard works
# on a different resolution (64x64), class count (200), and domain.
#
# Teacher:  ResNet-18 (modified for 64x64, ~60.23% top-1)
#           Auto-downloaded from HuggingFace: zeyuanyin/tiny-imagenet (rn18_100ep)
#           Cached to pretrained/tinyimagenet_resnet18.pt on first run.
# Students: resnet18 (same-arch), mobilenetv2 (cross-arch)
#
# PREREQUISITES:
#   1. Download Tiny-ImageNet:
#      wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./data/
#      unzip ./data/tiny-imagenet-200.zip -d ./data/
#   (Teacher weights are downloaded automatically — no manual training needed.)
#
# Available STEPS:
#   exp1   Soft-label main (2 students x T=1,3,5,10)           8 jobs
#   exp2   Hard-label BGS (2 students x T=5)                   2 jobs
#   exp3   Pruning attack (50%) + INT8 quantization             2 jobs
#   exp4   FP scan (N=100 random keys)                          1 job
#   exp5   Fidelity (obfuscation roundtrip)                     1 job
#   exp6   Overhead measurement                                 1 job
#   exp7   Own-Trigger Zero-Leakage Verify (train/test)         2 jobs
#   exp8   Hard-Label BGS: Pruning & FP Scan                    2 jobs
#   exp9   Clean Baseline (rw=0.0)                              1 job
#                                                    Total = 20 jobs
#
# Usage:
#   GPU_IDS="0,1" ./run_tinyimagenet_paper.sh
#   GPU_IDS="0" STEPS="exp7 exp8" ./run_tinyimagenet_paper.sh
#   GPU_IDS="0" DRY_RUN=1 ./run_tinyimagenet_paper.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
# 默认跑全量实验 exp1 到 exp9
STEPS="${STEPS:-exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_tinyimagenet}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# Tiny-ImageNet hyper-parameters
# -----------------------------------------------------------------
NQ=50000
DIST_EP=160          # Tiny-ImageNet is harder (200 cls, 64x64); cf. CIFAR-100=160
DIST_LR=0.002
DIST_BATCH=128
FT_EP=40             # fine-tune budget; cf. CIFAR-100=40
FT_LR=0.0005
FT_FRACS="0.0,0.01,0.05,0.10"

RW=0.10
D_LOGIT=15.0
BETA=0.7
V_TEMP=16.5          # recommended_vT(200) = max(5, 5*log10(200)+5) ≈ 16.5
DELTA_MIN=0.5
SWEEP_T=5            # single T used for exp2-9 (attacks / ablations)
TEMPS="1 3 5 10"     # temperature sweep for exp1 main table (cf. CIFAR-100 expA1)

# Hard-label BGS
HARD_TAU_Q=0.10
HARD_TAU_CALIB=3000

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_tinyimg_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# -----------------------------------------------------------------
# Pre-checks
# -----------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    # Check data exists
    if [ ! -d "./data/tiny-imagenet-200/train" ]; then
        echo "[ERROR] Tiny-ImageNet not found at ./data/tiny-imagenet-200/"
        echo "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./data/"
        echo "  unzip ./data/tiny-imagenet-200.zip -d ./data/"
        exit 1
    fi
    # Download pretrained teacher from HuggingFace if not cached
    if [ ! -f "./pretrained/tinyimagenet_resnet18.pt" ]; then
        echo "[INFO] Downloading pretrained ResNet-18 from HuggingFace (zeyuanyin/tiny-imagenet, rn18_100ep, 60.23% top-1)..."
        mkdir -p ./pretrained
        python - <<'PY'
import urllib.request, torch
from pathlib import Path
url = "https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_100ep/checkpoint.pth"
tmp = Path("/tmp/tinyimagenet_hf_ckpt.pth")
print("  Downloading checkpoint...")
urllib.request.urlretrieve(url, str(tmp))
raw = torch.load(tmp, map_location="cpu", weights_only=False)
# HuggingFace checkpoint: {"model": state_dict, "optimizer": ..., "epoch": ..., ...}
state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
out = Path("pretrained/tinyimagenet_resnet18.pt")
torch.save(state, str(out))
tmp.unlink(missing_ok=True)
print("  Cached state_dict to {}".format(out))
PY
    fi
    echo "[prep] Tiny-ImageNet ready."
fi


# =============================================================================
# EXP 1 -- Soft-label main table (2 students x T=1,3,5,10) = 8 jobs
# =============================================================================
if step_enabled exp1; then
    for STUDENT in resnet18 mobilenetv2; do
        for T in ${TEMPS}; do
            TAG="tinyimg_soft_r18_${STUDENT}_T${T}_b${BETA}_d${D_LOGIT}"
            add_job "${TAG}" --experiment surrogate_ft \
                --model tinyimagenet_resnet18 --student_arch "${STUDENT}" \
                --label_mode soft --trigger_mode rec_trigger \
                --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
                --verify_temperature "${V_TEMP}" \
                --temperatures "${T}" --nq "${NQ}" \
                --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
                --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
                --device cuda --tag "${TAG}" --seed 42
        done
    done
fi


# =============================================================================
# EXP 2 -- Hard-label BGS (2 pairs) = 2 jobs
# =============================================================================
if step_enabled exp2; then
    for STUDENT in resnet18 mobilenetv2; do
        TAG="tinyimg_bgs_r18_${STUDENT}"
        add_job "${TAG}" --experiment surrogate_ft \
            --model tinyimagenet_resnet18 --student_arch "${STUDENT}" \
            --label_mode hard --trigger_mode rec_trigger \
            --hard_label_mode bgs \
            --hard_tau_quantile "${HARD_TAU_Q}" --hard_tau_calib_samples "${HARD_TAU_CALIB}" \
            --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
            --verify_temperature "${V_TEMP}" \
            --temperatures "${SWEEP_T}" --nq "${NQ}" \
            --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
            --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
            --device cuda --tag "${TAG}" --seed 42
    done
fi


# =============================================================================
# EXP 3 -- Robustness: pruning 50% + INT8 quantization = 2 jobs
# =============================================================================
if step_enabled exp3; then
    TAG="tinyimg_prune_r18_resnet18"
    add_job "${TAG}" --experiment pruning_attack \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" \
        --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --prune_fracs "0.25,0.50,0.75" \
        --device cuda --tag "tinyimg_soft_r18_resnet18_T${SWEEP_T}_b${BETA}_d${D_LOGIT}" --seed 42

    TAG="tinyimg_int8_r18_resnet18"
    add_job "${TAG}" --experiment quantization_attack \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" \
        --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --device cuda --tag "tinyimg_soft_r18_resnet18_T${SWEEP_T}_b${BETA}_d${D_LOGIT}" --seed 42
fi


# =============================================================================
# EXP 4 -- FP scan (N=100 random keys) = 1 job
# =============================================================================
if step_enabled exp4; then
    TAG="tinyimg_fp_r18_resnet18"
    add_job "${TAG}" --experiment random_kw_fp \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --n_random_kw 100 \
        --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --device cuda --tag "tinyimg_soft_r18_resnet18_T${SWEEP_T}_b${BETA}_d${D_LOGIT}" --seed 42
fi


# =============================================================================
# EXP 5 -- Fidelity (obfuscation roundtrip) = 1 job
# =============================================================================
if step_enabled exp5; then
    TAG="tinyimg_fidelity_r18"
    add_job "${TAG}" --experiment fidelity \
        --model tinyimagenet_resnet18 \
        --device cuda
fi


# =============================================================================
# EXP 6 -- Overhead measurement = 1 job
# =============================================================================
if step_enabled exp6; then
    TAG="tinyimg_overhead_r18"
    add_job "${TAG}" --experiment overhead \
        --model tinyimagenet_resnet18 \
        --device cuda
fi


# =============================================================================
# EXP 7 -- Own-Trigger Zero-Leakage Verification (Trainset & Testset) = 2 jobs
# =============================================================================
if step_enabled exp7; then
    # 7.1 Trainset probe
    TAG="tinyimg_own_train_r18_resnet18"
    add_job "${TAG}" --experiment surrogate_ft \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft --trigger_mode own_trigger \
        --own_data_source trainset --own_trigger_size 0 \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed 42

    # 7.2 Testset probe
    TAG="tinyimg_own_test_r18_resnet18"
    add_job "${TAG}" --experiment surrogate_ft \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft --trigger_mode own_trigger \
        --own_data_source testset --own_trigger_size 0 \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed 42
fi


# =============================================================================
# EXP 8 -- Hard-Label BGS: Pruning & FP Scan = 2 jobs
# =============================================================================
if step_enabled exp8; then
    # Reuse the cached student from exp2
    REUSE_TAG="tinyimg_bgs_r18_resnet18"

    # 8.1 Pruning Attack against Hard-Label
    TAG="tinyimg_prune_hard_r18_resnet18"
    add_job "${TAG}" --experiment pruning_attack \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode hard --hard_label_mode bgs \
        --hard_tau_quantile "${HARD_TAU_Q}" --hard_tau_calib_samples "${HARD_TAU_CALIB}" \
        --prune_fracs "0.25,0.50,0.75" \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --device cuda --tag "${REUSE_TAG}" --seed 42

    # 8.2 FP Scan (N=100) against Hard-Label
    TAG="tinyimg_fp_hard_r18_resnet18"
    add_job "${TAG}" --experiment random_kw_fp \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode hard --hard_label_mode bgs \
        --hard_tau_quantile "${HARD_TAU_Q}" --hard_tau_calib_samples "${HARD_TAU_CALIB}" \
        --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --n_random_kw 100 \
        --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --device cuda --tag "${REUSE_TAG}" --seed 42
fi


# =============================================================================
# EXP 9 -- Clean Baseline (rw = 0.0) = 1 job
# =============================================================================
if step_enabled exp9; then
    TAG="tinyimg_clean_baseline"
    add_job "${TAG}" --experiment surrogate_ft \
        --model tinyimagenet_resnet18 --student_arch resnet18 \
        --label_mode soft --trigger_mode rec_trigger \
        --rw 0.0 --delta_logit "${D_LOGIT}" --beta "${BETA}" --delta_min "${DELTA_MIN}" \
        --verify_temperature "${V_TEMP}" --temperatures "${SWEEP_T}" --nq "${NQ}" \
        --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
        --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
        --device cuda --tag "${TAG}" --seed 42
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- Tiny-ImageNet-200 Generalization Experiments"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Params     : DIST_EP=${DIST_EP}, FT_EP=${FT_EP}, NQ=${NQ}, RW=${RW},"
echo "              D=${D_LOGIT}, BETA=${BETA}, V_TEMP=${V_TEMP}, TEMPS=${TEMPS}"
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
FAIL_FILE=$(mktemp /tmp/evalguard_tinyimg_fails.XXXXXX)

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
echo " DONE -- Tiny-ImageNet  $(date '+%Y-%m-%d %H:%M:%S')"
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