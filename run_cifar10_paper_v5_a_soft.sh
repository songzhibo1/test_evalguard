#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-10 Paper Experiments v5 — Part A (SOFT LABEL)
# =============================================================================
#
# This is File 1 of 3 (expA1 .. expA13).  exp numbering is continuous across:
#   Part A  run_cifar10_paper_v5_a_soft.sh     expA1 .. expA13  <-- this file
#   Part B  run_cifar10_paper_v5_b_hard.sh     expb1 .. expb5
#   Part C  run_cifar10_paper_v5_c_attacks.sh  expC1 .. expC7
#
# Mirrors the CIFAR-100 v5 Part A structure with CIFAR-10 hyper-parameters:
#   C=10  →  V_TEMP=10.0, D_LOGIT=5.0, BETA=0.5, DIST_EP=80, FT_EP=20
#
# Teachers from torch.hub (chenyaofo/pytorch-cifar-models):
#   cifar10_resnet20  (~92.6% top-1)
#   cifar10_vgg11_bn  (~92.4% top-1)
#
# Available STEPS in THIS file:
#   expA1   Main table (soft):    3 pairs x T=1,3,5,10            12 jobs
#   expA2   Cross-arch D/β combos: resnet20->vgg11 x T=1,3,5,10   8 jobs
#   expA3   V_TEMP sensitivity sweep                               7 jobs
#   expA4   Multi-seed reproducibility                             5 jobs
#   expA5   Soft r_w sweep                                         5 jobs
#   expA6   Soft δ_logit sweep                                     5 jobs
#   expA7   Soft β sweep                                           5 jobs
#   expA8   Soft n_q sweep                                         4 jobs
#   expA9   Soft trigger_size sweep                                5 jobs
#   expA10  Logit margin profiling (diagnostic, inline python)     1 task
#   expA11  r_w = 0.0 clean baseline                               1 job
#   expA12  own_trigger main table (3 pairs x T=5)                 3 jobs
#   expA13  own_trigger + testset (zero-leakage verify)            1 job
#                                              Total = 61 jobs + 1 task
#
# Usage:
#   GPU_IDS="0,1,2,3" ./run_cifar10_paper_v5_a_soft.sh
#   GPU_IDS="0,1"     STEPS="expA1 expA12"  ./run_cifar10_paper_v5_a_soft.sh
#   GPU_IDS="0"       DRY_RUN=1             ./run_cifar10_paper_v5_a_soft.sh
# =============================================================================

cd "$(dirname "$0")"

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-expA1 expA2 expA3 expA4 expA5 expA6 expA7 expA8 expA9 expA10 expA11 expA12 expA13}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_cifar10_v5}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# -----------------------------------------------------------------
# CIFAR-10 global hyper-parameters (v5)
# -----------------------------------------------------------------
NQ=50000
DIST_EP=80           # CIFAR-10 converges faster than CIFAR-100 (80 vs 160)
DIST_LR=0.002
DIST_BATCH=128
FT_EP=20             # cf. CIFAR-100=40; CIFAR-10 task is simpler
FT_LR=0.0005
FT_FRACS="0.0,0.01,0.05,0.10"

RW=0.10
D_LOGIT=5.0          # CIFAR-10 has smaller inter-class margins than CIFAR-100
BETA=0.5
V_TEMP=10.0          # recommended_vT(10) = max(5, 5*log10(10)+5) = 10.0
DELTA_MIN=0.5
SWEEP_T=5

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_c10v5a_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# soft_args MODEL STUDENT TEMPS RW D BETA V_TEMP TAG [EXTRA...]
soft_args() {
    local model=$1 student=$2 temps=$3 rw=$4 d=$5 beta=$6 vtemp=$7 tag=$8
    shift 8; local extra="$*"
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode soft --trigger_mode rec_trigger \
--rw ${rw} --delta_logit ${d} --beta ${beta} --delta_min ${DELTA_MIN} \
--verify_temperature ${vtemp} \
--temperatures ${temps} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} ${extra}"
}

# -----------------------------------------------------------------
# Pre-download guard
# -----------------------------------------------------------------
export TORCH_HUB_OFFLINE=1

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
# EXP A1 -- Main table (soft): 3 pairs x T=1,3,5,10 = 12 jobs
# =============================================================================
if step_enabled expA1; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        for T in 1 3 5 10; do
            TAG="c10v5_${SHORT_T}_${STUDENT}_T${T}_b${BETA}_d${D_LOGIT}"
            add_job "${TAG}" $(soft_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${V_TEMP}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP A2 -- Cross-arch D/β combos: resnet20->vgg11, T=1,3,5,10 = 8 jobs
# =============================================================================
if step_enabled expA2; then
    for COMBO in "0.5:5.0" "0.5:7.0"; do
        B_VAL="${COMBO%%:*}"; D_VAL="${COMBO##*:}"
        for T in 1 3 5 10; do
            TAG="c10v5_cross_b${B_VAL}_d${D_VAL}_T${T}"
            add_job "${TAG}" $(soft_args "cifar10_resnet20" "vgg11" "${T}" "${RW}" "${D_VAL}" "${B_VAL}" "${V_TEMP}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP A3 -- V_TEMP sensitivity sweep = 7 jobs
#   Fixed: resnet20->resnet20, T=5, D=5.0, BETA=0.5
# =============================================================================
if step_enabled expA3; then
    for VT_VAL in 1.0 3.0 5.0 10.0 15.0 20.0 30.0; do
        TAG="c10v5_vtemp_${VT_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" "${VT_VAL}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP A4 -- Multi-seed reproducibility (5 seeds) = 5 jobs
# =============================================================================
if step_enabled expA4; then
    for SEED in 42 123 256 777 2024; do
        TAG="c10v5_seed_${SEED}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" "${V_TEMP}" "${TAG}" "--seed ${SEED}")
    done
fi


# =============================================================================
# EXP A5 -- Soft r_w sweep = 5 jobs
# =============================================================================
if step_enabled expA5; then
    for RW_VAL in 0.02 0.05 0.10 0.20 0.50; do
        TAG="c10v5_rw_${RW_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW_VAL}" "${D_LOGIT}" "${BETA}" "${V_TEMP}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP A6 -- Soft δ_logit sweep = 5 jobs
#   CIFAR-10 margins are small; sweep around the D=5.0 baseline.
# =============================================================================
if step_enabled expA6; then
    for D_VAL in 1.0 2.0 3.0 5.0 7.0; do
        TAG="c10v5_d_${D_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_VAL}" "${BETA}" "${V_TEMP}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP A7 -- Soft β sweep = 5 jobs
# =============================================================================
if step_enabled expA7; then
    for B_VAL in 0.3 0.5 0.7 0.8 0.9; do
        TAG="c10v5_beta_${B_VAL}"
        add_job "${TAG}" $(soft_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${B_VAL}" "${V_TEMP}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP A8 -- Soft n_q (query budget) sweep = 4 jobs
# =============================================================================
if step_enabled expA8; then
    for NQ_VAL in 10000 20000 30000 50000; do
        TAG="c10v5_nq_${NQ_VAL}"
        ARGS="--experiment surrogate_ft \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode soft --trigger_mode rec_trigger \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ_VAL} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# EXP A9 -- Soft trigger_size sweep = 5 jobs
#   TS = 100 / 500 / 1000 / 2000 / 0 (= all watermarkable)
# =============================================================================
if step_enabled expA9; then
    for TS in 100 500 1000 2000 0; do
        TAG="c10v5_trig_${TS}"
        ARGS="--experiment surrogate_ft \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode soft --trigger_mode rec_trigger \
--rec_trigger_size ${TS} \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# EXP A10 -- Logit margin profiling (target-aware, diagnostic inline)
#   Output: ${LOG_DIR}/margin_profile_target_aware_*.json
# =============================================================================
if step_enabled expA10; then
    echo ""
    echo "================================================================"
    echo " EXP A10 -- Logit margin profiling (target-aware, CIFAR-10)"
    echo "================================================================"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY RUN] Would profile resnet20, vgg11"
    else
        python - "${LOG_DIR}" <<'PYMARGIN'
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import (cifar10_data, cifar10_resnet20, cifar10_vgg11)
from evalguard.crypto import keygen, kdf
from evalguard.watermark import derive_target_class

LOG_DIR = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, _, testloader = cifar10_data(batch_size=256)
NC = 10
K_w = kdf(keygen(256, seed=42), "watermark")

models = [("resnet20", cifar10_resnet20),
          ("vgg11",    cifar10_vgg11)]
combos = [{"D":5.0,"BETA":0.5},{"D":5.0,"BETA":0.7},
          {"D":7.0,"BETA":0.5},{"D":7.0,"BETA":0.7}]
DELTA_MIN = 0.5

for name, fn in models:
    print("\n--- {} ---".format(name))
    m, _ = fn(pretrained=True)
    m = m.to(device).eval()
    margins_top12, margins_target = [], []
    with torch.no_grad():
        for x, _ in testloader:
            logits = m(x.to(device)).cpu().numpy()
            top1 = logits.argmax(axis=1)
            sl = np.sort(logits, axis=1)[:, ::-1]
            margins_top12.append(sl[:,0] - sl[:,1])
            for i in range(logits.shape[0]):
                t = derive_target_class(K_w, int(top1[i]), NC)
                margins_target.append(float(logits[i, top1[i]] - logits[i, t]))
    margins_top12 = np.concatenate(margins_top12)
    margins_target = np.array(margins_target)
    out = {"model":name,"dataset":"cifar10","n_samples":int(len(margins_target)),
           "margin_top1_top2":{"mean":float(margins_top12.mean()),
                               "median":float(np.median(margins_top12)),
                               "p5":float(np.percentile(margins_top12,5)),
                               "p95":float(np.percentile(margins_top12,95))},
           "margin_top1_target":{"mean":float(margins_target.mean()),
                                 "median":float(np.median(margins_target)),
                                 "p5":float(np.percentile(margins_target,5)),
                                 "p95":float(np.percentile(margins_target,95))},
           "effective_delta_target_aware":[]}
    print("  top1-top2  mean={:.3f} median={:.3f}".format(margins_top12.mean(),np.median(margins_top12)))
    print("  top1-target mean={:.3f} median={:.3f}".format(margins_target.mean(),np.median(margins_target)))
    for c in combos:
        D,B = c["D"],c["BETA"]
        eff = np.minimum(D, B*margins_target)
        eff = np.where(margins_target<=0, 0.0, eff)
        valid = eff[eff >= DELTA_MIN]
        n_skip = int(np.sum(eff < DELTA_MIN))
        e = {"D":D,"BETA":B,"delta_min":DELTA_MIN,
             "n_total":int(len(eff)),"n_skipped":n_skip,
             "skip_rate":round(n_skip/len(eff),4),"n_valid":int(len(valid))}
        if len(valid):
            e["eff_delta_stats"] = {"mean":float(valid.mean()),
                                    "median":float(np.median(valid)),
                                    "frac_capped":float(np.mean(valid >= D-1e-6))}
        out["effective_delta_target_aware"].append(e)
        print("  D={} BETA={}: skip={}/{} ({:.1f}%) eff_mean={:.3f}".format(
            D,B,n_skip,len(eff),n_skip/len(eff)*100,
            float(valid.mean()) if len(valid) else 0.0))
    p = os.path.join(LOG_DIR, "margin_profile_target_aware_{}.json".format(name))
    with open(p,"w") as f: json.dump(out,f,indent=2)
    print("  saved", p)
    del m; torch.cuda.empty_cache()
print("\n[margin profiling] Done.")
PYMARGIN
    fi
fi


# =============================================================================
# EXP A11 -- r_w = 0.0 clean baseline = 1 job
# =============================================================================
if step_enabled expA11; then
    TAG="c10v5_rw_0.0"
    ARGS="--experiment surrogate_ft \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode soft --trigger_mode rec_trigger \
--rw 0.0 --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
    add_job "${TAG}" ${ARGS}
fi


# =============================================================================
# EXP A12 -- own_trigger main table (3 pairs x T=5) = 3 jobs
# =============================================================================
if step_enabled expA12; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        TAG="c10v5_own_${SHORT_T}_${STUDENT}"
        ARGS="--experiment surrogate_ft \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode soft --trigger_mode own_trigger \
--own_data_source trainset --own_trigger_size 0 \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
        add_job "${TAG}" ${ARGS}
    done
fi


# =============================================================================
# EXP A13 -- own_trigger + testset (zero-leakage verify) = 1 job
# =============================================================================
if step_enabled expA13; then
    TAG="c10v5_own_test_resnet20_resnet20"
    ARGS="--experiment surrogate_ft \
--model cifar10_resnet20 --student_arch resnet20 \
--label_mode soft --trigger_mode own_trigger \
--own_data_source testset --own_trigger_size 0 \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
    add_job "${TAG}" ${ARGS}
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard -- CIFAR-10 v5 Part A (SOFT)  expA1..expA13"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Params     : DIST_EP=${DIST_EP}, FT_EP=${FT_EP}, NQ=${NQ}, RW=${RW},"
echo "              D=${D_LOGIT}, BETA=${BETA}, V_TEMP=${V_TEMP}"
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
FAIL_FILE=$(mktemp /tmp/evalguard_c10v5a_fails.XXXXXX)

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
echo " DONE -- CIFAR-10 v5 Part A  $(date '+%Y-%m-%d %H:%M:%S')"
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