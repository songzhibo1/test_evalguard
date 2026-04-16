#!/bin/bash
export TORCH_HUB_OFFLINE=1
LOG_DIR="logs_cifar100_v5"
mkdir -p "${LOG_DIR}"

# 这 13 个失败任务的精确参数
TASKS=(
    "c100v5_vgg11_vgg11_T10|--experiment surrogate_ft --model cifar100_vgg11 --student_arch vgg11 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 10 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_vgg11_vgg11_T10 --seed 42"
    "c100v5_cross_b0.7_d15.0_T5|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet20 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.7 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_cross_b0.7_d15.0_T5 --seed 42"
    "c100v5_vtemp_5.0|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 5.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_vtemp_5.0 --seed 42"
    "c100v5_vtemp_15.0|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_vtemp_15.0 --seed 42"
    "c100v5_vtemp_30.0|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 30.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_vtemp_30.0 --seed 42"
    "c100v5_seed_123|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_seed_123 --seed 123"
    "c100v5_seed_256|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_seed_256 --seed 256"
    "c100v5_d_10.0|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 10.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_d_10.0 --seed 42"
    "c100v5_nq_10000|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 10000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_nq_10000 --seed 42"
    "c100v5_nq_50000|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_nq_50000 --seed 42"
    "c100v5_trig_500|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rec_trigger_size 500 --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_trig_500 --seed 42"
    "c100v5_trig_1000|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rec_trigger_size 1000 --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_trig_1000 --seed 42"
    "c100v5_trig_2000|--experiment surrogate_ft --model cifar100_resnet56 --student_arch resnet56 --label_mode soft --trigger_mode rec_trigger --rec_trigger_size 2000 --rw 0.1 --delta_logit 15.0 --beta 0.5 --delta_min 0.5 --verify_temperature 15.0 --temperatures 5 --nq 50000 --dist_epochs 160 --dist_lr 0.002 --dist_batch 128 --ft_epochs 40 --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 --device cuda --tag c100v5_trig_2000 --seed 42"
)

# ！！这里改成了 0 到 5，共 6 张显卡！！
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}

echo ">>> 准备将 13 个任务分配到 ${NUM_GPUS} 张显卡上..."

for i in "${!TASKS[@]}"; do
    IFS="|" read -r tag args <<< "${TASKS[$i]}"
    gpu_id=${GPUS[$((i % NUM_GPUS))]}
    
    echo ">>> [GPU${gpu_id}] Starting: ${tag}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python experiments.py ${args} > "${LOG_DIR}/${tag}.log" 2>&1 &
done

echo ">>> 13 个任务已全部发往后台运行！你可以输入 nvidia-smi 查看 6 张卡的负载。"
wait
echo "All 13 failed tasks completed!"