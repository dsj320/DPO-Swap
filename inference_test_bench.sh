
# # Set variables
# #设置工作环境
# PROJECT_ROOT="/data5/shuangjun.du/work/REFace"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


name="REFace"
# CKPT="/data5/shuangjun.du/work/REFace/logs/dpo_filtered_by_id_pose/2025-11-11T11-37-25_train_dpo/checkpoints/epoch=000002.ckpt"

# # ## CelebA ##
# Results_dir="results/CelebA/REFace/DPO_filtered_by_id_pose_step_4800"
# CONFIG="models/REFace/configs/project.yaml"


# # # Run inference·

# # CUDA_VISIBLE_DEVICES=2 python scripts/inference_test_bench.py \
# #     --outdir "${Results_dir}" \
# #     --config "${CONFIG}" \
# #     --ckpt "${CKPT}" \
# #     --scale 3 \
# #     --n_samples 22 \
# #     --dataset "CelebA" \
# #     --ddim_steps 50 \
# #     --start_idx 0 \
# #     --end_idx 500

# # CUDA_VISIBLE_DEVICES=3 python scripts/inference_test_bench.py \
# #     --outdir "${Results_dir}" \
# #     --config "${CONFIG}" \
# #     --ckpt "${CKPT}" \
# #     --scale 3 \
# #     --n_samples 22 \
# #     --dataset "CelebA" \
# #     --ddim_steps 50 \
# #     --start_idx 500 \
# #     --end_idx 1000



# # ## FFHQ ##

PROJECT_ROOT="/data5/shuangjun.du/work/REFace"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
CONFIG="models/REFace/configs/project_ffhq.yaml"  
Results_dir="results/FFHQ/REFace/ffhq_1"

CUDA_VISIBLE_DEVICES=0 python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "/data5/shuangjun.du/work/REFace/last.ckpt" \
    --scale 3 \
    --n_samples 2 \
    --dataset "FFHQ" \
    --ddim_steps 50


#     #!/usr/bin/env bash
# set -e

# # ===== 参数自己改 =====
# GPU_ID=2               # 想用哪块 GPU
# MIN_FREE_MEM=80000     # 至少要多少 MiB 空闲显存，比如 12000≈12GB
# CHECK_INTERVAL=30      # 每隔多少秒检查一次

# wait_for_gpu () {
#   local gpu_id=$1
#   local min_free=$2

#   while true; do
#     # 查询该 GPU 当前空闲显存（MiB）
#     free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')
#     # 也可以顺带看下利用率（不一定必须）
#     util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')

#     echo "[`date '+%F %T'`] GPU ${gpu_id}: free=${free_mem} MiB, util=${util}%"

#     if [ "${free_mem}" -ge "${min_free}" ]; then
#       echo "GPU ${gpu_id} 空间够了，开始执行任务。"
#       break
#     fi

#     sleep "${CHECK_INTERVAL}"
#   done
# }

# ====== 下面写你的真正命令 ======

# # 等 GPU 空
# wait_for_gpu "${GPU_ID}" "${MIN_FREE_MEM}"

# # 设置环境 & 跑命令
# PROJECT_ROOT="/data5/shuangjun.du/work/REFace"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# name="REFace"
# CKPT="/data5/shuangjun.du/work/REFace/last.ckpt"

# CONFIG="models/REFace/configs/project_ffhq.yaml"
# Results_dir="results/FFHQ/REFace/DPO_filtered_by_id_pose_step_4800"

# CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/inference_test_bench.py \
#     --outdir "${Results_dir}" \
#     --config "${CONFIG}" \
#     --ckpt "${CKPT}" \
#     --scale 3 \
#     --n_samples 10 \
#     --dataset "FFHQ" \
#     --ddim_steps 50




