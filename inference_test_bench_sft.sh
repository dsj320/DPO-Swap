
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
CONFIG="models/REFace/configs/project_ffhq_sft.yaml"  
Results_dir="results/FFHQ/REFace/sft_hybrid_diff_1_id_0.3_rec_20_lpips_0.2_step_9000"

CUDA_VISIBLE_DEVICES=1 python scripts/inference_test_bench_sft.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "/data5/shuangjun.du/work/REFace/logs/sft_hybrid_diff_1_id_0.3_rec_20_lpips_0.2/2025-12-23T08-43-41_train_sft/checkpoints/trainstep_checkpoints/epoch-epoch=000003-step-step=000008999.ckpt" \
    --scale 3 \
    --n_samples 3 \
    --dataset "FFHQ" \
    --ddim_steps 50




