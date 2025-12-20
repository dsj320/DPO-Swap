#!/bin/bash
# =====================================================
# SFT 训练脚本（有监督微调）
# =====================================================
export LD_LIBRARY_PATH=/data1/shuangjun.du/anaconda3/envs/REFace/lib:$LD_LIBRARY_PATH && echo "已设置 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
export WANDB_MODE=offline 

CUDA_VISIBLE_DEVICES=0 python -u main_dpo.py \
    --logdir logs/sft_test/sft_diff_1  \
    --pretrained_model /data3/shuangjun.du/FaceSwap/DPO-Swap/last.ckpt \
    --base configs/train_dpo.yaml \
    --scale_lr False \
    \
    `# ===== 训练模式配置 =====` \
    model.params.use_sft_loss=True \
    model.params.use_ema=False \
    \
    `# ===== 损失函数权重 =====` \
    model.params.aux_diffusion_weight=1.0 \
    model.params.aux_id_loss_weight=0 \
    model.params.aux_reconstruct_weight=0 \
    model.params.aux_lpips_loss_weight=0 \
    model.params.aux_lpips_multiscale=True \
    model.params.aux_reconstruct_ddim_steps=4 \
    \
    `# ===== 时间步权重 =====` \
    model.params.use_timestep_weighting=False \
    model.params.timestep_weight_scale=1.0 \
    \
    `# ===== 可视化配置 =====` \
    model.params.visualize_denoise_process=True \
    model.params.visualize_interval_steps=50 \
    \
    `# ===== 数据配置 =====` \
    data.params.batch_size=1 \
    data.params.train.params.data_manifest_path=./dataset/test.json \
    \
    `# ===== 测试配置 =====` \
    model.params.test_interval_steps=500 \
    model.params.test_num_samples=200 \
    model.params.test_batch_size=1 \
    \
    `# ===== GPU 配置 =====` \
    lightning.trainer.gpus=1 \
    lightning.trainer.accumulate_grad_batches=4 \
    \
    `# ===== ImageLogger 配置 =====` \
    lightning.callbacks.image_logger.params.batch_frequency=50 \
    lightning.callbacks.image_logger.params.max_images=2 \
    \
    `# ===== Wandb 配置 =====` \
    lightning.wandb.project="Face_Swapping_SFT" \
    lightning.wandb.run_name="sft_with_multistep_denoise_process" \
    lightning.wandb.tags='["sft","multistep-denoise-process"]' \
    lightning.wandb.notes="SFT训练：多步去噪过程可视化" \
    \
    `# ===== Checkpoint 保存配置（debug不保存）=====` \
    lightning.callbacks.checkpoint_callback.params.save_top_k=3 \
    lightning.callbacks.checkpoint_callback.params.save_last=True \
    \
    `# ===== 训练轮数 =====` \
    lightning.trainer.max_epochs=5000
