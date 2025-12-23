#!/bin/bash
# =====================================================
# DPO 训练脚本（偏好优化）
# =====================================================

export WANDB_MODE=offline 

CUDA_VISIBLE_DEVICES=1 python -u main_dpo.py \
    --logdir logs/dpo_loss_test \
    --pretrained_model /data5/shuangjun.du/work/REFace/last.ckpt \
    --base configs/train_dpo.yaml \
    --scale_lr False \
    \
    `# ===== 训练模式配置 =====` \
    model.params.use_sft_loss=False \
    model.params.use_auxiliary_losses=False \
    model.params.use_ema=False \
    \
    `# ===== DPO 核心参数 =====` \
    model.params.dpo_beta=2000 \
    model.params.dpo_loss_weight=1.0 \
    \
    `# ===== 辅助损失权重（对赢样本的约束）=====` \
    model.params.aux_diffusion_weight=1.0 \
    model.params.aux_id_loss_weight=0.3 \
    model.params.aux_reconstruct_weight=0 \
    model.params.aux_lpips_loss_weight=0.1 \
    model.params.aux_lpips_multiscale=True \
    model.params.aux_reconstruct_ddim_steps=4 \
    \
    `# ===== 时间步权重 =====` \
    model.params.use_timestep_weighting=False \
    model.params.timestep_weight_scale=1.0 \
    \
    `# ===== 可视化配置 =====` \
    model.params.visualize_denoise_process=False \
    model.params.visualize_interval_steps=500 \
    \
    `# ===== 数据配置 =====` \
    data.params.batch_size=10\
    data.params.train.params.data_manifest_path=/data5/shuangjun.du/work/REFace/dataset/dpo_data_filtered_1000.json \
    \
    `# ===== 测试配置 =====` \
    model.params.test_interval_steps=500 \
    model.params.test_num_samples=200 \
    model.params.test_batch_size=4 \
    \
    `# ===== GPU 配置 =====` \
    lightning.trainer.gpus=1 \
    lightning.trainer.accumulate_grad_batches=4 \
    \
    `# ===== ImageLogger 配置 =====` \
    lightning.callbacks.image_logger.params.batch_frequency=20 \
    lightning.callbacks.image_logger.params.max_images=2 \
    \
    `# ===== Wandb 配置 =====` \
    lightning.wandb.project="Face_Swapping_DPO" \
    lightning.wandb.run_name="dpo_multiscale_lpips" \
    lightning.wandb.tags='["dpo","preference-optimization","multiscale-lpips"]' \
    lightning.wandb.notes="DPO训练：基于偏好的强化学习，多尺度LPIPS辅助损失" \
    \
    `# ===== Checkpoint 保存配置（debug不保存）=====` \
    lightning.callbacks.checkpoint_callback.params.save_top_k=3 \
    lightning.callbacks.checkpoint_callback.params.save_last=True \
    \
    `# ===== 训练轮数 =====` \
    lightning.trainer.max_epochs=5000

