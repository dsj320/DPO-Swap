#!/bin/bash
# =====================================================
# SFT 训练脚本（有监督微调）
# =====================================================

export WANDB_MODE=online 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main_sft_13ch.py \
   --logdir /data6/shuangjun.du/logs/sft/sft_from_start_per_step \
    --resume /data6/shuangjun.du/logs/sft/sft_from_start_per_step/2026-01-07T22-16-00_train_sft_13ch/checkpoints/last.ckpt \
    --base configs/train_sft_13ch.yaml \
    --scale_lr False \
       \
    `# ===== 训练模式配置 =====` \
    model.params.use_sft_loss=True \
    model.params.use_ema=False \
    \
    `# ===== 损失函数权重 =====` \
    model.params.aux_diffusion_weight=1.0 \
    model.params.aux_id_loss_weight=0.4 \
    model.params.aux_reconstruct_weight=0 \
    model.params.aux_lpips_loss_weight=0.1 \
    model.params.aux_lpips_multiscale=True \
    model.params.aux_reconstruct_ddim_steps=4 \
    \
    `# ===== 数据配置 =====` \
    data.params.batch_size=2 \
    data.params.train.params.data_manifest_path=/data5/shuangjun.du/work/REFace/dataset/sft_final_with_celebahq.json \
    \
    `# ===== 测试配置 =====` \
    model.params.test_interval_steps=200000 \
    model.params.test_num_samples=200 \
    model.params.test_batch_size=1 \
    \
    `# ===== GPU 配置 =====` \
    lightning.trainer.gpus=4 \
    lightning.trainer.accumulate_grad_batches=4 \
    \
    `# ===== ImageLogger 配置 =====` \
    lightning.callbacks.image_logger.params.batch_frequency=200 \
    lightning.callbacks.image_logger.params.max_images=2 \
    \
    `# ===== Wandb 配置 =====` \
    lightning.wandb.project="Face_Swap_SFT" \
    lightning.wandb.run_name="sft_from_start_per_step" \
    lightning.wandb.tags='["sft","from_start"]' \
    lightning.wandb.notes="SFT训练：从开始训练，每步计算重构loss" \
    \
 `# ===== Checkpoint 保存配置（主checkpoint只保存last.ckpt用于断点恢复）=====` \
    lightning.callbacks.checkpoint_callback.params.save_top_k=0 \
    lightning.callbacks.checkpoint_callback.params.save_last=True \
    lightning.callbacks.checkpoint_callback.params.save_on_train_epoch_end=False \
    \
    `# ===== 按step保存checkpoint（带step名字，保存在trainstep_checkpoints子目录）=====` \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.target='pytorch_lightning.callbacks.ModelCheckpoint' \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.params.verbose=True \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.params.save_top_k=-1 \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.params.every_n_train_steps=2000 \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.params.save_weights_only=False \
    lightning.callbacks.metrics_over_trainsteps_checkpoint.params.filename='epoch-{epoch:06d}-step-{step:09d}' \
    \
    `# ===== 训练轮数 =====` \
    lightning.trainer.max_epochs=10

