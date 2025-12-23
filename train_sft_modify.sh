#!/bin/bash
# =====================================================
# SFT 训练脚本（有监督微调）
# =====================================================

export WANDB_MODE=offline 

CUDA_VISIBLE_DEVICES=1 python -u main_sft.py \
    --logdir logs/sft_hybrid_diff_1_id_0.3_rec_20_lpips_0.2 \
    --pretrained_model /data5/shuangjun.du/work/REFace/last.ckpt \
    --base configs/train_sft.yaml \
    --scale_lr False \
       \
    `# ===== 训练模式配置 =====` \
    model.params.use_sft_loss=True \
    model.params.use_ema=True \
    \
    `# ===== 损失函数权重 =====` \
    model.params.aux_diffusion_weight=1.0 \
    model.params.aux_id_loss_weight=0.3 \
    model.params.aux_reconstruct_weight=20 \
    model.params.aux_lpips_loss_weight=0.2 \
    model.params.aux_lpips_multiscale=True \
    model.params.aux_reconstruct_ddim_steps=4 \
    \
    `# ===== 可视化配置 =====` \
    model.params.visualize_denoise_process=False \
    model.params.visualize_interval_steps=50 \
    \
    `# ===== 数据配置 =====` \
    data.params.batch_size=3 \
    data.params.train.params.data_manifest_path=/data5/shuangjun.du/work/REFace/dataset/sft_data_filtered_with_ffhq.json \
    \
    `# ===== 测试配置 =====` \
    model.params.test_interval_steps=200 \
    model.params.test_num_samples=200 \
    model.params.test_batch_size=1 \
    \
    `# ===== GPU 配置 =====` \
    lightning.trainer.gpus=1 \
    lightning.trainer.accumulate_grad_batches=4 \
    \
    `# ===== ImageLogger 配置 =====` \
    lightning.callbacks.image_logger.params.batch_frequency=500 \
    lightning.callbacks.image_logger.params.max_images=1 \
    \
    `# ===== Wandb 配置 =====` \
    lightning.wandb.project="Face_Swapping_SFT" \
    lightning.wandb.run_name="sft_diff_1_id_0.3_rec_20_lpips_0.2" \
    lightning.wandb.tags='["sft","multistep-denoise-process","id_0.3","rec_20","lpips_0.2"]' \
    lightning.wandb.notes="SFT训练：多步去噪过程可视化" \
    \
    `# ===== Checkpoint 保存配置（debug不保存）=====` \
    lightning.callbacks.checkpoint_callback.params.save_top_k=3 \
    lightning.callbacks.checkpoint_callback.params.save_last=True \
    lightning.callbacks.checkpoint_callback.params.save_on_train_epoch_end=False \
    lightning.callbacks.checkpoint_callback.params.every_n_train_steps=1000 \
    \
    `# ===== 训练轮数 =====` \
    lightning.trainer.max_epochs=5000
