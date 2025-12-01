CUDA_VISIBLE_DEVICES=2,3 python -u main_dpo.py \
--logdir logs/dpo_filtered_by_id_pose_auxiliary_losses_without_dpo_loss \
--resume /data5/shuangjun.du/work/REFace/logs/dpo_filtered_by_id_pose_auxiliary_losses_without_dpo_loss/2025-11-24T23-57-47_train_dpo/checkpoints/last.ckpt \
--base configs/train_dpo.yaml \
--scale_lr False \
model.params.use_auxiliary_losses=True \
data.params.batch_size=3 \
model.params.aux_diffusion_weight=1.0 \
model.params.aux_id_loss_weight=0.3 \
model.params.aux_lpips_loss_weight=0.1 \
model.params.aux_reconstruct_ddim_steps=4 \
model.params.dpo_loss_weight=0 \
lightning.wandb.project="Face_Swapping_DPO_Auxiliary_Losses_without_dpo_loss" \
lightning.wandb.run_name="dpo_filtered_by_id_pose_auxiliary_·losses_without_dpo_loss" \
lightning.wandb.tags='["dpo","face-swap","auxiliary-losses","without-dpo-loss"]' \
lightning.wandb.notes="DPO training for face swapping with auxiliary losses without dpo loss"




