export WANDB_MODE=offline 
CUDA_VISIBLE_DEVICES=3 python -u main_dpo.py \
--logdir logs/dpo_filtered_data_with_sft_loss_rec_50_id_0.5_diffusion_1 \
--pretrained_model /data5/shuangjun.du/work/REFace/last.ckpt \
--base configs/train_dpo.yaml \
--scale_lr False \
lightning.trainer.gpus="0" \
model.params.test_interval_steps=300 \
model.params.test_num_samples=200 \
model.params.use_auxiliary_losses=False \
model.params.use_sft_loss=True \
data.params.batch_size=1 \
model.params.aux_diffusion_weight=1.0 \
model.params.aux_id_loss_weight=0.5 \
model.params.aux_lpips_loss_weight=50 \
model.params.aux_reconstruct_ddim_steps=4 \
model.params.dpo_loss_weight=1 \
lightning.wandb.project="Face_Swapping_SFT" \
lightning.wandb.run_name="dpo_filtered_data_with_sft_loss" \
lightning.wandb.tags='["sft","face-swap","filtered-data","with-sft-loss"]' \
lightning.wandb.notes="DPO training for face swapping with filtered data with sft loss"




