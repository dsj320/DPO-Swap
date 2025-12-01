
# Set variables
name="Swap_outs_5000_steps"
Results_dir="examples/FaceSwap/${name}/results"
Base_dir="examples/FaceSwap/${name}/Outs"
Results_out="examples/FaceSwap/${name}/results/results" 

device=2


CONFIG="models/REFace/configs/project_ffhq.yaml"
# CKPT="models/REFace/checkpoints/saved.ckpt"
CKPT="/data5/shuangjun.du/work/REFace/models/REFace/2025-10-28T22-19-08_train_dpo/checkpoints/last.ckpt"

#change this
target_path="examples/mydata/target"  
source_path="examples/mydata/source"




# Run inference
# ideal for small number of samples

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_selected.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 20 \
    --scale 3.5 \
    --ddim_steps 30



