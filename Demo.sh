
# Set variables
export HF_ENDPOINT=https://hf-mirror.com
#设置工作目录是/data5/shuangjun.du/work/REFace
PROJECT_ROOT="/data5/shuangjun.du/work/REFace"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
name="One_output"
Results_dir="examples/FaceSwap/${name}/results"
Base_dir="examples/FaceSwap/${name}/Outs"
Results_out="examples/FaceSwap/${name}/results/results" 
device=1

CONFIG="models/REFace/configs/project_ffhq.yaml"
# CKPT="/data5/shuangjun.du/work/REFace/logs/dpo_filtered_by_id_pose/2025-11-11T11-37-25_train_dpo/checkpoints/epoch=000011.ckpt"
CKPT="/data5/shuangjun.du/work/REFace/last.ckpt"
#change this
target_path="examples/FaceSwap/One_target"  ``
source_path="examples/FaceSwap/One_source"


# Run inference
# ideal for small number of samples

CUDA_VISIBLE_DEVICES=${device} python scripts/one_inference.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 1 \
    --scale 3.5 \
    --ddim_steps 50



