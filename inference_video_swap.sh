
##### EXPERIMENTAL #####

# Set variables
name="v5_elon_to_news_ep_19"
Results_dir="results_video/${name}"
Base_dir="results_video"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=0

CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"


current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"



CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "${Results_dir}" \
    --target_video "/data5/shuangjun.du/FaceSwap/datasets/result_video/BlendFace/video/256/all_cross_attribute/0051-0_0_to_0104-1.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "/data5/shuangjun.du/test_data/extracted_frames/0384-0_frame1.png" \
    --Base_dir "${Base_dir}" \
    --scale 3 \
    --ddim_steps 30 

    

