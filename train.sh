CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py \
--logdir models/REFace/ \
--resume /data5/shuangjun.du/work/REFace/models/REFace/2025-10-19T07-55-37_train/checkpoints/last.ckpt \
--base configs/train.yaml \
--scale_lr False 

