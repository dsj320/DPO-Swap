#!/usr/bin/env python3
from argparse import ArgumentParser
import glob, os, sys
from PIL import Image
from tqdm import tqdm
import numpy as np
from multiprocessing import get_context

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(".")
from pretrained.face_parsing.face_parsing_demo import (
    init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
)

def convert_bisenet_to_celeba(mask):
    m = {0:0,1:1,2:6,3:7,4:4,5:5,6:3,7:8,8:9,9:15,10:2,11:10,12:11,13:12,14:17,15:16,16:18,17:13,18:14}
    out = np.zeros_like(mask, dtype=np.uint8)
    for b,c in m.items(): out[mask==b] = c
    return out

_model = None
_in_dir = None
_out_dir = None
_vis_dir = None

def _init_worker(ckpt, in_dir, out_dir, vis_dir):
    global _model, _in_dir, _out_dir, _vis_dir
    _model = init_faceParsing_pretrained_model('default', ckpt, '')
    _in_dir = in_dir
    _out_dir = out_dir
    _vis_dir = vis_dir

def _process(img_path):
    try:
        # 检查输出文件是否已存在
        rel_dir = os.path.dirname(os.path.relpath(img_path, _in_dir))
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_sub = os.path.join(_out_dir, rel_dir)
        mask_path = os.path.join(out_sub, f"{base}.png")
        
        # 如果mask文件已存在，跳过处理
        if os.path.exists(mask_path):
            return "skipped"
        
        im = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        h, w = 512,512
        bisenet_mask = faceParsing_demo(_model, im, convert_to_seg12=False, model_name='default')
        celeba = convert_bisenet_to_celeba(np.asarray(bisenet_mask, dtype=np.uint8))
        if celeba.shape[:2] != (h, w):
            celeba = np.array(Image.fromarray(celeba).resize((w, h), Image.NEAREST), dtype=np.uint8)

        os.makedirs(out_sub, exist_ok=True)
        Image.fromarray(celeba).save(mask_path)

        if _vis_dir:
            vis_sub = os.path.join(_vis_dir, rel_dir); os.makedirs(vis_sub, exist_ok=True)
            vis = vis_parsing_maps(im, celeba)
            Image.fromarray(vis).save(os.path.join(vis_sub, f"{base}.png"))
        return True
    except:
        return False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--faceParsing_ckpt', type=str, default="Other_dependencies/face_parsing/79999_iter.pth")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_vis', action='store_true')
    args = parser.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = args.output_dir or os.path.join(os.path.dirname(in_dir.rstrip('/')), "CelebA_mask")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = out_dir + "_vis" if args.save_vis else None
    if vis_dir: os.makedirs(vis_dir, exist_ok=True)

    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    imgs = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if os.path.splitext(fn)[1] in exts:
                imgs.append(os.path.join(root, fn))
    imgs.sort(reverse=True)

    ctx = get_context("spawn")
    procs = 8
    with ctx.Pool(processes=procs, initializer=_init_worker, initargs=(args.faceParsing_ckpt, in_dir, out_dir, vis_dir)) as pool:
        for _ in tqdm(pool.imap_unordered(_process, imgs, chunksize=2), total=len(imgs)):
            pass


"""
python estimate_mask.py \
    --input_dir /data5/shuangjun.du/datasets/dpo_data/D_tgt \
    --output_dir /data5/shuangjun.du/datasets/dpo_data/D_tgt_mask \
    --save_vis \
    --faceParsing_ckpt Other_dependencies/face_parsing/79999_iter.pth 


python estimate_mask.py \
    --input_dir /data5/shuangjun.du/work/REFace/tmp \
    --output_dir /data5/shuangjun.du/work/REFace/tmp/output \
    --faceParsing_ckpt Other_dependencies/face_parsing/79999_iter.pth 

"""

