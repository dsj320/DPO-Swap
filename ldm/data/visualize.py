import os
import json
import torch
import shutil
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import os.path as osp
import random
import cv2 # 确保导入

# -----------------------------------------------------------------
# 1. 导入所有组件 (从你 dpo_dataset.py 文件)
# -----------------------------------------------------------------
try:
    # [!] 确保这里导入的是你项目中的原始 CelebAdataset
    # [!] 和我们最终修复的 DPOFaceDataset
    from dpo_dataset import CelebAdataset, DPOFaceDataset 
    from dpo_dataset import (
        get_tensor, get_tensor_clip, 
        un_norm, un_norm_clip, 
        rearrange, decow, A
    )
except ImportError as e:
    print(f"Error: 无法从 dpo_dataset.py 导入。 {e}")
    exit(1)

# -----------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------
def set_seed(seed):
    try:
        cv2.setRNGSeed(seed)
    except Exception:
        print("警告: cv2 未导入或 setRNGSeed 不可用。")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"设置所有随机种子 (torch, numpy, random, cv2) 为: {seed}")

def save_tensor_image(tensor, path, norm_type='-1to1'):
    tensor = tensor.cpu().float()
    if norm_type == '-1to1':
        tensor = un_norm(tensor)
    elif norm_type == 'clip':
        tensor = un_norm_clip(tensor)
    save_image(tensor, path)
    print(f"  ✅ 已保存: {path}")

# -----------------------------------------------------------------
# 3. [!!! 在这里配置你的真实数据路径 !!!]
# -----------------------------------------------------------------
# 使用你自己的 DPO 数据集中的一个样本
SRC_IMG_PATH  = "/data5/shuangjun.du/work/CelebAMask-HQ/CelebA-HQ-img/0.jpg" # 512px
SRC_MASK_PATH = "/data5/shuangjun.du/work/CelebAMask-HQ/CelebAMask_parsing_mask/0.png" # 512px
TGT_IMG_PATH  = "/data5/shuangjun.du/work/CelebAMask-HQ/CelebA-HQ-img/1.jpg" # 512px
TGT_MASK_PATH = "/data5/shuangjun.du/work/CelebAMask-HQ/CelebAMask_parsing_mask/1.png" # 512px

# -----------------------------------------------------------------
# 4. 配置参数
# -----------------------------------------------------------------
PRESERVE_LIST = [1, 2, 4, 5, 8, 9, 6, 7, 10, 11, 12, 17]
IMG_SIZE = 512 # 两个类的 args['image_size'] 都是 512
SEED = 42 # 使用相同的种子
OUTPUT_DIR = "_visual_comparison_final_output"
TEMP_DIR_CELEBA = "_temp_celeba_final"
TEMP_MANIFEST_DPO = "_temp_dpo_final.json"

def main():
    
    # --- 检查文件 ---
    paths_to_check = [SRC_IMG_PATH, SRC_MASK_PATH, TGT_IMG_PATH, TGT_MASK_PATH]
    missing_paths = [p for p in paths_to_check if not os.path.exists(p)]
    if missing_paths:
        print("❌ 错误: 一个或多个文件路径不正确。请检查第 3 部分中的路径。")
        for p in missing_paths: print(f"  未找到: {p}")
        return

    try:
        # --- 5. 创建 CelebA 的临时环境 (复制 TGT 数据) ---
        print("正在为 CelebAdataset 创建临时环境 (使用 TGT 数据)...")
        os.makedirs(os.path.join(TEMP_DIR_CELEBA, "CelebA-HQ-img"), exist_ok=True)
        os.makedirs(os.path.join(TEMP_DIR_CELEBA, "CelebA-HQ-mask", "Overall_mask"), exist_ok=True)
        celeba_temp_img = os.path.join(TEMP_DIR_CELEBA, "CelebA-HQ-img", "0.jpg")
        celeba_temp_mask = os.path.join(TEMP_DIR_CELEBA, "CelebA-HQ-mask", "Overall_mask", "0.png")
        shutil.copyfile(TGT_IMG_PATH, celeba_temp_img) # <-- 使用 TGT
        shutil.copyfile(TGT_MASK_PATH, celeba_temp_mask) # <-- 使用 TGT Mask
        print(f"  - CelebA 输入将使用: {TGT_IMG_PATH} 和 {TGT_MASK_PATH}")

        # --- 6. 创建 DPO 的临时 Manifest (使用所有真实路径, TGT 作为 win/lose) ---
        print("正在为 DPOFaceDataset 创建临时 manifest...")
        dpo_data = [{
            "path_B_source": SRC_IMG_PATH,   # 使用 SRC
            "path_B_mask":   SRC_MASK_PATH,  # 使用 SRC Mask
            "path_D_target": TGT_IMG_PATH,   # 使用 TGT
            "path_D_mask":   TGT_MASK_PATH,  # 使用 TGT Mask
            "path_A_chosen": TGT_IMG_PATH,   # Winner 使用 TGT
            "path_E_rejected": TGT_IMG_PATH    # Loser 使用 TGT
        }]
        with open(TEMP_MANIFEST_DPO, 'w') as f: json.dump(dpo_data, f)
        print(f"  - DPO 输入将使用真实路径, TGT 作为 win/lose。")

        # --- 7. 配置 ARGS ---
        celeba_args = {
            "dataset_dir": TEMP_DIR_CELEBA, "image_size": IMG_SIZE,
            "preserve_mask_src": PRESERVE_LIST, "remove_mask_tar": PRESERVE_LIST,
            "gray_outer_mask": True,
        }
        dpo_args = { 
            "image_size": IMG_SIZE, "preserve_mask_src": PRESERVE_LIST,
            "remove_mask_tar": PRESERVE_LIST,
        }

        # --- 8. 实例化和运行 ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # --- 运行 CelebA (使用 TGT 数据) ---
        print("\n" + "="*20 + " 正在加载 [原始 CelebA (使用 TGT)] " + "="*20)
        set_seed(SEED) # 确保随机性可比
        orig_dataset = CelebAdataset(state="train", **celeba_args) 
        orig_item = orig_dataset[0]
        
        # --- 运行 DPO (使用 SRC/TGT 数据) ---
        print("\n" + "="*20 + " 正在加载 [DPO (使用 SRC/TGT)] " + "="*20)
        set_seed(SEED) # 重新设置种子
        dpo_dataset = DPOFaceDataset(data_manifest_path=TEMP_MANIFEST_DPO, args=dpo_args) 
        dpo_item = dpo_dataset[0]

        # --- 9. 可视化对比 ---
        print(f"\n" + "="*20 + " 正在保存对比结果到: {OUTPUT_DIR} " + "="*20)
        
        # CelebA 输出
        print("\n--- CelebAdataset 输出 (基于 TGT 输入) ---")
        save_tensor_image(orig_item['ref_imgs'], os.path.join(OUTPUT_DIR, "orig_01_ref_imgs (from TGT).png"), 'clip')
        save_tensor_image(orig_item['inpaint_mask'], os.path.join(OUTPUT_DIR, "orig_02_inpaint_mask (from TGT_mask).png"), '0to1')
        save_tensor_image(orig_item['GT'], os.path.join(OUTPUT_DIR, "orig_03_GT (from TGT).png"), '-1to1')
        save_tensor_image(orig_item['inpaint_image'], os.path.join(OUTPUT_DIR, "orig_04_inpaint_image (GT x Mask).png"), '-1to1')

        # DPO 输出
        print("\n--- DPOFaceDataset 输出 (基于 SRC/TGT 输入) ---")
        save_tensor_image(dpo_item['ref_imgs'],  os.path.join(OUTPUT_DIR, "dpo_01_ref_imgs (from SRC).png"), 'clip')
        save_tensor_image(dpo_item['inpaint_mask'],  os.path.join(OUTPUT_DIR, "dpo_02_inpaint_mask (from TGT_mask).png"), '0to1')
        save_tensor_image(dpo_item['GT_w'], os.path.join(OUTPUT_DIR, "dpo_03_GT_w (from TGT).png"), '-1to1')
        save_tensor_image(dpo_item['GT_l'], os.path.join(OUTPUT_DIR, "dpo_04_GT_l (from TGT).png"), '-1to1') 
        save_tensor_image(dpo_item['inpaint_image'],  os.path.join(OUTPUT_DIR, "dpo_05_inpaint_image (GT_w x Mask).png"), '-1to1')

        print("\n" + "="*50)
        print("✅ 最终可视化对比脚本完成。")
        print(f"请检查 '{OUTPUT_DIR}' 文件夹中的图像。")
        print("主要检查点:")
        print("  - orig_01 和 dpo_01: 内容不同 (TGT vs SRC), 但随机增强应相同。")
        print("  - orig_02 和 dpo_02: 应该视觉上相同 (都来自 TGT_mask, 相同 Decow)。")
        print("  - orig_03 和 dpo_03: 应该视觉上相同 (都来自 TGT, 相同处理)。")
        print("  - dpo_04: 应该与 dpo_03 相同。")
        print("  - orig_04 和 dpo_05: 应该视觉上相同 (都是 GT * Mask)。")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # --- 10. 清理 ---
        if os.path.exists(TEMP_DIR_CELEBA): shutil.rmtree(TEMP_DIR_CELEBA)
        if os.path.exists(TEMP_MANIFEST_DPO): os.remove(TEMP_MANIFEST_DPO)
        print(f"\n已清理临时文件。")

if __name__ == "__main__":
    main()