"""
可视化 SFTFaceDataset (13ch版本) 返回的数据
用于检查数据集的逻辑是否正确
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image, make_grid
import torchvision.transforms as T

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ldm.data.sft_dataset_13ch import SFTFaceDataset

def un_norm(x):
    """反归一化：从 [-1, 1] 到 [0, 1]"""
    return (x + 1.0) / 2.0

def un_norm_clip(x):
    """反归一化CLIP：从CLIP归一化到 [0, 1]"""
    x = x * 1.0  # 避免修改原tensor
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073
    if reduce:
        x = x.squeeze(0)
    # CLIP归一化后的值可能在 [-2, 2] 范围，需要clamp到 [0, 1]
    x = torch.clamp(x, 0, 1)
    return x

def visualize_sample(sample, save_path=None, idx=0, label=None, mask_config=None):
    """
    可视化单个样本的所有字段
    
    Args:
        sample: 数据集返回的字典
        save_path: 保存路径（可选）
        idx: 样本索引
        label: 数据标签 (sft/recon)
        mask_config: mask配置信息字典
    """
    # 创建大图 (调整为3x3布局)
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. GT_w (Ground Truth Winner - 换脸结果)
    ax1 = fig.add_subplot(gs[0, 0])
    gt_w = un_norm(sample['GT_w']).clamp(0, 1)
    ax1.imshow(gt_w.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('GT_w (Winner)\npath_A_chosen', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. GT (Target/Base Image)
    ax2 = fig.add_subplot(gs[0, 1])
    gt = un_norm(sample['GT']).clamp(0, 1)
    ax2.imshow(gt.permute(1, 2, 0).cpu().numpy())
    ax2.set_title('GT (Target)\npath_D_target', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. mixed_3d_landmarks (3D关键点图)
    ax3 = fig.add_subplot(gs[0, 2])
    mixed_3d = un_norm(sample['mixed_3d_landmarks']).clamp(0, 1)
    ax3.imshow(mixed_3d.permute(1, 2, 0).cpu().numpy())
    ax3.set_title('mixed_3d_landmarks\n(3D guide)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. inpaint_image (masked image)
    ax4 = fig.add_subplot(gs[1, 0])
    inpaint_img = un_norm(sample['inpaint_image']).clamp(0, 1)
    ax4.imshow(inpaint_img.permute(1, 2, 0).cpu().numpy())
    ax4.set_title('inpaint_image\n(masked target)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. inpaint_mask
    ax5 = fig.add_subplot(gs[1, 1])
    inpaint_mask = sample['inpaint_mask']
    if inpaint_mask.shape[0] == 1:
        mask_vis = inpaint_mask.repeat(3, 1, 1)
    else:
        mask_vis = inpaint_mask
    ax5.imshow(mask_vis.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax5.set_title('inpaint_mask\npath_D_mask', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. inpaint_image with mask overlay
    ax6 = fig.add_subplot(gs[1, 2])
    inpaint_with_mask = inpaint_img.clone()
    mask_overlay = sample['inpaint_mask'].repeat(3, 1, 1)
    # 在mask区域添加红色半透明覆盖
    mask_region = (mask_overlay < 0.5).float()
    overlay = inpaint_with_mask * 0.7 + mask_region * torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1) * 0.3
    ax6.imshow(overlay.permute(1, 2, 0).cpu().numpy())
    ax6.set_title('inpaint_image\n+ mask overlay', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. ref_imgs (masked, for conditioning)
    ax7 = fig.add_subplot(gs[2, 0])
    ref_imgs = un_norm_clip(sample['ref_imgs']).clamp(0, 1)
    ax7.imshow(ref_imgs.permute(1, 2, 0).cpu().numpy())
    ax7.set_title('ref_imgs (masked)\npath_B_source (224x224)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 8. ref_img_raw (unmasked, for ID loss)
    ax8 = fig.add_subplot(gs[2, 1])
    ref_img_raw = un_norm_clip(sample['ref_img_raw']).clamp(0, 1)
    ax8.imshow(ref_img_raw.permute(1, 2, 0).cpu().numpy())
    ax8.set_title('ref_img_raw (unmasked)\npath_B_source (224x224)', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # 9. 对比：ref_imgs vs ref_img_raw
    ax9 = fig.add_subplot(gs[2, 2])
    ref_comparison = torch.cat([ref_imgs, ref_img_raw], dim=2)
    ax9.imshow(ref_comparison.permute(1, 2, 0).cpu().numpy())
    ax9.set_title('Ref Comparison:\nmasked (left) vs unmasked (right)', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    # 添加整体标题和信息
    label_str = f" [Label: {label}]" if label else ""
    mask_info = ""
    if mask_config:
        mask_info = f"\nMask Config: preserve_src={mask_config.get('preserve_src', 'N/A')}, remove_tar={mask_config.get('remove_tar', 'N/A')}"
    
    info_text = f"""Sample {idx}{label_str}{mask_info}
Shapes: GT_w={sample['GT_w'].shape}, GT={sample['GT'].shape}, inpaint_image={sample['inpaint_image'].shape}
        inpaint_mask={sample['inpaint_mask'].shape}, ref_imgs={sample['ref_imgs'].shape}, mixed_3d={sample['mixed_3d_landmarks'].shape}
Ranges: GT_w=[{sample['GT_w'].min():.2f}, {sample['GT_w'].max():.2f}], mask=[{sample['inpaint_mask'].min():.2f}, {sample['inpaint_mask'].max():.2f}]
        ref_imgs=[{sample['ref_imgs'].min():.2f}, {sample['ref_imgs'].max():.2f}]"""
    
    plt.suptitle(f'SFTFaceDataset (13ch) Sample Visualization\n{info_text}', 
                 fontsize=11, fontweight='bold', y=0.99)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """主函数：加载数据集并可视化几个样本"""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize SFTFaceDataset (13ch)')
    parser.add_argument('--manifest', type=str, 
                       default='./dataset/sft_final.json',
                       help='Path to data manifest JSON file')
    parser.add_argument('--base_3d_path', type=str,
                       default='./models/third_party',
                       help='Path to 3D model directory')
    parser.add_argument('--config', type=str,
                       default='./configs/train_sft_13ch.yaml',
                       help='Path to config YAML file (for args)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize from start and end (default: 5)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for visualization (ignored if --verify_both is used)')
    parser.add_argument('--verify_both', action='store_true', default=True,
                       help='Verify both beginning and end of dataset (default: True for sft+recon validation)')
    parser.add_argument('--no_verify_both', dest='verify_both', action='store_false',
                       help='Disable verify_both mode, only visualize from start_idx')
    parser.add_argument('--filter_label', type=str, default=None, choices=['sft', 'recon', None],
                       help='Only visualize samples with specific label (sft/recon/None for all)')
    parser.add_argument('--show_label_stats', action='store_true',
                       help='Show detailed statistics about labels and mask configs')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 从config文件读取参数 - 使用新的动态mask配置
    dataset_args = {
        'state': 'train',
        'arbitrary_mask_percent': 0.5,
        'image_size': 512,
        'data_seed': 42,
        'gray_outer_mask': True,
        'ref_imgs_augmentation': True,
        
        # ⭐ 使用新的动态mask配置
        'preserve_mask_src_sft': [1, 2, 4, 5, 6, 7, 10, 11, 12],  # SFT: 核心人脸
        'remove_mask_tar_sft': [1, 2, 4, 5, 6, 7, 10, 11, 12],
        'preserve_mask_src_recon': [1, 2, 4, 5, 8, 9, 6, 7, 10, 11, 12, 17],  # Recon: 完整人脸
        'remove_mask_tar_recon': [1, 2, 4, 5, 8, 9, 6, 7, 10, 11, 12, 17],
    }
    
    # 读取JSON文件以获取label信息
    print("=" * 80)
    print("Loading data manifest to check labels...")
    try:
        with open(args.manifest, 'r') as f:
            data_list = json.load(f)
        print(f"Loaded {len(data_list)} samples from manifest")
        
        # 统计label分布
        label_counts = {}
        for item in data_list:
            label = item.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")
    except Exception as e:
        print(f"Warning: Could not read manifest file: {e}")
        data_list = []
    
    print("=" * 80)
    print("Loading SFTFaceDataset (13ch)...")
    print("=" * 80)
    
    try:
        # 创建数据集
        dataset = SFTFaceDataset(
            data_manifest_path=args.manifest,
            base_3d_path=args.base_3d_path,
            args=dataset_args
        )
        
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        
        # 确定要可视化的索引列表
        if args.verify_both:
            # 验证前面和后面的样本
            first_indices = list(range(min(args.num_samples, len(dataset))))
            last_indices = list(range(max(0, len(dataset) - args.num_samples), len(dataset)))
            indices_to_visualize = first_indices + last_indices
            print(f"Verifying BOTH beginning and end of dataset:")
            print(f"  - First {len(first_indices)} samples: {first_indices}")
            print(f"  - Last {len(last_indices)} samples: {last_indices}")
            print(f"  - Total: {len(indices_to_visualize)} samples")
        else:
            # 只从指定位置开始可视化
            indices_to_visualize = list(range(args.start_idx, min(args.start_idx + args.num_samples, len(dataset))))
            print(f"Visualizing {len(indices_to_visualize)} samples starting from index {args.start_idx}")
        
        print("=" * 80)
        
        # 可视化样本
        success_count = 0
        for i, idx in enumerate(indices_to_visualize):
            if idx >= len(dataset):
                print(f"Warning: Index {idx} exceeds dataset size ({len(dataset)})")
                continue
            
            print(f"\nProcessing sample {i+1}/{len(indices_to_visualize)} (index {idx})...")
            
            try:
                # 获取label信息
                label = 'unknown'
                mask_config = None
                if idx < len(data_list):
                    label = data_list[idx].get('label', 'unknown')
                    
                    # 根据label确定mask配置
                    if label == 'recon':
                        mask_config = {
                            'preserve_src': dataset_args['preserve_mask_src_recon'],
                            'remove_tar': dataset_args['remove_mask_tar_recon']
                        }
                    else:  # sft
                        mask_config = {
                            'preserve_src': dataset_args['preserve_mask_src_sft'],
                            'remove_tar': dataset_args['remove_mask_tar_sft']
                        }
                
                sample = dataset[idx]
                
                # 保存可视化
                position = 'first' if idx < args.num_samples else 'last'
                save_path = os.path.join(args.output_dir, f'sample_{idx:04d}_{label}_{position}.png')
                visualize_sample(sample, save_path=save_path, idx=idx, label=label, mask_config=mask_config)
                
                print(f"  ✓ Sample {idx} (label: {label}) visualized successfully")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print(f"Visualization complete!")
        print(f"  - Successfully visualized: {success_count}/{len(indices_to_visualize)} samples")
        print(f"  - Results saved to: {args.output_dir}")
        if args.verify_both:
            print(f"  - Verified first {args.num_samples} and last {args.num_samples} samples")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

