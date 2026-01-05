"""
可视化 CelebA 数据集返回的数据
用于检查数据集的逻辑是否正确

使用方法：
    # 默认：验证前5个样本
    python ldm/data/visualize_celebA.py
    
    # 验证前10个样本
    python ldm/data/visualize_celebA.py --num_samples 10
    
    # 从指定位置开始验证
    python ldm/data/visualize_celebA.py --start_idx 100 --num_samples 5
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as T

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ldm.data.celebA import CelebAdataset

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

def visualize_sample(sample, save_path=None, idx=0, has_nomask=False):
    """
    可视化单个样本的所有字段
    
    Args:
        sample: 数据集返回的字典
        save_path: 保存路径（可选）
        idx: 样本索引
        has_nomask: 是否有ref_imgs_nomask字段
    """
    # 根据是否有nomask字段决定布局
    if has_nomask:
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. GT (Ground Truth)
    ax1 = fig.add_subplot(gs[0, 0])
    gt = un_norm(sample['GT']).clamp(0, 1)
    ax1.imshow(gt.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('GT (Ground Truth)\nComplete Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. inpaint_image (masked image)
    ax2 = fig.add_subplot(gs[0, 1])
    inpaint_img = un_norm(sample['inpaint_image']).clamp(0, 1)
    ax2.imshow(inpaint_img.permute(1, 2, 0).cpu().numpy())
    ax2.set_title('inpaint_image\n(Masked Image)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. inpaint_mask
    ax3 = fig.add_subplot(gs[0, 2])
    inpaint_mask = sample['inpaint_mask']
    if inpaint_mask.shape[0] == 1:
        mask_vis = inpaint_mask.repeat(3, 1, 1)
    else:
        mask_vis = inpaint_mask
    ax3.imshow(mask_vis.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax3.set_title('inpaint_mask\n(1=keep, 0=inpaint)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. ref_imgs
    if has_nomask:
        ax4 = fig.add_subplot(gs[0, 3])
    else:
        ax4 = fig.add_subplot(gs[1, 0])
    
    ref_imgs = sample['ref_imgs']
    # 检查是否是CLIP归一化
    if ref_imgs.min() < -1.5 or ref_imgs.max() > 1.5:
        ref_imgs_vis = un_norm_clip(ref_imgs).clamp(0, 1)
    else:
        ref_imgs_vis = un_norm(ref_imgs).clamp(0, 1)
    ax4.imshow(ref_imgs_vis.permute(1, 2, 0).cpu().numpy())
    ax4.set_title('ref_imgs\n(Reference Image)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. inpaint_image with mask overlay
    if has_nomask:
        ax5 = fig.add_subplot(gs[1, 0])
    else:
        ax5 = fig.add_subplot(gs[1, 1])
    inpaint_with_mask = inpaint_img.clone()
    mask_overlay = sample['inpaint_mask'].repeat(3, 1, 1)
    # 在mask区域添加红色半透明覆盖
    mask_region = (mask_overlay < 0.5).float()
    overlay = inpaint_with_mask * 0.7 + mask_region * torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1) * 0.3
    ax5.imshow(overlay.permute(1, 2, 0).cpu().numpy())
    ax5.set_title('inpaint_image\n+ mask overlay (red)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. GT with mask overlay
    if has_nomask:
        ax6 = fig.add_subplot(gs[1, 1])
    else:
        ax6 = fig.add_subplot(gs[1, 2])
    gt_with_mask = gt.clone()
    gt_overlay = gt_with_mask * 0.7 + mask_region * torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1) * 0.3
    ax6.imshow(gt_overlay.permute(1, 2, 0).cpu().numpy())
    ax6.set_title('GT\n+ mask overlay (green)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. ref_imgs_nomask (if available)
    if has_nomask and 'ref_imgs_nomask' in sample:
        ax7 = fig.add_subplot(gs[1, 2])
        ref_imgs_nomask = sample['ref_imgs_nomask']
        if ref_imgs_nomask.min() < -1.5 or ref_imgs_nomask.max() > 1.5:
            ref_imgs_nomask_vis = un_norm_clip(ref_imgs_nomask).clamp(0, 1)
        else:
            ref_imgs_nomask_vis = un_norm(ref_imgs_nomask).clamp(0, 1)
        ax7.imshow(ref_imgs_nomask_vis.permute(1, 2, 0).cpu().numpy())
        ax7.set_title('ref_imgs_nomask\n(Unmasked Reference)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 8. Comparison: ref_imgs vs ref_imgs_nomask
        ax8 = fig.add_subplot(gs[1, 3])
        # Resize to same size for comparison
        if ref_imgs_vis.shape != ref_imgs_nomask_vis.shape:
            ref_imgs_nomask_vis_resized = T.Resize(ref_imgs_vis.shape[-2:])(ref_imgs_nomask_vis)
        else:
            ref_imgs_nomask_vis_resized = ref_imgs_nomask_vis
        ref_comparison = torch.cat([ref_imgs_vis, ref_imgs_nomask_vis_resized], dim=2)
        ax8.imshow(ref_comparison.permute(1, 2, 0).cpu().numpy())
        ax8.set_title('Ref Comparison:\nmasked (left) vs unmasked (right)', fontsize=12, fontweight='bold')
        ax8.axis('off')
    
    # 添加整体标题和信息
    fields = ['GT', 'inpaint_image', 'inpaint_mask', 'ref_imgs']
    if has_nomask and 'ref_imgs_nomask' in sample:
        fields.append('ref_imgs_nomask')
    
    info_text = f"""Sample {idx}
Fields: {', '.join(fields)}
Shapes: GT={sample['GT'].shape}, inpaint_image={sample['inpaint_image'].shape}, 
        inpaint_mask={sample['inpaint_mask'].shape}, ref_imgs={sample['ref_imgs'].shape}
Ranges: GT=[{sample['GT'].min():.2f}, {sample['GT'].max():.2f}], 
        mask=[{sample['inpaint_mask'].min():.2f}, {sample['inpaint_mask'].max():.2f}]
        ref_imgs=[{sample['ref_imgs'].min():.2f}, {sample['ref_imgs'].max():.2f}]"""
    
    plt.suptitle(f'CelebA Dataset Sample Visualization\n{info_text}', 
                 fontsize=11, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """主函数：加载数据集并可视化几个样本"""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CelebA Dataset')
    parser.add_argument('--dataset_dir', type=str, 
                       default='./dataset/FaceData/CelebAMask-HQ/',
                       help='Path to CelebA dataset directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (default: 5)')
    parser.add_argument('--output_dir', type=str, default='./visualizations_celebA',
                       help='Output directory for visualizations')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for visualization')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--gray_outer_mask', action='store_true', default=True,
                       help='Use gray outer mask mode (default: True)')
    parser.add_argument('--no_gray_outer_mask', dest='gray_outer_mask', action='store_false',
                       help='Disable gray outer mask mode')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据集参数 (通过 **kwargs 传递)
    dataset_kwargs = {
        'dataset_dir': args.dataset_dir,
        'image_size': args.image_size,
        'gray_outer_mask': args.gray_outer_mask,
        'preserve_mask_src': [1, 2, 4, 5, 8, 9, 6, 7, 10, 11, 12, 17],  # 核心人脸区域
        'remove_mask_tar': [1, 2, 4, 5, 8, 9, 6, 7, 10, 11, 12, 17],  # 完整人脸区域
    }
    
    print("=" * 80)
    print("Loading CelebA Dataset...")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Gray outer mask: {args.gray_outer_mask}")
    print("=" * 80)
    
    try:
        # 创建数据集 - state是位置参数，其他通过kwargs传递
        dataset = CelebAdataset(
            state='train',
            arbitrary_mask_percent=0.5,
            **dataset_kwargs
        )
        
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        print(f"Visualizing {args.num_samples} samples starting from index {args.start_idx}")
        print("=" * 80)
        
        # 检测数据集是否有nomask字段
        sample_0 = dataset[0]
        has_nomask = 'ref_imgs_nomask' in sample_0
        print(f"Dataset has ref_imgs_nomask field: {has_nomask}")
        print("=" * 80)
        
        # 可视化样本
        success_count = 0
        for i in range(args.num_samples):
            idx = args.start_idx + i
            if idx >= len(dataset):
                print(f"Warning: Index {idx} exceeds dataset size ({len(dataset)})")
                break
            
            print(f"\nProcessing sample {i+1}/{args.num_samples} (index {idx})...")
            
            try:
                sample = dataset[idx]
                
                # 保存可视化
                mode_str = 'gray' if args.gray_outer_mask else 'black'
                save_path = os.path.join(args.output_dir, f'celebA_sample_{idx:04d}_{mode_str}.png')
                visualize_sample(sample, save_path=save_path, idx=idx, has_nomask=has_nomask)
                
                print(f"  ✓ Sample {idx} visualized successfully")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print(f"Visualization complete!")
        print(f"  - Successfully visualized: {success_count}/{args.num_samples} samples")
        print(f"  - Results saved to: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

