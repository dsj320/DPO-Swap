#!/usr/bin/env python3
"""
使用官方pytorch-fid计算FID
"""
import subprocess
import sys
import argparse
from pathlib import Path

def check_pytorch_fid():
    """检查是否安装了pytorch-fid"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def install_pytorch_fid():
    """安装pytorch-fid"""
    print("正在安装pytorch-fid...")
    try:
        subprocess.run(
            ["pip", "install", "pytorch-fid"],
            check=True
        )
        print("✓ pytorch-fid安装成功")
        return True
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        print("请手动安装: pip install pytorch-fid")
        return False

def calculate_fid(path1, path2, batch_size=50, device="cuda"):
    """
    使用官方pytorch-fid计算FID
    
    Args:
        path1: 真实图像路径
        path2: 生成图像路径
        batch_size: 批次大小
        device: 计算设备
    """
    # 检查路径
    if not Path(path1).exists():
        print(f"错误：路径不存在 {path1}")
        return None
    if not Path(path2).exists():
        print(f"错误：路径不存在 {path2}")
        return None
    
    # 构建命令
    cmd = [
        "python", "-m", "pytorch_fid",
        str(path1),
        str(path2),
        "--batch-size", str(batch_size),
        "--device", device
    ]
    
    print("="*60)
    print("使用官方pytorch-fid计算FID")
    print("="*60)
    print(f"真实图像: {path1}")
    print(f"生成图像: {path2}")
    print(f"批次大小: {batch_size}")
    print(f"设备: {device}")
    print(f"\n命令: {' '.join(cmd)}")
    print("="*60)
    print()
    
    # 执行命令
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ FID计算完成")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("✗ FID计算失败")
            print("="*60)
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ 计算超时")
        return False
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="使用官方pytorch-fid计算FID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path', type=str, nargs=2,
                       help='两个图像目录的路径（真实数据集 生成数据集）')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--install', action='store_true',
                       help='安装pytorch-fid')
    
    args = parser.parse_args()
    
    # 检查安装
    if args.install or not check_pytorch_fid():
        if not check_pytorch_fid():
            print("pytorch-fid未安装")
            if not install_pytorch_fid():
                sys.exit(1)
        else:
            print("✓ pytorch-fid已安装")
    
    # 计算FID
    success = calculate_fid(
        args.path[0],
        args.path[1],
        batch_size=args.batch_size,
        device=args.device
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

