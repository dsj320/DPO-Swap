#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测脚本 - 用于评测训练过程中生成的 test_samples
支持自动评测 FID, ID相似度, Pose距离, Expression距离
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class MetricsEvaluator:
    """指标评测器"""
    
    def __init__(self, test_samples_dir, dataset_type='celeba', device='cuda:3', max_samples=None):
        """
        初始化评测器
        
        Args:
            test_samples_dir: 测试样本目录路径
            dataset_type: 数据集类型 ('celeba' 或 'ffhq')
            device: 使用的 GPU 设备
            max_samples: 最大样本数量（None 表示评测全部）
        """
        self.test_samples_dir = Path(test_samples_dir)
        self.dataset_type = dataset_type.lower()
        self.device = device
        self.max_samples = max_samples
        self.project_root = PROJECT_ROOT
        
        # 检查测试样本目录
        if not self.test_samples_dir.exists():
            raise ValueError(f"测试样本目录不存在: {self.test_samples_dir}")
        
        # 统计样本数量
        self.sample_count = len(list(self.test_samples_dir.glob("*.png")))
        if self.sample_count == 0:
            raise ValueError(f"测试样本目录中没有 PNG 图像: {self.test_samples_dir}")
        
        print(f"✓ 找到 {self.sample_count} 个测试样本")
        
        # 配置数据集路径
        self._setup_dataset_paths()
        
        # 创建输出目录
        self._setup_output_dir()
    
    def _setup_dataset_paths(self):
        """配置数据集路径"""
        if self.dataset_type == 'celeba':
            print("✓ 使用 CelebA 数据集配置")
            self.source_path = self.project_root / "dataset/FaceData/CelebAMask-HQ/Val_target"
            self.target_path = self.project_root / "dataset/FaceData/CelebAMask-HQ/Val"
            self.source_mask_path = self.project_root / "dataset/FaceData/CelebAMask-HQ/target_mask"
            self.target_mask_path = self.project_root / "dataset/FaceData/CelebAMask-HQ/src_mask"
            self.dataset_path = self.project_root / "dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"
        elif self.dataset_type == 'ffhq':
            print("✓ 使用 FFHQ 数据集配置")
            self.source_path = self.project_root / "dataset/FaceData/FFHQ/Val"
            self.target_path = self.project_root / "dataset/FaceData/FFHQ/Val_target"
            self.source_mask_path = self.project_root / "dataset/FaceData/FFHQ/src_mask"
            self.target_mask_path = self.project_root / "dataset/FaceData/FFHQ/target_mask"
            self.dataset_path = self.project_root / "dataset/FaceData/FFHQ/images512"
        else:
            raise ValueError(f"未知的数据集类型: {self.dataset_type}")
    
    def _setup_output_dir(self):
        """创建输出目录"""
        # 输出目录（与 test_samples 同级）
        metrics_dir_name = f"metrics_{self.test_samples_dir.name}"
        self.metrics_dir = self.test_samples_dir.parent / metrics_dir_name
        self.metrics_dir.mkdir(exist_ok=True)
        
        # 输出文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.metrics_dir / f"evaluation_{timestamp}.txt"
        self.id_output_file = self.metrics_dir / f"id_retrieval_{timestamp}.txt"
        
        print(f"✓ 评测结果将保存到: {self.output_file}")
        
        # 写入基本信息
        with open(self.output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("评测测试样本\n")
            f.write("=" * 60 + "\n")
            f.write(f"样本目录: {self.test_samples_dir}\n")
            f.write(f"样本数量: {self.sample_count}\n")
            f.write(f"数据集类型: {self.dataset_type}\n")
            f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.max_samples:
                f.write(f"最大样本数: {self.max_samples}\n")
            f.write("=" * 60 + "\n\n")
    
    def _run_command(self, cmd, description):
        """运行命令并记录输出"""
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
        
        # 写入日志
        with open(self.output_file, 'a') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{description}\n")
            f.write(f"{'='*60}\n")
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)
        
        # 提取 GPU 编号
        if ':' in self.device:
            gpu_id = self.device.split(':')[1]
        else:
            gpu_id = '0'
        
        env['CUDA_VISIBLE_DEVICES'] = gpu_id
        
        try:
            # 运行命令并实时输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1
            )
            
            # 实时读取并显示输出
            with open(self.output_file, 'a') as f:
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"⚠️  警告: 命令执行返回非零状态码: {process.returncode}")
            
            with open(self.output_file, 'a') as f:
                f.write("\n")
            
            return True
            
        except Exception as e:
            print(f"✗ 错误: {e}")
            with open(self.output_file, 'a') as f:
                f.write(f"错误: {e}\n\n")
            return False
    
    def evaluate_fid(self):
        """评测 FID Score"""
        cmd = [
            'python', 'eval_tool/fid/fid_score.py',
            '--device', self.device,
            str(self.dataset_path),
            str(self.test_samples_dir)
        ]
        
        if self.max_samples:
            cmd.extend(['--max-samples', str(self.max_samples)])
        
        return self._run_command(cmd, "1. 计算 FID Score (与真实数据集比较)")
    
    def evaluate_id_similarity(self):
        """评测 ID 相似度"""
        cmd = [
            'python', 'eval_tool/ID_retrieval/ID_retrieval.py',
            '--device', self.device,
            str(self.source_path),
            str(self.test_samples_dir),
            str(self.source_mask_path),
            str(self.target_mask_path),
            '--dataset', self.dataset_type,
            '--print_sim', 'True',
            '--arcface', 'True',
            '--output', str(self.id_output_file)
        ]
        
        if self.max_samples:
            cmd.extend(['--max-samples', str(self.max_samples)])
        
        return self._run_command(cmd, "2. 计算 ID 相似度 (Arcface - 与源图像比较)")
    
    def evaluate_pose(self):
        """评测 Pose Distance"""
        cmd = [
            'python', 'eval_tool/Pose/pose_compare.py',
            '--device', self.device,
            str(self.target_path),
            str(self.test_samples_dir)
        ]
        
        if self.max_samples:
            cmd.extend(['--max-samples', str(self.max_samples)])
        
        return self._run_command(cmd, "3. 计算 Pose Distance (与目标图像比较)")
    
    def evaluate_expression(self):
        """评测 Expression Distance"""
        cmd = [
            'python', 'eval_tool/Expression/expression_compare_face_recon.py',
            '--device', self.device,
            str(self.target_path),
            str(self.test_samples_dir)
        ]
        
        if self.max_samples:
            cmd.extend(['--max-samples', str(self.max_samples)])
        
        return self._run_command(cmd, "4. 计算 Expression Distance (与目标图像比较)")
    
    def evaluate_all(self):
        """运行所有评测"""
        print(f"\n{'='*60}")
        print("开始评测所有指标")
        print(f"{'='*60}\n")
        
        results = {
            'FID': self.evaluate_fid(),
            'ID Similarity': self.evaluate_id_similarity(),
            'Pose': self.evaluate_pose(),
            'Expression': self.evaluate_expression()
        }
        
        # 完成信息
        print(f"\n{'='*60}")
        print("评测完成！")
        print(f"{'='*60}")
        print(f"✓ 结果保存在: {self.output_file}")
        print(f"✓ ID 检索详情: {self.id_output_file}")
        
        with open(self.output_file, 'a') as f:
            f.write("=" * 60 + "\n")
            f.write("评测完成！\n")
            f.write("=" * 60 + "\n")
        
        # 显示评测状态
        print("\n评测状态:")
        for metric, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {metric}")
        
        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='评测训练过程中生成的测试样本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'test_samples_dir',
        type=str,
        help='测试样本目录路径'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='celeba',
        choices=['celeba', 'ffhq'],
        help='数据集类型'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:3',
        help='使用的 GPU 设备 (例如: cuda:0, cuda:1)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大样本数量 (None 表示评测全部)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'fid', 'id', 'pose', 'expression'],
        help='要评测的指标 (默认: all)'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建评测器
        evaluator = MetricsEvaluator(
            test_samples_dir=args.test_samples_dir,
            dataset_type=args.dataset,
            device=args.device,
            max_samples=args.max_samples
        )
        
        # 运行评测
        if 'all' in args.metrics:
            success = evaluator.evaluate_all()
        else:
            success = True
            if 'fid' in args.metrics:
                success &= evaluator.evaluate_fid()
            if 'id' in args.metrics:
                success &= evaluator.evaluate_id_similarity()
            if 'pose' in args.metrics:
                success &= evaluator.evaluate_pose()
            if 'expression' in args.metrics:
                success &= evaluator.evaluate_expression()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()






