"""
验证阶段自动生成图片并评估的回调
支持：
1. 在指定global_step后触发验证
2. 只生成前N对验证图片（默认100对）
3. 自动调用评估脚本计算指标
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ValidationEvaluationCallback(Callback):
    """
    在验证阶段生成图片并自动评估
    """
    def __init__(
        self, 
        eval_every_n_steps=300,      # 每隔多少步进行一次评估
        max_val_samples=100,          # 最多评估多少对图片
        batch_size=4,                 # 生成时的batch size
        ddim_steps=50,                # DDIM采样步数
        save_dir=None,                # 保存目录，None则使用checkpoint目录
        run_evaluation=True,          # 是否运行评估脚本
        device=0,                     # 评估使用的GPU
        start_step=0,                 # 从哪个step开始评估
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.max_val_samples = max_val_samples
        self.batch_size = batch_size
        self.ddim_steps = ddim_steps
        self.save_dir = save_dir
        self.run_evaluation = run_evaluation
        self.device = device
        self.start_step = start_step
        self.last_eval_step = -1
        
    def should_evaluate(self, global_step):
        """判断是否应该在当前步数进行评估"""
        if global_step < self.start_step:
            return False
        if global_step == self.last_eval_step:
            return False
        if (global_step % self.eval_every_n_steps) == 0:
            return True
        return False
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时触发评估"""
        global_step = pl_module.global_step
        
        if not self.should_evaluate(global_step):
            return
            
        print(f"\n{'='*80}")
        print(f"🔍 Starting Validation Evaluation at step {global_step}")
        print(f"{'='*80}\n")
        
        self.last_eval_step = global_step
        
        # 1. 生成验证图片
        output_dir = self._generate_validation_images(trainer, pl_module, global_step)
        
        # 2. 运行评估
        if self.run_evaluation and output_dir is not None:
            self._run_evaluation_scripts(output_dir, global_step)
    
    def _generate_validation_images(self, trainer, pl_module, global_step):
        """生成验证集图片"""
        print(f"📸 Generating validation images (max {self.max_val_samples} samples)...")
        
        # 设置保存目录
        if self.save_dir is None:
            log_dir = Path(trainer.logger.save_dir)
            output_dir = log_dir / f"validation_eval_step_{global_step:06d}"
        else:
            output_dir = Path(self.save_dir) / f"step_{global_step:06d}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        # 获取验证数据加载器
        val_dataloader = trainer.val_dataloaders[0] if trainer.val_dataloaders else None
        if val_dataloader is None:
            print("⚠️  No validation dataloader found!")
            return None
        
        # 设置模型为评估模式
        was_training = pl_module.training
        if was_training:
            pl_module.eval()
        
        # 生成图片
        total_generated = 0
        max_batches = (self.max_val_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    # 处理数据格式
                    # test_bench_dataset 返回元组: (image, prior, dict, index)
                    if isinstance(batch, (tuple, list)) and len(batch) == 4:
                        image_tensor, prior_tensor, batch_dict, index_str = batch
                        # 构建符合 log_images 期望的字典格式
                        batch = {
                            'image': image_tensor.to(pl_module.device),
                            'GT': prior_tensor.to(pl_module.device),
                            'inpaint_image': batch_dict['inpaint_image'].to(pl_module.device),
                            'inpaint_mask': batch_dict['inpaint_mask'].to(pl_module.device),
                            'ref': batch_dict['ref_imgs'].squeeze(1).to(pl_module.device),  # 去掉多余维度
                        }
                    # 处理字典格式（DPO数据集）
                    elif isinstance(batch, dict):
                        batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    
                    # 生成图片（使用EMA模型）
                    with pl_module.ema_scope("Validation"):
                        images_dict = pl_module.log_images(
                            batch,
                            N=min(batch['image'].shape[0], self.batch_size),
                            sample=True,
                            ddim_steps=self.ddim_steps,
                            ddim_eta=1.0
                        )
                    
                    # 保存生成的图片
                    if 'output_current' in images_dict:
                        output_images = images_dict['output_current']
                    elif 'samples' in images_dict:
                        output_images = images_dict['samples']
                    else:
                        print(f"⚠️  No output images in batch {batch_idx}")
                        continue
                    
                    batch_size_actual = output_images.shape[0]
                    for idx in range(batch_size_actual):
                        if total_generated >= self.max_val_samples:
                            break
                        
                        # 保存图片（格式：000000000000.png）
                        img_path = output_dir / f"{total_generated:012d}.png"
                        self._save_image(output_images[idx], img_path)
                        total_generated += 1
                    
                    print(f"  Progress: {total_generated}/{self.max_val_samples} images generated", end='\r')
                    
                except Exception as e:
                    print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n✅ Generated {total_generated} validation images")
        
        # 恢复训练模式
        if was_training:
            pl_module.train()
        
        return output_dir
    
    def _save_image(self, tensor, path):
        """保存单张图片"""
        # tensor: [C, H, W], range [-1, 1]
        img = tensor.cpu().numpy()
        img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def _run_evaluation_scripts(self, output_dir, global_step):
        """运行评估脚本"""
        print(f"\n{'='*80}")
        print(f"📊 Running Evaluation Scripts")
        print(f"{'='*80}\n")
        
        # 数据路径（根据你的配置）
        source_path = "dataset/FaceData/CelebAMask-HQ/Val_target"
        target_path = "dataset/FaceData/CelebAMask-HQ/Val"
        source_mask_path = "dataset/FaceData/CelebAMask-HQ/target_mask"
        target_mask_path = "dataset/FaceData/CelebAMask-HQ/src_mask"
        dataset_path = "dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"
        
        results_file = output_dir / "evaluation_results.txt"
        
        # 写入评估信息头部
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"="*80 + "\n")
            f.write(f"Global Step: {global_step}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Max Samples: {self.max_val_samples}\n")
            f.write(f"="*80 + "\n\n")
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.device)
        
        # 1. FID Score
        print("📈 Computing FID score...")
        self._run_metric_evaluation(
            "FID Score",
            ["python", "eval_tool/fid/fid_score.py",
             "--device", "cuda",
             "--max-samples", str(self.max_val_samples),
             dataset_path, str(output_dir)],
            results_file, env
        )
        
        # 2. ID Similarity
        print("👤 Computing ID similarity...")
        self._run_metric_evaluation(
            "ID Similarity (Arcface)",
            ["python", "eval_tool/ID_retrieval/ID_retrieval.py",
             "--device", "cuda",
             "--max-samples", str(self.max_val_samples),
             "--dataset", "ffhq",
             "--arcface", "True",
             source_path, str(output_dir), source_mask_path, target_mask_path],
            results_file, env
        )
        
        # 3. Pose Comparison
        print("🤸 Computing pose consistency...")
        self._run_metric_evaluation(
            "Pose Comparison",
            ["python", "eval_tool/Pose/pose_compare.py",
             "--device", "cuda",
             "--max-samples", str(self.max_val_samples),
             target_path, str(output_dir)],
            results_file, env
        )
        
        # 4. Expression Comparison
        print("😊 Computing expression consistency...")
        self._run_metric_evaluation(
            "Expression Comparison",
            ["python", "eval_tool/Expression/expression_compare_face_recon.py",
             "--device", "cuda",
             "--max-samples", str(self.max_val_samples),
             target_path, str(output_dir)],
            results_file, env
        )
        
        print(f"\n{'='*80}")
        print(f"✅ Evaluation Complete!")
        print(f"📄 Results saved to: {results_file}")
        print(f"{'='*80}\n")
    
    def _run_metric_evaluation(self, metric_name, cmd, results_file, env):
        """运行单个指标的评估"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=600  # 10分钟超时
            )
            
            with open(results_file, 'a') as f:
                f.write(f"\n{metric_name}:\n")
                f.write("-" * 80 + "\n")
                f.write(result.stdout + "\n")
                if result.stderr:
                    f.write("Stderr:\n" + result.stderr + "\n")
            
            # 提取关键数值并打印
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:  # 打印最后3行（通常包含结果）
                print(f"  {line}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Timeout computing {metric_name}")
            with open(results_file, 'a') as f:
                f.write(f"\n{metric_name}: TIMEOUT\n")
        except Exception as e:
            print(f"  ⚠️  Error computing {metric_name}: {e}")
            with open(results_file, 'a') as f:
                f.write(f"\n{metric_name}: ERROR - {e}\n")

