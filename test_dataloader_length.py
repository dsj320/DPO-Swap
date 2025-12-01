import sys
import os
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# 加载配置
config = OmegaConf.load("configs/train_dpo.yaml")

# 创建数据模块
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

print("=" * 60)
print("数据集信息:")
for k in data.datasets:
    print(f"  {k}: {data.datasets[k].__class__.__name__}, 长度={len(data.datasets[k])}")

print("\n" + "=" * 60)
print("DataLoader配置:")
print(f"  batch_size: {data.batch_size}")
print(f"  num_workers: {data.num_workers}")
print(f"  use_worker_init_fn: {data.use_worker_init_fn}")

print("\n" + "=" * 60)
print("训练DataLoader信息:")
train_loader = data.train_dataloader()
print(f"  DataLoader长度: {len(train_loader)}")
print(f"  数据集长度: {len(data.datasets['train'])}")
print(f"  Batch size: {train_loader.batch_size}")
print(f"  Num workers: {train_loader.num_workers}")
print(f"  理论batch数: {len(data.datasets['train']) / train_loader.batch_size:.2f}")

print("\n" + "=" * 60)
print("分析:")
expected_steps = len(data.datasets['train']) / data.batch_size
print(f"  预期每个epoch的步数: {expected_steps:.0f}")
print(f"  实际DataLoader报告的长度: {len(train_loader)}")
print(f"  差异: {len(train_loader) - expected_steps:.0f}")
if len(train_loader) != expected_steps:
    ratio = len(train_loader) / expected_steps
    print(f"  比例: {ratio:.2f}x")
    if abs(ratio - 6.25) < 0.1:
        print("  ⚠️  可能是num_workers导致的问题! (24 workers / 4 GPUs ≈ 6)")


