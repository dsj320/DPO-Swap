#!/usr/bin/env python3
"""
通道扩展脚本：将模型从9通道扩展到13通道
功能：在现有9通道权重后追加4个新的零值通道
"""
import torch
import json
from datetime import datetime
import os

print("=" * 80)
print("通道扩展脚本：9通道 → 13通道")
print("=" * 80)

# 配置参数
INPUT_CHECKPOINT = './last.ckpt'  # 输入：当前的9通道checkpoint
OUTPUT_CHECKPOINT = './last_13channel.ckpt'  # 输出：扩展后的13通道checkpoint
TARGET_KEY = 'model.diffusion_model.input_blocks.0.0.weight'

# 原始通道数和目标通道数
ORIGINAL_CHANNELS = 9
TARGET_CHANNELS = 13
ADDITIONAL_CHANNELS = TARGET_CHANNELS - ORIGINAL_CHANNELS

print(f"\n📁 输入文件: {INPUT_CHECKPOINT}")
print(f"📁 输出文件: {OUTPUT_CHECKPOINT}")
print(f"📊 通道数变化: {ORIGINAL_CHANNELS} → {TARGET_CHANNELS} (新增 {ADDITIONAL_CHANNELS} 个通道)")

# 1. 加载现有的9通道checkpoint
print(f"\n{'='*80}")
print("步骤 1: 加载当前checkpoint")
print(f"{'='*80}")

if not os.path.exists(INPUT_CHECKPOINT):
    print(f"❌ 错误: 找不到输入文件 {INPUT_CHECKPOINT}")
    exit(1)

try:
    ckpt_file = torch.load(INPUT_CHECKPOINT, map_location='cpu')
    print(f"✅ Checkpoint加载成功")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 2. 获取原始权重
print(f"\n{'='*80}")
print("步骤 2: 提取输入层权重")
print(f"{'='*80}")

if 'state_dict' not in ckpt_file:
    print(f"❌ 错误: checkpoint中没有'state_dict'键")
    exit(1)

if TARGET_KEY not in ckpt_file['state_dict']:
    print(f"❌ 错误: 找不到目标层 {TARGET_KEY}")
    exit(1)

original_weight = ckpt_file['state_dict'][TARGET_KEY]
print(f"✅ 目标层: {TARGET_KEY}")
print(f"✅ 原始权重形状: {original_weight.shape}")
print(f"   - 输出通道: {original_weight.shape[0]}")
print(f"   - 输入通道: {original_weight.shape[1]}")
print(f"   - 卷积核大小: {original_weight.shape[2]}x{original_weight.shape[3]}")

# 验证原始通道数
if original_weight.shape[1] != ORIGINAL_CHANNELS:
    print(f"⚠️  警告: 期望输入通道为{ORIGINAL_CHANNELS}，但实际为{original_weight.shape[1]}")
    response = input(f"是否继续？(y/n): ")
    if response.lower() != 'y':
        print("操作取消")
        exit(0)
    ORIGINAL_CHANNELS = original_weight.shape[1]
    ADDITIONAL_CHANNELS = TARGET_CHANNELS - ORIGINAL_CHANNELS
    print(f"✅ 已更新: 从{ORIGINAL_CHANNELS}通道扩展到{TARGET_CHANNELS}通道")

print(f"\n📊 权重统计信息:")
print(f"   - 均值: {original_weight.mean().item():.6f}")
print(f"   - 标准差: {original_weight.std().item():.6f}")
print(f"   - 最小值: {original_weight.min().item():.6f}")
print(f"   - 最大值: {original_weight.max().item():.6f}")

# 3. 创建额外通道的零填充
print(f"\n{'='*80}")
print("步骤 3: 创建新增通道（零初始化）")
print(f"{'='*80}")

# 额外通道的形状: [out_channels, additional_channels, kernel_h, kernel_w]
expansion_shape = (
    original_weight.shape[0],  # 输出通道保持不变 (320)
    ADDITIONAL_CHANNELS,        # 新增的输入通道 (4)
    original_weight.shape[2],  # 卷积核高度 (3)
    original_weight.shape[3]   # 卷积核宽度 (3)
)

zero_channels = torch.zeros(expansion_shape, dtype=original_weight.dtype)
print(f"✅ 零填充张量形状: {zero_channels.shape}")
print(f"   - 这将为{ADDITIONAL_CHANNELS}个新通道创建权重")
print(f"   - 初始化方式: 全零（保证新通道初始时不影响输出）")

# 4. 拼接权重
print(f"\n{'='*80}")
print("步骤 4: 拼接原始权重和新通道")
print(f"{'='*80}")

new_weight = torch.cat((original_weight, zero_channels), dim=1)
print(f"✅ 拼接完成")
print(f"   原始形状: {original_weight.shape}")
print(f"   新增形状: {zero_channels.shape}")
print(f"   最终形状: {new_weight.shape}")
print(f"   通道数变化: {original_weight.shape[1]} → {new_weight.shape[1]}")

# 验证拼接结果
print(f"\n📊 新权重统计信息:")
print(f"   - 均值: {new_weight.mean().item():.6f}")
print(f"   - 标准差: {new_weight.std().item():.6f}")
print(f"   - 最小值: {new_weight.min().item():.6f}")
print(f"   - 最大值: {new_weight.max().item():.6f}")

# 验证前9个通道没有改变
if torch.equal(new_weight[:, :ORIGINAL_CHANNELS, :, :], original_weight):
    print(f"✅ 验证通过: 原始{ORIGINAL_CHANNELS}个通道的权重保持不变")
else:
    print(f"❌ 警告: 原始通道的权重发生了变化！")

# 验证新通道全为零
if torch.all(new_weight[:, ORIGINAL_CHANNELS:, :, :] == 0):
    print(f"✅ 验证通过: 新增{ADDITIONAL_CHANNELS}个通道的权重均为零")
else:
    print(f"❌ 警告: 新通道的权重不全为零！")

# 5. 更新checkpoint
print(f"\n{'='*80}")
print("步骤 5: 更新checkpoint")
print(f"{'='*80}")

ckpt_file['state_dict'][TARGET_KEY] = new_weight
print(f"✅ 权重已更新")

# 计算参数变化
original_params = original_weight.numel()
new_params = new_weight.numel()
param_increase = new_params - original_params

print(f"\n📊 参数统计:")
print(f"   原始参数数: {original_params:,}")
print(f"   新参数数: {new_params:,}")
print(f"   增加参数: {param_increase:,} (+{(param_increase/original_params)*100:.2f}%)")

# 6. 保存新checkpoint
print(f"\n{'='*80}")
print("步骤 6: 保存扩展后的checkpoint")
print(f"{'='*80}")

try:
    torch.save(ckpt_file, OUTPUT_CHECKPOINT)
    file_size = os.path.getsize(OUTPUT_CHECKPOINT) / (1024**3)  # GB
    print(f"✅ 保存成功: {OUTPUT_CHECKPOINT}")
    print(f"   文件大小: {file_size:.2f} GB")
except Exception as e:
    print(f"❌ 保存失败: {e}")
    exit(1)

# 7. 验证保存的文件
print(f"\n{'='*80}")
print("步骤 7: 验证保存的checkpoint")
print(f"{'='*80}")

try:
    verify_ckpt = torch.load(OUTPUT_CHECKPOINT, map_location='cpu')
    verify_weight = verify_ckpt['state_dict'][TARGET_KEY]
    print(f"✅ 文件验证成功")
    print(f"   验证权重形状: {verify_weight.shape}")
    
    if verify_weight.shape == new_weight.shape:
        print(f"✅ 形状匹配: {verify_weight.shape}")
    else:
        print(f"❌ 形状不匹配!")
        print(f"   期望: {new_weight.shape}")
        print(f"   实际: {verify_weight.shape}")
except Exception as e:
    print(f"❌ 验证失败: {e}")

# 8. 生成修改报告
print(f"\n{'='*80}")
print("📝 修改报告")
print(f"{'='*80}")

report = {
    "modification_info": {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "expand_channels_9_to_13.py",
        "operation": "Channel Expansion"
    },
    "files": {
        "input_checkpoint": INPUT_CHECKPOINT,
        "output_checkpoint": OUTPUT_CHECKPOINT
    },
    "modification_details": {
        "target_layer": TARGET_KEY,
        "original_shape": list(original_weight.shape),
        "new_shape": list(new_weight.shape),
        "channel_change": {
            "original": ORIGINAL_CHANNELS,
            "target": TARGET_CHANNELS,
            "added": ADDITIONAL_CHANNELS
        },
        "parameter_change": {
            "original": original_params,
            "new": new_params,
            "increase": param_increase,
            "increase_percentage": round((param_increase/original_params)*100, 4)
        },
        "initialization": "zeros (zero padding for new channels)"
    }
}

report_file = "channel_expansion_report.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ 修改报告已保存: {report_file}")

# 9. 下一步操作提示
print(f"\n{'='*80}")
print("✅ 扩展完成！")
print(f"{'='*80}")
print(f"\n📋 下一步操作:")
print(f"   1. 修改配置文件 configs/train_sft.yaml")
print(f"      第73行: in_channels: 9  →  in_channels: {TARGET_CHANNELS}")
print(f"")
print(f"   2. 使用新的checkpoint进行训练:")
print(f"      方法A: 重命名文件")
print(f"         mv {OUTPUT_CHECKPOINT} last.ckpt")
print(f"")
print(f"      方法B: 修改配置中的 ref_ckpt_path")
print(f"         ref_ckpt_path: ./{OUTPUT_CHECKPOINT}")
print(f"")
print(f"   3. 验证配置是否正确:")
print(f"      python scripts/verify_channel_config.py")
print(f"")
print(f"💡 提示:")
print(f"   - 新增的{ADDITIONAL_CHANNELS}个通道初始权重为0")
print(f"   - 这确保了在训练初期不会破坏原有的预训练权重")
print(f"   - 训练过程中，这{ADDITIONAL_CHANNELS}个通道的权重会逐渐学习")
print(f"")
print(f"{'='*80}")

