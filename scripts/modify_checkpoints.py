import torch
import json
from datetime import datetime

print("=== Stable Diffusion模型通道扩展修改脚本 ===")
print("功能：将输入通道从4通道扩展到9通道")

# Load pretrained model checkpoint
pretrained_model_path = 'pretrained_models/sd-v1-4.ckpt'
print(f"\n1. 加载预训练模型: {pretrained_model_path}")
ckpt_file = torch.load(pretrained_model_path, map_location='cpu')
print("   模型加载完成")

# Get original weight tensor from the first input block
target_key = 'model.diffusion_model.input_blocks.0.0.weight'
original_weight = ckpt_file['state_dict'][target_key]
print(f"\n2. 获取原始权重张量: {target_key}")
print(f"   原始权重形状: {original_weight.shape}")
print(f"   原始权重的统计信息:")
print(f"   - 均值: {original_weight.mean().item():.6f}")
print(f"   - 标准差: {original_weight.std().item():.6f}")
print(f"   - 最小值: {original_weight.min().item():.6f}")
print(f"   - 最大值: {original_weight.max().item():.6f}")

# Create zero padding tensor for additional channels
print(f"\n3. 创建额外通道的零填充张量")
channel_expansion_size = (320, 5, 3, 3)  # 5 additional channels
zero_data = torch.zeros(channel_expansion_size)
print(f"   零填充张量形状: {zero_data.shape}")
print(f"   这是为了增加5个输入通道（从4通道扩展到9通道）")

# Concatenate original weights with zero padding
print(f"\n4. 拼接原始权重和零填充:")
print(f"   维度1上拼接（通道维度）: dim=1")
new_weight = torch.cat((original_weight, zero_data), dim=1)
print(f"   新权重形状: {new_weight.shape}")
print(f"   通道数变化: {original_weight.shape[1]} -> {new_weight.shape[1]}")

# Verify the concatenation results
print(f"\n5. 验证拼接结果:")
print(f"   新权重的统计信息:")
print(f"   - 均值: {new_weight.mean().item():.6f}")
print(f"   - 标准差: {new_weight.std().item():.6f}")
print(f"   - 最小值: {new_weight.min().item():.6f}")
print(f"   - 最大值: {new_weight.max().item():.6f}")

# Update checkpoint with modified weight
print(f"\n6. 更新检查点文件:")
ckpt_file['state_dict'][target_key] = new_weight
print(f"   权重已更新到检查点中")

# Save modified checkpoint
output_path = "pretrained_models/sd-v1-4-modified-9channel.ckpt"
print(f"\n7. 保存修改后的模型: {output_path}")
torch.save(ckpt_file, output_path)
print("   保存完成!")

print(f"\n=== 修改总结 ===")
print(f"- 输入通道数: {original_weight.shape[1]} -> {new_weight.shape[1]}")
print(f"- 新增了 {new_weight.shape[1] - original_weight.shape[1]} 个零值通道")
print(f"- 输出文件: {output_path}")
print("这是为了适配9通道输入的新模型架构")

# ===== 保存完整模型结构信息 =====
print(f"\n8. 分析并保存完整模型结构:")

# Function to analyze all layers in state_dict
def analyze_model_structure(state_dict, model_name):
    print(f"   正在分析{model_name}的完整结构...")
    
    total_parameters = 0
    layer_info = {}
    key_categories = {
        "input_blocks": [],
        "middle_block": [],
        "output_blocks": [],
        "pos_embed": [],
        "time_embed": [],
        "out": [],
        "others": []
    }
    
    # Analyze each parameter in state_dict
    for key, tensor in state_dict.items():
        param_count = tensor.numel()
        total_parameters += param_count
        
        layer_info[key] = {
            "shape": tensor.shape.tolist() if hasattr(tensor.shape, 'tolist') else list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "parameter_count": param_count,
            "statistics": {
                "mean": float(tensor.float().mean().item()),
                "std": float(tensor.float().std().item()),
                "min": float(tensor.float().min().item()),
                "max": float(tensor.float().max().item())
            }
        }
        
        # Categorize layers
        if "input_blocks" in key:
            key_categories["input_blocks"].append(key)
        elif "middle_block" in key:
            key_categories["middle_block"].append(key)
        elif "output_blocks" in key:
            key_categories["output_blocks"].append(key)
        elif "pos_embed" in key:
            key_categories["pos_embed"].append(key)
        elif "time_embed" in key:
            key_categories["time_embed"].append(key)
        elif "out" in key:
            key_categories["out"].append(key)
        else:
            key_categories["others"].append(key)
    
    # Calculate summary statistics
    summary_stats = {
        "total_layers": len(state_dict),
        "total_parameters": total_parameters,
        "key_distribution": {k: len(v) for k, v in key_categories.items()},
        "layer_categories": key_categories
    }
    
    print(f"   分析完成: {len(state_dict)}层, {total_parameters:,}个参数")
    return layer_info, summary_stats

# Analyze original model structure
print(f"   正在重新加载原始模型进行分析...")
original_ckpt = torch.load(pretrained_model_path, map_location='cpu')
original_layer_info, original_summary = analyze_model_structure(original_ckpt['state_dict'], "原始模型")

# Prepare complete original model structure
original_structure = {
    "model_info": {
        "file_path": pretrained_model_path,
        "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Stable Diffusion",
        "version": "1.4",
        "description": "原始的4通道输入SD模型"
    },
    "summary": original_summary,
    "all_layers": original_layer_info,
    "modified_layer": {
        "key": target_key,
        "original_shape": list(original_weight.shape),
        "description": "将被修改的层（从4通道扩展到9通道）",
        "parameter_count": original_weight.numel()
    }
}

# Analyze modified model structure
print(f"   正在重新加载修改后的模型进行分析...")
modified_ckpt = torch.load(output_path, map_location='cpu')
modified_layer_info, modified_summary = analyze_model_structure(modified_ckpt['state_dict'], "修改后模型")

# Calculate modification impact
parameter_increase = modified_summary["total_parameters"] - original_summary["total_parameters"]

# Prepare complete modified model structure
modified_structure = {
    "model_info": {
        "file_path": output_path,
        "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Stable Diffusion (Modified for REFace)",
        "version": "1.4-9channel",
        "description": "修改后的9通道输入SD模型，适配REFace面部替换任务",
        "modification_type": "通道扩展"
    },
    "summary": modified_summary,
    "all_layers": modified_layer_info,
    "modified_layer": {
        "key": target_key,
        "original_shape": list(original_weight.shape),
        "new_shape": list(new_weight.shape),
        "description": "扩展至9通道的输入卷积层",
        "parameter_count": new_weight.numel(),
        "parameter_increase": new_weight.numel() - original_weight.numel()
    },
    "modification_details": {
        "channel_expansion": {
            "original_channels": original_weight.shape[1],
            "new_channels": new_weight.shape[1],
            "added_channels": new_weight.shape[1] - original_weight.shape[1],
            "expansion_tensor_shape": channel_expansion_size
        },
        "total_model_changes": {
            "total_parameter_increase": parameter_increase,
            "percentage_increase": round((parameter_increase / original_summary["total_parameters"]) * 100, 4)
        },
        "expansion_method": "零填充通道拼接",
        "affected_layer_position": "输入层第一部分"
    }
}

# Save structure information to files
original_file = "model_structure_original.json"
modified_file = "model_structure_modified.json"
comparison_file = "model_structure_comparison.json"

# Save original model structure
with open(original_file, 'w', encoding='utf-8') as f:
    json.dump(original_structure, f, indent=2, ensure_ascii=False)
print(f"   原始模型结构已保存到: {original_file}")

# Save modified model structure  
with open(modified_file, 'w', encoding='utf-8') as f:
    json.dump(modified_structure, f, indent=2, ensure_ascii=False)
print(f"   修改后模型结构已保存到: {modified_file}")

# Create comprehensive comparison structure
comparison_structure = {
    "comparison_info": {
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modification_type": "REFace - Complete Model Structure Analysis",
        "purpose": "完整分析Stable Diffusion模型的通道扩展修改",
        "analysis_scope": "完整模型结构对比"
    },
    "model_overview": {
        "original": {
            "model_type": "Stable Diffusion 1.4",
            "total_layers": original_summary["total_layers"],
            "total_parameters": original_summary["total_parameters"],
            "input_channels": "4 (RGB + condition)",
            "key_categories": original_summary["key_distribution"]
        },
        "modified": {
            "model_type": "Stable Diffusion 1.4-9channel",
            "total_layers": modified_summary["total_layers"],
            "total_parameters": modified_summary["total_parameters"],
            "input_channels": "9 (extended for REFace)",
            "key_categories": modified_summary["key_distribution"]
        }
    },
    "detailed_changes": {
        "layer_changes": {
            "modified_layer_key": target_key,
            "shape_change": f"{original_weight.shape} -> {new_weight.shape}",
            "parameter_change": {
                "original": original_weight.numel(),
                "modified": new_weight.numel(),
                "increase": new_weight.numel() - original_weight.numel(),
                "increase_percentage": round(((new_weight.numel() - original_weight.numel()) / original_weight.numel()) * 100, 2)
            }
        },
        "model_total_changes": {
            "total_parameter_change": parameter_increase,
            "total_increase_percentage": round((parameter_increase / original_summary["total_parameters"]) * 100, 4),
            "unchanged_layers": original_summary["total_layers"] - 1,
            "modified_layers": 1
        },
        "expansion_details": {
            "method": "零填充通道拼接",
            "original_input_channels": original_weight.shape[1],
            "new_input_channels": new_weight.shape[1],
            "added_channels": new_weight.shape[1] - original_weight.shape[1],
            "expansion_tensor_size": f"{channel_expansion_size} (zeros)"
        }
    },
    "layer_category_comparison": {
        "input_blocks": {
            "original_count": original_summary["key_distribution"]["input_blocks"],
            "modified_count": modified_summary["key_distribution"]["input_blocks"],
            "change": "unchanged (参数增加仅在第一层)"
        },
        "middle_block": {
            "original_count": original_summary["key_distribution"]["middle_block"],
            "modified_count": modified_summary["key_distribution"]["middle_block"],
            "change": "unchanged"
        },
        "output_blocks": {
            "original_count": original_summary["key_distribution"]["output_blocks"],
            "modified_count": modified_summary["key_distribution"]["output_blocks"],
            "change": "unchanged"
        }
    },
    "complete_structures": {
        "original_model": original_structure,
        "modified_model": modified_structure
    }
}

# Save comparison structure
with open(comparison_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_structure, f, indent=2, ensure_ascii=False)
print(f"   模型对比信息已保存到: {comparison_file}")

print(f"\n=== 文件输出清单 ===")
print(f"1. 修改后的模型: {output_path}")
print(f"2. 原始模型完整结构: {original_file} ({original_summary['total_layers']}层, {original_summary['total_parameters']:,}参数)")
print(f"3. 修改后模型完整结构: {modified_file} ({modified_summary['total_layers']}层, {modified_summary['total_parameters']:,}参数)")
print(f"4. 完整模型对比分析: {comparison_file}")
print(f"\n=== 修改影响总结 ===")
print(f"- 模型层数: {original_summary['total_layers']} 层保持不变")
print(f"- 总参数数: {original_summary['total_parameters']:,} -> {modified_summary['total_parameters']:,}")
print(f"- 参数增加: +{parameter_increase:,} ({round((parameter_increase / original_summary['total_parameters']) * 100, 4)}%)")
print(f"- 仅修改层数: 1层 (model.diffusion_model.input_blocks.0.0)")
print(f"\n所有完整结构信息已保存完成，包含模型中每一层的详细信息!")