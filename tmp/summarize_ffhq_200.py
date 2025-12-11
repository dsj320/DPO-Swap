#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 FFHQ 200 张图片的评估指标
"""

import re
import numpy as np

# 读取 FID Score
with open('/data5/shuangjun.du/work/REFace/tmp/fid_score_ffhq_200.txt', 'r') as f:
    fid_content = f.read()
    fid_match = re.search(r'FID Score:\s*([\d.]+)', fid_content)
    fid_score = float(fid_match.group(1)) if fid_match else None

# 读取 ID Retrieval
with open('/data5/shuangjun.du/work/REFace/tmp/id_retrieval_ffhq_200_without_mask.txt', 'r') as f:
    id_content = f.read()
    
    # Top-1, Top-5 accuracy
    top1_match = re.search(r'Top-1 accuracy:\s*([\d.]+)%', id_content)
    top5_match = re.search(r'Top-5 accuracy:\s*([\d.]+)%', id_content)
    mean_id_match = re.search(r'Mean ID feat:\s*([\d.]+)', id_content)
    
    top1_acc = float(top1_match.group(1)) if top1_match else None
    top5_acc = float(top5_match.group(1)) if top5_match else None
    mean_id_feat = float(mean_id_match.group(1)) if mean_id_match else None
    
    # 提取所有相似度值
    similarities = []
    for line in id_content.split('\n'):
        if ':' in line and line.strip()[0].isdigit():
            try:
                sim_value = float(line.split(':')[1].strip())
                similarities.append(sim_value)
            except:
                pass

# 读取 Expression
with open('/data5/shuangjun.du/work/REFace/tmp/expression_compare_ffhq_200.txt', 'r') as f:
    expr_content = f.read()
    expr_match = re.search(r'Expression_value:\s*([\d.]+)', expr_content)
    expression_value = float(expr_match.group(1)) if expr_match else None

# 读取 Pose
with open('/data5/shuangjun.du/work/REFace/tmp/pose_compare_ffhq_200.txt', 'r') as f:
    pose_content = f.read()
    pose_match = re.search(r'Pose_value:\s*([\d.]+)', pose_content)
    pose_value = float(pose_match.group(1)) if pose_match else None

# 计算 ID 相似度统计
if similarities:
    similarities = np.array(similarities)
    id_mean = np.mean(similarities)
    id_std = np.std(similarities)
    id_min = np.min(similarities)
    id_max = np.max(similarities)
    id_median = np.median(similarities)
    id_q25 = np.percentile(similarities, 25)
    id_q75 = np.percentile(similarities, 75)
else:
    id_mean = id_std = id_min = id_max = id_median = id_q25 = id_q75 = None

# 生成汇总报告
print("=" * 80)
print("FFHQ 200 张图片评估指标汇总")
print("=" * 80)
print()

print("📊 1. FID Score (Fréchet Inception Distance)")
print("-" * 80)
if fid_score is not None:
    print(f"   FID Score: {fid_score:.6f}")
    print(f"   说明: 值越小越好，< 20 表示质量较好")
else:
    print("   ❌ 未找到 FID Score")
print()

print("🆔 2. ID Retrieval (身份检索)")
print("-" * 80)
if top1_acc is not None:
    print(f"   Top-1 Accuracy: {top1_acc:.2f}%")
if top5_acc is not None:
    print(f"   Top-5 Accuracy: {top5_acc:.2f}%")
if mean_id_feat is not None:
    print(f"   Mean ID Feature: {mean_id_feat:.2f}")
print()
if similarities is not None and len(similarities) > 0:
    print("   ID 相似度统计 (200 个样本):")
    print(f"   - 平均值 (Mean):     {id_mean:.6f}")
    print(f"   - 标准差 (Std):      {id_std:.6f}")
    print(f"   - 最小值 (Min):      {id_min:.6f}")
    print(f"   - 最大值 (Max):      {id_max:.6f}")
    print(f"   - 中位数 (Median):   {id_median:.6f}")
    print(f"   - 25% 分位数 (Q25):  {id_q25:.6f}")
    print(f"   - 75% 分位数 (Q75):  {id_q75:.6f}")
    print(f"   说明: 相似度越高越好，通常 > 0.5 表示身份保持较好")
else:
    print("   ❌ 未找到相似度数据")
print()

print("😊 3. Expression (表情保持)")
print("-" * 80)
if expression_value is not None:
    print(f"   Expression Value: {expression_value:.6f}")
    print(f"   说明: 值越小越好，表示生成图像与目标图像的表情更相似")
else:
    print("   ❌ 未找到 Expression 值")
print()

print("📐 4. Pose (姿态保持)")
print("-" * 80)
if pose_value is not None:
    print(f"   Pose Value: {pose_value:.6f}")
    print(f"   说明: 值越小越好，表示生成图像与目标图像的姿态更相似")
else:
    print("   ❌ 未找到 Pose 值")
print()

print("=" * 80)
print("📈 综合评估")
print("=" * 80)
print()

# 评估等级
if fid_score is not None:
    if fid_score < 10:
        fid_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif fid_score < 20:
        fid_grade = "良好 ⭐⭐⭐⭐"
    elif fid_score < 30:
        fid_grade = "中等 ⭐⭐⭐"
    else:
        fid_grade = "需改进 ⭐⭐"
    print(f"FID Score: {fid_score:.6f} - {fid_grade}")

if top1_acc is not None:
    if top1_acc >= 95:
        id_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif top1_acc >= 90:
        id_grade = "良好 ⭐⭐⭐⭐"
    elif top1_acc >= 80:
        id_grade = "中等 ⭐⭐⭐"
    else:
        id_grade = "需改进 ⭐⭐"
    print(f"ID Top-1 Accuracy: {top1_acc:.2f}% - {id_grade}")

if id_mean is not None:
    if id_mean >= 0.6:
        sim_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif id_mean >= 0.5:
        sim_grade = "良好 ⭐⭐⭐⭐"
    elif id_mean >= 0.4:
        sim_grade = "中等 ⭐⭐⭐"
    else:
        sim_grade = "需改进 ⭐⭐"
    print(f"ID 平均相似度: {id_mean:.6f} - {sim_grade}")

if expression_value is not None:
    if expression_value < 1.0:
        expr_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif expression_value < 1.5:
        expr_grade = "良好 ⭐⭐⭐⭐"
    elif expression_value < 2.0:
        expr_grade = "中等 ⭐⭐⭐"
    else:
        expr_grade = "需改进 ⭐⭐"
    print(f"Expression: {expression_value:.6f} - {expr_grade}")

if pose_value is not None:
    if pose_value < 3.0:
        pose_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif pose_value < 4.0:
        pose_grade = "良好 ⭐⭐⭐⭐"
    elif pose_value < 5.0:
        pose_grade = "中等 ⭐⭐⭐"
    else:
        pose_grade = "需改进 ⭐⭐"
    print(f"Pose: {pose_value:.6f} - {pose_grade}")

print()
print("=" * 80)


