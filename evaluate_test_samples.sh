#!/bin/bash

# ========================================
# 评测脚本 - 用于评测训练过程中生成的 test_samples
# ========================================

# 设置项目根目录到 PYTHONPATH
PROJECT_ROOT="/data5/shuangjun.du/work/REFace"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ========================================
# 配置参数
# ========================================

# 要评测的样本目录（可以修改）
TEST_SAMPLES_DIR="/data5/shuangjun.du/datasets/benchmark/result/SimSwap/FFHQ"

# 是否保存FID统计信息（设置为"--save-stats"启用，留空""禁用）
# SAVE_STATS="--save-stats"  # 启用保存统计信息（首次运行或强制更新时使用）
SAVE_STATS=""  # 禁用（使用已缓存的统计信息）

# GPU 设备
DEVICE=0

# 是否限制样本数量（留空则评测全部）
MAX_SAMPLES=1000 # 例如设置为 100 则只评测前 100 个样本

# 数据集路径（根据你的数据集选择 CelebA 或 FFHQ）
DATASET_TYPE="ffhq"  # "celeba" 或 "ffhq"

# ========================================
# 数据集路径配置
# ========================================

if [ "$DATASET_TYPE" = "celeba" ]; then
    echo "使用 CelebA 数据集配置"
    SOURCE_PATH="dataset/FaceData/CelebAMask-HQ/Val_target"
    TARGET_PATH="dataset/FaceData/CelebAMask-HQ/Val"
    SOURCE_MASK_PATH="dataset/FaceData/CelebAMask-HQ/target_mask"
    TARGET_MASK_PATH="dataset/FaceData/CelebAMask-HQ/src_mask"
    DATASET_PATH="dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"
elif [ "$DATASET_TYPE" = "ffhq" ]; then
    echo "使用 FFHQ 数据集配置"
    SOURCE_PATH="dataset/FaceData/FFHQ/Val"
    TARGET_PATH="dataset/FaceData/FFHQ/Val_target"
    SOURCE_MASK_PATH="dataset/FaceData/FFHQ/src_mask"
    TARGET_MASK_PATH="dataset/FaceData/FFHQ/target_mask"
    DATASET_PATH="dataset/FaceData/FFHQ/images512"
else
    echo "错误: 未知的数据集类型 $DATASET_TYPE"
    exit 1
fi

# ========================================
# 检查测试样本目录是否存在
# ========================================

if [ ! -d "$TEST_SAMPLES_DIR" ]; then
    echo "错误: 测试样本目录不存在: $TEST_SAMPLES_DIR"
    exit 1
fi

# 统计样本数量（支持png和jpg）
SAMPLE_COUNT=$(ls -1 "$TEST_SAMPLES_DIR"/*.{png,jpg,jpeg} 2>/dev/null | wc -l)
if [ "$SAMPLE_COUNT" -eq 0 ]; then
    echo "错误: 测试样本目录中没有图像文件 (png/jpg): $TEST_SAMPLES_DIR"
    exit 1
fi

echo "找到 $SAMPLE_COUNT 个测试样本"

# ========================================
# 创建输出目录和文件
# ========================================

# 输出目录（与 test_samples 同级的 metrics 目录）
METRICS_DIR=$(dirname "$TEST_SAMPLES_DIR")/metrics_$(basename "$TEST_SAMPLES_DIR")
mkdir -p "$METRICS_DIR"

# 输出文件名（带时间戳）
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${METRICS_DIR}/evaluation_${CURRENT_TIME}.txt"

echo "评测结果将保存到: $OUTPUT_FILE"
echo "========================================" | tee "$OUTPUT_FILE"
echo "评测测试样本" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "样本目录: $TEST_SAMPLES_DIR" | tee -a "$OUTPUT_FILE"
echo "样本数量: $SAMPLE_COUNT" | tee -a "$OUTPUT_FILE"
echo "数据集类型: $DATASET_TYPE" | tee -a "$OUTPUT_FILE"
echo "评测时间: $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ========================================
# 1. FID Score
# ========================================

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "1. 计算 FID Score (与真实数据集比较)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if [ -n "$MAX_SAMPLES" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/fid/fid_score.py --device cuda \
        --max-samples ${MAX_SAMPLES} \
        ${SAVE_STATS} \
        "${DATASET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
else
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/fid/fid_score.py --device cuda \
        ${SAVE_STATS} \
        "${DATASET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ========================================
# 2. ID Similarity (Arcface)
# ========================================

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "2. 计算 ID 相似度 (Arcface - 与源图像比较)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

ID_OUTPUT_FILE="${METRICS_DIR}/id_retrieval_${CURRENT_TIME}.txt"

if [ -n "$MAX_SAMPLES" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        --max-samples ${MAX_SAMPLES} \
        "${SOURCE_PATH}" \
        "${TEST_SAMPLES_DIR}" \
        "${SOURCE_MASK_PATH}" \
        "${TARGET_MASK_PATH}" \
        --dataset "${DATASET_TYPE}" \
        --print_sim True \
        --arcface True \
        --output "${ID_OUTPUT_FILE}" 2>&1 | tee -a "$OUTPUT_FILE"
else
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${SOURCE_PATH}" \
        "${TEST_SAMPLES_DIR}" \
        "${SOURCE_MASK_PATH}" \
        "${TARGET_MASK_PATH}" \
        --dataset "${DATASET_TYPE}" \
        --print_sim True \
        --arcface True \
        --output "${ID_OUTPUT_FILE}" 2>&1 | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ========================================
# 3. Pose Distance
# ========================================

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "3. 计算 Pose Distance (与目标图像比较)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if [ -n "$MAX_SAMPLES" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/Pose/pose_compare.py --device cuda \
        --max-samples ${MAX_SAMPLES} \
        "${TARGET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
else
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/Pose/pose_compare.py --device cuda \
        "${TARGET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ========================================
# 4. Expression Distance
# ========================================

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "4. 计算 Expression Distance (与目标图像比较)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if [ -n "$MAX_SAMPLES" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        --max-samples ${MAX_SAMPLES} \
        "${TARGET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
else
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        "${TARGET_PATH}" \
        "${TEST_SAMPLES_DIR}" 2>&1 | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ========================================
# 完成
# ========================================

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "评测完成！" | tee -a "$OUTPUT_FILE"
echo "结果保存在: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

# 显示结果摘要
echo ""
echo "========================================"
echo "评测结果摘要"
echo "========================================"
grep -E "(FID:|mean:|Top-1:|Top-5:|mean_rotation_distance|mean_distance)" "$OUTPUT_FILE"

