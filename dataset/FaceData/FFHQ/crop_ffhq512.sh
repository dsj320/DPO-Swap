#!/bin/bash

# FFHQ 512 裁剪脚本
# 使用 faceswap_pipeline_5pts.py 进行人脸检测、对齐和裁剪
# 如果检测/裁剪失败，复制原图并resize到512x512

set -e  # 遇到错误立即退出

# ======================== 配置参数 ========================
# 源图像目录
SOURCE_DIR="/data5/shuangjun.du/work/REFace/dataset/FaceData/FFHQ/images512"

# 输出目录
PARAMS_PATH="${SOURCE_DIR}/params_simswap.json"
ALIGN_DIR="/data5/shuangjun.du/work/REFace/dataset/FaceData/FFHQ/images512_crop_align_512_simswap"

# Pipeline 脚本路径
PIPELINE_SCRIPT="/data5/shuangjun.du/FaceSwap/utils/crop/faceswap_pipeline_5pts.py"

# 参数配置
SIZE=512              # 裁剪后的尺寸
DET_THRESH=0.05       # 检测阈值（越小越宽松）
GPU_ID=3              # GPU ID

# ======================== 检查目录 ========================
echo "=========================================="
echo "FFHQ 512 人脸裁剪 Pipeline"
echo "=========================================="

if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "❌ 错误: Pipeline 脚本不存在: $PIPELINE_SCRIPT"
    exit 1
fi

echo "✓ 源目录: $SOURCE_DIR"
echo "✓ 输出对齐目录: $ALIGN_DIR"
echo "✓ 参数文件: $PARAMS_PATH"
echo "✓ 裁剪尺寸: ${SIZE}x${SIZE}"
echo ""

# 统计源目录中的图像数量
IMG_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
echo "📊 源目录图像数量: $IMG_COUNT"
echo ""

# ======================== 步骤1: Prepare (检测并保存参数) ========================
echo "=========================================="
echo "步骤 1/3: 检测人脸并保存参数"
echo "=========================================="
echo "运行命令:"
echo "python $PIPELINE_SCRIPT prepare \\"
echo "  --target_dir $SOURCE_DIR \\"
echo "  --params_path $PARAMS_PATH \\"
echo "  --size $SIZE \\"
echo "  --det_thresh $DET_THRESH \\"
echo "  --gpu_id $GPU_ID"
echo ""

python "$PIPELINE_SCRIPT" prepare \
  --target_dir "$SOURCE_DIR" \
  --params_path "$PARAMS_PATH" \
  --size $SIZE \
  --det_thresh $DET_THRESH \
  --gpu_id $GPU_ID

if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败: 人脸检测出错"
    exit 1
fi

echo ""
echo "✓ 步骤1完成: 参数已保存到 $PARAMS_PATH"
echo ""

# 检查参数文件
if [ ! -f "$PARAMS_PATH" ]; then
    echo "❌ 错误: 参数文件未生成: $PARAMS_PATH"
    exit 1
fi

# 统计检测到的人脸数量
DETECTED_COUNT=$(python3 -c "import json; data=json.load(open('$PARAMS_PATH')); print(len(data['targets']))")
echo "📊 检测到人脸: $DETECTED_COUNT / $IMG_COUNT"
echo ""

# ======================== 步骤2: Crop (裁剪对齐) ========================
echo "=========================================="
echo "步骤 2/3: 裁剪并对齐人脸"
echo "=========================================="
echo "运行命令:"
echo "python $PIPELINE_SCRIPT crop \\"
echo "  --target_dir $SOURCE_DIR \\"
echo "  --params_path $PARAMS_PATH \\"
echo "  --align_dir $ALIGN_DIR"
echo ""

python "$PIPELINE_SCRIPT" crop \
  --target_dir "$SOURCE_DIR" \
  --params_path "$PARAMS_PATH" \
  --align_dir "$ALIGN_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败: 裁剪对齐出错"
    exit 1
fi

echo ""
echo "✓ 步骤2完成: 裁剪后的图像已保存到 $ALIGN_DIR"
echo ""

# ======================== 步骤3: 处理失败的图像 (复制并resize) ========================
echo "=========================================="
echo "步骤 3/3: 处理未检测到的图像"
echo "=========================================="

# 创建临时Python脚本处理失败的图像
TEMP_SCRIPT="/tmp/process_failed_images_$$.py"
cat > "$TEMP_SCRIPT" << 'EOF'
import os
import json
import cv2
import sys
from pathlib import Path

def process_failed_images(source_dir, params_path, align_dir, target_size):
    """处理未检测到人脸的图像：复制并resize到目标尺寸"""
    
    # 读取参数文件
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    detected_files = set(params['targets'].keys())
    
    # 获取所有源文件
    source_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        source_files.extend(Path(source_dir).glob(ext))
    
    source_files = [f.name for f in source_files if not f.name.startswith('.')]
    
    # 找出未检测到的文件
    failed_files = [f for f in source_files if f not in detected_files]
    
    if not failed_files:
        print("✓ 所有图像都成功检测并裁剪")
        return 0
    
    print(f"⚠️  发现 {len(failed_files)} 个未检测到人脸的图像")
    print(f"   正在处理: 复制原图并resize到 {target_size}x{target_size}...\n")
    
    os.makedirs(align_dir, exist_ok=True)
    
    success_count = 0
    for idx, filename in enumerate(failed_files, 1):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(align_dir, filename)
        
        # 如果已经存在（可能是之前处理过的），跳过
        if os.path.exists(dst_path):
            success_count += 1
            continue
        
        try:
            # 读取图像
            img = cv2.imread(src_path)
            if img is None:
                print(f"  [{idx}/{len(failed_files)}] ❌ 无法读取: {filename}")
                continue
            
            # Resize到目标尺寸
            if img.shape[:2] != (target_size, target_size):
                img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            else:
                img_resized = img
            
            # 保存
            cv2.imwrite(dst_path, img_resized)
            success_count += 1
            
            if idx % 10 == 0:
                print(f"  已处理: {idx}/{len(failed_files)}")
                
        except Exception as e:
            print(f"  [{idx}/{len(failed_files)}] ❌ 处理失败: {filename}, 错误: {e}")
    
    print(f"\n✓ 成功处理 {success_count}/{len(failed_files)} 个失败图像")
    return success_count

if __name__ == '__main__':
    source_dir = sys.argv[1]
    params_path = sys.argv[2]
    align_dir = sys.argv[3]
    target_size = int(sys.argv[4])
    
    process_failed_images(source_dir, params_path, align_dir, target_size)
EOF

echo "运行命令:"
echo "python $TEMP_SCRIPT \\"
echo "  $SOURCE_DIR \\"
echo "  $PARAMS_PATH \\"
echo "  $ALIGN_DIR \\"
echo "  $SIZE"
echo ""

python3 "$TEMP_SCRIPT" "$SOURCE_DIR" "$PARAMS_PATH" "$ALIGN_DIR" $SIZE

# 清理临时脚本
rm -f "$TEMP_SCRIPT"

echo ""
echo "✓ 步骤3完成"
echo ""

# ======================== 完成统计 ========================
echo "=========================================="
echo "✓ 全部完成！"
echo "=========================================="

# 统计输出文件数量
if [ -d "$ALIGN_DIR" ]; then
    OUTPUT_COUNT=$(find "$ALIGN_DIR" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
    echo "📊 最终统计:"
    echo "  - 源图像数量: $IMG_COUNT"
    echo "  - 检测到人脸: $DETECTED_COUNT"
    echo "  - 输出文件数量: $OUTPUT_COUNT"
    
    if [ $OUTPUT_COUNT -eq $IMG_COUNT ]; then
        echo "  ✓ 所有图像都已处理完成！"
    else
        echo "  ⚠️  输出文件数量与源文件不匹配"
    fi
    
    echo ""
    echo "📁 结果路径:"
    echo "  - 参数文件: $PARAMS_PATH"
    echo "  - 对齐图像: $ALIGN_DIR"
else
    echo "⚠️  警告: 输出目录不存在"
fi

echo ""
echo "💡 使用提示:"
echo "  - 成功裁剪的图像: 使用人脸对齐算法，人脸居中"
echo "  - 失败的图像: 直接resize到${SIZE}x${SIZE}，保持原始内容"
echo "  - 如果只想重新裁剪（跳过检测），直接运行步骤2的命令"
echo "  - 参数文件保存了所有人脸的检测结果，可重复使用"
echo ""
