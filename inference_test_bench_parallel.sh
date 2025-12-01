#!/bin/bash

# 并行推理脚本示例
# 用于加速大规模测试集的推理过程
# 使用方法：bash inference_test_bench_parallel.sh

# ============= 配置参数 =============
CONFIG="configs/train.yaml"
CKPT="last.ckpt"
DATASET="CelebA"  # 或 FFHQ, FF++
DATASET_DIR="dataset/FaceData/CelebAMask-HQ"
OUTDIR="results/CelebA/REFace/parallel_test"
DDIM_STEPS=50
SCALE=5
BATCH_SIZE=5

# 数据集大小（根据实际情况调整）
# CelebA test: 1000 samples
# FFHQ test: 1000 samples
TOTAL_SAMPLES=1000

# 并行进程数（根据可用GPU数量调整）
NUM_PROCESSES=4

# 每个进程处理的样本数
SAMPLES_PER_PROCESS=$((TOTAL_SAMPLES / NUM_PROCESSES))

# ============= 启动并行推理 =============
echo "========================================="
echo "🚀 Starting Parallel Inference"
echo "========================================="
echo "Dataset: $DATASET"
echo "Total samples: $TOTAL_SAMPLES"
echo "Processes: $NUM_PROCESSES"
echo "Samples per process: $SAMPLES_PER_PROCESS"
echo "========================================="
echo ""

# 启动多个进程
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    # 计算当前进程的起始和结束索引
    START_IDX=$((i * SAMPLES_PER_PROCESS))
    
    # 最后一个进程处理剩余所有样本
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        END_IDX=$TOTAL_SAMPLES
    else
        END_IDX=$(((i + 1) * SAMPLES_PER_PROCESS))
    fi
    
    # 根据进程数分配GPU
    GPU_ID=$i
    
    echo "🔹 Process $i: GPU $GPU_ID, samples [$START_IDX, $END_IDX)"
    
    # 启动后台进程
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/inference_test_bench.py \
        --config $CONFIG \
        --ckpt $CKPT \
        --dataset $DATASET \
        --dataset_dir $DATASET_DIR \
        --outdir $OUTDIR \
        --ddim_steps $DDIM_STEPS \
        --scale $SCALE \
        --n_samples $BATCH_SIZE \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --device_ID $GPU_ID \
        > logs/inference_process_${i}.log 2>&1 &
    
    # 保存进程ID
    PIDS[$i]=$!
    
    # 短暂延迟避免同时启动
    sleep 2
done

echo ""
echo "✅ All processes launched!"
echo "📊 Monitoring progress..."
echo ""

# 等待所有进程完成
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Process $i completed successfully"
    else
        echo "❌ Process $i failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "========================================="
echo "🎉 All inference processes completed!"
echo "📁 Results saved to: $OUTDIR"
echo "========================================="




