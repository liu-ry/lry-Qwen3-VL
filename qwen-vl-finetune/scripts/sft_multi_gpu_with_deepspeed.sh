#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "多 GPU 训练（使用 DeepSpeed Stage 1）"
echo "=========================================="

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

echo "[INFO] 分布式配置:"
echo "  MASTER_ADDR:     $MASTER_ADDR"
echo "  MASTER_PORT:     $MASTER_PORT"
echo "  NPROC_PER_NODE:  $NPROC_PER_NODE"

# DeepSpeed configuration
deepspeed=$PROJECT_DIR/scripts/zero1.json

echo "[INFO] DeepSpeed 配置: $deepspeed"
if [ ! -f "$deepspeed" ]; then
    echo "[ERROR] DeepSpeed 配置文件不存在: $deepspeed"
    exit 1
fi

# Model configuration
llm=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
echo "[INFO] 模型: $llm"

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=8

echo "[INFO] 训练超参数:"
echo "  学习率:          $lr"
echo "  批大小:          $batch_size"
echo "  梯度累积步数:    $grad_accum_steps"
echo "  有效批大小:      $((batch_size * grad_accum_steps * NPROC_PER_NODE))"

# Training entry point
entry_file=$PROJECT_DIR/qwenvl/train/train_qwen.py

if [ ! -f "$entry_file" ]; then
    echo "[ERROR] 训练脚本不存在: $entry_file"
    exit 1
fi

# Dataset configuration
datasets=${DATASETS:-"first_test"}
echo "[INFO] 数据集: $datasets"

# Output configuration
run_name="qwen3vl_deepspeed"
output_dir="./output_deepspeed"

echo "[INFO] 输出: $output_dir"
mkdir -p "$output_dir"

# Training arguments - 使用 DeepSpeed
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --ddp_backend gloo \
    --run_name ${run_name}"

echo ""
echo "=========================================="
echo "启动多 GPU 训练（DeepSpeed Stage 1）"
echo "=========================================="
echo ""

# RTX 5090 NCCL 和 CUDA 优化
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=0
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=32

echo "[INFO] 环境变量已设置：禁用梯度检查点、使用Gloo后端、NCCL优化配置"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练成功完成！"
else
    echo "✗ 训练失败，退出代码: $TRAIN_EXIT_CODE"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
