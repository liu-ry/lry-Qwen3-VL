#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "单 GPU 训练（用于调试）"
echo "=========================================="

# 选择使用的 GPU
GPU_ID=${GPU_ID:-0}

echo "[INFO] 使用 GPU: $GPU_ID"

# Model configuration
llm=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
echo "[INFO] 模型: $llm"

# Training hyperparameters（减小以节省内存）
lr=1e-5
batch_size=1
grad_accum_steps=4

# Dataset configuration
datasets=${DATASETS:-"first_test"}
echo "[INFO] 数据集: $datasets"

# Output configuration
run_name="qwen3vl_single_gpu"
output_dir="./output_single_gpu"

echo "[INFO] 输出目录: $output_dir"
mkdir -p "$output_dir"

# Training entry point
entry_file=$PROJECT_DIR/qwenvl/train/train_qwen.py

if [ ! -f "$entry_file" ]; then
    echo "[ERROR] 训练脚本不存在: $entry_file"
    exit 1
fi

# Training arguments - 不使用 DeepSpeed，用标准训练
args="
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
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name}"

echo ""
echo "=========================================="
echo "启动单 GPU 训练"
echo "=========================================="
echo ""

# 单 GPU 训练，不使用 torchrun
CUDA_VISIBLE_DEVICES=$GPU_ID \
python ${entry_file} ${args}

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
