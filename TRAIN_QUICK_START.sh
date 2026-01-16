#!/usr/bin/env bash
# 快速训练命令参考

# ===== 单GPU训练 =====
# GPU_ID=6 bash qwen-vl-finetune/scripts/sft_single_gpu.sh

# ===== 多GPU训练 (推荐) =====
# CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash qwen-vl-finetune/scripts/sft_multi_gpu_no_deepspeed.sh

# ===== 多GPU + DeepSpeed (显存优化) =====
# CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash qwen-vl-finetune/scripts/sft_multi_gpu_with_deepspeed.sh

# ===== 自定义参数示例 =====
# CUDA_VISIBLE_DEVICES=2,3,4,5 NPROC_PER_NODE=4 bash qwen-vl-finetune/scripts/sft_multi_gpu_no_deepspeed.sh

# ===== 检查GPU状态 =====
# nvidia-smi

# ===== 查看训练日志 =====
# tail -f qwen-vl-finetune/wandb/run-*/logs/debug.log

echo "🚀 RTX 5090 多GPU训练命令参考已加载"
echo ""
echo "快速开始："
echo "  单GPU:   GPU_ID=0 bash qwen-vl-finetune/scripts/sft_single_gpu.sh"
echo "  多GPU:   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash qwen-vl-finetune/scripts/sft_multi_gpu_no_deepspeed.sh"
echo ""