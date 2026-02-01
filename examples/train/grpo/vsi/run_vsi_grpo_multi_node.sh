#!/bin/bash
# VSI-Bench GRPO Multi-Node Training Script for Qwen3-VL
#
# 4 nodes x 8 GPUs = 32 GPUs total
# Colocate mode (training and inference on same GPUs)
# Full parameter training (no LoRA), using FSDP2 (no DeepSpeed)
#
# Usage:
#   # On each node, set NODE_RANK and MASTER_ADDR:
#   NODE_RANK=0 MASTER_ADDR=<master_ip> bash run_vsi_grpo_multi_node.sh  # Node 0
#   NODE_RANK=1 MASTER_ADDR=<master_ip> bash run_vsi_grpo_multi_node.sh  # Node 1
#   NODE_RANK=2 MASTER_ADDR=<master_ip> bash run_vsi_grpo_multi_node.sh  # Node 2
#   NODE_RANK=3 MASTER_ADDR=<master_ip> bash run_vsi_grpo_multi_node.sh  # Node 3
#
# Note: All nodes must have the same configuration except NODE_RANK

set -e

# ============================================================
# Multi-Node Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=4
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=8

echo "============================================================"
echo "Multi-Node Configuration:"
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "============================================================"

# ============================================================
# Configuration - Modify these paths for your environment
# ============================================================
MODEL="/upfs/models/Qwen/Qwen3-VL-8B-Instruct"
DATASET_PATH="/upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json"
OUTPUT_DIR="output/vsi_grpo_multi_node"

# CoT System Prompt
SYSTEM_PROMPT="examples/train/grpo/vsi/vsi_cot_prompt.txt"

# ============================================================
# Training Hyperparameters (from run_vsi_grpo.sh)
# ============================================================
LEARNING_RATE=1e-6
BETA=0.001
NUM_GENERATIONS=8
MAX_COMPLETION_LENGTH=512
BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
NUM_EPOCHS=1

# Image processing - reduce MAX_PIXELS to save memory with 32 frames
export MAX_PIXELS=200704  # ~256*28*28

# ============================================================
# Training with Colocate Mode + FSDP2 (Full Parameter, No DeepSpeed)
# ============================================================
MAX_PIXELS=${MAX_PIXELS} \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --system ${SYSTEM_PROMPT} \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
    --dataset ${DATASET_PATH} \
    --load_from_cache_file true \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations ${NUM_GENERATIONS} \
    --temperature 1.0 \
    --sleep_level 1 \
    --log_completions true \
    --num_iterations 1 \
    --beta ${BETA} \
    --overlong_filter true \
    --max_grad_norm 1.0

echo "Training complete on node ${NODE_RANK}!"
