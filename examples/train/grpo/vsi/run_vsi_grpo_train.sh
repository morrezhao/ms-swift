#!/bin/bash
# VSI-Bench GRPO - Multi-Node Training Script (2 nodes)
#
# Full parameter training (no LoRA), using FSDP2 (no DeepSpeed)
#
# Usage:
#   # Node 1: Training master (rank 0)
#   NODE_RANK=0 MASTER_ADDR=<master_ip> bash run_vsi_grpo_train.sh
#
#   # Node 2: Training worker (rank 1)
#   NODE_RANK=1 MASTER_ADDR=<master_ip> bash run_vsi_grpo_train.sh

set -e

# ============================================================
# Multi-Node Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=8
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=etho

echo "============================================================"
echo "Multi-Node Training Configuration:"
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
DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json"  # 32-frame version
VAL_DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/test/vsi_test.json"  # VSI-Bench test set
OUTPUT_DIR="output/vsi_grpo_multi_node"

# CoT System Prompt
SYSTEM_PROMPT="examples/train/grpo/vsi/vsi_cot_prompt.txt"

# ============================================================
# vLLM Server Configuration (2 rollout nodes)
# Replace with actual rollout node IPs
# ============================================================
VLLM_SERVER_HOST_1=${VLLM_SERVER_HOST_1:-"<rollout_node1_ip>"}
VLLM_SERVER_HOST_2=${VLLM_SERVER_HOST_2:-"<rollout_node2_ip>"}
VLLM_SERVER_PORT_1=${VLLM_SERVER_PORT_1:-8000}
VLLM_SERVER_PORT_2=${VLLM_SERVER_PORT_2:-8000}


NUM_FRAMES=${NUM_FRAMES:-32}

# Validate NUM_FRAMES
if [[ ! "$NUM_FRAMES" =~ ^(1|2|4|8|16|32)$ ]]; then
    echo "Error: NUM_FRAMES must be 1, 2, 4, 8, 16, or 32. Got: $NUM_FRAMES"
    exit 1
fi

echo "Using NUM_FRAMES=$NUM_FRAMES"

# Determine dataset paths based on NUM_FRAMES
if [ "$NUM_FRAMES" -eq 32 ]; then
    DATASET_PATH="$DATASET_PATH_BASE"
    VAL_DATASET_PATH="$VAL_DATASET_PATH_BASE"
else
    # Generate resampled dataset path
    DATASET_DIR=$(dirname "$DATASET_PATH_BASE")
    DATASET_NAME=$(basename "$DATASET_PATH_BASE" .json)
    DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}_${NUM_FRAMES}f.json"

    VAL_DATASET_DIR=$(dirname "$VAL_DATASET_PATH_BASE")
    VAL_DATASET_NAME=$(basename "$VAL_DATASET_PATH_BASE" .json)
    VAL_DATASET_PATH="${VAL_DATASET_DIR}/${VAL_DATASET_NAME}_${NUM_FRAMES}f.json"

    # Create resampled datasets if they don't exist
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Creating ${NUM_FRAMES}-frame training dataset..."
        python "$SCRIPT_DIR/resample_frames.py" \
            --input "$DATASET_PATH_BASE" \
            --output "$DATASET_PATH" \
            --num_frames "$NUM_FRAMES"
    fi

    if [ ! -f "$VAL_DATASET_PATH" ]; then
        echo "Creating ${NUM_FRAMES}-frame validation dataset..."
        python "$SCRIPT_DIR/resample_frames.py" \
            --input "$VAL_DATASET_PATH_BASE" \
            --output "$VAL_DATASET_PATH" \
            --num_frames "$NUM_FRAMES"
    fi
fi

echo "Training dataset: $DATASET_PATH"
echo "Validation dataset: $VAL_DATASET_PATH"
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
# GRPO Training with FSDP2 (Full Parameter, No DeepSpeed)
# ============================================================
MAX_PIXELS=${MAX_PIXELS} \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --system ${SYSTEM_PROMPT} \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host ${VLLM_SERVER_HOST_1} ${VLLM_SERVER_HOST_2} \
    --vllm_server_port ${VLLM_SERVER_PORT_1} ${VLLM_SERVER_PORT_2} \
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
    --log_completions true \
    --num_iterations 1 \
    --beta ${BETA} \
    --overlong_filter true \
    --max_grad_norm 1.0

echo "Training complete on node ${NODE_RANK}!"
