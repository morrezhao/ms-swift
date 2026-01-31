#!/bin/bash
# VSI-Bench GRPO Training Script for Qwen3-VL
#
# This script trains Qwen3-VL models on VSI-Bench data using GRPO algorithm.
# Pre-extracted video frames are used as input.
#
# Reward function:
# - Numeric questions (distance, counting, size): MRA (Mean Relative Accuracy)
# - Multiple-choice questions (direction, rel_distance): 0/1 accuracy
#
# Usage:
#   bash run_vsi_grpo.sh              # Use default 32 frames
#   NUM_FRAMES=8 bash run_vsi_grpo.sh # Use 8 frames (saves memory)

# ============================================================
# Configuration - Modify these paths for your environment
# ============================================================
MODEL="/upfs/models/Qwen/Qwen3-VL-8B-Instruct"
DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json"  # 32-frame version
VAL_DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/test/vsi_test.json"  # VSI-Bench test set
OUTPUT_DIR="output/vsi_grpo"

# ============================================================
# Frame Configuration
# Supported values: 1, 2, 4, 8, 16, 32 (must be power of 2)
# Fewer frames = less memory, faster training, but potentially lower accuracy
# ============================================================
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

# Evaluation settings
EVAL_STEPS=1
EVAL_BATCH_SIZE=96

# Training hyperparameters
LEARNING_RATE=1e-6
BETA=0.001
NUM_GENERATIONS=8
MAX_COMPLETION_LENGTH=512
BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
NUM_EPOCHS=1
# Global batch size = BATCH_SIZE × GRADIENT_ACCUMULATION × NUM_GPUS = 16 × 1 × 8 = 128

# CoT System Prompt
SYSTEM_PROMPT="examples/train/grpo/vsi/vsi_cot_prompt.txt"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8

# Image processing - reduce MAX_PIXELS to save memory with 32 frames
export MAX_PIXELS=200704  # ~256*28*28

# ============================================================
# Option 1: External vLLM Server Mode (Recommended for large models)
# Split GPUs between training and inference
# ============================================================

# First, start the vLLM rollout server in background
CUDA_VISIBLE_DEVICES=6,7 swift rollout \
    --model ${MODEL} \
    --vllm_tensor_parallel_size 2 \
    --vllm_disable_mm_preprocessor_cache true \
    --vllm_max_lora_rank 64 \
    --port 8000 &

echo "Waiting for vLLM server to start..."
until ss -lnt | grep -q ":8000"; do sleep 1; done
echo "vLLM server is ready!"

# Then run training on remaining GPUs
MAX_PIXELS=${MAX_PIXELS} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --system ${SYSTEM_PROMPT} \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
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

    # --eval_strategy "no" \
    # --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    # --val_dataset ${VAL_DATASET_PATH} \
    # --eval_steps ${EVAL_STEPS} \

# ============================================================
# Option 2: Colocate Mode (Training and inference on same GPUs)
# Simpler setup but requires more careful memory management
# ============================================================

# MAX_PIXELS=${MAX_PIXELS} \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# NPROC_PER_NODE=${NPROC_PER_NODE} \
# swift rlhf \
#     --rlhf_type grpo \
#     --model ${MODEL} \
#     --external_plugins examples/train/grpo/vsi/vsi_reward.py \
#     --reward_funcs vsi_reward \
#     --system ${SYSTEM_PROMPT} \
#     --use_vllm true \
#     --vllm_mode colocate \
#     --vllm_gpu_memory_utilization 0.5 \
#     --vllm_tensor_parallel_size 4 \
#     --tuner_type lora \
#     --lora_rank 64 \
#     --lora_alpha 128 \
#     --torch_dtype bfloat16 \
#     --dataset ${DATASET_PATH} \
#     --load_from_cache_file true \
#     --max_completion_length ${MAX_COMPLETION_LENGTH} \
#     --num_train_epochs ${NUM_EPOCHS} \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --per_device_eval_batch_size ${BATCH_SIZE} \
#     --learning_rate ${LEARNING_RATE} \
#     --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
#     --save_strategy steps \
#     --eval_strategy steps \
#     --val_dataset ${VAL_DATASET_PATH} \
#     --eval_steps ${EVAL_STEPS} \
#     --save_steps 200 \
#     --save_total_limit 5 \
#     --logging_steps 10 \
#     --output_dir ${OUTPUT_DIR} \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 4 \
#     --num_generations ${NUM_GENERATIONS} \
#     --temperature 1.0 \
#     --sleep_level 1 \
#     --log_completions true \
#     --num_iterations 1 \
#     --beta ${BETA} \
#     --overlong_filter true \
#     --max_grad_norm 1.0

echo "Training complete! Results saved to ${OUTPUT_DIR}"
