#!/bin/bash
# VSI-Bench GRPO Training Script for Qwen3-VL
#
# This script trains Qwen3-VL models on VSI-Bench data using GRPO algorithm.
# Pre-extracted video frames (32 frames) are used as input.
#
# Reward function:
# - Numeric questions (distance, counting, size): MRA (Mean Relative Accuracy)
# - Multiple-choice questions (direction, rel_distance): 0/1 accuracy
#
# Usage:
#   bash run_vsi_grpo.sh

# ============================================================
# Configuration - Modify these paths for your environment
# ============================================================
MODEL="/upfs/models/Qwen/Qwen3-VL-8B-Instruct"
DATASET_PATH="/upfs/enhan/code/ms-swift/vsi_data/processed/scannet.json"  # Your VSI training data
FRAMES_DIR="/upfs/enhan/data/processed_data/ScanNet/color/train"                  # Pre-extracted frames directory
OUTPUT_DIR="output/vsi_grpo"
NUM_FRAMES=32

# Training hyperparameters
LEARNING_RATE=1e-6
BETA=0.001
NUM_GENERATIONS=8
MAX_COMPLETION_LENGTH=1024
BATCH_SIZE=4
GRADIENT_ACCUMULATION=2
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

# First, start the vLLM rollout server in a separate terminal:
CUDA_VISIBLE_DEVICES=6,7 swift rollout \
    --model ${MODEL} \
    --vllm_data_parallel_size 2 \
    --port 8000 &

until ss -lnt | grep -q ":8000"; do sleep 1; done
# Then run training:
VLLM_MM_INPUT_CACHE_GIB=20 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset ${DATASET_PATH} \
    --load_from_cache_file true \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --save_strategy 'steps' \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations ${NUM_GENERATIONS} \
    --temperature 1.0 \
    --log_completions true \
    --num_iterations 1 \
    --beta ${BETA}

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
#     --reward_funcs vsi_reward format \
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
#     --save_strategy 'steps' \
#     --eval_strategy 'steps' \
#     --eval_steps 50 \
#     --save_steps 200 \
#     --save_total_limit 5 \
#     --logging_steps 1 \
#     --output_dir ${OUTPUT_DIR} \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 4 \
#     --num_generations ${NUM_GENERATIONS} \
#     --temperature 1.0 \
#     --sleep_level 1 \
#     --log_completions true \
#     --num_iterations 1 \
#     --beta ${BETA} \
#     --max_grad_norm 1.0

echo "Training complete! Results saved to ${OUTPUT_DIR}"
