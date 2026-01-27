#!/bin/bash
# VSI-Bench GKD (Generalized Knowledge Distillation) Training Script
#
# On-policy distillation: Student (small Qwen-VL) generates, Teacher (big Qwen-VL) guides
#
# Usage:
#   bash run_vsi_gkd.sh
#   NUM_FRAMES=8 bash run_vsi_gkd.sh  # Use fewer frames

# ============================================================
# Model Configuration
# ============================================================
# Student model (small) - will be trained
STUDENT_MODEL="/upfs/models/Qwen/Qwen2.5-VL-3B-Instruct"

# Teacher model (big) - provides soft labels
TEACHER_MODEL="/upfs/models/Qwen/Qwen2.5-VL-7B-Instruct"

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json"
VAL_DATASET_PATH_BASE="/upfs/enhan/code/ms-swift/vsi_data/test/vsi_test.json"
OUTPUT_DIR="output/vsi_gkd"

# ============================================================
# Frame Configuration
# ============================================================
NUM_FRAMES=${NUM_FRAMES:-8}

# Validate NUM_FRAMES
if [[ ! "$NUM_FRAMES" =~ ^(1|2|4|8|16|32)$ ]]; then
    echo "Error: NUM_FRAMES must be 1, 2, 4, 8, 16, or 32. Got: $NUM_FRAMES"
    exit 1
fi

echo "Using NUM_FRAMES=$NUM_FRAMES"

# Determine dataset paths
if [ "$NUM_FRAMES" -eq 32 ]; then
    DATASET_PATH="$DATASET_PATH_BASE"
    VAL_DATASET_PATH="$VAL_DATASET_PATH_BASE"
else
    DATASET_DIR=$(dirname "$DATASET_PATH_BASE")
    DATASET_NAME=$(basename "$DATASET_PATH_BASE" .json)
    DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}_${NUM_FRAMES}f.json"

    VAL_DATASET_DIR=$(dirname "$VAL_DATASET_PATH_BASE")
    VAL_DATASET_NAME=$(basename "$VAL_DATASET_PATH_BASE" .json)
    VAL_DATASET_PATH="${VAL_DATASET_DIR}/${VAL_DATASET_NAME}_${NUM_FRAMES}f.json"

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
# Training Hyperparameters
# ============================================================
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
NUM_EPOCHS=1
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=512

# GKD specific parameters
LMBDA=1.0              # 1.0 = 100% on-policy (student always generates)
SEQ_KD=false           # Not used when lmbda=1.0
TEMPERATURE=1.0        # Temperature for generation and KD

# ============================================================
# GPU Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8

# Image processing

export MAX_PIXELS=200704

# ============================================================
# Option 1: Server Mode (Recommended)
# Student uses vLLM for fast generation, Teacher on separate GPUs
# ============================================================

# Start vLLM server for student model generation (4 GPUs)
echo "Starting vLLM server for student model..."
CUDA_VISIBLE_DEVICES=4,5,6,7 swift rollout \
    --model ${STUDENT_MODEL} \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.9 \
    --port 8000 &

echo "Waiting for vLLM server..."
until ss -lnt | grep -q ":8000"; do sleep 1; done
echo "vLLM server ready!"

# Run GKD training (4 GPUs)
MAX_PIXELS=${MAX_PIXELS} \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --system examples/train/grpo/vsi/vsi_cot_prompt.txt \
    --dataset ${DATASET_PATH} \
    --val_dataset ${VAL_DATASET_PATH} \
    --load_from_cache_file true \
    --lmbda ${LMBDA} \
    --seq_kd ${SEQ_KD} \
    --temperature ${TEMPERATURE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --max_length ${MAX_LENGTH} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000

# ============================================================
# Option 2: Colocate Mode (Alternative)
# All models share GPUs, uses sleep/wake mechanism
# ============================================================

# MAX_PIXELS=${MAX_PIXELS} \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# NPROC_PER_NODE=${NPROC_PER_NODE} \
# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# swift rlhf \
#     --rlhf_type gkd \
#     --model ${STUDENT_MODEL} \
#     --teacher_model ${TEACHER_MODEL} \
#     --tuner_type lora \
#     --lora_rank 64 \
#     --lora_alpha 128 \
#     --torch_dtype bfloat16 \
#     --dataset ${DATASET_PATH} \
#     --val_dataset ${VAL_DATASET_PATH} \
#     --load_from_cache_file true \
#     --lmbda ${LMBDA} \
#     --seq_kd ${SEQ_KD} \
#     --temperature ${TEMPERATURE} \
#     --num_train_epochs ${NUM_EPOCHS} \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --per_device_eval_batch_size ${BATCH_SIZE} \
#     --learning_rate ${LEARNING_RATE} \
#     --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
#     --max_length ${MAX_LENGTH} \
#     --max_completion_length ${MAX_COMPLETION_LENGTH} \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_strategy steps \
#     --save_steps 200 \
#     --save_total_limit 3 \
#     --logging_steps 10 \
#     --output_dir ${OUTPUT_DIR} \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 4 \
#     --deepspeed zero3 \
#     --teacher_deepspeed zero3_offload \
#     --use_vllm true \
#     --vllm_mode colocate \
#     --vllm_gpu_memory_utilization 0.3 \
#     --sleep_level 1

echo "GKD Training complete! Results saved to ${OUTPUT_DIR}"
