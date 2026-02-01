#!/bin/bash
# VSI-Bench GKD - Multi-Node Training Script (2 nodes)
#
# Cross-model GKD: Student (Qwen2.5-VL-7B) generates, Teacher (Qwen3-VL-8B) guides
# Full parameter training (no LoRA), using FSDP2 (no DeepSpeed)
#
# Usage:
#   # Node 1: Training master (rank 0)
#   NODE_RANK=0 MASTER_ADDR=<master_ip> bash run_vsi_gkd_train.sh
#
#   # Node 2: Training worker (rank 1)
#   NODE_RANK=1 MASTER_ADDR=<master_ip> bash run_vsi_gkd_train.sh

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
# Model Configuration
# ============================================================
STUDENT_MODEL="/upfs/models/Qwen/Qwen2.5-VL-7B-Instruct"
TEACHER_MODEL="/upfs/models/Qwen/Qwen3-VL-8B-Instruct"

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_PATH="/upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json"
VAL_DATASET_PATH="/upfs/enhan/code/ms-swift/vsi_data/test/vsi_test.json"
OUTPUT_DIR="output/vsi_gkd_multi_node"

# ============================================================
# vLLM Server Configuration (2 rollout nodes)
# Replace with actual rollout node IPs
# ============================================================
VLLM_SERVER_HOST_1=${VLLM_SERVER_HOST_1:-"<rollout_node1_ip>"}
VLLM_SERVER_HOST_2=${VLLM_SERVER_HOST_2:-"<rollout_node2_ip>"}
VLLM_SERVER_PORT_1=${VLLM_SERVER_PORT_1:-8001}
VLLM_SERVER_PORT_2=${VLLM_SERVER_PORT_2:-8002}

# ============================================================
# Training Hyperparameters (from run_vsi_gkd.sh)
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

# Image processing
export MAX_PIXELS=200704

# ============================================================
# GKD Training with FSDP2 (Full Parameter, No DeepSpeed)
# ============================================================
MAX_PIXELS=${MAX_PIXELS} \
CROSS_MODEL_GKD=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --external_plugins examples/train/grpo/vsi/cross_model_gkd.py \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
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
    --vllm_server_host ${VLLM_SERVER_HOST_1} ${VLLM_SERVER_HOST_2} \
    --vllm_server_port ${VLLM_SERVER_PORT_1} ${VLLM_SERVER_PORT_2}

echo "GKD Training complete on node ${NODE_RANK}!"
