#!/bin/bash
# VSI-Bench GKD - vLLM Rollout Server Script (2 nodes)
#
# Start this on 2 dedicated nodes before running training.
# The vLLM server handles student model generation.
#
# Usage:
#   # Node 1: Rollout server 1
#   ROLLOUT_PORT=8001 bash run_vsi_gkd_rollout.sh
#
#   # Node 2: Rollout server 2
#   ROLLOUT_PORT=8002 bash run_vsi_gkd_rollout.sh

set -e

# ============================================================
# Configuration
# ============================================================
STUDENT_MODEL="/upfs/models/Qwen/Qwen2.5-VL-7B-Instruct"
ROLLOUT_PORT=${ROLLOUT_PORT:-8001}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-4}   # Tensor parallel size
VLLM_DP_SIZE=${VLLM_DP_SIZE:-2}   # Data parallel size (TP * DP = 8 GPUs)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "============================================================"
echo "Starting vLLM Rollout Server for GKD:"
echo "  Model: $STUDENT_MODEL"
echo "  Port: $ROLLOUT_PORT"
echo "  Tensor Parallel Size: $VLLM_TP_SIZE"
echo "  Data Parallel Size: $VLLM_DP_SIZE"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

swift rollout \
    --model ${STUDENT_MODEL} \
    --vllm_tensor_parallel_size ${VLLM_TP_SIZE} \
    --vllm_data_parallel_size ${VLLM_DP_SIZE} \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_disable_mm_preprocessor_cache true \
    --port ${ROLLOUT_PORT}
