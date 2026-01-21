#!/bin/bash
# VSI-Bench Evaluation Script
#
# This script evaluates multimodal LLMs on VSI-Bench for spatial reasoning.
#
# Prerequisites:
# 1. Download VSI-Bench videos/frames from https://github.com/vision-x-nyu/thinking-in-space
# 2. Install required dependencies: pip install decord av moviepy
#
# Usage:
#   bash eval_vsi_bench.sh

# Configuration
MODEL_NAME="Qwen3-VL-32B-Instruct"
MODEL="/upfs/models/Qwen/${MODEL_NAME}"
VIDEO_DIR="/upfs/enhan/data/nyu_visionx/VSI-Bench"
OUTPUT_DIR="/upfs/enhan/vsi_bench_output/${MODEL_NAME}"
NUM_FRAMES=32
EVAL_LIMIT=100  # Set to empty or remove for full evaluation

# ============================================================
# Option 1: Single GPU with Transformers backend (default)
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python -m swift.pipelines.eval.run_vsi_bench \
#     --model ${MODEL} \
#     --infer_backend transformers \
#     --video_dir ${VIDEO_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --output_dir ${OUTPUT_DIR} \
#     --eval_limit ${EVAL_LIMIT} \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --verbose

# ============================================================
# Option 2: Multiple GPUs with Transformers (auto device_map)
# ============================================================
# Model will be automatically distributed across GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m swift.pipelines.eval.run_vsi_bench \
#     --model ${MODEL} \
#     --infer_backend transformers \
#     --video_dir ${VIDEO_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --output_dir ${OUTPUT_DIR} \
#     --eval_limit ${EVAL_LIMIT} \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --verbose

# ============================================================
# Option 3: Multiple GPUs with vLLM (tensor parallelism)
# Recommended for faster inference
# ============================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m swift.pipelines.eval.run_vsi_bench \
    --model ${MODEL} \
    --infer_backend vllm \
    --tensor_parallel_size 4 \
    --video_dir ${VIDEO_DIR} \
    --num_frames ${NUM_FRAMES} \
    --output_dir ${OUTPUT_DIR} \
    --eval_limit ${EVAL_LIMIT} \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --verbose

# ============================================================
# Option 4: With LoRA adapters
# ============================================================
# python -m swift.pipelines.eval.run_vsi_bench \
#     --model ${MODEL} \
#     --adapters /path/to/lora/adapter \
#     --video_dir ${VIDEO_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --output_dir ${OUTPUT_DIR}

echo "Evaluation complete! Results saved to ${OUTPUT_DIR}"
