#!/bin/bash
# VSI-Bench Evaluation Script
#
# This script evaluates multimodal LLMs on VSI-Bench for spatial reasoning.
#
# Prerequisites:
# 1. Download VSI-Bench videos from https://github.com/vision-x-nyu/thinking-in-space
# 2. Install required dependencies: pip install decord av moviepy
#
# Usage:
#   bash eval_vsi_bench.sh

# Configuration
MODEL_NAME="Qwen3-VL-32B-Instruct"
MODEL="/upfs/models/Qwen/${MODEL_NAME}"
VIDEO_DIR="/upfs/enhan/data/nyu-visionx/VSI-Bench"
FRAMES_DIR="/upfs/enhan/data/nyu-visionx/VSI-Bench-frames"  # Pre-extracted frames (faster)
DATA_PATH="/upfs/enhan/data/nyu-visionx/VSI-Bench/test.jsonl"  # Local dataset file (JSON/JSONL)
OUTPUT_DIR="/upfs/enhan/vsi_bench_output/${MODEL_NAME}"
NUM_FRAMES=32
EVAL_LIMIT=100  # Set to empty or remove for full evaluation

# ============================================================
# Step 0: Extract frames from videos (run once, highly recommended)
# This saves significant time during evaluation by avoiding repeated video decoding
# ============================================================
# python examples/eval/vsi_bench/extract_frames.py \
#     --video_dir ${VIDEO_DIR} \
#     --output_dir ${FRAMES_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --skip_existing
# Note: By default, uses all available CPUs. Use --num_workers N to limit.

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
# Recommended for faster inference with batch processing
# Use FRAMES_DIR (pre-extracted frames) instead of VIDEO_DIR for faster loading
# ============================================================
# Set VLLM_MM_INPUT_CACHE_GIB to increase multimodal cache size (default 4GB)
# This prevents LRU cache eviction errors when processing many images per batch
VLLM_MM_INPUT_CACHE_GIB=40 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m swift.pipelines.eval.run_vsi_bench \
    --model ${MODEL} \
    --infer_backend vllm \
    --tensor_parallel_size 8 \
    --data_path ${DATA_PATH} \
    --video_dir ${FRAMES_DIR} \
    --num_frames ${NUM_FRAMES} \
    --output_dir ${OUTPUT_DIR} \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --batch_size 64
# Note:
# - Use FRAMES_DIR for pre-extracted frames (faster) or VIDEO_DIR for raw videos
# - batch_size=0 means process all samples at once (fastest with vLLM)
# - Set batch_size to a smaller value (e.g., 32, 64) if you encounter OOM errors

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
