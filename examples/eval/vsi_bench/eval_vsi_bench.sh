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
MODEL="/upfs/models/Qwen/Qwen3-VL-32B-Instruct"  
VIDEO_DIR="/upfs/enhan/data/nyu_visionx/VSI-Bench"  
OUTPUT_DIR="/upfs/enhan/vsi_bench_output/${MODEL}"
NUM_FRAMES=32
EVAL_LIMIT=100  # Set to null or remove for full evaluation

# Run evaluation
python -m swift.pipelines.eval.run_vsi_bench \
    --model ${MODEL} \
    --video_dir ${VIDEO_DIR} \
    --num_frames ${NUM_FRAMES} \
    --output_dir ${OUTPUT_DIR} \
    --eval_limit ${EVAL_LIMIT} \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --verbose

# For models with LoRA adapters:
# python -m swift.pipelines.eval.run_vsi_bench \
#     --model ${MODEL} \
#     --adapters /path/to/lora/adapter \
#     --video_dir ${VIDEO_DIR} \
#     --num_frames ${NUM_FRAMES} \
#     --output_dir ${OUTPUT_DIR}

echo "Evaluation complete! Results saved to ${OUTPUT_DIR}"
