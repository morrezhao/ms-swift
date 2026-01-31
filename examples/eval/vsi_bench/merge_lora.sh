#!/bin/bash
# Merge LoRA adapter with base model
#
# This script merges a trained LoRA adapter with the base model to create
# a standalone model that can be used for inference without loading adapters.
#
# Usage:
#   bash merge_lora.sh

# Configuration
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
BASE_MODEL="/upfs/models/Qwen/${MODEL_NAME}"

# LoRA adapter path (output from training)
ALGO_NAME="grpo"   # Options: grpo, gkd, sft
CKPT_STEP="1000"   # Checkpoint step number
ADAPTER_PATH="/upfs/enhan/output/${MODEL_NAME}_${ALGO_NAME}/checkpoint-${CKPT_STEP}"

# Output path for merged model
OUTPUT_DIR="/upfs/enhan/merged_models/${MODEL_NAME}_${ALGO_NAME}_step${CKPT_STEP}"

# ============================================================
# Merge LoRA with base model
# ============================================================
swift merge-lora \
    --model ${BASE_MODEL} \
    --adapters ${ADAPTER_PATH} \
    --output_dir ${OUTPUT_DIR}

echo "Merged model saved to: ${OUTPUT_DIR}"
echo ""
echo "You can now use the merged model for evaluation:"
echo "  MODEL=\"${OUTPUT_DIR}\""
