# VSI-Bench GRPO Training for Qwen3-VL

This example demonstrates how to train Qwen3-VL models on VSI-Bench spatial reasoning data using the GRPO algorithm.

## Overview

VSI-Bench (Visual Spatial Intelligence Benchmark) contains questions about:
- **Numeric questions** (evaluated with MRA - Mean Relative Accuracy):
  - `object_abs_distance`: Distance between objects (meters)
  - `object_counting`: Count of objects
  - `object_size_estimation`: Object dimensions (cm)
  - `room_size_estimation`: Room area (square meters)

- **Multiple-choice questions** (evaluated with 0/1 accuracy):
  - `object_rel_direction_v1/v2/v3`: Relative direction questions (A/B/C/D)
  - `object_rel_distance_v1/v2/v3`: Relative distance comparison (A/B/C/D)

## Reward Function

The reward function (`vsi_reward.py`) automatically selects the appropriate metric:
- **Numeric questions**: Uses MRA (Mean Relative Accuracy) with thresholds [0.5, 0.55, ..., 0.95]
- **Multiple-choice questions**: Uses exact match (0/1 accuracy)

## Prerequisites

1. **Extract video frames** (if not already done):
   ```bash
   # Example using the provided extraction script
   python examples/eval/vsi_bench/extract_frames.py \
       --video_dir /path/to/videos \
       --output_dir /path/to/frames \
       --num_frames 32
   ```

2. **Preprocess training data**:
   ```bash
   python examples/train/grpo/vsi/preprocess_vsi_data.py \
       --input_path vsi_data/train_formatted.json \
       --frames_dir /path/to/extracted_frames \
       --output_path vsi_data/train_grpo.json \
       --num_frames 32
   ```

## Training

### Option 1: Using Shell Script (Recommended)

Edit `run_vsi_grpo.sh` to set your paths, then:
```bash
bash examples/train/grpo/vsi/run_vsi_grpo.sh
```

### Option 2: Using YAML Config

Edit `vsi_grpo_config.yaml` to set your paths, then:
```bash
swift rlhf --config examples/train/grpo/vsi/vsi_grpo_config.yaml
```

### Option 3: Direct Command

```bash
# Colocate mode (simpler setup)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=262144 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset vsi_data/train_grpo.json \
    --max_completion_length 256 \
    --num_generations 8 \
    --learning_rate 1e-6 \
    --beta 0.001 \
    --output_dir output/vsi_grpo
```

### External vLLM Server Mode (for larger models)

Start vLLM server first:
```bash
CUDA_VISIBLE_DEVICES=6,7 swift rollout \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --vllm_data_parallel_size 2
```

Then run training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --external_plugins examples/train/grpo/vsi/vsi_reward.py \
    --reward_funcs vsi_reward \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    ...
```

## Data Format

After preprocessing, the data should be in this format:
```json
{
    "images": ["/path/to/frames/scene0191_00/frame_0000.jpg", ...],
    "messages": [
        {"role": "user", "content": "<image>\nThese are frames of a video.\n...question..."},
        {"role": "assistant", "content": "answer"}
    ],
    "solution": "answer",
    "question_type": "object_abs_distance"
}
```

## Hyperparameters

Key hyperparameters to tune:
- `learning_rate`: 1e-6 (start low for stability)
- `beta`: 0.001 (KL penalty coefficient)
- `num_generations`: 8 (more generations = better reward estimation but slower)
- `max_completion_length`: 256 (VSI answers are typically short)
- `MAX_PIXELS`: 262144 (reduce for memory, ~512x512 per frame)

## Files

- `vsi_reward.py`: Custom reward function (MRA for numeric, 0/1 for MC)
- `preprocess_vsi_data.py`: Data preprocessing script
- `run_vsi_grpo.sh`: Training shell script
- `vsi_grpo_config.yaml`: YAML configuration file
- `train_vsi_grpo.py`: Python training script (alternative)
- `vsi_dataset.py`: Dataset registration module

## Evaluation

After training, evaluate using the VSI-Bench evaluation script:
```bash
python -m swift.pipelines.eval.run_vsi_bench \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --adapters output/vsi_grpo/checkpoint-xxx \
    --video_dir /path/to/frames \
    --output_dir output/eval
```
