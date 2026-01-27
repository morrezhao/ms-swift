#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench GRPO Training Script for Qwen3-VL

This script trains Qwen3-VL models on VSI-Bench data using GRPO algorithm.
Pre-extracted video frames (32 frames) are used as input.

Reward function:
- Numeric questions (distance, counting, size, room_size): MRA (Mean Relative Accuracy)
- Multiple-choice questions (direction, relative distance): 0/1 accuracy

Usage:
    python train_vsi_grpo.py --config config.yaml
    # or
    python train_vsi_grpo.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --dataset_path /path/to/train_formatted.json \
        --frames_dir /path/to/frames
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import reward function (registers automatically)
import vsi_reward  # noqa: F401

from swift.dataset import register_dataset, DatasetMeta
from swift.dataset.preprocessor import ResponsePreprocessor
from swift.llm import rlhf_main, RLHFArguments


class VSIPreprocessor(ResponsePreprocessor):
    """Preprocessor for VSI-Bench GRPO training data."""

    def __init__(self, frames_dir: str, num_frames: int = 32):
        super().__init__()
        self.frames_dir = frames_dir
        self.num_frames = num_frames

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Get video path and extract scene name
        video_path = row.get('video', '')
        scene_name = os.path.splitext(os.path.basename(video_path))[0]

        # Get frame paths (sorted order = temporal order)
        scene_frames_dir = os.path.join(self.frames_dir, scene_name)
        images = []
        if os.path.isdir(scene_frames_dir):
            frame_files = sorted([
                f for f in os.listdir(scene_frames_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            # Use all frames directly, sample only if more than expected
            if len(frame_files) > self.num_frames:
                indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
                frame_files = [frame_files[i] for i in indices]
            images = [os.path.join(scene_frames_dir, f) for f in frame_files]

        # Extract query and response from conversations
        conversations = row.get('conversations', [])
        query = ''
        response = ''
        for conv in conversations:
            if conv.get('from') == 'human':
                query = conv.get('value', '')
            elif conv.get('from') == 'gpt':
                response = conv.get('value', '')

        row.update({
            'query': query,
            'response': response,
            'images': images,
            'solution': response,
            'question_type': row.get('question_type', ''),
        })

        return super().preprocess(row)


def register_vsi_dataset(dataset_path: str, frames_dir: str, num_frames: int = 32):
    """Register VSI dataset for training."""
    register_dataset(
        DatasetMeta(
            ms_dataset_id=dataset_path,
            preprocess_func=VSIPreprocessor(frames_dir=frames_dir, num_frames=num_frames),
            tags=['qa', 'spatial', 'vsi'],
        ),
        dataset_name='vsi_train'
    )


def main():
    parser = argparse.ArgumentParser(description='VSI-Bench GRPO Training')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                        help='Model name or path')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to VSI training data JSON file')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing extracted video frames')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video')
    parser.add_argument('--output_dir', type=str, default='output/vsi_grpo',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    args, remaining = parser.parse_known_args()

    # Register the VSI dataset
    register_vsi_dataset(args.dataset_path, args.frames_dir, args.num_frames)

    # Build RLHF arguments
    rlhf_args = RLHFArguments(
        rlhf_type='grpo',
        model=args.model,
        dataset=['vsi_train'],
        output_dir=args.output_dir,
        reward_funcs=['vsi_reward'],
        external_plugins=[os.path.join(os.path.dirname(__file__), 'vsi_reward.py')],
        num_generations_eval=1,  # Use single generation for faster evaluation
    )

    # Override with config file if provided
    if args.config:
        rlhf_main(args.config)
    else:
        # Run with default or command-line arguments
        rlhf_main(rlhf_args)


if __name__ == '__main__':
    main()
