#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Test Data Preprocessing Script

Converts VSI-Bench test.jsonl to GRPO evaluation format.

Input format (test.jsonl):
{"id": 166, "dataset": "arkitscenes", "scene_name": "47430468",
 "question_type": "object_counting", "question": "How many chair(s) are in this room?",
 "ground_truth": "4", "options": null}

Output format:
{
    "id": "166",
    "images": ["/path/to/frames/47430468/000000.jpg", ...],
    "messages": [
        {"role": "user", "content": "<image>\\n...question..."},
        {"role": "assistant", "content": "4"}
    ],
    "solution": "4",
    "question_type": "object_counting",
    "data_source": "arkitscenes",
    "scene_name": "47430468"
}

Usage:
    python preprocess_vsi_test.py \\
        --input_path /path/to/test.jsonl \\
        --frames_dir /path/to/VSI-Bench-frames \\
        --output_path /path/to/vsi_bench_test.json \\
        --num_frames 32
"""
import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List


def get_frame_paths(dataset: str, scene_name: str, frames_dir: str, num_frames: int = 32) -> List[str]:
    """Get sorted frame paths for a scene.

    Directory structure: frames_dir/dataset/scene_name/*.jpg
    """
    scene_frames_dir = os.path.join(frames_dir, dataset, scene_name)

    if not os.path.isdir(scene_frames_dir):
        return []

    frame_files = sorted([
        f for f in os.listdir(scene_frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frame_files:
        return []

    if len(frame_files) > num_frames:
        indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
        frame_files = [frame_files[i] for i in indices]

    return [os.path.join(scene_frames_dir, f) for f in frame_files]


def convert_sample(sample: Dict[str, Any], frames_dir: str, num_frames: int = 32) -> Dict[str, Any]:
    """Convert a single test sample to GRPO format."""
    dataset = sample.get('dataset', '')
    scene_name = sample.get('scene_name', '')
    images = get_frame_paths(dataset, scene_name, frames_dir, num_frames)

    # Build user content
    question = sample.get('question', '')
    options = sample.get('options')

    content = "<image>\nThese are frames of a video.\n" + question
    if options:
        content += "\n" + "\n".join(options)
    content += "\nPlease think step by step, then answer the question using a single word or phrase."

    ground_truth = str(sample.get('ground_truth', ''))

    return {
        'id': str(sample.get('id', '')),
        'images': images,
        'messages': [
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': ground_truth}
        ],
        'solution': ground_truth,
        'question_type': sample.get('question_type', ''),
        'data_source': sample.get('dataset', ''),
        'scene_name': scene_name,
    }


def preprocess_test_data(
    input_path: str,
    frames_dir: str,
    output_path: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True
) -> None:
    """Preprocess VSI-Bench test data."""
    print(f"Loading data from {input_path}...")

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples")

    converted_data = []
    skipped = 0

    for i, sample in enumerate(data):
        converted = convert_sample(sample, frames_dir, num_frames)

        if skip_missing_frames and not converted['images']:
            skipped += 1
            continue

        converted_data.append(converted)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")

    print(f"\nTotal: {len(data)}, Converted: {len(converted_data)}, Skipped: {skipped}")

    # Distribution
    print("\nDataset distribution:")
    for src, cnt in sorted(Counter(s['data_source'] for s in converted_data).items()):
        print(f"  {src}: {cnt}")

    print("\nQuestion type distribution:")
    for qtype, cnt in sorted(Counter(s['question_type'] for s in converted_data).items()):
        print(f"  {qtype}: {cnt}")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess VSI-Bench test data')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to test.jsonl file')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing extracted frames')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video (default: 32)')
    parser.add_argument('--no_skip_missing', action='store_true',
                        help='Do not skip samples with missing frames')

    args = parser.parse_args()

    preprocess_test_data(
        input_path=args.input_path,
        frames_dir=args.frames_dir,
        output_path=args.output_path,
        num_frames=args.num_frames,
        skip_missing_frames=not args.no_skip_missing
    )


if __name__ == '__main__':
    main()
