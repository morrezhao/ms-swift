#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Data Preprocessing Script

This script converts the VSI training data format from video paths to
pre-extracted frame paths for GRPO training. Supports multiple datasets.

Input format (train_formatted.json):
{
    "id": "...",
    "data_source": "scannet",
    "scene_name": "scene0191_00",
    "question_type": "object_abs_distance",
    "video": "/path/to/videos/scene0191_00.mp4",
    "conversations": [
        {"from": "human", "value": "<image>\\n...question..."},
        {"from": "gpt", "value": "answer"}
    ]
}

Output format (for GRPO training):
{
    "images": ["/path/to/frames/scene0191_00/frame_0000.jpg", ...],
    "messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"}  # will be removed during training
    ],
    "solution": "answer",
    "question_type": "object_abs_distance"
}

Usage:
    # Single dataset:
    python preprocess_vsi_data.py \\
        --input_path /path/to/train_formatted.json \\
        --frames_dir /path/to/extracted_frames \\
        --output_path /path/to/train_grpo.json \\
        --num_frames 32

    # Multiple datasets (combined and shuffled):
    python preprocess_vsi_data.py \\
        --datasets scannet:/upfs/enhan/code/ms-swift/vsi_data/formatted_qa_scannet.json:/upfs/enhan/data/processed_data/ScanNet/color/train \\
        --datasets arkitscenes:/upfs/enhan/code/ms-swift/vsi_data/formatted_qa_arkitscenes.json:/upfs/enhan/data/processed_data/ARKitScenes/color/train \\
        --datasets scannetpp:/upfs/enhan/code/ms-swift/vsi_data/formatted_qa_scannetpp.json:/upfs/enhan/data/processed_data/ARKitScenes/color/train \\
        --output_path /upfs/enhan/code/ms-swift/vsi_data/processed/combined_train.json \\
        --num_frames 32 \\
        --shuffle
"""
import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def get_frame_paths(scene_name: str, frames_dir: str, num_frames: int = 32) -> List[str]:
    """Get sorted frame paths for a scene.

    Args:
        scene_name: Name of the scene (e.g., 'scene0191_00')
        frames_dir: Base directory containing frame subdirectories
        num_frames: Number of frames to use (default 32, frames are pre-extracted)

    Returns:
        List of frame paths in sorted order.
        Frames are assumed to be pre-extracted and already in correct order when sorted.
        Frame filenames can be any format (e.g., 000000.jpg, 000179.jpg, etc.)
    """
    scene_frames_dir = os.path.join(frames_dir, scene_name)

    if not os.path.isdir(scene_frames_dir):
        # print(f"Warning: Frame directory not found: {scene_frames_dir}")
        return []

    # Get all frame files and sort them (sorting gives correct temporal order)
    frame_files = sorted([
        f for f in os.listdir(scene_frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frame_files:
        # print(f"Warning: No frames found in {scene_frames_dir}")
        return []

    # Use all frames directly (assuming exactly num_frames pre-extracted frames)
    # If there are more frames, uniformly sample to get num_frames
    if len(frame_files) > num_frames:
        indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
        frame_files = [frame_files[i] for i in indices]
    elif len(frame_files) < num_frames:
        print(f"Warning: Only {len(frame_files)} frames found in {scene_frames_dir}, expected {num_frames}")

    return [os.path.join(scene_frames_dir, f) for f in frame_files]


def convert_sample(sample: Dict[str, Any], frames_dir: str, num_frames: int = 32) -> Dict[str, Any]:
    """Convert a single sample to GRPO training format.

    Args:
        sample: Original sample from train_formatted.json
        frames_dir: Base directory containing frame subdirectories
        num_frames: Number of frames to use

    Returns:
        Converted sample in GRPO format
    """
    # Get scene name from video path
    video_path = sample.get('video', '')
    scene_name = os.path.splitext(os.path.basename(video_path))[0]

    # Get frame paths
    images = get_frame_paths(scene_name, frames_dir, num_frames)

    # Extract query and response from conversations
    conversations = sample.get('conversations', [])
    query = ''
    response = ''
    for conv in conversations:
        if conv.get('from') == 'human':
            query = conv.get('value', '')
        elif conv.get('from') == 'gpt':
            response = conv.get('value', '')

    # Build messages in standard format
    messages = [
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': response}
    ]

    return {
        'id': sample.get('id', ''),
        'images': images,
        'messages': messages,
        'solution': response,
        'question_type': sample.get('question_type', ''),
        'data_source': sample.get('data_source', ''),
        'scene_name': scene_name,
    }


def process_single_dataset(
    input_path: str,
    frames_dir: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True,
    dataset_name: str = ''
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Process a single dataset.

    Args:
        input_path: Path to input JSON file
        frames_dir: Base directory containing frame subdirectories
        num_frames: Number of frames per video
        skip_missing_frames: Whether to skip samples with missing frames
        dataset_name: Name of the dataset for logging

    Returns:
        Tuple of (converted_samples, total_count, skipped_count)
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name or input_path}")
    print(f"Input: {input_path}")
    print(f"Frames dir: {frames_dir}")
    print(f"{'='*60}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    converted_data = []
    skipped = 0

    for i, sample in enumerate(data):
        converted = convert_sample(sample, frames_dir, num_frames)

        if skip_missing_frames and not converted['images']:
            skipped += 1
            continue

        converted_data.append(converted)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples...")

    print(f"  Converted: {len(converted_data)}, Skipped: {skipped}")

    return converted_data, len(data), skipped


def preprocess_dataset(
    input_path: str,
    frames_dir: str,
    output_path: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True
) -> None:
    """Preprocess a single VSI dataset (legacy interface).

    Args:
        input_path: Path to input JSON file (train_formatted.json)
        frames_dir: Base directory containing frame subdirectories
        output_path: Path to output JSON file
        num_frames: Number of frames per video
        skip_missing_frames: Whether to skip samples with missing frames
    """
    converted_data, total, skipped = process_single_dataset(
        input_path, frames_dir, num_frames, skip_missing_frames
    )

    print(f"\nConversion complete!")
    print(f"Total samples: {total}")
    print(f"Converted samples: {len(converted_data)}")
    print(f"Skipped (missing frames): {skipped}")

    # Save output
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print("Done!")
    print_sample(converted_data)


def preprocess_multiple_datasets(
    datasets: List[Tuple[str, str, str]],  # List of (name, json_path, frames_dir)
    output_path: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42
) -> None:
    """Preprocess multiple VSI datasets and combine them.

    Args:
        datasets: List of (dataset_name, json_path, frames_dir) tuples
        output_path: Path to output JSON file
        num_frames: Number of frames per video
        skip_missing_frames: Whether to skip samples with missing frames
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for shuffling
    """
    all_converted_data = []
    total_samples = 0
    total_skipped = 0

    # Process each dataset
    for dataset_name, json_path, frames_dir in datasets:
        converted_data, total, skipped = process_single_dataset(
            json_path, frames_dir, num_frames, skip_missing_frames, dataset_name
        )
        all_converted_data.extend(converted_data)
        total_samples += total
        total_skipped += skipped

    print(f"\n{'='*60}")
    print("COMBINED SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples across all datasets: {total_samples}")
    print(f"Total converted: {len(all_converted_data)}")
    print(f"Total skipped: {total_skipped}")

    # Print dataset distribution
    print("\nDataset distribution:")
    from collections import Counter
    source_counts = Counter(s['data_source'] for s in all_converted_data)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    # Print question type distribution
    print("\nQuestion type distribution:")
    qtype_counts = Counter(s['question_type'] for s in all_converted_data)
    for qtype, count in sorted(qtype_counts.items()):
        print(f"  {qtype}: {count}")

    # Shuffle if requested
    if shuffle:
        print(f"\nShuffling dataset with seed={seed}...")
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_converted_data)
        print("Shuffled!")

    # Save output
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_converted_data, f, indent=2, ensure_ascii=False)

    print("Done!")
    print_sample(all_converted_data)


def print_sample(converted_data: List[Dict[str, Any]]) -> None:
    """Print a sample for verification."""
    if converted_data:
        print("\n" + "=" * 60)
        print("Sample output:")
        print("=" * 60)
        sample = converted_data[0]
        print(f"ID: {sample['id']}")
        print(f"Data source: {sample['data_source']}")
        print(f"Question type: {sample['question_type']}")
        print(f"Number of images: {len(sample['images'])}")
        if sample['images']:
            print(f"First image: {sample['images'][0]}")
            print(f"Last image: {sample['images'][-1]}")
        print(f"Solution: {sample['solution']}")


def parse_dataset_spec(spec: str) -> Tuple[str, str, str]:
    """Parse a dataset specification string.

    Format: name:json_path:frames_dir

    Args:
        spec: Dataset specification string

    Returns:
        Tuple of (name, json_path, frames_dir)
    """
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid dataset specification: {spec}\n"
            f"Expected format: name:json_path:frames_dir"
        )
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess VSI-Bench data for GRPO training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single dataset:
    python preprocess_vsi_data.py \\
        --input_path /path/to/train_formatted.json \\
        --frames_dir /path/to/extracted_frames \\
        --output_path /path/to/train_grpo.json

    # Multiple datasets (combined and shuffled):
    python preprocess_vsi_data.py \\
        --datasets scannet:/path/to/scannet.json:/path/to/ScanNet/color/train \\
        --datasets arkitscenes:/path/to/arkitscenes.json:/path/to/ARKitScenes/color/train \\
        --datasets scannetpp:/path/to/scannetpp.json:/path/to/ScanNetpp/color/train \\
        --output_path /path/to/combined_train.json \\
        --shuffle
        """
    )

    # Single dataset mode
    parser.add_argument('--input_path', type=str,
                        help='Path to input train_formatted.json file (single dataset mode)')
    parser.add_argument('--frames_dir', type=str,
                        help='Directory containing extracted frames (single dataset mode)')

    # Multiple datasets mode
    parser.add_argument('--datasets', type=str, action='append',
                        help='Dataset specification in format "name:json_path:frames_dir". '
                             'Can be specified multiple times for multiple datasets.')

    # Common arguments
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSON file for GRPO training')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video (default: 32)')
    parser.add_argument('--no_skip_missing', action='store_true',
                        help='Do not skip samples with missing frames')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the combined dataset (only for multiple datasets mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    # Determine mode
    if args.datasets:
        # Multiple datasets mode
        datasets = [parse_dataset_spec(spec) for spec in args.datasets]
        preprocess_multiple_datasets(
            datasets=datasets,
            output_path=args.output_path,
            num_frames=args.num_frames,
            skip_missing_frames=not args.no_skip_missing,
            shuffle=args.shuffle,
            seed=args.seed
        )
    elif args.input_path and args.frames_dir:
        # Single dataset mode
        preprocess_dataset(
            input_path=args.input_path,
            frames_dir=args.frames_dir,
            output_path=args.output_path,
            num_frames=args.num_frames,
            skip_missing_frames=not args.no_skip_missing
        )
    else:
        parser.error("Either --input_path and --frames_dir, or --datasets must be specified")


if __name__ == '__main__':
    main()
