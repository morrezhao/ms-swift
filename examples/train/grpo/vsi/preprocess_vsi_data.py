#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Data Preprocessing Script

This script converts the VSI training data format from video paths to
pre-extracted frame paths for GRPO training.

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
    python preprocess_vsi_data.py \\
        --input_path /path/to/train_formatted.json \\
        --frames_dir /path/to/extracted_frames \\
        --output_path /path/to/train_grpo.json \\
        --num_frames 32
"""
import argparse
import json
import os
from typing import Any, Dict, List


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
        print(f"Warning: Frame directory not found: {scene_frames_dir}")
        return []

    # Get all frame files and sort them (sorting gives correct temporal order)
    frame_files = sorted([
        f for f in os.listdir(scene_frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frame_files:
        print(f"Warning: No frames found in {scene_frames_dir}")
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


def preprocess_dataset(
    input_path: str,
    frames_dir: str,
    output_path: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True
) -> None:
    """Preprocess the entire VSI dataset.

    Args:
        input_path: Path to input JSON file (train_formatted.json)
        frames_dir: Base directory containing frame subdirectories
        output_path: Path to output JSON file
        num_frames: Number of frames per video
        skip_missing_frames: Whether to skip samples with missing frames
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} samples...")
    converted_data = []
    skipped = 0

    for i, sample in enumerate(data):
        converted = convert_sample(sample, frames_dir, num_frames)

        if skip_missing_frames and not converted['images']:
            skipped += 1
            continue

        converted_data.append(converted)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")

    print(f"\nConversion complete!")
    print(f"Total samples: {len(data)}")
    print(f"Converted samples: {len(converted_data)}")
    print(f"Skipped (missing frames): {skipped}")

    # Save output
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print("Done!")

    # Print sample for verification
    if converted_data:
        print("\n" + "=" * 60)
        print("Sample output:")
        print("=" * 60)
        sample = converted_data[0]
        print(f"ID: {sample['id']}")
        print(f"Question type: {sample['question_type']}")
        print(f"Number of images: {len(sample['images'])}")
        if sample['images']:
            print(f"First image: {sample['images'][0]}")
            print(f"Last image: {sample['images'][-1]}")
        print(f"Solution: {sample['solution']}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess VSI-Bench data for GRPO training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python preprocess_vsi_data.py \\
        --input_path /path/to/train_formatted.json \\
        --frames_dir /path/to/extracted_frames \\
        --output_path /path/to/train_grpo.json
        """
    )
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input train_formatted.json file')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing extracted frames (organized by scene name)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSON file for GRPO training')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video (default: 32)')
    parser.add_argument('--no_skip_missing', action='store_true',
                        help='Do not skip samples with missing frames')

    args = parser.parse_args()

    preprocess_dataset(
        input_path=args.input_path,
        frames_dir=args.frames_dir,
        output_path=args.output_path,
        num_frames=args.num_frames,
        skip_missing_frames=not args.no_skip_missing
    )


if __name__ == '__main__':
    main()
