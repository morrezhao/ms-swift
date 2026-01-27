#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Resample Frames from Pre-processed VSI Dataset.

This script creates a new dataset with fewer frames by uniformly sampling
from the existing 32-frame preprocessed data. Much faster than re-processing
from raw videos.

Usage:
    python resample_frames.py \
        --input /path/to/combined_train.json \
        --output /path/to/combined_train_8frames.json \
        --num_frames 8

Supported frame counts: 1, 2, 4, 8, 16, 32
"""
import argparse
import json
import os
from typing import List

VALID_FRAME_COUNTS = {1, 2, 4, 8, 16, 32}


def uniform_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    """Get indices for uniform sampling."""
    if num_frames >= total_frames:
        return list(range(total_frames))

    indices = []
    for i in range(num_frames):
        idx = int(i * total_frames / num_frames)
        indices.append(idx)
    return indices


def resample_dataset(input_path: str, output_path: str, num_frames: int) -> None:
    """Resample frames in a preprocessed dataset.

    Args:
        input_path: Path to input JSON (with 32 frames)
        output_path: Path to output JSON (with num_frames)
        num_frames: Number of frames to keep (1, 2, 4, 8, 16, or 32)
    """
    if num_frames not in VALID_FRAME_COUNTS:
        raise ValueError(f"num_frames must be one of {sorted(VALID_FRAME_COUNTS)}, got {num_frames}")

    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")
    print(f"Resampling from existing frames to {num_frames} frames...")

    resampled_data = []
    skipped = 0

    for sample in data:
        images = sample.get('images', [])

        if not images:
            skipped += 1
            continue

        total_frames = len(images)
        if total_frames < num_frames:
            print(f"Warning: Sample has only {total_frames} frames, keeping all")
            resampled_data.append(sample)
            continue

        # Uniform sampling
        indices = uniform_sample_indices(total_frames, num_frames)
        sampled_images = [images[i] for i in indices]

        # Create new sample with sampled images
        new_sample = sample.copy()
        new_sample['images'] = sampled_images
        resampled_data.append(new_sample)

    print(f"Resampled: {len(resampled_data)}, Skipped: {skipped}")

    # Save output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resampled_data, f, indent=2, ensure_ascii=False)

    print("Done!")

    # Print sample for verification
    if resampled_data:
        sample = resampled_data[0]
        print(f"\nSample verification:")
        print(f"  Number of images: {len(sample['images'])}")
        if sample['images']:
            print(f"  First image: {sample['images'][0]}")
            print(f"  Last image: {sample['images'][-1]}")


def main():
    parser = argparse.ArgumentParser(
        description='Resample frames from preprocessed VSI dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create 8-frame version:
    python resample_frames.py --input train.json --output train_8f.json --num_frames 8

    # Create multiple versions:
    for N in 1 2 4 8 16; do
        python resample_frames.py --input train.json --output train_${N}f.json --num_frames $N
    done
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input JSON (preprocessed with 32 frames)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output JSON')
    parser.add_argument('--num_frames', '-n', type=int, required=True,
                        choices=sorted(VALID_FRAME_COUNTS),
                        help='Number of frames to keep (1, 2, 4, 8, 16, or 32)')

    args = parser.parse_args()
    resample_dataset(args.input, args.output, args.num_frames)


if __name__ == '__main__':
    main()
