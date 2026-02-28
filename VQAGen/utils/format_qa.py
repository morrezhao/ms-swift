#!/usr/bin/env python
"""
Convert code_qa_generator output to GRPO training format.

Input format (from code_qa_generator):

  MC type (e.g. object_rel_direction_v1):
  {
      "id": 0, "dataset": "scannetpp", "scene_name": "scene0191_00",
      "question_type": "object_rel_direction_v1",
      "question": "Standing at the table facing the door, where is the chair?",
      "options": ["A. front-left", "B. front-right", "C. back-left", "D. back-right"],
      "ground_truth": "front-left", "mc_answer": "A",
      "code": "def compute_answer(): ...", "generated_by": "llm_code"
  }

  Numeric type (e.g. object_abs_distance):
  {
      "id": 1, "dataset": "scannetpp", "scene_name": "scene0191_00",
      "question_type": "object_abs_distance",
      "question": "What is the distance between the chair and the table?",
      "options": [], "ground_truth": "1.2", "mc_answer": null,
      "code": "def compute_answer(): ...", "generated_by": "llm_code"
  }

Output format (for GRPO training):

  MC type -> solution = mc_answer, user content includes options:
  {
      "id": 0,
      "images": ["/path/to/frames/scene0191_00/000000.jpg", ...],
      "messages": [
          {"role": "user", "content": "Standing at the table...\\nA. front-left\\nB. ..."},
          {"role": "assistant", "content": "A"}
      ],
      "solution": "A",
      "question_type": "object_rel_direction_v1",
      "data_source": "scannetpp", "scene_name": "scene0191_00"
  }

  Numeric type -> solution = ground_truth, no options in user content:
  {
      "id": 1,
      "images": ["/path/to/frames/scene0191_00/000000.jpg", ...],
      "messages": [
          {"role": "user", "content": "What is the distance between the chair and the table?"},
          {"role": "assistant", "content": "1.2"}
      ],
      "solution": "1.2",
      "question_type": "object_abs_distance",
      "data_source": "scannetpp", "scene_name": "scene0191_00"
  }

Usage:
    python -m utils.format_qa \\
        --datasets scannet:data/qa_output/train/qa_code_generated_scannet.json:/upfs/enhan/data/processed_data/ScanNet/color/train \\
        --datasets scannetpp:data/qa_output/train/qa_code_generated_scannetpp.json:/upfs/enhan/data/processed_data/ScanNetpp/color/train \\
        --datasets arkitscenes:data/qa_output/train/qa_code_generated_arkitscenes.json:/upfs/enhan/data/processed_data/ARKitScenes/color/train \\
        --output_path data/qa_output/train_grpo.json --shuffle
"""
import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def get_frame_paths(scene_name: str, frames_dir: str, num_frames: int = 32) -> List[str]:
    """Get sorted frame paths for a scene.

    Args:
        scene_name: Name of the scene (e.g., 'scene0191_00')
        frames_dir: Base directory containing frame subdirectories
        num_frames: Number of frames to use (default 32, frames are pre-extracted)

    Returns:
        List of frame paths in sorted order.
    """
    scene_frames_dir = os.path.join(frames_dir, scene_name)

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
    elif len(frame_files) < num_frames:
        print(f"Warning: Only {len(frame_files)} frames found in {scene_frames_dir}, expected {num_frames}")

    return [os.path.join(scene_frames_dir, f) for f in frame_files]


def convert_sample(sample: Dict[str, Any], frames_dir: str, num_frames: int = 32) -> Dict[str, Any]:
    """Convert a code_qa_generator output sample to GRPO training format."""
    scene_name = sample.get('scene_name', '')
    question = sample.get('question', '')
    options = sample.get('options', [])
    ground_truth = sample.get('ground_truth', '')
    mc_answer = sample.get('mc_answer')

    images = get_frame_paths(scene_name, frames_dir, num_frames)

    if options:
        options_text = '\n'.join(options)
        user_content = f"{question}\n{options_text}"
        solution = str(mc_answer) if mc_answer else str(ground_truth)
    else:
        user_content = question
        solution = str(ground_truth)

    return {
        'id': sample.get('id', ''),
        'images': images,
        'messages': [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': solution}
        ],
        'solution': solution,
        'question_type': sample.get('question_type', ''),
        'data_source': sample.get('dataset', ''),
        'scene_name': scene_name,
    }


def process_and_save(
    datasets: List[Tuple[str, str, str]],
    output_path: str,
    num_frames: int = 32,
    skip_missing_frames: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = 42
) -> None:
    """Process one or more datasets, combine, and save to output file.

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

    for dataset_name, json_path, frames_dir in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name or json_path}")
        print(f"Input: {json_path}")
        print(f"Frames dir: {frames_dir}")
        print(f"{'='*60}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples")
        total_samples += len(data)
        skipped = 0

        for i, sample in enumerate(data):
            converted = convert_sample(sample, frames_dir, num_frames)

            if skip_missing_frames and not converted['images']:
                skipped += 1
                continue

            all_converted_data.append(converted)

            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(data)} samples...")

        total_skipped += skipped
        print(f"  Converted: {len(data) - skipped}, Skipped: {skipped}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Converted: {len(all_converted_data)}")
    print(f"Skipped: {total_skipped}")

    print("\nDataset distribution:")
    source_counts = Counter(s['data_source'] for s in all_converted_data)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("\nQuestion type distribution:")
    qtype_counts = Counter(s['question_type'] for s in all_converted_data)
    for qtype, count in sorted(qtype_counts.items()):
        print(f"  {qtype}: {count}")

    if shuffle:
        print(f"\nShuffling dataset with seed={seed}...")
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_converted_data)

    # Re-assign sequential IDs to ensure uniqueness across all datasets
    for i, sample in enumerate(all_converted_data):
        sample['id'] = i

    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_converted_data, f, indent=2, ensure_ascii=False)

    print("Done!")
    if all_converted_data:
        sample = all_converted_data[0]
        print(f"\nSample: id={sample['id']}, source={sample['data_source']}, "
              f"type={sample['question_type']}, images={len(sample['images'])}, "
              f"solution={sample['solution']}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert code_qa_generator output to GRPO training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--datasets', type=str, action='append', required=True,
                        help='Dataset spec: "name:json_path:frames_dir". Can be repeated.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames per video (default: 32)')
    parser.add_argument('--no_skip_missing', action='store_true',
                        help='Do not skip samples with missing frames')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the output dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    datasets = []
    for spec in args.datasets:
        parts = spec.split(':')
        if len(parts) != 3:
            parser.error(f"Invalid dataset spec: {spec}. Expected format: name:json_path:frames_dir")
        datasets.append(tuple(parts))

    process_and_save(
        datasets=datasets,
        output_path=args.output_path,
        num_frames=args.num_frames,
        skip_missing_frames=not args.no_skip_missing,
        shuffle=args.shuffle,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
