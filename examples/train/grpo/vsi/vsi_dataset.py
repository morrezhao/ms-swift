# Copyright (c) ModelScope Contributors. All rights reserved.
# VSI-Bench Dataset Registration for GRPO Training
"""
Dataset preprocessor for VSI-Bench training data.

This converts video paths to pre-extracted frame paths (32 frames).
The frames should be extracted in advance and stored in a directory structure like:
    frames_dir/
        scene0191_00/
            000000.jpg
            000179.jpg
            ...  (32 frames total, sorted order = temporal order)
"""
import os
from typing import Any, Dict, List

from swift.dataset import register_dataset, DatasetMeta, SubsetDataset
from swift.dataset.preprocessor import ResponsePreprocessor


class VSIPreprocessor(ResponsePreprocessor):
    """Preprocessor for VSI-Bench GRPO training data.

    Converts video field to images field (list of frame paths).
    Passes through question_type for reward function.
    """

    def __init__(self, frames_dir: str = None, num_frames: int = 32):
        """
        Args:
            frames_dir: Base directory containing extracted frames.
                        If None, will try to infer from video path.
            num_frames: Number of frames to use (default: 32)
        """
        super().__init__()
        self.frames_dir = frames_dir
        self.num_frames = num_frames

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Get video path and extract scene name
        video_path = row.get('video', '')
        scene_name = os.path.splitext(os.path.basename(video_path))[0]

        # Determine frames directory
        if self.frames_dir:
            scene_frames_dir = os.path.join(self.frames_dir, scene_name)
        else:
            # Assume frames are in a parallel 'frames' directory
            video_dir = os.path.dirname(video_path)
            base_dir = os.path.dirname(video_dir)
            scene_frames_dir = os.path.join(base_dir, 'frames', scene_name)

        # Get frame paths (sorted order = temporal order)
        images = []
        if os.path.isdir(scene_frames_dir):
            frame_files = sorted([
                f for f in os.listdir(scene_frames_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            # Use all frames directly (assuming pre-extracted with correct count)
            # Uniformly sample only if more frames than expected
            if len(frame_files) > self.num_frames:
                indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
                frame_files = [frame_files[i] for i in indices]
            images = [os.path.join(scene_frames_dir, f) for f in frame_files]

        # Build the output row
        # Extract query and response from conversations
        conversations = row.get('conversations', [])
        query = ''
        response = ''
        for conv in conversations:
            if conv.get('from') == 'human':
                query = conv.get('value', '')
            elif conv.get('from') == 'gpt':
                response = conv.get('value', '')

        # Update row with processed data
        row.update({
            'query': query,
            'response': response,
            'images': images,
            # Keep solution for reward function (same as response/ground_truth)
            'solution': response,
            # Keep question_type for reward function
            'question_type': row.get('question_type', ''),
        })

        return super().preprocess(row)


def register_vsi_dataset(
    dataset_path: str,
    frames_dir: str,
    num_frames: int = 32,
    dataset_name: str = 'vsi_train'
):
    """Register VSI dataset for GRPO training.

    Args:
        dataset_path: Path to the JSON dataset file
        frames_dir: Directory containing extracted frames
        num_frames: Number of frames per video (default: 32)
        dataset_name: Name to register the dataset as
    """
    register_dataset(
        DatasetMeta(
            ms_dataset_id=dataset_path,  # Use local path
            subsets=[
                SubsetDataset(
                    name='default',
                    subset='default',
                    split=['train'],
                ),
            ],
            preprocess_func=VSIPreprocessor(
                frames_dir=frames_dir,
                num_frames=num_frames
            ),
            tags=['qa', 'spatial', 'video', 'vsi'],
        ),
        dataset_name=dataset_name
    )
