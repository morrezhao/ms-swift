# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench: Benchmark for Spatial Reasoning in Video Understanding

Dataset: nyu-visionx/VSI-Bench on HuggingFace
Paper: Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

This module provides support for VSI-Bench evaluation in ms-swift.
VSI-Bench tests spatial reasoning capabilities through 8 tasks across 3 categories:
- Configurational: Spatial layout questions
- Measurement Estimation: Numerical distance/dimension questions
- Spatiotemporal: Questions involving time and space relationships

Answer types:
- Multiple-Choice Answer (MCA): Standard accuracy
- Numerical Answer (NA): Mean Relative Accuracy (MRA)
"""
import os
from typing import Any, Dict, Optional

from ..preprocessor import RowPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset


class VSIBenchPreprocessor(RowPreprocessor):
    """Preprocessor for VSI-Bench dataset.

    VSI-Bench format:
    {
        "id": 956,
        "dataset": "arkitscenes",
        "scene_name": "47430468",
        "question_type": "object_abs_distance",
        "question": "Measuring from the closest point...",
        "ground_truth": "1.9",
        "options": null  # or ["A. front-left", "B. back-right", ...]
    }
    """

    def __init__(
        self,
        *,
        num_frames: int = 32,
        video_dir: Optional[str] = None,
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """Initialize VSI-Bench preprocessor.

        Args:
            num_frames: Number of frames to sample from video (default: 32 for open-source MLLMs)
            video_dir: Directory containing video files or frames
            columns: Column mapping
        """
        super().__init__(columns=columns, **kwargs)
        self.num_frames = num_frames
        self.video_dir = video_dir

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a VSI-Bench sample.

        Args:
            row: Raw data row from the dataset

        Returns:
            Preprocessed row with messages and video/images
        """
        question = row.get('question', '')
        options = row.get('options')
        ground_truth = row.get('ground_truth', '')
        question_type = row.get('question_type', '')
        scene_name = row.get('scene_name', '')
        dataset_name = row.get('dataset', '')

        # Build the query with options if it's a multiple-choice question
        if options:
            # Multiple-choice question
            query = question + '\n' + '\n'.join(options)
            # Extract the letter answer (A, B, C, D)
            response = ground_truth
        else:
            # Numerical answer question
            query = question
            response = str(ground_truth)

        # Handle video/frames
        videos = None
        images = None

        # Try to find video or frames
        if self.video_dir:
            # Check for video file
            video_path = os.path.join(self.video_dir, dataset_name, scene_name)
            video_file = f'{video_path}.mp4'
            if os.path.exists(video_file):
                videos = [video_file]
            else:
                # Check for frame directory
                frame_dir = video_path
                if os.path.isdir(frame_dir):
                    # Load frames as images
                    frame_files = sorted([
                        os.path.join(frame_dir, f)
                        for f in os.listdir(frame_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))
                    ])
                    if frame_files:
                        # Sample frames uniformly
                        if len(frame_files) > self.num_frames:
                            indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
                            frame_files = [frame_files[i] for i in indices]
                        images = frame_files

        # Build messages
        messages = [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': response}
        ]

        result = {
            'messages': messages,
        }

        if videos:
            result['videos'] = videos
        elif images:
            result['images'] = images

        return result


class VSIBenchEvalPreprocessor(RowPreprocessor):
    """Preprocessor for VSI-Bench evaluation (inference only, no ground truth in messages).

    This preprocessor is used for evaluation where we only need the question
    and want the model to generate an answer.
    """

    def __init__(
        self,
        *,
        num_frames: int = 32,
        video_dir: Optional[str] = None,
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        super().__init__(columns=columns, **kwargs)
        self.num_frames = num_frames
        self.video_dir = video_dir

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a VSI-Bench sample for evaluation."""
        question = row.get('question', '')
        options = row.get('options')
        question_type = row.get('question_type', '')
        scene_name = row.get('scene_name', '')
        dataset_name = row.get('dataset', '')
        sample_id = row.get('id', '')
        ground_truth = row.get('ground_truth', '')

        # Build the query
        if options:
            query = question + '\n' + '\n'.join(options)
            # Add instruction for multiple-choice
            query += '\n\nPlease answer with just the letter (A, B, C, or D).'
        else:
            # Numerical answer - add instruction
            query += '\n\nPlease provide the numerical answer only.'

        # Handle video/frames
        videos = None
        images = None

        if self.video_dir:
            video_path = os.path.join(self.video_dir, dataset_name, scene_name)
            video_file = f'{video_path}.mp4'
            if os.path.exists(video_file):
                videos = [video_file]
            else:
                frame_dir = video_path
                if os.path.isdir(frame_dir):
                    frame_files = sorted([
                        os.path.join(frame_dir, f)
                        for f in os.listdir(frame_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))
                    ])
                    if frame_files:
                        if len(frame_files) > self.num_frames:
                            indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
                            frame_files = [frame_files[i] for i in indices]
                        images = frame_files

        # Only user message for evaluation
        messages = [
            {'role': 'user', 'content': query},
        ]

        result = {
            'messages': messages,
            # Store metadata for evaluation
            '__vsi_id': sample_id,
            '__vsi_question_type': question_type,
            '__vsi_ground_truth': ground_truth,
            '__vsi_is_multiple_choice': options is not None,
        }

        if videos:
            result['videos'] = videos
        elif images:
            result['images'] = images

        return result


# Register the VSI-Bench dataset
register_dataset(
    DatasetMeta(
        hf_dataset_id='nyu-visionx/VSI-Bench',
        subsets=[
            SubsetDataset(
                name='train',
                split=['train'],
                preprocess_func=VSIBenchPreprocessor(),
            ),
            SubsetDataset(
                name='eval',
                split=['test'],
                preprocess_func=VSIBenchEvalPreprocessor(),
            ),
        ],
        tags=['video', 'spatial-reasoning', 'multi-modal', 'vqa', 'benchmark'],
    ))
