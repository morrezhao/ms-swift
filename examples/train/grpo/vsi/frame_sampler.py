# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Frame Sampler Plugin for VSI-Bench GRPO Training.

This plugin allows dynamic frame sampling during training without re-preprocessing data.
Supports NUM_FRAMES = 1, 2, 4, 8, 16, 32 (must be power of 2 and <= 32).

Usage:
    In training script, set environment variable:
        export VSI_NUM_FRAMES=8

    Then use --custom_dataset_info to register the dataset with frame sampling.
"""
import os
from typing import Any, Dict, List, Optional

from swift.dataset import RowPreprocessor

# Valid frame counts (powers of 2, <= 32)
VALID_FRAME_COUNTS = {1, 2, 4, 8, 16, 32}


def uniform_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    """Get indices for uniform sampling from total_frames to num_frames.

    Args:
        total_frames: Total number of available frames
        num_frames: Number of frames to sample

    Returns:
        List of indices to sample
    """
    if num_frames >= total_frames:
        return list(range(total_frames))

    # Uniform sampling
    indices = []
    for i in range(num_frames):
        idx = int(i * total_frames / num_frames)
        indices.append(idx)
    return indices


def sample_frames_from_list(images: List[str], num_frames: int) -> List[str]:
    """Sample frames uniformly from image list.

    Args:
        images: List of image paths (typically 32 frames)
        num_frames: Number of frames to keep

    Returns:
        Sampled list of image paths
    """
    if not images:
        return images

    total_frames = len(images)
    if num_frames >= total_frames:
        return images

    indices = uniform_sample_indices(total_frames, num_frames)
    return [images[i] for i in indices]


class VSIFrameSamplerPreprocessor(RowPreprocessor):
    """Preprocessor that samples frames from VSI data.

    Reads NUM_FRAMES from environment variable VSI_NUM_FRAMES.
    Default is 32 (no sampling).
    """

    def __init__(self, num_frames: Optional[int] = None, **kwargs):
        """Initialize the preprocessor.

        Args:
            num_frames: Number of frames to sample. If None, reads from
                       VSI_NUM_FRAMES environment variable (default: 32).
        """
        super().__init__(**kwargs)
        self._num_frames = num_frames

    @property
    def num_frames(self) -> int:
        """Get the number of frames to sample."""
        if self._num_frames is not None:
            return self._num_frames
        return int(os.environ.get('VSI_NUM_FRAMES', '32'))

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a row by sampling frames.

        Args:
            row: Dataset row containing 'images' field

        Returns:
            Row with sampled images
        """
        num_frames = self.num_frames

        if num_frames not in VALID_FRAME_COUNTS:
            raise ValueError(
                f"VSI_NUM_FRAMES must be one of {sorted(VALID_FRAME_COUNTS)}, got {num_frames}"
            )

        # Sample frames from images list
        if 'images' in row and isinstance(row['images'], list):
            row['images'] = sample_frames_from_list(row['images'], num_frames)

        return row


def sample_frames(row: Dict[str, Any]) -> Dict[str, Any]:
    """Dataset transform function to sample frames.

    Reads NUM_FRAMES from environment variable VSI_NUM_FRAMES.
    Default is 32 (no sampling).

    Args:
        row: Dataset row containing 'images' field

    Returns:
        Row with sampled images
    """
    num_frames = int(os.environ.get('VSI_NUM_FRAMES', '32'))

    if num_frames not in VALID_FRAME_COUNTS:
        raise ValueError(
            f"VSI_NUM_FRAMES must be one of {sorted(VALID_FRAME_COUNTS)}, got {num_frames}"
        )

    if 'images' in row and isinstance(row['images'], list):
        row['images'] = sample_frames_from_list(row['images'], num_frames)

    return row


def get_frame_sampler(num_frames: int):
    """Factory function to create a frame sampler with specific frame count.

    Args:
        num_frames: Number of frames to sample (1, 2, 4, 8, 16, or 32)

    Returns:
        Transform function that samples frames
    """
    if num_frames not in VALID_FRAME_COUNTS:
        raise ValueError(
            f"num_frames must be one of {sorted(VALID_FRAME_COUNTS)}, got {num_frames}"
        )

    def transform(row: Dict[str, Any]) -> Dict[str, Any]:
        if 'images' in row and isinstance(row['images'], list):
            row['images'] = sample_frames_from_list(row['images'], num_frames)
        return row

    return transform


# Pre-defined samplers for convenience
sample_1_frame = get_frame_sampler(1)
sample_2_frames = get_frame_sampler(2)
sample_4_frames = get_frame_sampler(4)
sample_8_frames = get_frame_sampler(8)
sample_16_frames = get_frame_sampler(16)
sample_32_frames = get_frame_sampler(32)  # No-op, keeps all frames
