# Copyright (c) ModelScope Contributors. All rights reserved.
# VSI-Bench Reward Functions for GRPO Training
"""
Reward functions for VSI-Bench spatial reasoning tasks.

Question types:
- Numeric (use MRA): object_abs_distance, object_counting, object_size_estimation, room_size_estimation
- Multiple-choice (use 0/1): object_rel_direction_v1/v2/v3, object_rel_distance_v1/v2/v3
"""
import re
from typing import List, Optional

import numpy as np

from swift.rewards import ORM, orms

# Question type categories
NUMERIC_QUESTION_TYPES = {
    'object_abs_distance',
    'object_counting',
    'object_size_estimation',
    'room_size_estimation',
}

MC_QUESTION_TYPES = {
    'object_rel_direction_v1',
    'object_rel_direction_v2',
    'object_rel_direction_v3',
    'object_rel_distance_v1',
    'object_rel_distance_v2',
    'object_rel_distance_v3',
}


def extract_answer_from_tags(response: str) -> Optional[str]:
    """Extract answer from <answer> tags if present."""
    if response is None:
        return None
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_numeric_answer(response: str) -> Optional[float]:
    """Extract a numeric answer from model response.

    First tries to extract from <answer> tags (for CoT format),
    then falls back to VSI-Bench official extraction method.
    """
    if response is None:
        return None

    response = response.strip()
    if not response:
        return None

    # First try to extract from <answer> tags
    answer_content = extract_answer_from_tags(response)
    if answer_content is not None:
        response = answer_content

    # Try first word
    first_word = response.split(' ')[0].rstrip('.').strip()
    try:
        return float(first_word)
    except (ValueError, TypeError):
        pass

    # Fallback: find first number in the response
    match = re.search(r'[-+]?\d*\.?\d+', response)
    if match:
        try:
            return float(match.group())
        except (ValueError, TypeError):
            pass

    return None


def extract_choice_answer(response: str) -> Optional[str]:
    """Extract a multiple-choice answer (A/B/C/D) from model response.

    First tries to extract from <answer> tags (for CoT format),
    then falls back to first word extraction.
    """
    if response is None:
        return None

    response = response.strip()
    if not response:
        return None

    # First try to extract from <answer> tags
    answer_content = extract_answer_from_tags(response)
    if answer_content is not None:
        response = answer_content

    # Extract the first word (should be A, B, C, or D)
    answer = response.split(' ')[0].rstrip('.').strip()
    return answer if answer else None


def compute_mra_single(pred: float, gt: float, thresholds: List[float] = None) -> float:
    """Compute Mean Relative Accuracy (MRA) for a single sample.

    MRA is defined as per VSI-Bench paper:
    For confidence thresholds C = {0.5, 0.55, ..., 0.95} (10 thresholds),
    MRA = (1/|C|) * sum_{theta in C} 1(|pred - gt| / gt < 1 - theta)
    """
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if pred is None or gt is None:
        return 0.0

    try:
        pred = float(pred)
        gt = float(gt)

        if gt == 0:
            return 1.0 if pred == 0 else 0.0

        # Compute relative error
        relative_error = abs(pred - gt) / abs(gt)

        # For each threshold theta, check if relative_error < (1 - theta)
        threshold_scores = []
        for theta in thresholds:
            tolerance = 1.0 - theta
            if relative_error < tolerance:
                threshold_scores.append(1.0)
            else:
                threshold_scores.append(0.0)

        return float(np.mean(threshold_scores))

    except (ValueError, TypeError):
        return 0.0


class VSIRewardFunction(ORM):
    """
    Reward function for VSI-Bench training.

    For numeric questions: uses MRA (Mean Relative Accuracy)
    For multiple-choice questions: uses 0/1 accuracy
    """

    def __call__(self, completions, solution, question_type, **kwargs) -> List[float]:
        """
        Compute rewards for VSI-Bench samples.

        Args:
            completions: List of model generated responses
            solution: List of ground truth answers
            question_type: List of question types

        Returns:
            List of reward scores
        """
        rewards = []

        for completion, gt, qtype in zip(completions, solution, question_type):
            if qtype in NUMERIC_QUESTION_TYPES:
                # Use MRA for numeric questions
                pred_value = extract_numeric_answer(completion)
                gt_value = extract_numeric_answer(str(gt))
                reward = compute_mra_single(pred_value, gt_value)
            elif qtype in MC_QUESTION_TYPES:
                # Use 0/1 accuracy for multiple-choice questions
                pred_answer = extract_choice_answer(completion)
                gt_answer = extract_choice_answer(str(gt))
                if pred_answer is not None and gt_answer is not None:
                    reward = 1.0 if pred_answer.upper() == gt_answer.upper() else 0.0
                else:
                    reward = 0.0
            else:
                # Unknown question type, default to string match
                reward = 1.0 if completion.strip() == str(gt).strip() else 0.0

            rewards.append(reward)

        return rewards


# Register the reward function
orms['vsi_reward'] = VSIRewardFunction
