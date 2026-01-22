# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Evaluation Module

This module provides evaluation metrics and utilities for VSI-Bench benchmark.

Metrics:
- Multiple-Choice Answer (MCA): Standard accuracy
- Numerical Answer (NA): Mean Relative Accuracy (MRA)

Question Types in VSI-Bench:
1. Configurational (Multiple-Choice):
   - object_rel_direction_hard
   - object_rel_direction_easy
   - room_size_estimate (can be both)

2. Measurement Estimation (Numerical):
   - object_abs_distance
   - object_size_estimate
   - room_size_estimate

3. Spatiotemporal:
   - counting
   - appearance_order
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from swift.utils import get_logger

logger = get_logger()

# Question type categories
NUMERIC_QUESTION_TYPES = {
    'object_abs_distance',
    'object_size_estimate',
    'room_size_estimate',  # Can be numeric
}

MULTIPLE_CHOICE_QUESTION_TYPES = {
    'object_rel_direction_hard',
    'object_rel_direction_easy',
    'counting',
    'appearance_order',
}

# All question types
ALL_QUESTION_TYPES = NUMERIC_QUESTION_TYPES | MULTIPLE_CHOICE_QUESTION_TYPES


def extract_numeric_answer(response: str) -> Optional[float]:
    """Extract a numeric answer from model response.

    Args:
        response: Model's response string

    Returns:
        Extracted numeric value or None if not found
    """
    if response is None:
        return None

    response = response.strip()

    # Try to extract number directly
    # Match patterns like: "1.9", "1.9 meters", "approximately 1.9", etc.
    patterns = [
        r'[-+]?\d*\.?\d+',  # Basic number pattern
        r'(\d+(?:\.\d+)?)\s*(?:m|meters?|cm|centimeters?)?',  # With units
        r'(?:approximately|about|around|roughly)?\s*([-+]?\d*\.?\d+)',  # With qualifiers
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                # Take the first valid number found
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else None
                    if match:
                        return float(match)
            except (ValueError, TypeError):
                continue

    return None


def extract_choice_answer(response: str) -> Optional[str]:
    """Extract a multiple-choice answer (A, B, C, D) from model response.

    Args:
        response: Model's response string

    Returns:
        Extracted choice letter or None if not found
    """
    if response is None:
        return None

    response = response.strip().upper()

    # Direct match for single letter
    if response in ['A', 'B', 'C', 'D']:
        return response

    # Match patterns like "A.", "A)", "(A)", "Answer: A", etc.
    patterns = [
        r'^([A-D])\s*[\.\)\:]',  # A. or A) or A:
        r'\(([A-D])\)',  # (A)
        r'(?:answer|choice|option)[\s:]*([A-D])',  # Answer: A
        r'^([A-D])(?:\s|$)',  # A at start
        r'(?:^|\s)([A-D])(?:\s|$)',  # Standalone A
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Check if response starts with an option text
    for letter in ['A', 'B', 'C', 'D']:
        if response.startswith(letter + '.') or response.startswith(letter + ' '):
            return letter

    return None


def compute_mra(predictions: List[float], ground_truths: List[float], thresholds: Optional[List[float]] = None) -> float:
    """Compute Mean Relative Accuracy (MRA) for numerical answers.

    MRA is defined as per VSI-Bench paper (Thinking in Space):
    For confidence thresholds C = {0.5, 0.55, ..., 0.95} (10 thresholds),
    MRA = (1/|C|) * sum_{theta in C} 1(|pred - gt| / gt < 1 - theta)

    This means for each sample:
    - theta=0.5: relative error must be < 50% to score 1
    - theta=0.95: relative error must be < 5% to score 1
    - Average over all thresholds gives the sample's MRA score

    Args:
        predictions: List of predicted numeric values
        ground_truths: List of ground truth numeric values
        thresholds: List of confidence thresholds (default: [0.5, 0.55, ..., 0.95])

    Returns:
        MRA score between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError('Predictions and ground truths must have the same length')

    if len(predictions) == 0:
        return 0.0

    # Default thresholds from VSI-Bench paper
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    scores = []
    for pred, gt in zip(predictions, ground_truths):
        if pred is None or gt is None:
            scores.append(0.0)
            continue

        try:
            pred = float(pred)
            gt = float(gt)

            if gt == 0:
                # Handle zero ground truth case
                # If both are zero, perfect match
                if pred == 0:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
                continue

            # Compute relative error: |pred - gt| / |gt|
            relative_error = abs(pred - gt) / abs(gt)

            # For each threshold theta, check if relative_error < (1 - theta)
            # theta=0.5 -> tolerance=0.5 (50% error allowed)
            # theta=0.95 -> tolerance=0.05 (5% error allowed)
            threshold_scores = []
            for theta in thresholds:
                tolerance = 1.0 - theta
                if relative_error < tolerance:
                    threshold_scores.append(1.0)
                else:
                    threshold_scores.append(0.0)

            # Average over all thresholds for this sample
            sample_score = np.mean(threshold_scores)
            scores.append(sample_score)

        except (ValueError, TypeError):
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute accuracy for multiple-choice answers.

    Args:
        predictions: List of predicted choices (A, B, C, D)
        ground_truths: List of ground truth choices

    Returns:
        Accuracy score between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError('Predictions and ground truths must have the same length')

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred is not None and gt is not None:
            # Normalize both to uppercase single letter
            pred_norm = pred.strip().upper()[:1] if pred else ''
            gt_norm = gt.strip().upper()[:1] if gt else ''
            if pred_norm == gt_norm:
                correct += 1

    return correct / len(predictions)


class VSIBenchEvaluator:
    """Evaluator for VSI-Bench benchmark.

    This class handles the evaluation of model predictions on VSI-Bench,
    computing appropriate metrics for each question type.
    """

    def __init__(self, mra_thresholds: Optional[List[float]] = None):
        """Initialize the evaluator.

        Args:
            mra_thresholds: List of confidence thresholds for MRA computation
                           (default: [0.5, 0.55, ..., 0.95] as per VSI-Bench paper)
        """
        self.mra_thresholds = mra_thresholds
        self.results: List[Dict[str, Any]] = []

    def add_prediction(
        self,
        sample_id: Union[int, str],
        question_type: str,
        prediction: str,
        ground_truth: str,
        is_multiple_choice: bool,
        raw_response: Optional[str] = None,
    ):
        """Add a single prediction result.

        Args:
            sample_id: Unique identifier for the sample
            question_type: Type of question
            prediction: Model's prediction (extracted answer)
            ground_truth: Ground truth answer
            is_multiple_choice: Whether this is a multiple-choice question
            raw_response: Original model response before extraction
        """
        self.results.append({
            'id': sample_id,
            'question_type': question_type,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'is_multiple_choice': is_multiple_choice,
            'raw_response': raw_response,
        })

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all predictions and compute metrics.

        Returns:
            Dictionary containing evaluation metrics:
            - overall_score: Weighted average of MCA accuracy and NA MRA
            - mca_accuracy: Accuracy for multiple-choice questions
            - na_mra: MRA for numerical answer questions
            - by_question_type: Metrics broken down by question type
            - by_dataset: Metrics broken down by source dataset
        """
        if not self.results:
            logger.warning('No results to evaluate')
            return {}

        # Separate results by answer type
        mca_results = [r for r in self.results if r['is_multiple_choice']]
        na_results = [r for r in self.results if not r['is_multiple_choice']]

        # Compute MCA accuracy
        mca_accuracy = 0.0
        if mca_results:
            mca_preds = [extract_choice_answer(r['prediction']) for r in mca_results]
            mca_gts = [r['ground_truth'] for r in mca_results]
            mca_accuracy = compute_accuracy(mca_preds, mca_gts)

        # Compute NA MRA
        na_mra = 0.0
        if na_results:
            na_preds = [extract_numeric_answer(r['prediction']) for r in na_results]
            na_gts = [extract_numeric_answer(r['ground_truth']) for r in na_results]
            na_mra = compute_mra(na_preds, na_gts, self.mra_thresholds)

        # Compute overall score (weighted average based on sample count)
        total = len(self.results)
        mca_weight = len(mca_results) / total if total > 0 else 0
        na_weight = len(na_results) / total if total > 0 else 0
        overall_score = mca_weight * mca_accuracy + na_weight * na_mra

        # Compute metrics by question type
        by_question_type = {}
        for qt in set(r['question_type'] for r in self.results):
            qt_results = [r for r in self.results if r['question_type'] == qt]
            qt_mca = [r for r in qt_results if r['is_multiple_choice']]
            qt_na = [r for r in qt_results if not r['is_multiple_choice']]

            qt_metrics = {'count': len(qt_results)}

            if qt_mca:
                preds = [extract_choice_answer(r['prediction']) for r in qt_mca]
                gts = [r['ground_truth'] for r in qt_mca]
                qt_metrics['mca_accuracy'] = compute_accuracy(preds, gts)
                qt_metrics['mca_count'] = len(qt_mca)

            if qt_na:
                preds = [extract_numeric_answer(r['prediction']) for r in qt_na]
                gts = [extract_numeric_answer(r['ground_truth']) for r in qt_na]
                qt_metrics['na_mra'] = compute_mra(preds, gts, self.mra_thresholds)
                qt_metrics['na_count'] = len(qt_na)

            by_question_type[qt] = qt_metrics

        return {
            'overall_score': overall_score,
            'mca_accuracy': mca_accuracy,
            'mca_count': len(mca_results),
            'na_mra': na_mra,
            'na_count': len(na_results),
            'total_count': total,
            'by_question_type': by_question_type,
        }

    def save_results(self, output_path: str):
        """Save detailed results to a JSON file.

        Args:
            output_path: Path to save the results
        """
        output = {
            'evaluation': self.evaluate(),
            'predictions': self.results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f'VSI-Bench results saved to {output_path}')

    def print_report(self):
        """Print a formatted evaluation report."""
        metrics = self.evaluate()

        print('\n' + '=' * 60)
        print('VSI-Bench Evaluation Report')
        print('=' * 60)
        print(f"\nOverall Score: {metrics.get('overall_score', 0):.4f}")
        print(f"\nMultiple-Choice Answer (MCA):")
        print(f"  Accuracy: {metrics.get('mca_accuracy', 0):.4f}")
        print(f"  Count: {metrics.get('mca_count', 0)}")
        print(f"\nNumerical Answer (NA):")
        print(f"  MRA: {metrics.get('na_mra', 0):.4f}")
        print(f"  Count: {metrics.get('na_count', 0)}")

        print('\n' + '-' * 60)
        print('By Question Type:')
        print('-' * 60)
        for qt, qt_metrics in metrics.get('by_question_type', {}).items():
            print(f"\n  {qt}:")
            print(f"    Count: {qt_metrics.get('count', 0)}")
            if 'mca_accuracy' in qt_metrics:
                print(f"    MCA Accuracy: {qt_metrics['mca_accuracy']:.4f} (n={qt_metrics.get('mca_count', 0)})")
            if 'na_mra' in qt_metrics:
                print(f"    NA MRA: {qt_metrics['na_mra']:.4f} (n={qt_metrics.get('na_count', 0)})")

        print('\n' + '=' * 60)


def evaluate_vsi_bench(
    predictions: List[Dict[str, Any]],
    mra_thresholds: Optional[List[float]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to evaluate VSI-Bench predictions.

    Args:
        predictions: List of prediction dictionaries, each containing:
            - id: Sample ID
            - question_type: Type of question
            - prediction: Model's prediction
            - ground_truth: Ground truth answer
            - is_multiple_choice: Whether it's a multiple-choice question
        mra_thresholds: List of confidence thresholds for MRA computation
                        (default: [0.5, 0.55, ..., 0.95] as per VSI-Bench paper)
        output_path: Optional path to save detailed results

    Returns:
        Evaluation metrics dictionary
    """
    evaluator = VSIBenchEvaluator(mra_thresholds=mra_thresholds)

    for pred in predictions:
        evaluator.add_prediction(
            sample_id=pred['id'],
            question_type=pred['question_type'],
            prediction=pred['prediction'],
            ground_truth=pred['ground_truth'],
            is_multiple_choice=pred['is_multiple_choice'],
            raw_response=pred.get('raw_response'),
        )

    if output_path:
        evaluator.save_results(output_path)

    evaluator.print_report()
    return evaluator.evaluate()
