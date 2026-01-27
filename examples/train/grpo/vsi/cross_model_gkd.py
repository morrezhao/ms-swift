# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Cross-Model GKD Trainer for Multimodal Models.

This module patches the GKD trainer to support knowledge distillation between
different model families (e.g., Qwen2.5-VL -> Qwen3-VL) by processing images
separately for student and teacher models.

Both student and teacher see the images, but processed through their own visual encoders.

Usage:
    Add to --external_plugins in training command:
    --external_plugins examples/train/grpo/vsi/cross_model_gkd.py

    Set environment variable:
    CROSS_MODEL_GKD=1
"""
import os
import torch
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from swift.trainers import disable_gradient_checkpointing
from swift.utils import get_logger, to_device

logger = get_logger()

# Global storage for teacher template (initialized during training)
_teacher_template = None


def init_teacher_template(trainer):
    """Initialize teacher template if not already done."""
    global _teacher_template
    if _teacher_template is not None:
        return _teacher_template

    from swift.template import get_template
    from swift.utils import get_model_meta

    teacher_model = trainer.accelerator.unwrap_model(trainer.teacher_model)
    model_meta = get_model_meta(teacher_model)

    # Get template for teacher model
    _teacher_template = get_template(
        model_meta.template,
        trainer.processing_class,
        default_system=trainer.template.default_system,
        max_length=trainer.template.max_length,
        truncation_strategy=trainer.template.truncation_strategy,
        model=teacher_model,
    )
    logger.info(f"Initialized teacher template: {model_meta.template}")
    return _teacher_template


def encode_for_teacher(trainer, inputs: List[Dict], student_encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Re-encode inputs for teacher model using teacher's template.

    Args:
        trainer: The GKD trainer instance
        inputs: Original input data (with raw images)
        student_encoded: Student's encoded inputs (for reference)

    Returns:
        Teacher's encoded inputs with properly processed images
    """
    teacher_template = init_teacher_template(trainer)

    # Encode each input using teacher's template
    batch_encoded = []
    for data in inputs:
        encoded = teacher_template.encode(data, return_length=True)
        batch_encoded.append(encoded)

    # Collate into batch
    teacher_inputs = to_device(
        teacher_template.data_collator(batch_encoded),
        trainer.teacher_model.device
    )

    return teacher_inputs


def find_response_token_positions(input_ids: torch.Tensor, labels: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """Find positions of response tokens (where labels != -100).

    Args:
        input_ids: [batch_size, seq_len]
        labels: [batch_size, seq_len], -100 for non-response positions

    Returns:
        List of lists, each containing (position, token_id) tuples for response tokens
    """
    batch_size = input_ids.shape[0]
    response_info = []

    for b in range(batch_size):
        info = []
        for pos in range(labels.shape[1]):
            if labels[b, pos] != -100:
                token_id = input_ids[b, pos].item()
                info.append((pos, token_id))
        response_info.append(info)

    return response_info


def align_teacher_positions(
    teacher_input_ids: torch.Tensor,
    student_response_info: List[List[tuple]],
    student_seq_len: int
) -> List[List[int]]:
    """Find corresponding positions in teacher sequence for student's response tokens.

    Strategy: Align from the end of sequence (right-align).
    Since response tokens are at the end and have identical token IDs in both
    student and teacher, we just need to compute the offset from sequence end.

    Alignment diagram:
        Student: [visual_tokens_A][prompt][response]
                                              ^--- offset_from_end = 0
        Teacher: [visual_tokens_B][prompt][response]
                                              ^--- teacher_len - 1 - offset_from_end

    Args:
        teacher_input_ids: [batch_size, seq_len]
        student_response_info: List of [(pos, token_id), ...] for each batch item
        student_seq_len: Length of student sequence

    Returns:
        List of teacher positions corresponding to each student response token
    """
    batch_size = teacher_input_ids.shape[0]
    teacher_positions = []

    for b in range(batch_size):
        positions = []
        if not student_response_info[b]:
            teacher_positions.append(positions)
            continue

        teacher_len = teacher_input_ids.shape[1]

        # Right-align: compute offset from end for each response position
        for student_pos, _ in student_response_info[b]:
            # Offset from end in student sequence
            offset_from_end = student_seq_len - 1 - student_pos
            # Corresponding position in teacher sequence
            teacher_pos = teacher_len - 1 - offset_from_end
            if teacher_pos >= 0:
                positions.append(teacher_pos)

        teacher_positions.append(positions)

    return teacher_positions


def compute_loss_cross_model_with_images(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Compute loss where both student and teacher see images.

    Key: KL loss is computed only on student-generated tokens, with proper alignment.

    Alignment strategy:
    1. Identify response tokens in student's sequence (labels != -100)
    2. Find the same token sequence in teacher's output
    3. Compute KL only on those aligned positions
    """
    from swift.rlhf_trainers.gkd_trainer import DataSource

    data_source = inputs.pop('_data_source', DataSource.DATASET)
    raw_inputs = inputs.pop('_raw_inputs', None)

    model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
    student_input_ids = inputs['input_ids']
    student_labels = inputs['labels']

    # Student forward
    if self.args.sft_alpha > 0:
        model_inputs['labels'] = student_labels
    outputs_student = model(**model_inputs)
    model_inputs.pop('labels', None)

    # Find response token positions in student's sequence
    student_response_info = find_response_token_positions(student_input_ids, student_labels)

    # Teacher forward
    load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
    with torch.no_grad(), load_context, disable_gradient_checkpointing(
            self.teacher_model, self.args.gradient_checkpointing_kwargs):

        if raw_inputs is not None:
            try:
                teacher_inputs = encode_for_teacher(self, raw_inputs, model_inputs)
                teacher_input_ids = teacher_inputs['input_ids']
                outputs_teacher = self.teacher_model(**teacher_inputs)
            except Exception as e:
                logger.warning(f"Teacher encoding failed: {e}, using student inputs")
                teacher_input_ids = student_input_ids
                outputs_teacher = self.teacher_model(**model_inputs)
        else:
            teacher_input_ids = student_input_ids
            outputs_teacher = self.teacher_model(**model_inputs)

    # Find aligned positions in teacher's sequence (right-align)
    student_seq_len = student_input_ids.shape[1]
    teacher_positions = align_teacher_positions(teacher_input_ids, student_response_info, student_seq_len)

    # Extract logits at aligned positions and compute loss
    batch_size = student_input_ids.shape[0]
    all_student_logits = []
    all_teacher_logits = []

    for b in range(batch_size):
        if not student_response_info[b] or not teacher_positions[b]:
            continue

        # Get student logits at response positions (shifted by 1 for next-token prediction)
        student_positions = [pos for pos, _ in student_response_info[b]]

        # For next-token prediction, we need logits at position i to predict token at i+1
        # So we use positions [:-1] to predict tokens at positions [1:]
        if len(student_positions) > 1:
            stu_logit_positions = student_positions[:-1]  # Positions to get logits from
            stu_logits = outputs_student.logits[b, stu_logit_positions, :]  # [num_tokens-1, vocab]

            # Teacher positions (also shifted)
            tea_positions = teacher_positions[b]
            if len(tea_positions) > 1:
                tea_logit_positions = tea_positions[:-1]
                # Make sure positions are valid
                tea_logit_positions = [p for p in tea_logit_positions if p < outputs_teacher.logits.shape[1]]
                if tea_logit_positions:
                    tea_logits = outputs_teacher.logits[b, tea_logit_positions, :]

                    # Align lengths
                    min_len = min(stu_logits.shape[0], tea_logits.shape[0])
                    all_student_logits.append(stu_logits[:min_len])
                    all_teacher_logits.append(tea_logits[:min_len])

    if not all_student_logits:
        logger.warning("No aligned tokens found, returning zero loss")
        loss = outputs_student.logits.new_zeros(())
        if return_outputs:
            return (loss, outputs_student)
        return loss

    # Concatenate all logits
    shifted_student_logits = torch.cat(all_student_logits, dim=0)[None]  # [1, total_tokens, vocab]
    shifted_teacher_logits = torch.cat(all_teacher_logits, dim=0)[None]

    # Compute JSD loss
    loss = self.generalized_jsd_loss(
        student_logits=shifted_student_logits,
        teacher_logits=shifted_teacher_logits,
        beta=self.beta,
    )

    # Add SFT loss if enabled
    if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
        loss = loss + self.args.sft_alpha * outputs_student.loss

    if return_outputs:
        return (loss, outputs_student)
    return loss


# Store original methods
_original_compute_loss = GKDTrainer.compute_loss
_original_prepare_batch_inputs = GKDTrainer._prepare_batch_inputs


def patched_prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
    """Patched to store raw inputs for teacher re-encoding."""
    use_cross_model = os.environ.get('CROSS_MODEL_GKD', '0') == '1'

    # Call original method
    encoded = _original_prepare_batch_inputs(self, inputs, encode_prompt_only)

    # Store raw inputs for cross-model re-encoding
    if use_cross_model:
        encoded['_raw_inputs'] = deepcopy(inputs)

    return encoded


def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Patched compute_loss that handles cross-model distillation with images."""
    use_cross_model = os.environ.get('CROSS_MODEL_GKD', '0') == '1'

    if use_cross_model:
        return compute_loss_cross_model_with_images(self, model, inputs, return_outputs, num_items_in_batch)
    else:
        return _original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)


# Apply patches
GKDTrainer.compute_loss = patched_compute_loss
GKDTrainer._prepare_batch_inputs = patched_prepare_batch_inputs

cross_model_enabled = os.environ.get('CROSS_MODEL_GKD', '0') == '1'
logger.info(f"Cross-model GKD patch loaded. CROSS_MODEL_GKD={'enabled' if cross_model_enabled else 'disabled'}")
if cross_model_enabled:
    logger.info("Both student and teacher will see images (processed by their own visual encoders).")
