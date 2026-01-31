# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Cross-Model GKD Trainer for Multimodal Models (Fixed Signature).

Changes:
1. Fixed TypeError by matching `compute_loss` signature exactly with transformers implementation.
2. CLEARS hooks from the teacher model (removing Student's hooks).
3. DOES NOT register new hooks (relies on Qwen3-VL's native forward).
4. Re-encodes raw inputs using Teacher's processor.
"""
import os
import torch
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, Any, List, Tuple

from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from swift.trainers import disable_gradient_checkpointing
from swift.utils import get_logger, to_device

logger = get_logger()

_teacher_template = None
_hooks_cleared = False

def init_teacher_template(trainer):
    global _teacher_template
    if _teacher_template is not None:
        return _teacher_template

    from swift import get_processor, get_template

    teacher_path = getattr(trainer.args, 'teacher_model_id', '/upfs/models/Qwen/Qwen3-VL-8B-Instruct')
    if isinstance(teacher_path, list):
        teacher_path = teacher_path[0]
        
    logger.info(f"Initializing Teacher Template for: {teacher_path}")

    processor = get_processor(teacher_path, trust_remote_code=True)

    if hasattr(trainer.args, 'min_pixels') and trainer.args.min_pixels:
        if hasattr(processor, 'image_processor'):
            processor.image_processor.min_pixels = trainer.args.min_pixels

    if hasattr(trainer.args, 'max_pixels') and trainer.args.max_pixels:
        if hasattr(processor, 'image_processor'):
            processor.image_processor.max_pixels = trainer.args.max_pixels

    _teacher_template = get_template(processor)
    return _teacher_template


def clear_teacher_hooks(trainer):
    """
    只做清理，不注册。
    """
    global _hooks_cleared
    if _hooks_cleared:
        return

    teacher_model = trainer.teacher_model

    if hasattr(teacher_model, '_forward_pre_hooks'):
        num_hooks = len(teacher_model._forward_pre_hooks)
        if num_hooks > 0:
            logger.info(f"Clearing {num_hooks} hooks from Teacher Model.")
            teacher_model._forward_pre_hooks.clear()

    _hooks_cleared = True


def encode_for_teacher(trainer, inputs: List[Dict]) -> Dict[str, torch.Tensor]:
    teacher_template = init_teacher_template(trainer)

    batch_encoded = []
    for data in inputs:
        encoded = teacher_template.encode(data, return_length=True)
        batch_encoded.append(encoded)

    teacher_inputs = to_device(
        teacher_template.data_collator(batch_encoded),
        trainer.teacher_model.device
    )

    return teacher_inputs


def find_response_token_positions(input_ids: torch.Tensor, labels: torch.Tensor) -> List[List[Tuple[int, int]]]:
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


def align_teacher_positions(teacher_input_ids: torch.Tensor, student_response_info: List[List[tuple]], student_seq_len: int) -> List[List[int]]:
    batch_size = teacher_input_ids.shape[0]
    teacher_positions = []
    for b in range(batch_size):
        positions = []
        if not student_response_info[b]:
            teacher_positions.append(positions)
            continue
        teacher_len = teacher_input_ids.shape[1]
        for student_pos, _ in student_response_info[b]:
            offset_from_end = student_seq_len - 1 - student_pos
            teacher_pos = teacher_len - 1 - offset_from_end
            if teacher_pos >= 0:
                positions.append(teacher_pos)
        teacher_positions.append(positions)
    return teacher_positions


def compute_loss_cross_model_with_images(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    from swift.rlhf_trainers.gkd_trainer import DataSource

    clear_teacher_hooks(self)

    data_source = inputs.pop('_data_source', DataSource.DATASET)
    raw_inputs = inputs.pop('_raw_inputs', None)

    model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
    student_input_ids = inputs['input_ids']
    student_labels = inputs['labels']

    if self.args.sft_alpha > 0:
        model_inputs['labels'] = student_labels
    outputs_student = model(**model_inputs)
    model_inputs.pop('labels', None)

    load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
    with torch.no_grad(), load_context, disable_gradient_checkpointing(
            self.teacher_model, self.args.gradient_checkpointing_kwargs):

        if raw_inputs is not None:
            try:
                teacher_inputs = encode_for_teacher(self, raw_inputs)
                teacher_input_ids = teacher_inputs['input_ids']
                outputs_teacher = self.teacher_model(**teacher_inputs)
            except Exception as e:
                logger.error(f"Teacher forward failed: {e}")
                teacher_input_ids = student_input_ids
                outputs_teacher = self.teacher_model(**model_inputs)
        else:
            teacher_input_ids = student_input_ids
            outputs_teacher = self.teacher_model(**model_inputs)

    student_response_info = find_response_token_positions(student_input_ids, student_labels)
    student_seq_len = student_input_ids.shape[1]
    teacher_positions = align_teacher_positions(teacher_input_ids, student_response_info, student_seq_len)

    batch_size = student_input_ids.shape[0]
    all_student_logits = []
    all_teacher_logits = []

    for b in range(batch_size):
        if not student_response_info[b] or not teacher_positions[b]:
            continue
        student_indices = [pos for pos, _ in student_response_info[b]]
        teacher_indices = teacher_positions[b]
        
        valid_len = min(len(student_indices), len(teacher_indices))
        if valid_len > 0:
            stu_logits = outputs_student.logits[b, student_indices[:valid_len], :]
            tea_logits = outputs_teacher.logits[b, teacher_indices[:valid_len], :]
            all_student_logits.append(stu_logits)
            all_teacher_logits.append(tea_logits)

    if not all_student_logits:
        loss = outputs_student.logits.new_zeros(())
        if return_outputs:
            return (loss, outputs_student)
        return loss

    shifted_student_logits = torch.cat(all_student_logits, dim=0)[None]
    shifted_teacher_logits = torch.cat(all_teacher_logits, dim=0)[None]

    # 手动对齐词表维度
    s_v = shifted_student_logits.shape[-1]
    t_v = shifted_teacher_logits.shape[-1]
    if s_v != t_v:
        target_v = max(s_v, t_v)
        def pad_to_v(tensor, target):
            curr = tensor.shape[-1]
            if curr < target:
                # 补齐维度并填充极小值
                pad = torch.full((*tensor.shape[:-1], target - curr), -10000.0, 
                                 dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad], dim=-1)
            return tensor
        
        shifted_student_logits = pad_to_v(shifted_student_logits, target_v)
        shifted_teacher_logits = pad_to_v(shifted_teacher_logits, target_v)
        logger.info(f"Manual alignment applied: {s_v}/{t_v} -> {target_v}")

    loss = self.generalized_jsd_loss(
        student_logits=shifted_student_logits,
        teacher_logits=shifted_teacher_logits,
        beta=self.beta,
    )

    if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
        loss = loss + self.args.sft_alpha * outputs_student.loss

    if return_outputs:
        return (loss, outputs_student)
    return loss


# --- Fix: Use Explicit Function Definition Instead of Lambda ---
# We store the original method first
_original_compute_loss = getattr(GKDTrainer, '_original_compute_loss', GKDTrainer.compute_loss)
GKDTrainer._original_compute_loss = _original_compute_loss

# Explicitly define the patched method with the correct argument name
def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    use_cross_model = os.environ.get('CROSS_MODEL_GKD', '0') == '1'
    if use_cross_model:
        return compute_loss_cross_model_with_images(self, model, inputs, return_outputs, num_items_in_batch)
    else:
        # Forward to original, passing num_items_in_batch explicitly
        return _original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

GKDTrainer.compute_loss = patched_compute_loss


# Prepare Batch Inputs Patch
_original_prepare_batch_inputs = getattr(GKDTrainer, '_original_prepare_batch_inputs', GKDTrainer._prepare_batch_inputs)
GKDTrainer._original_prepare_batch_inputs = _original_prepare_batch_inputs

def patched_prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False):
    encoded = _original_prepare_batch_inputs(self, inputs, encode_prompt_only)
    if os.environ.get('CROSS_MODEL_GKD') == '1':
        encoded['_raw_inputs'] = deepcopy(inputs)
    return encoded

GKDTrainer._prepare_batch_inputs = patched_prepare_batch_inputs

logger.info("Cross-model GKD patch loaded (Signature Fixed).")