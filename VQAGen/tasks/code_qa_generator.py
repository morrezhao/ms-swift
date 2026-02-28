"""
Code-based LLM QA Generator.

This module generates QA pairs by having the LLM write Python code to compute answers.
The code is then executed against the scene metadata to ensure correctness.
"""

import os
import sys
import json
import argparse
import base64
import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm import LLMClient, VLLMClient, OllamaClient, SceneContextFormatter, CockpitClient
from llm.code_output_parser import CodeOutputParser, CodeGeneratedQA
from llm.code_prompts import (
    CODE_SYSTEM_PROMPT, CODE_USER_PROMPT_TEMPLATE,
    VALID_QUESTION_TYPES as PROMPT_VALID_TYPES,
    NUMERIC_ONLY_TYPES as PROMPT_NUMERIC_TYPES,
    MC_QUESTION_TYPES as PROMPT_MC_TYPES,
)
# MC prompts are used via qa_validator module
from llm.script_executor import ScriptExecutor, ExecutionResult
from utils.format_qa import get_frame_paths
import re

logger = logging.getLogger(__name__)


# Patterns to detect instance-index references in questions (Rule 4)
INSTANCE_INDEX_PATTERNS = [
    r'\binstance\s*\d+\b',  # "instance 1", "instance 2"
    r'\binstance\s+#?\d+\b',  # "instance #1"
    r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(table|chair|sofa|lamp|desk|bed|shelf|cabinet|door|window|plant|tv|monitor|screen|clock|picture|painting|mirror|curtain|rug|carpet|pillow|cushion|blanket|towel|bottle|cup|mug|glass|plate|bowl|vase|book|box|bag|basket|bin|trash|can|pot|pan|sink|toilet|shower|bathtub|fridge|refrigerator|oven|microwave|stove|washer|dryer|printer|computer|keyboard|mouse|phone|speaker|fan|heater|ac|air\s+conditioner)\b',  # "the second table"
    r'\b(the|a)\s+\d+(st|nd|rd|th)\s+\w+\b',  # "the 2nd chair"
]

# Compile patterns for efficiency
INSTANCE_INDEX_COMPILED = [re.compile(p, re.IGNORECASE) for p in INSTANCE_INDEX_PATTERNS]


# Valid question types from vsibench (excluding spatial_temporal_appearance_order)
# Import from code_prompts.py for consistency
VALID_QUESTION_TYPES = PROMPT_VALID_TYPES

# Question types that have numeric answers and don't need MC options
NUMERIC_ONLY_QUESTION_TYPES = PROMPT_NUMERIC_TYPES

# Question types that need MC options
MC_QUESTION_TYPES = PROMPT_MC_TYPES



def contains_instance_index_reference(question: str) -> bool:
    """
    Check if a question contains instance-index references.

    Rule 4: Do not generate questions that refer to object instance indices:
    - "instance 1 / instance 2 / the second table / the third chair"
    - any question requiring stable numbering of multiple instances

    Returns:
        True if the question contains forbidden instance references
    """
    for pattern in INSTANCE_INDEX_COMPILED:
        if pattern.search(question):
            return True
    return False


def is_valid_qa_item(qa: dict) -> bool:
    """
    Validate a QA item according to all rules.

    Returns:
        True if the QA item is valid, False if it should be discarded
    """
    question_type = qa.get('question_type', '')

    # Rule 1: Check for unknown/invalid ground truth
    gt = qa.get('ground_truth')
    if isinstance(gt, str):
        gt_lower = gt.lower().strip()
        if gt_lower in {'unknown', 'n/a', 'none', 'null', 'undefined', 'ambiguous', '', 'skip', 'equal'}:
            logger.warning(f"Discarding QA with invalid ground_truth: '{gt}' (question_type={question_type})")
            return False

    # Rule 4: Check for instance-index references in question
    question = qa.get('question', '')
    if contains_instance_index_reference(question):
        logger.warning(f"Discarding QA with instance-index reference: '{question[:100]}...' (question_type={question_type})")
        return False

    # For numeric-only question types, skip option validation
    if question_type in NUMERIC_ONLY_QUESTION_TYPES:
        logger.debug(f"Skipping option validation for numeric-only question_type={question_type}")
        # Discard object_abs_distance with 0.0m (overlapping bounding boxes)
        if question_type == 'object_abs_distance':
            try:
                if float(qa.get('ground_truth', '')) == 0.0:
                    logger.warning(f"Discarding object_abs_distance with 0.0m")
                    return False
            except (ValueError, TypeError):
                pass
        return True

    # Rule 2: Check for placeholder options
    options = qa.get('options', [])
    for opt in options:
        opt_text = opt.split('. ', 1)[-1] if '. ' in opt else opt
        opt_lower = opt_text.lower().strip()
        if opt_lower in {'option_1', 'option_2', 'option_3', 'option_4', 'n/a', 'unknown', 'ambiguous'}:
            logger.warning(f"Discarding QA with placeholder option: '{opt}' (question_type={question_type})")
            return False

    # Check that we have at least 2 options (only for non-numeric question types)
    if len(options) < 2:
        logger.warning(f"Discarding QA with fewer than 2 options: {options} (question_type={question_type})")
        return False

    # Check for duplicate options (Rule 5)
    normalized_options = []
    for opt in options:
        opt_text = opt.split('. ', 1)[-1] if '. ' in opt else opt
        # Normalize: remove trailing .0 for integers
        try:
            val = float(opt_text)
            if val == int(val):
                opt_text = str(int(val))
        except (ValueError, TypeError):
            pass
        normalized_options.append(opt_text.lower().strip())

    if len(normalized_options) != len(set(normalized_options)):
        logger.warning(f"Discarding QA with duplicate options: {options} (question_type={question_type})")
        return False

    return True


@dataclass
class CodeGenerationConfig:
    """Configuration for code-based LLM QA generation."""
    temperature: float = 0.7
    scene_timeout: int = 600  # 10 minutes per scene
    question_type: Optional[str] = None  # Specific question type to generate (None = mixed)
    num_frames: int = 0  # Number of scene frames to include as images (0 = text-only)


class CodeQAGenerator:
    """
    Code-based LLM QA Generator.

    The LLM generates Python code to compute answers, which is then executed
    to get verified answer values.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: CodeGenerationConfig = None,
        use_llm_mc: bool = False,
        mc_config: "QAValidatorConfig" = None
    ):
        """
        Initialize the Code QA Generator.

        Args:
            llm_client: LLM client instance (VLLMClient or OllamaClient)
            config: Generation configuration
            use_llm_mc: Whether to use LLM for MC option generation
            mc_config: Configuration for MC generator (only used if use_llm_mc=True)
        """
        self.llm_client = llm_client
        self.config = config or CodeGenerationConfig()
        self.context_formatter = SceneContextFormatter()
        self.output_parser = CodeOutputParser()
        self.script_executor = ScriptExecutor()

        self.option_letters = ['A', 'B', 'C', 'D']

        # LLM MC generation
        self.use_llm_mc = use_llm_mc
        self.qa_validator = None

        if use_llm_mc:
            from llm.qa_validator import QAValidator, QAValidatorConfig
            if mc_config is None:
                mc_config = QAValidatorConfig()
            self.qa_validator = QAValidator(llm_client, mc_config)

        # Will be set during run()
        self.scene_annos = None
        self.frame_annos = None
        self.all_qa_list = []
        self.processed_data_path = None
        self.dataset = None
        self.split_type = None

    def _load_json(self, file_path: str) -> Optional[Dict]:
        """Load JSON file."""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def generate_scene_qa(
        self,
        scene_name: str,
        scene_info: Dict,
        frame_info: Optional[Dict],
        question_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate a QA pair for a single scene using code execution.

        Args:
            scene_name: Name of the scene
            scene_info: Scene-level metadata
            frame_info: Frame-level metadata (optional)
            question_type: Specific question type to generate (overrides config)

        Returns:
            List of validated QA dictionaries
        """
        target_question_type = question_type or self.config.question_type

        # Format scene context for LLM
        scene_context = self.context_formatter.format_scene_context(
            scene_name, scene_info, frame_info
        )

        validated_qas = []

        logger.info(f"[Scene {scene_name}] Generating QA of type: {target_question_type or 'mixed'}")

        try:
            # Build user prompt
            user_prompt = CODE_USER_PROMPT_TEMPLATE.format(
                scene_context=scene_context,
            )
            # If a specific question type is requested, append constraint
            if target_question_type:
                user_prompt += f"\n\nIMPORTANT: Generate the QA pair using question_type = `{target_question_type}` ONLY."

            # Build user message content (text-only or multimodal with images)
            image_content_parts = None
            if self.config.num_frames > 0 and self.processed_data_path and self.split_type:
                frames_dir = os.path.join(self.processed_data_path, 'color', self.split_type)
                frame_paths = get_frame_paths(scene_name, frames_dir, num_frames=self.config.num_frames)
                if frame_paths:
                    image_content_parts = []
                    for fp in frame_paths:
                        try:
                            with open(fp, 'rb') as img_f:
                                b64 = base64.b64encode(img_f.read()).decode('utf-8')
                            image_content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                        except Exception as e:
                            logger.warning(f"[Scene {scene_name}] Failed to load frame {fp}: {e}")
                    if not image_content_parts:
                        image_content_parts = None
                    else:
                        logger.info(f"[Scene {scene_name}] Loaded {len(image_content_parts)} images")
                else:
                    logger.warning(f"[Scene {scene_name}] No frames found, falling back to text-only")

            if image_content_parts:
                user_content = image_content_parts + [{"type": "text", "text": user_prompt}]
                user_message = {"role": "user", "content": user_content}
            else:
                user_message = {"role": "user", "content": user_prompt}

            messages = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                user_message
            ]

            # Call LLM
            import time
            llm_start_time = time.time()
            logger.info(f"[Scene {scene_name}] 开始调用 LLM...")
            llm_response = self.llm_client.generate(
                messages=messages,
                temperature=self.config.temperature
            )
            llm_elapsed = time.time() - llm_start_time
            logger.info(f"[Scene {scene_name}] LLM 调用完成，耗时 {llm_elapsed:.2f}s，响应长度: {len(llm_response) if llm_response else 0}")

            # Parse output
            logger.debug(f"[Scene {scene_name}] 解析 LLM 输出...")
            parsed_output = self.output_parser.parse(llm_response)
            logger.info(f"[Scene {scene_name}] 解析完成，获得 {len(parsed_output.generated_qas)} 个 QA")

            # Execute code for each QA
            for qa_idx, qa in enumerate(parsed_output.generated_qas):
                question_type = qa.question_type
                logger.info(f"[Scene {scene_name}] 执行 QA {qa_idx + 1}/{len(parsed_output.generated_qas)}: question_type={question_type}")
                result = self._execute_qa_code(qa, scene_info, frame_info)

                if result.success:
                    # Execution passed - format for output
                    logger.info(f"[Scene {scene_name}] QA 执行成功: question_type={question_type}, result={result.result}")
                    qa.computed_answer = result.result
                    # Format QA (with optional LLM MC generation via qa_validator)
                    formatted_qa = self._format_validated_qa(
                        qa, scene_name, scene_info,
                        image_content_parts=image_content_parts
                    )
                    if formatted_qa is not None:
                        logger.info(f"[Scene {scene_name}] QA 格式化完成: question_type={question_type}")
                        validated_qas.append(formatted_qa)
                    else:
                        logger.warning(f"[Scene {scene_name}] QA 被丢弃: question_type={question_type}")
                else:
                    logger.warning(f"[Scene {scene_name}] QA 执行失败: question_type={question_type}, error={result.error_message}")

            logger.info(f"Scene {scene_name}: Generated {len(validated_qas)} valid QAs")

        except Exception as e:
            logger.error(f"Scene {scene_name}: Generation error - {e}")

        return validated_qas

    def _execute_qa_code(
        self,
        qa: CodeGeneratedQA,
        scene_info: Dict,
        frame_info: Optional[Dict],
    ) -> ExecutionResult:
        """Execute code for a single QA pair."""
        logger.debug(f"[ExecuteQA] 开始执行代码: {qa.question_type}")
        result = self.script_executor.execute(
            code=qa.code,
            scene_info=scene_info,
            frame_info=frame_info,
        )
        logger.debug(f"[ExecuteQA] 执行完成: success={result.success}, result={result.result}")
        return result

    def _format_validated_qa(
        self,
        qa: CodeGeneratedQA,
        scene_name: str,
        scene_info: Dict,
        image_content_parts: Optional[List] = None
    ) -> Optional[Dict]:
        """Format validated QA for final output.

        Args:
            qa: Parsed QA with computed answer
            scene_name: Name of the scene
            scene_info: Scene metadata
            image_content_parts: Optional image content parts for multimodal LLM validation

        Returns:
            Formatted QA dict, or None if the QA should be discarded
        """
        question_type = qa.question_type
        logger.info(f"[FormatQA] 开始格式化 QA: question_type={question_type}, answer={qa.computed_answer}")
        verified_answer = qa.computed_answer

        # Validate question_type
        if question_type not in VALID_QUESTION_TYPES:
            logger.warning(f"[FormatQA] Invalid question_type: '{question_type}', valid types: {VALID_QUESTION_TYPES}")
            return None

        # Discard object_abs_distance with 0.0m (overlapping bounding boxes)
        if question_type == 'object_abs_distance':
            try:
                if float(verified_answer) == 0.0:
                    logger.warning(f"[FormatQA] Discarding object_abs_distance with 0.0m distance")
                    return None
            except (ValueError, TypeError):
                pass

        # Rule 1: Check for unknown/invalid ground truth
        if isinstance(verified_answer, str):
            gt_lower = verified_answer.lower().strip()
            if gt_lower in {'unknown', 'n/a', 'none', 'null', 'undefined', 'ambiguous', '', 'skip', 'equal'}:
                logger.warning(f"[FormatQA] Discarding QA with invalid ground_truth: '{verified_answer}' (question_type={question_type})")
                return None

        # Rule 4: Check for instance-index references in question
        if contains_instance_index_reference(qa.question):
            logger.warning(f"[FormatQA] Discarding QA with instance-index reference: '{qa.question[:100]}...' (question_type={question_type})")
            return None

        # Check if this question type needs MC options
        is_numeric_only = question_type in NUMERIC_ONLY_QUESTION_TYPES

        if is_numeric_only:
            # Numeric-only question type: validate but skip MC generation
            logger.info(f"[FormatQA] question_type={question_type} is numeric-only")

            # Validate using LLM if enabled
            if self.use_llm_mc and self.qa_validator:
                logger.debug(f"[FormatQA] 使用 LLM 验证数值型 QA (question_type={question_type})...")
                accepted, rejection_reason, confidence = self.qa_validator.validate_qa(
                    question=qa.question,
                    question_type=question_type,
                    ground_truth=verified_answer,
                    scene_info=scene_info,
                    image_content_parts=image_content_parts
                )

                if not accepted:
                    logger.info(f"[FormatQA] LLM 拒绝数值型 QA: {rejection_reason} (question_type={question_type})")
                    return None

                logger.info(f"[FormatQA] LLM 接受数值型 QA (confidence={confidence:.2f})")

            formatted_qa = {
                "dataset": scene_info.get("dataset", self.dataset or "unknown"),
                "scene_name": scene_name,
                "question_type": question_type,
                "video_path": scene_info.get("video_path", ""),
                "question": qa.question,
                "options": [],  # No MC options for numeric-only
                "ground_truth": verified_answer,
                "mc_answer": None,  # No MC answer
                "code": qa.code,
                "generated_by": "llm_code"
            }
            logger.info(f"[FormatQA] Numeric QA formatted: question_type={question_type}, ground_truth={verified_answer}")
            return formatted_qa

        # Use LLM-provided options for MC types
        options = qa.options
        if not options or not isinstance(options, list) or len(options) < 2:
            logger.warning(f"[FormatQA] LLM 未提供有效选项，丢弃 (question_type={question_type})")
            return None

        # Ensure ground truth is included
        if str(verified_answer) not in [str(o) for o in options]:
            options.append(verified_answer)

        # Randomly shuffle options so answer position is uniformly distributed
        random.shuffle(options)
        mc_answer = self.option_letters[options.index(verified_answer)]

        # Validate with LLM if enabled
        if self.use_llm_mc and self.qa_validator:
            accepted, rejection_reason, confidence = self.qa_validator.validate_qa(
                question=qa.question,
                question_type=question_type,
                ground_truth=verified_answer,
                scene_info=scene_info,
                image_content_parts=image_content_parts
            )
            if not accepted:
                logger.info(f"[FormatQA] LLM 拒绝 MC QA: {rejection_reason} (question_type={question_type})")
                return None

        # Format options with letters
        formatted_options = [
            f"{self.option_letters[i]}. {opt}"
            for i, opt in enumerate(options)
        ]

        formatted_qa = {
            "dataset": scene_info.get("dataset", self.dataset or "unknown"),
            "scene_name": scene_name,
            "question_type": question_type,
            "video_path": scene_info.get("video_path", ""),
            "question": qa.question,
            "options": formatted_options,
            "ground_truth": verified_answer,
            "mc_answer": mc_answer,
            "code": qa.code,
            "generated_by": "llm_code"
        }

        # Final validation check
        if not is_valid_qa_item(formatted_qa):
            logger.warning(f"[FormatQA] QA failed final validation, discarding (question_type={question_type})")
            return None

        logger.info(f"[FormatQA] MC QA formatted successfully: question_type={question_type}, mc_answer={mc_answer}")
        return formatted_qa

    def run(
        self,
        processed_data_path: str,
        dataset: str,
        split_type: str,
        output_dir: str,
        num_workers: int = 1,
        file_name: str = "qa_code_generated.json",
        limit: int = None
    ):
        """
        Run QA generation for all scenes.

        Args:
            processed_data_path: Path to processed data directory
            dataset: Dataset name (e.g., 'scannet', 'scannetpp')
            split_type: Split type ('train', 'val', 'test')
            output_dir: Output directory for results
            num_workers: Number of parallel workers (for LLM calls)
            file_name: Output JSON file name (default: qa_code_generated.json)
            limit: Limit number of scenes to process (default: all)
        """
        # Store for later use
        self.processed_data_path = processed_data_path
        self.dataset = dataset
        self.split_type = split_type

        # Setup logging
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

        # Load metadata
        metadata_path = os.path.join(processed_data_path, 'metadata', split_type)
        dataset_lower = dataset.lower()
        scene_meta_path = os.path.join(
            metadata_path, f"{dataset_lower}_metadata_{split_type}.json"
        )
        frame_meta_path = os.path.join(
            metadata_path, f"{dataset_lower}_frame_metadata_{split_type}.json"
        )

        self.scene_annos = self._load_json(scene_meta_path)
        self.frame_annos = self._load_json(frame_meta_path)

        if not self.scene_annos:
            logger.error(f"Failed to load scene metadata from {scene_meta_path}")
            return

        logger.info(f"Loaded scene metadata: {len(self.scene_annos)} scenes")

        # Use all scenes from metadata
        scene_list = list(self.scene_annos.keys())
        if limit and limit > 0:
            scene_list = scene_list[:limit]
            logger.info(f"Limited to {len(scene_list)} scenes (limit={limit})")
        else:
            logger.info(f"Processing all {len(scene_list)} scenes")

        # Process scenes
        all_results = []
        total_answer_counts = Counter()

        def process_scene(scene_name: str):
            import time
            scene_start_time = time.time()
            scene_name = scene_name.strip()
            logger.info(f"[ProcessScene] 开始处理场景: {scene_name}")
            try:
                scene_info = self.scene_annos.get(scene_name, {})
                frame_info = self.frame_annos.get(scene_name, {}) if self.frame_annos else {}

                if not scene_info:
                    logger.warning(f"Scene {scene_name} not found in metadata")
                    return [], Counter()

                logger.debug(f"[ProcessScene] {scene_name}: 加载元数据完成，开始生成 QA...")
                qas = self.generate_scene_qa(scene_name, scene_info, frame_info)
                logger.info(f"[ProcessScene] {scene_name}: 生成了 {len(qas)} 个 QA，耗时 {time.time() - scene_start_time:.2f}s")

                # Count answers
                local_counts = Counter()
                for qa in qas:
                    mc = qa.get('mc_answer', '')
                    if mc in self.option_letters:
                        local_counts[mc] += 1

                return qas, local_counts

            except Exception as e:
                import traceback
                elapsed = time.time() - scene_start_time
                logger.error(f"[ProcessScene] {scene_name}: 处理失败，耗时 {elapsed:.2f}s，错误: {e}")
                logger.error(f"[ProcessScene] {scene_name}: 详细堆栈:\n{traceback.format_exc()}")
                return [], Counter()

        # Use ThreadPoolExecutor for I/O-bound LLM calls
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_scene, scene_name): scene_name
                    for scene_name in scene_list
                }

                # Process results with progress bar
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing scenes"
                ):
                    scene_name = futures[future]
                    try:
                        qas, counts = future.result(timeout=self.config.scene_timeout)
                        all_results.extend(qas)
                        total_answer_counts.update(counts)
                    except FutureTimeoutError:
                        logger.error(f"Scene {scene_name} timed out after {self.config.scene_timeout}s")
                    except Exception as e:
                        logger.error(f"Scene {scene_name} failed with error: {e}")
        else:
            for scene_name in tqdm.tqdm(scene_list, desc="Processing scenes"):
                try:
                    qas, counts = process_scene(scene_name)
                    all_results.extend(qas)
                    total_answer_counts.update(counts)
                except Exception as e:
                    logger.error(f"Scene {scene_name} failed with error: {e}")

        # Assign sequential IDs
        for i, qa in enumerate(all_results):
            qa["id"] = i

        self.all_qa_list = all_results
        self.answer_counts_log = dict(sorted(total_answer_counts.items()))

        # Count question types
        question_type_counts = Counter()
        numeric_only_count = 0
        mc_count = 0
        for qa in all_results:
            qt = qa.get('question_type', 'unknown')
            question_type_counts[qt] += 1
            if qt in NUMERIC_ONLY_QUESTION_TYPES:
                numeric_only_count += 1
            else:
                mc_count += 1

        logger.info(f"Total QA pairs generated: {len(self.all_qa_list)}")
        logger.info(f"Question type distribution: {dict(sorted(question_type_counts.items()))}")
        logger.info(f"Numeric-only QAs (no MC): {numeric_only_count}")
        logger.info(f"MC QAs: {mc_count}")
        logger.info(f"Answer distribution (MC only): {self.answer_counts_log}")

        # Save results
        self._save_results(output_dir, split_type, file_name)

    def run_multiple(
        self,
        dataset_specs: List[tuple],
        split_type: str,
        output_dir: str,
        num_workers: int = 1,
        file_name: str = "qa_code_generated.json",
        limit: int = None
    ):
        """
        Run QA generation for multiple datasets, combining results into one output file.

        Args:
            dataset_specs: List of (dataset_name, processed_data_path) tuples
            split_type: Split type ('train', 'val', 'test')
            output_dir: Output directory for results
            num_workers: Number of parallel workers
            file_name: Output JSON file name
            limit: Limit number of scenes per dataset
        """
        all_combined_results = []

        for dataset_name, processed_data_path in dataset_specs:
            logger.info(f"{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"  processed_data_path: {processed_data_path}")
            logger.info(f"{'='*60}")

            # Reset per-dataset state
            self.all_qa_list = []

            self.run(
                processed_data_path=processed_data_path,
                dataset=dataset_name,
                split_type=split_type,
                output_dir=output_dir,
                num_workers=num_workers,
                file_name=f"qa_code_generated_{dataset_name}.json",
                limit=limit
            )

            all_combined_results.extend(self.all_qa_list)

        # Re-assign sequential IDs across all datasets
        for i, qa in enumerate(all_combined_results):
            qa["id"] = i

        self.all_qa_list = all_combined_results
        logger.info(f"Combined total: {len(self.all_qa_list)} QA pairs across {len(dataset_specs)} datasets")

        # Save combined results
        self._save_results(output_dir, split_type, file_name)

    def _save_results(self, output_dir: str, split_type: str, file_name: str = "qa_code_generated.json"):
        """Save generated QA pairs to JSON file."""
        output_path = os.path.join(
            output_dir, split_type, file_name
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w') as f:
                json.dump(self.all_qa_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.all_qa_list)} QA pairs to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def create_llm_client(backend: str, base_url: str, model: str, timeout: int = 180):
    """
    Factory function to create LLM client.

    Args:
        backend: 'vllm', 'ollama', or 'cockpit' (for Gemini/GPT/Claude via Cockpit API)
        base_url: Server URL (not used for cockpit backend)
        model: Model name (e.g., 'Qwen/Qwen3-32B' for vllm, 'gemini-2.5-pro' for cockpit)
        timeout: Request timeout in seconds
    """
    if backend == 'cockpit':
        return CockpitClient(model=model, timeout=timeout)
    elif backend == 'vllm':
        return VLLMClient(base_url=base_url, model=model, timeout=timeout)
    elif backend == 'ollama':
        return OllamaClient(base_url=base_url, model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description='Code-based LLM QA Generator')

    # Data paths (single dataset mode)
    parser.add_argument('--processed_data_path', type=str, default=None,
                        help='Path to processed data directory (single dataset mode)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (single dataset mode: scannet, scannetpp, etc.)')

    # Multi-dataset mode
    parser.add_argument('--datasets', type=str, action='append', default=None,
                        help='Dataset spec: "name:processed_data_path". '
                             'Can be specified multiple times. All scenes from metadata are processed.')

    # Common
    parser.add_argument('--split_type', type=str, required=True,
                        help='Split type (train, val, test)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of scenes to process (default: all)')

    # LLM settings
    parser.add_argument('--llm_backend', type=str, default='vllm',
                        choices=['vllm', 'ollama', 'cockpit'],
                        help='LLM backend: vllm, ollama, or cockpit (for Gemini/GPT/Claude)')
    parser.add_argument('--llm_base_url', type=str,
                        default='http://localhost:8000/v1',
                        help='LLM server URL (not used for cockpit backend)')
    parser.add_argument('--llm_model', type=str,
                        default='Qwen/Qwen3-32B',
                        help='Model name (e.g., Qwen/Qwen3-32B for vllm, gemini-2.5-pro for cockpit)')

    # Image settings
    parser.add_argument('--num_frames', type=int, default=0,
                        help='Number of scene frames to send as images (0 = text-only, default: 0)')

    # Generation settings
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='LLM sampling temperature')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--llm_timeout', type=int, default=180,
                        help='Timeout per LLM call in seconds (default: 180)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--file_name', type=str, default='qa_code_generated.json',
                        help='Output JSON file name (default: qa_code_generated.json)')

    # Question type settings
    valid_types_str = ', '.join(sorted(VALID_QUESTION_TYPES))
    parser.add_argument('--question_type', type=str, default=None,
                        help=f'Specific question type to generate. Valid types: {valid_types_str}. '
                             'If not specified, generates a mix of all types.')

    # LLM MC generation settings
    parser.add_argument('--use_llm_mc', action='store_true',
                        help='Use LLM for MC option generation instead of rules')
    parser.add_argument('--mc_temperature', type=float, default=0.5,
                        help='Temperature for LLM MC generation (default: 0.5)')
    parser.add_argument('--mc_min_confidence', type=float, default=0.3,
                        help='Minimum confidence threshold for LLM MC acceptance (default: 0.3)')

    args = parser.parse_args()

    # 根据参数设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True  # 强制重新配置
    )
    # 减少 httpx 和 httpcore 的日志噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger.info(f"日志级别: {'DEBUG' if args.debug else 'INFO'}")

    # Create LLM client
    llm_client = create_llm_client(
        backend=args.llm_backend,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=args.llm_timeout
    )

    # Validate question_type if specified
    if args.question_type:
        if args.question_type not in VALID_QUESTION_TYPES:
            logger.error(f"Invalid question_type: {args.question_type}")
            logger.error(f"Valid types: {sorted(VALID_QUESTION_TYPES)}")
            return

    # Create config
    config = CodeGenerationConfig(
        temperature=args.temperature,
        question_type=args.question_type,
        num_frames=args.num_frames,
    )

    logger.info(f"Generation config: question_type={args.question_type or 'mixed'}, "
                f"use_llm_mc={args.use_llm_mc}, num_frames={args.num_frames}")

    # Create MC config if using LLM MC
    mc_config = None
    if args.use_llm_mc:
        from llm.qa_validator import QAValidatorConfig
        mc_config = QAValidatorConfig(
            temperature=args.mc_temperature,
            min_confidence=args.mc_min_confidence
        )
        logger.info(f"启用 LLM MC 验证: temperature={args.mc_temperature}")

    # Run generator
    generator = CodeQAGenerator(
        llm_client,
        config,
        use_llm_mc=args.use_llm_mc,
        mc_config=mc_config
    )

    if args.datasets:
        # Multi-dataset mode: name:processed_data_path
        dataset_specs = []
        for spec in args.datasets:
            parts = spec.split(':')
            if len(parts) != 2:
                logger.error(f"Invalid dataset spec: {spec}. Expected format: name:processed_data_path")
                return
            dataset_specs.append(tuple(parts))

        logger.info(f"Multi-dataset mode: {len(dataset_specs)} datasets")
        generator.run_multiple(
            dataset_specs=dataset_specs,
            split_type=args.split_type,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            file_name=args.file_name,
            limit=args.limit
        )
    elif args.processed_data_path and args.dataset:
        # Single dataset mode
        generator.run(
            processed_data_path=args.processed_data_path,
            dataset=args.dataset,
            split_type=args.split_type,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            file_name=args.file_name,
            limit=args.limit
        )
    else:
        parser.error("Either --datasets or (--processed_data_path, --dataset) must be specified")


if __name__ == '__main__':
    main()
