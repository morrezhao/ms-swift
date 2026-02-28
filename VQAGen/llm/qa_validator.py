"""
QA Validator — LLM-driven accept/reject validation for generated QA pairs.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .llm_client import LLMClient
from .mc_output_parser import MCOutputParser
from .mc_prompts import VALIDATION_SYSTEM_PROMPT, VALIDATION_USER_PROMPT_TEMPLATE
from .context_formatter import SceneContextFormatter

logger = logging.getLogger(__name__)


@dataclass
class QAValidatorConfig:
    """Configuration for QA validator."""
    temperature: float = 0.5
    fallback_to_rules: bool = True
    min_confidence: float = 0.3


class QAValidator:
    """
    LLM-driven QA validator.

    Evaluates question quality and decides accept/reject.
    MC options are provided by the code generator directly.
    """

    def __init__(self, llm_client: LLMClient, config: QAValidatorConfig = None):
        self.llm_client = llm_client
        self.config = config or QAValidatorConfig()
        self.output_parser = MCOutputParser()
        self.context_formatter = SceneContextFormatter()

    def validate_qa(
        self,
        question: str,
        question_type: str,
        ground_truth: Union[str, int, float],
        scene_info: Dict
    ) -> Tuple[bool, Optional[str], float]:
        """
        Validate a QA pair using LLM.

        Returns:
            Tuple of (accepted, rejection_reason, confidence)
        """
        scene_context = self._format_scene_context(scene_info)

        user_prompt = VALIDATION_USER_PROMPT_TEMPLATE.format(
            scene_context=scene_context,
            question_type=question_type,
            question=question,
            ground_truth=ground_truth
        )

        try:
            messages = [
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            llm_response = self.llm_client.generate(
                messages=messages,
                temperature=self.config.temperature
            )

            result = self.output_parser.parse(llm_response)

            if result.confidence < self.config.min_confidence:
                logger.warning(
                    f"Low confidence ({result.confidence:.2f}) "
                    f"for validation"
                )
                return (
                    False,
                    f"Low confidence: {result.confidence:.2f}",
                    result.confidence
                )

            if result.is_accepted():
                logger.info(
                    f"QA validated and accepted "
                    f"(confidence={result.confidence:.2f})"
                )
                return True, None, result.confidence
            else:
                logger.info(f"QA rejected: {result.rejection_reason}")
                return (
                    False,
                    result.rejection_reason,
                    result.confidence
                )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, f"Validation error: {str(e)}", 0.0

    def _format_scene_context(self, scene_info: Dict) -> str:
        """Format scene info into context string."""
        if not scene_info:
            return "No scene information available."

        parts = []

        object_counts = scene_info.get('object_counts', {})
        if object_counts:
            counts_str = ", ".join(
                f"{cat}: {count}"
                for cat, count in sorted(object_counts.items())
            )
            parts.append(f"Object counts: {counts_str}")

        if 'room_size' in scene_info:
            parts.append(f"Room size: {scene_info['room_size']} sqm")

        object_bboxes = scene_info.get('object_bboxes', {})
        if object_bboxes:
            parts.append(
                f"Object categories: {', '.join(object_bboxes.keys())}"
            )

        return (
            "\n".join(parts)
            if parts
            else "Scene metadata available but minimal details."
        )
