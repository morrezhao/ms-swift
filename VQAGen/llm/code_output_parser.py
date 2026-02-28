"""Parser for code-based LLM output."""

import json
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CodeGeneratedQA:
    """Single QA pair with code for answer computation."""
    question_type: str
    question: str
    code: str
    # LLM-provided options for MC types (None for numeric types)
    options: Optional[List[str]] = None
    # Computed after execution
    computed_answer: Optional[any] = None

    def to_dict(self) -> dict:
        result = {
            'question_type': self.question_type,
            'question': self.question,
            'code': self.code,
        }
        if self.options is not None:
            result['options'] = self.options
        if self.computed_answer is not None:
            result['computed_answer'] = self.computed_answer
        return result


@dataclass
class CodeGenerationOutput:
    """Complete code-based LLM generation output."""
    generated_qas: List[CodeGeneratedQA]
    raw_output: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'generated_qas': [qa.to_dict() for qa in self.generated_qas],
        }


class CodeOutputParser:
    """Parser for code-based LLM JSON output."""

    # Basic question types (for reference, not enforced)
    BASIC_QUESTION_TYPES = {
        'object_counting',
        'object_abs_distance',
        'room_size',  # room_size uses point cloud, not bounding box
    }

    # Extended question types (complex, multi-hop, hybrid)
    EXTENDED_QUESTION_TYPES = {
        # Comparison
        'object_distance_comparison',
        'object_count_comparison',
        # Multi-hop reasoning
        'multi_hop_distance',
        'multi_hop_spatial',
        # Hybrid (combining multiple computations)
        'hybrid_distance_count',
        # Relative spatial
        'relative_position',
        'relative_direction',
        # Aggregation
        'closest_object',
        'farthest_object',
        # Custom (LLM can define new types)
        'custom',
    }

    def __init__(self, strict_types: bool = False):
        """
        Initialize parser.

        Args:
            strict_types: If True, only allow predefined question types.
                         If False (default), allow any question type.
        """
        self.strict_types = strict_types
        self.all_valid_types = self.BASIC_QUESTION_TYPES | self.EXTENDED_QUESTION_TYPES

    def parse(self, llm_output: str) -> CodeGenerationOutput:
        """
        Parse LLM output string into structured format.

        Args:
            llm_output: Raw LLM output string (expected to contain JSON with code)

        Returns:
            CodeGenerationOutput object

        Raises:
            ValueError: If parsing fails or output is invalid
        """
        # Extract JSON from output (handle markdown code blocks)
        json_str = self._extract_json(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # Parse generated QAs — support both flat and legacy array format
        generated_qas = []
        if 'generated_qas' in data:
            raw_qas = data['generated_qas']
            if not isinstance(raw_qas, list):
                raise ValueError("'generated_qas' must be a list")
        elif 'question_type' in data:
            raw_qas = [data]
        else:
            raise ValueError("Expected 'question_type' (flat format) or 'generated_qas' (array format)")

        for i, qa_data in enumerate(raw_qas):
            try:
                qa = self._parse_single_qa(qa_data)
                generated_qas.append(qa)
            except ValueError as e:
                raise ValueError(f"Error parsing QA {i}: {e}")

        return CodeGenerationOutput(
            generated_qas=generated_qas,
            raw_output=llm_output
        )

    def _extract_json(self, text: str, max_length: int = 500000) -> str:
        """
        Extract JSON from text, handling markdown code blocks.

        Uses bracket matching instead of greedy regex to avoid catastrophic backtracking.

        Args:
            text: Input text containing JSON
            max_length: Maximum text length to process (prevents memory issues)

        Returns:
            Extracted JSON string

        Raises:
            ValueError: If no valid JSON found
        """
        # Limit text length to prevent memory issues
        if len(text) > max_length:
            text = text[:max_length]

        # Remove thinking tags if present (for Qwen3 etc.)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

        # Try to find JSON in code blocks first (non-greedy)
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)

        if matches:
            # Return the first code block that looks like JSON
            for match in matches:
                stripped = match.strip()
                if stripped.startswith('{'):
                    return stripped

        # Use bracket matching instead of greedy regex to find JSON
        json_str = self._find_json_by_bracket_matching(text)
        if json_str:
            return json_str

        raise ValueError("No JSON object found in LLM output")

    def _find_json_by_bracket_matching(self, text: str) -> Optional[str]:
        """
        Find JSON object using bracket matching to avoid regex backtracking.

        Args:
            text: Input text

        Returns:
            JSON string or None if not found
        """
        # Find first '{'
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Use bracket matching to find the end
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

        return None

    def _parse_single_qa(self, qa_data: dict) -> CodeGeneratedQA:
        """Parse a single QA dict into CodeGeneratedQA object."""
        if not isinstance(qa_data, dict):
            raise ValueError("QA data must be a dictionary")

        # Required fields
        question_type = qa_data.get('question_type')
        question = qa_data.get('question')
        code = qa_data.get('code')

        if not question_type:
            raise ValueError("Missing 'question_type'")
        if not question:
            raise ValueError("Missing 'question'")
        if not code:
            raise ValueError("Missing 'code'")

        # Validate question type (only if strict mode)
        if self.strict_types and question_type not in self.all_valid_types:
            raise ValueError(
                f"Invalid question_type '{question_type}'. "
                f"Must be one of: {self.all_valid_types}"
            )

        # Validate code has compute_answer function
        if 'def compute_answer' not in code and 'compute_answer' not in code:
            raise ValueError(
                "Code must define a 'compute_answer()' function"
            )

        # Optional: options for MC types
        options = qa_data.get('options')
        if options is not None and not isinstance(options, list):
            options = None

        return CodeGeneratedQA(
            question_type=question_type,
            question=question,
            code=code,
            options=options
        )

    def extract_code_from_text(self, text: str) -> str:
        """
        Extract Python code from text (for cases where code is in markdown blocks).

        Args:
            text: Text potentially containing code blocks

        Returns:
            Extracted Python code
        """
        # Try python code blocks first
        python_block_pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(python_block_pattern, text)

        if matches:
            for match in matches:
                stripped = match.strip()
                if 'def compute_answer' in stripped:
                    return stripped

        # Return original if no code block found
        return text
