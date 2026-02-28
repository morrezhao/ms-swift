"""
MC Output Parser for LLM-driven multiple choice option generation.

This module parses LLM responses for MC option generation, handling
both accept/reject decisions and distractor generation.
"""

import json
import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class MCDecision(Enum):
    """Decision enum for MC generation."""
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class MCGenerationResult:
    """Result of MC generation from LLM."""
    decision: MCDecision
    rejection_reason: Optional[str]
    distractors: Optional[List[Union[str, int, float]]]
    confidence: float
    raw_output: Optional[str] = None

    def is_accepted(self) -> bool:
        """Check if the question was accepted."""
        return self.decision == MCDecision.ACCEPT

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "rejection_reason": self.rejection_reason,
            "distractors": self.distractors,
            "confidence": self.confidence,
            "raw_output": self.raw_output
        }


class MCOutputParser:
    """Parser for MC generation LLM output."""

    def __init__(self):
        pass

    def parse(self, llm_output: str) -> MCGenerationResult:
        """
        Parse LLM output for MC generation.

        Expected format:
        {
            "decision": "accept" | "reject",
            "rejection_reason": "reason if rejected, null otherwise",
            "distractors": ["opt1", "opt2", "opt3"] | null,
            "confidence": 0.0-1.0
        }

        Args:
            llm_output: Raw LLM output string

        Returns:
            MCGenerationResult with parsed data
        """
        if not llm_output:
            logger.warning("Empty LLM output for MC generation")
            return MCGenerationResult(
                decision=MCDecision.REJECT,
                rejection_reason="Empty LLM response",
                distractors=None,
                confidence=0.0,
                raw_output=llm_output
            )

        # Extract JSON from response (handle markdown code blocks)
        json_str = self._extract_json(llm_output)

        if not json_str:
            logger.warning(f"Failed to extract JSON from MC output: {llm_output[:200]}...")
            return MCGenerationResult(
                decision=MCDecision.REJECT,
                rejection_reason="Failed to parse LLM response as JSON",
                distractors=None,
                confidence=0.0,
                raw_output=llm_output
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return MCGenerationResult(
                decision=MCDecision.REJECT,
                rejection_reason=f"JSON decode error: {e}",
                distractors=None,
                confidence=0.0,
                raw_output=llm_output
            )

        return self._parse_json_data(data, llm_output)

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code block first
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                # Verify it's valid JSON-like
                json_str = json_str.strip()
                if json_str.startswith('{') and json_str.endswith('}'):
                    return json_str

        return None

    def _parse_json_data(self, data: dict, raw_output: str) -> MCGenerationResult:
        """Parse JSON data into MCGenerationResult."""
        # Parse decision
        decision_str = data.get('decision', '').lower().strip()
        if decision_str == 'accept':
            decision = MCDecision.ACCEPT
        elif decision_str == 'reject':
            decision = MCDecision.REJECT
        else:
            logger.warning(f"Unknown decision value: {decision_str}, defaulting to reject")
            decision = MCDecision.REJECT

        # Parse rejection reason
        rejection_reason = data.get('rejection_reason')
        if rejection_reason and not isinstance(rejection_reason, str):
            rejection_reason = str(rejection_reason)

        # Parse distractors
        distractors = data.get('distractors')
        if distractors is not None:
            if not isinstance(distractors, list):
                logger.warning(f"Distractors is not a list: {distractors}")
                distractors = None
            else:
                # Convert to appropriate types
                distractors = self._normalize_distractors(distractors)

        # Parse confidence
        confidence = data.get('confidence', 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return MCGenerationResult(
            decision=decision,
            rejection_reason=rejection_reason,
            distractors=distractors,
            confidence=confidence,
            raw_output=raw_output
        )

    def _normalize_distractors(
        self,
        distractors: List
    ) -> List[Union[str, int, float]]:
        """Normalize distractor values."""
        normalized = []
        for d in distractors:
            if d is None:
                continue

            # Try to convert to number if it looks like one
            if isinstance(d, str):
                d_stripped = d.strip()
                try:
                    # Try int first
                    if '.' not in d_stripped:
                        normalized.append(int(d_stripped))
                    else:
                        val = float(d_stripped)
                        # Keep as int if it's a whole number
                        if val == int(val):
                            normalized.append(int(val))
                        else:
                            normalized.append(val)
                except ValueError:
                    normalized.append(d_stripped)
            elif isinstance(d, (int, float)):
                # Keep int as int, convert float to int if whole number
                if isinstance(d, float) and d == int(d):
                    normalized.append(int(d))
                else:
                    normalized.append(d)
            else:
                normalized.append(str(d))

        return normalized
