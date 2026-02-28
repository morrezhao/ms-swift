"""LLM Client Abstract Base Class"""

from abc import ABC, abstractmethod
from typing import List, Dict


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 4096) -> str:
        """
        Generate text from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used"""
        pass
