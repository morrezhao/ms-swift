"""vLLM Client Implementation using OpenAI-compatible API"""

import logging
from typing import List, Dict, Optional
from openai import OpenAI, APITimeoutError, APIConnectionError
import httpx

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default timeout in seconds
DEFAULT_TIMEOUT = 180  # 3 minutes (reduced from 5)
MAX_RETRIES = 2


class VLLMClient(LLMClient):
    """vLLM client using OpenAI-compatible API"""

    def __init__(self,
                 base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen3-32B",
                 api_key: str = "dummy",
                 timeout: int = DEFAULT_TIMEOUT,
                 max_retries: int = MAX_RETRIES):
        """
        Initialize vLLM client.

        Args:
            base_url: vLLM server URL (OpenAI-compatible endpoint)
            model: Model name/path
            api_key: API key (can be dummy for local vLLM)
            timeout: Request timeout in seconds (default: 180)
            max_retries: Maximum number of retries on connection errors
        """
        # Configure timeout for OpenAI client
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=30.0),
            max_retries=0  # We handle retries ourselves for better control
        )
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self,
                 messages: List[Dict],
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: Optional[int] = None) -> str:
        """
        Generate text using vLLM server.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Optional per-request timeout (uses default if not specified)

        Returns:
            Generated text content

        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Use per-request timeout if specified
        request_timeout = httpx.Timeout(timeout or self.timeout, connect=30.0)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout
                )
                return response.choices[0].message.content

            except APITimeoutError as e:
                last_error = e
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    continue
            except APIConnectionError as e:
                last_error = e
                logger.warning(f"LLM connection error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except Exception as e:
                last_error = e
                logger.error(f"LLM unexpected error: {e}")
                raise

        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")
