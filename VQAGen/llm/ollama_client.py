"""Ollama Client Implementation"""

from typing import List, Dict, Optional
import requests

from .llm_client import LLMClient

# Default timeout in seconds
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_CONNECT_TIMEOUT = 30  # Connection timeout


class OllamaClient(LLMClient):
    """Ollama client for local LLM inference"""

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "qwen3:32b",
                 timeout: int = DEFAULT_TIMEOUT,
                 connect_timeout: int = DEFAULT_CONNECT_TIMEOUT):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Model name (e.g., 'llama3.1:70b', 'qwen2.5:72b')
            timeout: Request timeout in seconds (default: 300)
            connect_timeout: Connection timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self._model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self,
                 messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: Optional[int] = None) -> str:
        """
        Generate text using Ollama server.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Optional per-request timeout (uses default if not specified)

        Returns:
            Generated text content
        """
        # Use per-request timeout or default
        request_timeout = (self.connect_timeout, timeout or self.timeout)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            },
            timeout=request_timeout
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
