import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, AsyncGenerator, Generator, List
# from dotenv import load_dotenv
import os
import concurrent.futures
import functools
import httpx
import json
import re
import logging

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class CockpitLLMParams:
    temperature: float = 0.8
    max_tokens: int = 10000
    top_p: float = 1
    timeout: int = 180000
    stream: bool = False
    extend_params: Optional[dict] = None
    user_id: Optional[str] = None

    def to_dict(self):
        """转换为字典并过滤None值"""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        return data

    def to_camel_dict(self):
        return snake_to_camel_case(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        """从字典转换为对象"""
        return cls(**camel_to_snake_case(data))


async def async_call_cockpit_llm(messages, tools, tool_choice, model_name, llm_params: CockpitLLMParams | Dict = CockpitLLMParams()):
    """
    用于调用GPT/Claude/Gemini等外部模型

    Args:
        messages: 消息内容
        tools: 工具列表
        tool_choice: 工具模式
        model_name: 模型名称（支持友好名称，如 'gpt-4o', 'claude-3.5-sonnet' 等）
        llm_params: 模型配置, 支持字典和CockpitLLMParams对象
    """
    logger.info(f"[Cockpit] 开始调用 LLM，模型: {model_name}")
    url, headers, data = prepare_cockpit_request(messages, tools, tool_choice, model_name, llm_params)
    logger.debug(f"[Cockpit] 请求 URL: {url}")

    # data = {
    #     "model": 'gemini-2.5-pro',
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "What'\''s the weather like in Boston today?"
    #         }
    #     ],
    #     "tools": [
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "get_current_weather",
    #                 "description": "Get the current weather in a given location",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "location": {
    #                             "type": "string",
    #                             "description": "The city and state, e.g. San Francisco, CA"
    #                         },
    #                         "unit": {
    #                             "type": "string",
    #                             "enum": ["celsius", "fahrenheit"]
    #                         }
    #                     },
    #                     "required": ["location"]
    #                 }
    #             }
    #         }
    #     ],
    #     "tool_choice": "auto"
    # }

    # data = {
    #     "model": 'GEMINI-25-PRO',
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "What'\''s the weather like in Boston today?"
    #         }
    #     ],
    #     "tools": [
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "get_current_weather",
    #                 "description": "Get the current weather in a given location",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "location": {
    #                             "type": "string",
    #                             "description": "The city and state, e.g. San Francisco, CA"
    #                         },
    #                         "unit": {
    #                             "type": "string",
    #                             "enum": ["celsius", "fahrenheit"]
    #                         }
    #                     },
    #                     "required": ["location"]
    #                 }
    #             }
    #         }
    #     ],
    #     "tool_choice": "auto",
    #     'temperature': 0.8,
    #     'maxTokens': 10000,
    #     'topP': 1,
    #     'timeout': 180000,
    #     'stream': False,
    # }

    # print(data)

    try:
        logger.info(f"[Cockpit] 发送 HTTP 请求到 {url}...")
        res = await make_async_http_request(url=url, headers=headers, data=data)
        logger.info(f"[Cockpit] HTTP 请求完成，检查响应...")
        if not res.get("success", True):  # 默认为True，防止字段不存在的情况
            logger.error(f"[Cockpit] API 返回失败: {res}")
            raise RuntimeError(f"Cockpit API调用失败: {res}")
        logger.info(f"[Cockpit] API 调用成功，模型: {model_name}")
        return camel_to_snake_case(res["data"])
    except asyncio.TimeoutError as e:
        logger.error(f"[Cockpit] 请求超时，model_name: {model_name}")
        return None
    except Exception as e:
        logger.error(f"[Cockpit] 请求失败，model_name: {model_name} -> {data.get('model', 'unknown')}，错误信息: {str(e)}")
        return None


async def make_async_http_request(url, headers, data=None, method="post", retry_params=None, timeout=180, **kwargs):
    """
    发送异步HTTP请求
    :param url: 请求URL
    :param headers: 请求头
    :param data: 请求数据
    :param retry_params: 重试参数 dict，可包含 max_retries, delay, backoff, exceptions
    :param timeout: 超时时间（秒）
    :return: 响应结果
    """

    async def request():
        # 使用 httpx.Timeout 对象来设置更细粒度的超时
        timeout_config = httpx.Timeout(
            connect=30.0,      # 连接超时 30 秒
            read=timeout,      # 读取超时使用传入的 timeout
            write=30.0,        # 写入超时 30 秒
            pool=10.0          # 连接池超时 10 秒
        )
        logger.debug(f"[HTTP] 开始请求 {url}，超时设置: connect=30s, read={timeout}s")

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            if method.upper() == "POST":
                logger.debug(f"[HTTP] 发送 POST 请求...")
                response = await client.post(url, headers=headers, json=data, **kwargs)
            else:
                logger.debug(f"[HTTP] 发送 GET 请求...")
                response = await client.get(url, headers=headers, **kwargs)

            logger.debug(f"[HTTP] 收到响应，状态码: {response.status_code}")
            response.raise_for_status()  # 抛出HTTP错误以触发重试

            if response.status_code == 200:
                res = response.json()
                logger.debug(f"[HTTP] 响应解析成功")
                return res
            else:
                error_text = response.text[:500] if response.text else "无响应内容"
                logger.error(f"[HTTP] 非200状态码: {response.status_code}, 响应: {error_text}")
                raise Exception(f"请求返回非200状态码: {response.status_code}, {error_text}")

    # 默认重试参数
    retry_config = {
        'max_retries': 3,
        'delay': 1,
        'backoff': 2,
        'exceptions': (Exception,)
    }

    # 更新重试参数
    if retry_params:
        retry_config.update(retry_params)

    # 使用配置的参数执行重试
    return await async_retry(
        max_retries=retry_config['max_retries'],
        delay=retry_config['delay'],
        backoff=retry_config['backoff'],
        exceptions=retry_config['exceptions']
    )(request)()


def async_retry(func=None, max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    异步重试装饰器
    :param func: 异步函数
    :param max_retries: 最大重试次数
    :param delay: 初始延迟时间（秒）
    :param backoff: 退避系数，每次重试后延迟时间乘以该系数
    :param exceptions: 需要重试的异常类型
    :return: 函数执行结果
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while True:
                try:
                    logger.debug(f"[Retry] 执行请求，尝试 {retries + 1}/{max_retries}...")
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"[Retry] 已达到最大重试次数 ({max_retries})，放弃重试")
                        raise e

                    logger.warning(f"[Retry] 请求失败，正在进行第 {retries} 次重试，错误: {str(e)[:200]}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def prepare_cockpit_request(messages, tools, tool_choice, model_name, llm_params: Optional[CockpitLLMParams | Dict] = None):
    # cockpit_api_key = os.getenv("COCKPIT_API_KEY")
    cockpit_api_key = "eyJ0b2tlbiI6ImlZdU1vOG9qT1pPdjV6cmlRZWl3SUNrUUJjSWJ4UWxSIiwid29ya3NwYWNlSWQiOiIyMDI1MDcxNjAwMzAwMTI5NjgwIiwiYml6TmFtZSI6Ind1d2VpX2FpX3NlYXJjaCJ9"
    if not cockpit_api_key:
        raise ValueError("COCKPIT_API_KEY环境变量未设置或为空，请检查.env文件配置")

    url = 'https://ibotservice.alipayplus.com/api/v1/modelCall/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {cockpit_api_key}',
        'Origin': 'http://ilmprod-front-deploy_model-dev.page.alipay.net'
    }

    if isinstance(messages, str):
        messages = [
            {
                "role": "user",
                "content": messages
            }
        ]
    # Pass messages as-is; snake_to_camel_case would break multimodal
    # content keys like "image_url" -> "imageUrl"

    try:
        cockpit_model_name = get_cockpit_model_name(model_name)
    except ValueError as e:
        print(f"模型名称错误: {e}")
        return url, headers, {}

    # 处理参数
    if llm_params is None:
        llm_params = CockpitLLMParams()

    # 校验api key
    if isinstance(llm_params, CockpitLLMParams):
        params_dict = llm_params.to_camel_dict()
    elif isinstance(llm_params, dict):
        default_params = CockpitLLMParams(stream=False).to_camel_dict()
        params_dict = snake_to_camel_case(camel_to_snake_case(llm_params, max_depth=1))
        params_dict = {**default_params, **params_dict}
    else:
        raise ValueError(f"llm_params类型错误: {type(llm_params)}")

    data = {
        "model": cockpit_model_name,
        "messages": messages,
        **params_dict
    }
    if tools is not None:
        data["tools"] = tools
        data["tool_choice"] = tool_choice

    return url, headers, data


def camel_to_snake_case(data, max_depth=10, current_depth=0):
    # Prevent infinite recursion
    if current_depth > max_depth:
        return data

    # Handle None input
    if data is None:
        return None

    if isinstance(data, dict):
        # Only convert keys if they are strings, but leave values as is unless they're nested dicts/lists
        result = {}
        for k, v in data.items():
            # Convert key only if it's a string
            if isinstance(k, str):
                # Convert camelCase key to snake_case
                new_key = re.sub(r"([a-z])([A-Z])", r"\1_\2", k).lower()
            else:
                new_key = k

            # Only recursively convert values if they are dict or list
            if isinstance(v, (dict, list)):
                new_value = camel_to_snake_case(v, max_depth, current_depth + 1)
            else:
                # Keep primitive values (strings, numbers, booleans, None) as-is
                new_value = v

            result[new_key] = new_value
        return result
    elif isinstance(data, list):
        # Convert list elements only if they are dict or list
        return [
            (
                camel_to_snake_case(item, max_depth, current_depth + 1)
                if isinstance(item, (dict, list))
                else item
            )
            for item in data
        ]
    else:
        # For primitive types (str, int, float, bool, None), return as-is
        return data


def snake_to_camel_case(data, max_depth=10, current_depth=0):
    # Prevent infinite recursion
    if current_depth > max_depth:
        return data

    # Handle None input
    if data is None:
        return None

    if isinstance(data, dict):
        # Only convert keys if they are strings, but leave values as is unless they're nested dicts/lists
        result = {}
        for k, v in data.items():
            # Convert key only if it's a string
            if isinstance(k, str):
                # Convert snake_case key to camelCase
                components = k.split("_")
                new_key = components[0] + "".join(x.title() for x in components[1:])
            else:
                new_key = k

            # Only recursively convert values if they are dict or list
            if isinstance(v, (dict, list)):
                new_value = snake_to_camel_case(v, max_depth, current_depth + 1)
            else:
                # Keep primitive values (strings, numbers, booleans, None) as-is
                new_value = v

            result[new_key] = new_value
        return result
    elif isinstance(data, list):
        # Convert list elements only if they are dict or list
        return [
            (
                snake_to_camel_case(item, max_depth, current_depth + 1)
                if isinstance(item, (dict, list))
                else item
            )
            for item in data
        ]
    else:
        # For primitive types (str, int, float, bool, None), return as-is
        return data


"""
Cockpit模型映射配置
根据图片中的信息整理的模型映射关系
"""
import warnings

COCKPIT_MODEL_MAPPING = {
    # GPT系列
    "gpt-3.5-turbo": "GPT-35-TURBO-1106",
    "gpt-4o": "GPT4O",
    "gpt-4o-mini": "GPT4O-MINI",
    "o3-mini": "O3-MINI",
    "gpt-4.1": "GPT-41",
    "gpt-4o-intention-v32": "GPT4O-INTENTION-V32",
    "gpt-4.1-nano": "GPT-41-NANO",
    "gpt-4.1-mini": "GPT-41-MINI",

    # Qwen系列
    "qwen-turbo": "QWEN-TURBO",
    "qwen-plus": "QWEN-PLUS",
    "qwen-max": "QWEN-MAX",
    "qwen-long": "QWEN-LONG",
    "qwen-vl-max": "QWEN-VL-MAX",
    "qwen-vl-plus": "QWEN-VL-PLUS",
    "qwen2.5-72b-instruct": "QWEN-25-72B-INSTRUCT",
    "qwq-32b": "QWQ-32B",
    "qwen2.5-vl-72b-instruct": "QWEN-25-VL-72B-INSTRUCT",
    "qwen2.5-32b-instruct": "QWEN-25-32B-INSTRUCT",
    "qwen3-235b-a22b": "QWEN-3-235B-A22B",
    "qwen3-30b-a3b": "QWEN-3-30B-A3B",

    # Llama系列
    "llama-3.1-70b-instruct": "LLAMA-31-70B",
    "llama-3.1-8b-instruct": "LLAMA-31-8B",
    "llama-3.1-405b-instruct": "LLAMA-31-405B",

    # Gemini系列
    "gemini-1.5-pro": "GEMINI-15-PRO",
    "gemini-1.0-pro": "GEMINI-10-PRO",
    "gemini-1.5-flash": "GEMINI-15-FLASH",
    "gemini-2.0-pro": "GEMINI-20-PRO",
    "gemini-2.0-flash": "GEMINI-20-FLASH",
    "gemini-2.0-flash-thinking": "GEMINI-20-FLASH-THINKING",
    "gemini-2.5-pro": "GEMINI-25-PRO",

    # Claude系列
    "claude-3.5-sonnet": "CLAUDE-35-SONNET",
    "claude-3-sonnet": "CLAUDE-3-SONNET",
    "claude-3.5-sonnet-v2": "CLAUDE-35-SONNET-V2",
    "claude-3.5-haiku": "CLAUDE-35-HAIKU",
    "claude-3.7-sonnet": "CLAUDE-37-SONNET",

    # Bailling系列
    "bailling-65b-0315": "BAILLING-65B-0315",

    # DeepSeek系列
    "deepseek-r1": "DEEPSEEK-R1",
    "deepseek-v3": "BAILLING-DEEPSEEK-V3",
    "deepseek-r1-bailling": "BAILLING-DEEPSEEK-R1",
}

# 模型类型映射
MODEL_TYPES = {
    # 对话模型
    "DIALOGUE": [
        "GPT-35-TURBO-1106", "O3-MINI", "GPT4O-INTENTION-V32",
        "QWEN-TURBO", "QWEN-PLUS", "QWEN-MAX", "QWEN-LONG",
        "QWEN-25-72B-INSTRUCT", "QWQ-32B", "QWEN-25-32B-INSTRUCT",
        "QWEN-3-235B-A22B", "QWEN-3-30B-A3B",
        "LLAMA-31-70B", "LLAMA-31-8B", "LLAMA-31-405B",
        "GEMINI-10-PRO", "GEMINI-20-FLASH-THINKING",
        "CLAUDE-35-SONNET", "CLAUDE-3-SONNET", "CLAUDE-35-HAIKU", "CLAUDE-37-SONNET",
        "BAILLING-65B-0315", "DEEPSEEK-R1", "BAILLING-DEEPSEEK-V3", "BAILLING-DEEPSEEK-R1"
    ],

    # 多模态模型
    "MULTIMODAL": [
        "GPT4O", "GPT4O-MINI", "GPT-41", "GPT-41-NANO", "GPT-41-MINI",
        "QWEN-VL-MAX", "QWEN-VL-PLUS", "QWEN-25-VL-72B-INSTRUCT",
        "GEMINI-15-PRO", "GEMINI-15-FLASH", "GEMINI-20-PRO", "GEMINI-20-FLASH", "GEMINI-25-PRO",
        "CLAUDE-35-SONNET-V2"
    ]
}


def get_cockpit_model_name(user_model_name: str) -> str:
    """
    根据用户友好的模型名称获取Cockpit API需要的模型名称

    Args:
        user_model_name: 用户使用的模型名称

    Returns:
        Cockpit API对应的模型名称

    Raises:
        ValueError: 如果模型名称不支持
    """
    if user_model_name in COCKPIT_MODEL_MAPPING:
        return COCKPIT_MODEL_MAPPING[user_model_name]
    elif user_model_name in COCKPIT_MODEL_MAPPING.values():
        return user_model_name
    else:
        warnings.warn(f"model name not found: {user_model_name}, but still call Cockpit API")
        return user_model_name


def is_multimodal_model(model_name: str) -> bool:
    """
    判断模型是否支持多模态

    Args:
        model_name: 模型名称（用户友好名称或Cockpit模型名称）

    Returns:
        bool: 是否为多模态模型
    """
    cockpit_model_name = get_cockpit_model_name(model_name)
    return cockpit_model_name in MODEL_TYPES["MULTIMODAL"]


def get_supported_models():
    """获取所有支持的模型列表"""
    return list(COCKPIT_MODEL_MAPPING.keys())


def get_models_by_provider(provider: str):
    """
    根据提供商获取模型列表

    Args:
        provider: 提供商名称 (gpt, qwen, llama, gemini, claude, bailling, deepseek)

    Returns:
        list: 该提供商的模型列表
    """
    provider = provider.lower()
    models = []

    for user_name in COCKPIT_MODEL_MAPPING.keys():
        if user_name.startswith(provider):
            models.append(user_name)

    return models


class CockpitClient(LLMClient):
    """
    Cockpit API client for accessing GPT, Claude, Gemini, Qwen models.

    This client wraps the Cockpit async API into a synchronous interface
    compatible with the LLMClient base class.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.8,
        max_tokens: int = 10000,
        timeout: int = 180,
        max_retries: int = 3
    ):
        """
        Initialize Cockpit client.

        Args:
            model: Model name (e.g., 'gemini-2.5-pro', 'gpt-4o', 'claude-3.5-sonnet')
            temperature: Default sampling temperature
            max_tokens: Default max tokens
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self._model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate model name
        try:
            self._cockpit_model = get_cockpit_model_name(model)
        except ValueError as e:
            logger.warning(f"Model validation warning: {e}")
            self._cockpit_model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[Dict],
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None
    ) -> str:
        """
        Generate text using Cockpit API (synchronous wrapper).

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Maximum tokens (uses default if not specified)
            timeout: Request timeout (uses default if not specified)

        Returns:
            Generated text content

        Raises:
            RuntimeError: If API call fails after retries
        """
        import time
        start_time = time.time()

        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        req_timeout = timeout if timeout is not None else self.timeout

        logger.info(f"[CockpitClient] 开始生成，模型: {self._model}, 超时: {req_timeout}s")

        llm_params = CockpitLLMParams(
            temperature=temp,
            max_tokens=tokens,
            timeout=req_timeout * 1000,  # Convert to milliseconds
            stream=False
        )

        # Run async function in sync context
        try:
            logger.debug(f"[CockpitClient] 检查 event loop...")
            try:
                loop = asyncio.get_running_loop()
                logger.debug(f"[CockpitClient] 检测到运行中的 event loop，使用 nest_asyncio")
                # If there's already an event loop running, use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(
                    asyncio.wait_for(self._async_generate(messages, llm_params), timeout=req_timeout + 30)
                )
            except RuntimeError:
                # No event loop running
                logger.debug(f"[CockpitClient] 没有运行中的 event loop，创建新的")
                result = asyncio.run(
                    asyncio.wait_for(self._async_generate(messages, llm_params), timeout=req_timeout + 30)
                )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[CockpitClient] asyncio 超时，已等待 {elapsed:.2f}s")
            raise RuntimeError(f"Cockpit API call timed out after {elapsed:.2f}s for model {self._model}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[CockpitClient] 调用失败，已耗时 {elapsed:.2f}s，错误: {e}")
            raise

        elapsed = time.time() - start_time
        if result is None:
            logger.error(f"[CockpitClient] API 返回 None，已耗时 {elapsed:.2f}s")
            raise RuntimeError(f"Cockpit API call failed for model {self._model}")

        logger.info(f"[CockpitClient] 生成完成，耗时 {elapsed:.2f}s，响应长度: {len(result) if result else 0}")
        return result

    async def _async_generate(
        self,
        messages: List[Dict],
        llm_params: CockpitLLMParams
    ) -> Optional[str]:
        """Internal async generation method."""
        logger.debug(f"[CockpitClient] _async_generate 开始...")
        result = await async_call_cockpit_llm(
            messages=messages,
            tools=None,
            tool_choice=None,
            model_name=self._model,
            llm_params=llm_params
        )

        if result is None:
            logger.warning(f"[CockpitClient] async_call_cockpit_llm 返回 None")
            return None

        logger.debug(f"[CockpitClient] 解析响应，类型: {type(result)}")

        # Extract content from response
        if isinstance(result, dict):
            # Handle different response formats
            if 'choices' in result:
                choices = result.get('choices', [])
                logger.debug(f"[CockpitClient] 响应包含 choices，数量: {len(choices)}")
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    logger.debug(f"[CockpitClient] 提取到 content，长度: {len(content)}")
                    return content
            elif 'message' in result:
                content = result['message'].get('content', '')
                logger.debug(f"[CockpitClient] 从 message 提取 content，长度: {len(content)}")
                return content
            elif 'content' in result:
                logger.debug(f"[CockpitClient] 直接使用 content 字段")
                return result['content']
            else:
                logger.warning(f"[CockpitClient] 未知的响应格式，keys: {result.keys()}")

        logger.warning(f"[CockpitClient] 无法解析响应，将转换为字符串")
        return str(result) if result else None


if __name__ == "__main__":
    # Test async API
    result = asyncio.run(async_call_cockpit_llm(
        messages='hello',
        tools=None,
        tool_choice=None,
        model_name='gemini-2.5-pro',
    ))
    print("Async result:", result)

    # Test sync client
    client = CockpitClient(model='gemini-2.5-pro')
    response = client.generate([{"role": "user", "content": "Hello, how are you?"}])
    print("Sync result:", response)