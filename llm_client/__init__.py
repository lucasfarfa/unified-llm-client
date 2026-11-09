"""Cliente asíncrono unificado para OpenAI y Anthropic.

Este archivo convierte la carpeta ``llm_client`` en un paquete de Python y
define qué se exporta. Así, desde afuera alcanza con::

    from llm_client import AsyncLLMManager, ChatMessage
"""

from llm_client.manager import AsyncLLMManager
from llm_client.schemas import (
    ChatMessage,
    LLMError,
    ModelConfig,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)

# ``__all__`` lista la API pública del paquete (lo que se importa con ``*``).
__all__ = [
    "AsyncLLMManager",
    "ChatMessage",
    "LLMError",
    "ModelConfig",
    "ModelResponse",
    "StreamChunk",
    "TokenUsage",
]
