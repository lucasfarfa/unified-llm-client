"""Punto de entrada único de la librería: ``AsyncLLMManager``.

Patrones de diseño que aplica:
    - **Factory (fábrica):** a partir de ``config.provider`` decide QUÉ clase
      de cliente instanciar, usando un diccionario (registro) en lugar de una
      cadena de ``if/elif``.
    - **Facade (fachada):** expone una interfaz simple (``generate`` y
      ``stream``) y esconde los detalles de cada SDK. ``main.py`` habla solo
      con el Manager, nunca con ``OpenAIClient`` ni ``AnthropicClient``.
"""

import logging
import os
from collections.abc import AsyncIterator

from pydantic import ValidationError

from llm_client.anthropic_client import AnthropicClient
from llm_client.base import BaseLLMClient
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk

logger = logging.getLogger(__name__)

# Registro de proveedores: nombre -> clase del cliente.
# Para agregar un proveedor nuevo alcanza con implementar su cliente
# (subclase de BaseLLMClient) y sumar UNA línea acá (y a ProviderName en
# schemas.py). El resto del código no cambia.
PROVIDER_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

# Variable de entorno de donde se lee la API key de cada proveedor.
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Variable de entorno con el nombre del modelo de cada proveedor.
MODEL_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}

# Modelos por defecto, verificados en la documentación oficial (nov. 2026):
#   - OpenAI:    https://developers.openai.com/api/docs/models
#   - Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
# Elegimos los modelos chicos/económicos porque alcanzan para una demo.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-5-5",
}


class AsyncLLMManager:
    """Fachada asíncrona que unifica el acceso a distintos proveedores de LLM.

    Example:
        >>> manager = AsyncLLMManager.from_env(provider="openai")
        >>> response = await manager.generate([ChatMessage(role="user", content="Hola")])

    Attributes:
        config: Configuración validada del modelo.
    """

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        """Construye el manager y el cliente del proveedor indicado.

        Args:
            config: Configuración del modelo (validada por Pydantic).
            api_key: API key explícita. Si es None, se lee de la variable de
                entorno del proveedor (p. ej. ``OPENAI_API_KEY``).

        Raises:
            ValueError: Si el proveedor no está registrado o si falta la key.
                Fallamos al CONSTRUIR (y no al llamar a la API) para que un
                error de configuración se detecte lo antes posible.
        """
        self.config = config

        # Factory: buscamos la clase en el registro en vez de usar if/elif.
        client_class = PROVIDER_REGISTRY.get(config.provider)
        if client_class is None:
            supported = ", ".join(PROVIDER_REGISTRY)
            raise ValueError(
                f"Proveedor no soportado: '{config.provider}'. Opciones: {supported}."
            )

        key = api_key or os.getenv(API_KEY_ENV_VARS[config.provider])
        if not key:
            raise ValueError(
                f"Falta la API key de '{config.provider}'. Definí "
                f"{API_KEY_ENV_VARS[config.provider]} en el archivo .env."
            )

        self._client: BaseLLMClient = client_class(config, key)
        logger.info("Manager listo: provider=%s model=%s", config.provider, config.model)

    @classmethod
    def from_env(
        cls, provider: str | None = None, api_key: str | None = None
    ) -> "AsyncLLMManager":
        """Construye el manager leyendo la configuración de variables de entorno.

        ``@classmethod`` recibe la clase (``cls``) en vez de una instancia;
        se usa como "constructor alternativo": ``AsyncLLMManager.from_env()``.

        Variables leídas: ``LLM_PROVIDER``, ``OPENAI_MODEL`` /
        ``ANTHROPIC_MODEL``, ``LLM_TEMPERATURE``, ``LLM_MAX_TOKENS``,
        ``LLM_MAX_RETRIES``, ``LLM_TIMEOUT`` y la API key del proveedor.

        Args:
            provider: Proveedor a usar. Si es None, se usa ``LLM_PROVIDER``
                (default ``openai``).
            api_key: API key explícita (útil para pruebas). Si es None, se lee
                del entorno.

        Returns:
            Un ``AsyncLLMManager`` listo para usar.

        Raises:
            ValueError: Si el proveedor no es soportado, falta la API key o la
                configuración no pasa la validación de Pydantic.
        """
        provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
        if provider not in PROVIDER_REGISTRY:
            supported = ", ".join(PROVIDER_REGISTRY)
            raise ValueError(f"Proveedor no soportado: '{provider}'. Opciones: {supported}.")

        timeout = os.getenv("LLM_TIMEOUT")
        try:
            # Las variables de entorno son siempre strings ("0.7", "512").
            # Pydantic los convierte al tipo declarado (float, int) y valida
            # los rangos, así que no hace falta hacer float()/int() a mano.
            config = ModelConfig(
                provider=provider,
                model=os.getenv(MODEL_ENV_VARS[provider]) or DEFAULT_MODELS[provider],
                temperature=os.getenv("LLM_TEMPERATURE", "0.7"),
                max_tokens=os.getenv("LLM_MAX_TOKENS", "512"),
                max_retries=os.getenv("LLM_MAX_RETRIES", "2"),
                timeout=timeout or None,
            )
        except ValidationError as exc:
            # Re-lanzamos como ValueError con contexto, para que quien llama
            # sepa que el problema está en el .env.
            raise ValueError(f"Configuración inválida en el .env:\n{exc}") from exc

        return cls(config, api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa delegando en el cliente.

        Args:
            messages: Conversación a enviar.

        Returns:
            ``ModelResponse`` (con ``success=False`` si hubo error).
        """
        return await self._client.generate(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Emite la respuesta fragmento a fragmento delegando en el cliente.

        Es un generador asíncrono que "re-emite" cada chunk del cliente. Así
        ``main.py`` puede hacer ``async for chunk in manager.stream(...)``.

        Args:
            messages: Conversación a enviar.

        Yields:
            ``StreamChunk`` con texto, o uno final con ``error``.
        """
        async for chunk in self._client.stream(messages):
            yield chunk
