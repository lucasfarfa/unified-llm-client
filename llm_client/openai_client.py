"""Cliente asíncrono para OpenAI (Chat Completions API).

Implementa ``BaseLLMClient`` usando ``AsyncOpenAI``, la versión asíncrona del
SDK oficial. Nunca usamos ``OpenAI`` (la síncrona) dentro de un ``async def``:
bloquearía el event loop mientras espera la respuesta HTTP.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from llm_client.base import BaseLLMClient
from llm_client.schemas import (
    ChatMessage,
    LLMError,
    ModelConfig,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def map_openai_error(exc: Exception) -> LLMError:
    """Traduce una excepción del SDK de OpenAI a un ``LLMError`` normalizado.

    El ORDEN de los ``isinstance`` importa porque hay herencia entre
    excepciones del SDK:
      - ``APITimeoutError`` hereda de ``APIConnectionError``.
      - ``RateLimitError`` y ``AuthenticationError`` heredan de
        ``APIStatusError``.
    Por eso se chequean primero las más específicas.

    Args:
        exc: Excepción capturada.

    Returns:
        El ``LLMError`` equivalente.
    """
    # rate_limit -> retryable: superamos el límite de requests/tokens por
    # minuto; esperando un poco, la misma request funciona.
    if isinstance(exc, openai.RateLimitError):
        return LLMError(type="rate_limit", message=str(exc), provider="openai", retryable=True)
    # authentication -> NO retryable: la key es inválida; reintentar con la
    # misma key va a fallar siempre. Hay que corregir la configuración.
    if isinstance(exc, openai.AuthenticationError):
        return LLMError(type="authentication", message=str(exc), provider="openai", retryable=False)
    # timeout -> retryable: el servidor tardó demasiado, puede ser momentáneo.
    if isinstance(exc, openai.APITimeoutError):
        return LLMError(type="timeout", message=str(exc), provider="openai", retryable=True)
    # connection -> retryable: problema de red (DNS, sin internet, etc.).
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(type="connection", message=str(exc), provider="openai", retryable=True)
    # api_error -> depende del código HTTP: los 5xx son fallas del servidor
    # (transitorias, conviene reintentar); los 4xx son errores de NUESTRA
    # request (modelo inexistente, parámetro inválido): reintentar no sirve.
    if isinstance(exc, openai.APIStatusError):
        return LLMError(
            type="api_error",
            message=f"HTTP {exc.status_code}: {exc.message}",
            provider="openai",
            retryable=exc.status_code >= 500,
        )
    # unknown -> NO retryable: no sabemos qué pasó, asumimos lo seguro.
    return LLMError(type="unknown", message=repr(exc), provider="openai", retryable=False)


class OpenAIClient(BaseLLMClient):
    """Cliente de OpenAI que cumple la interfaz ``BaseLLMClient``."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        """Crea el cliente ``AsyncOpenAI`` con timeout y reintentos.

        Args:
            config: Configuración del modelo.
            api_key: API key de OpenAI.
        """
        super().__init__(config, api_key)
        # ``max_retries``: el SDK reintenta solo (con espera exponencial) los
        # errores transitorios: conexión, 408, 409, 429 y 5xx. Si se agotan los
        # reintentos, recién ahí lanza la excepción que capturamos abajo.
        # ``timeout``: si es None, el SDK usa su default (10 minutos).
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": config.max_retries}
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        self._client = AsyncOpenAI(**kwargs)

    def _build_params(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Arma los parámetros comunes a ``generate`` y ``stream``.

        Diferencia entre SDKs: OpenAI recibe los mensajes ``system`` dentro de
        la lista ``messages``, igual que los de ``user`` y ``assistant``. Por
        eso acá se pasan tal cual (Anthropic, en cambio, los separa).

        Args:
            messages: Conversación validada.

        Returns:
            Diccionario de parámetros para ``chat.completions.create``.
        """
        params: dict[str, Any] = {
            "model": self.config.model,
            # ``model_dump()`` convierte cada ChatMessage en un dict
            # {"role": ..., "content": ...}, el formato que espera la API.
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            # En Chat Completions el nombre vigente es ``max_completion_tokens``
            # (``max_tokens`` quedó deprecado y los modelos nuevos lo rechazan).
            "max_completion_tokens": self.config.max_tokens,
        }
        # Parámetros opcionales: solo se envían si están definidos.
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Pide una respuesta completa a OpenAI.

        Args:
            messages: Conversación a enviar.

        Returns:
            ``ModelResponse`` con el texto y el uso de tokens, o con
            ``success=False`` y el error si la llamada falló.
        """
        try:
            # ``await``: mientras esperamos la respuesta HTTP, el event loop
            # queda libre para atender otras tareas.
            completion = await self._client.chat.completions.create(
                **self._build_params(messages)
            )
            usage = None
            if completion.usage is not None:
                # Diferencia entre SDKs: OpenAI los llama prompt_tokens /
                # completion_tokens; los normalizamos a input/output.
                usage = TokenUsage(
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                )
            return ModelResponse(
                content=completion.choices[0].message.content,
                provider="openai",
                model=completion.model,
                usage=usage,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001 - la librería nunca debe romper
            # Un solo ``except Exception`` alcanza porque ``map_openai_error``
            # clasifica cada tipo (RateLimitError, APIConnectionError, etc.).
            error = map_openai_error(exc)
            logger.error("OpenAI generate falló (%s): %s", error.type, error.message)
            return ModelResponse(
                provider="openai", model=self.config.model, success=False, error=error
            )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Pide una respuesta a OpenAI y la emite fragmento a fragmento.

        Al tener ``yield`` dentro de ``async def``, este método es un
        generador asíncrono: se consume con ``async for``.

        Args:
            messages: Conversación a enviar.

        Yields:
            ``StreamChunk`` con texto; ante un error, un último chunk con
            ``error`` y el stream termina.
        """
        # El try envuelve TODO, incluido el ``async for``: el error puede
        # llegar al abrir la conexión o a mitad del stream (p. ej. se corta la
        # red después de recibir la mitad del texto).
        try:
            response_stream = await self._client.chat.completions.create(
                **self._build_params(messages), stream=True
            )
            # Cada iteración espera (sin bloquear) el próximo chunk de la red.
            async for chunk in response_stream:
                # Algunos chunks llegan sin ``choices`` (p. ej. el de uso de
                # tokens al final), así que los salteamos.
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                # El primer delta suele traer solo el ``role`` y el último solo
                # el ``finish_reason``: en ambos ``content`` es None.
                if text:
                    yield StreamChunk(text=text)
        except Exception as exc:  # noqa: BLE001
            error = map_openai_error(exc)
            logger.error("OpenAI stream falló (%s): %s", error.type, error.message)
            # En vez de lanzar la excepción, la emitimos como último chunk.
            yield StreamChunk(error=error)
