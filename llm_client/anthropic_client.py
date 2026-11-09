"""Cliente asíncrono para Anthropic (Messages API).

Implementa ``BaseLLMClient`` usando ``AsyncAnthropic``, la versión asíncrona
del SDK oficial. Acá se resuelven las diferencias de Anthropic respecto de
OpenAI: los mensajes ``system`` van aparte, ``max_tokens`` es obligatorio y el
streaming se maneja con un context manager asíncrono.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

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


def map_anthropic_error(exc: Exception) -> LLMError:
    """Traduce una excepción del SDK de Anthropic a un ``LLMError``.

    Igual que en OpenAI, el orden importa por la herencia entre excepciones:
    ``APITimeoutError`` hereda de ``APIConnectionError``, y ``RateLimitError``
    y ``AuthenticationError`` heredan de ``APIStatusError``.

    Args:
        exc: Excepción capturada.

    Returns:
        El ``LLMError`` equivalente.
    """
    # rate_limit -> retryable: es temporal, esperar y reintentar funciona.
    if isinstance(exc, anthropic.RateLimitError):
        return LLMError(type="rate_limit", message=str(exc), provider="anthropic", retryable=True)
    # authentication -> NO retryable: la key está mal, hay que corregirla.
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMError(
            type="authentication", message=str(exc), provider="anthropic", retryable=False
        )
    # timeout -> retryable: el servidor tardó, puede ser momentáneo.
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError(type="timeout", message=str(exc), provider="anthropic", retryable=True)
    # connection -> retryable: problema de red.
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError(type="connection", message=str(exc), provider="anthropic", retryable=True)
    # api_error -> 5xx (incluye 529 "overloaded" de Anthropic) es transitorio;
    # 4xx es un error de nuestra request y reintentar no lo arregla.
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(
            type="api_error",
            message=f"HTTP {exc.status_code}: {exc.message}",
            provider="anthropic",
            retryable=exc.status_code >= 500,
        )
    # unknown -> NO retryable: ante la duda, no reintentamos.
    return LLMError(type="unknown", message=repr(exc), provider="anthropic", retryable=False)


class AnthropicClient(BaseLLMClient):
    """Cliente de Anthropic que cumple la interfaz ``BaseLLMClient``."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        """Crea el cliente ``AsyncAnthropic`` con timeout y reintentos.

        Args:
            config: Configuración del modelo.
            api_key: API key de Anthropic.
        """
        super().__init__(config, api_key)
        # ``max_retries``: el SDK reintenta solo los errores transitorios
        # (conexión, 408, 409, 429 y 5xx) con espera exponencial.
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": config.max_retries}
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        self._client = AsyncAnthropic(**kwargs)

    def _build_params(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Arma los parámetros comunes a ``generate`` y ``stream``.

        Diferencias con OpenAI que se resuelven acá:
          1. Anthropic NO acepta mensajes con rol ``system`` dentro de
             ``messages``: las instrucciones de sistema van en el parámetro
             ``system``. Por eso separamos la lista en dos.
          2. ``max_tokens`` es OBLIGATORIO (en OpenAI es opcional).
          3. Los modelos Claude vigentes (familia 5.x) NO aceptan ``temperature``,
             ``top_p`` ni ``top_k``: la documentación oficial indica omitirlos
             y el SDK 2.x directamente ya no los tiene como parámetros. Por eso
             NO los enviamos. ``ModelConfig`` igual valida su rango (0..1) para
             que una config incoherente se detecte temprano y para mantener la
             misma interfaz que OpenAI.

        Args:
            messages: Conversación validada.

        Returns:
            Diccionario de parámetros para ``messages.create`` / ``stream``.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [m.model_dump() for m in messages if m.role != "system"]

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": chat_messages,
            "max_tokens": self.config.max_tokens,
            # Los modelos Claude actuales "piensan" antes de responder
            # (thinking adaptativo) y esos tokens CUENTAN dentro de max_tokens.
            # Con ``effort: "low"`` el modelo piensa poco en preguntas simples,
            # así queda espacio para el texto de la respuesta.
            "output_config": {"effort": "low"},
        }
        # Si hay varios mensajes system, los unimos en un solo texto.
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        # Notá que NO agregamos temperature ni top_p (ver punto 3 del docstring).
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Pide una respuesta completa a Anthropic.

        Args:
            messages: Conversación a enviar.

        Returns:
            ``ModelResponse`` con el texto y el uso de tokens, o con
            ``success=False`` y el error si la llamada falló.
        """
        try:
            message = await self._client.messages.create(**self._build_params(messages))
            # Diferencia entre SDKs: OpenAI devuelve un string en
            # ``choices[0].message.content``; Anthropic devuelve una LISTA de
            # bloques de contenido (texto, thinking, tool_use...). Nos quedamos
            # solo con los bloques de tipo "text" y los concatenamos.
            text = "".join(block.text for block in message.content if block.type == "text")
            return ModelResponse(
                content=text,
                provider="anthropic",
                model=message.model,
                # Anthropic ya usa los nombres input_tokens / output_tokens.
                usage=TokenUsage(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                ),
                success=True,
            )
        except Exception as exc:  # noqa: BLE001 - la librería nunca debe romper
            error = map_anthropic_error(exc)
            logger.error("Anthropic generate falló (%s): %s", error.type, error.message)
            return ModelResponse(
                provider="anthropic", model=self.config.model, success=False, error=error
            )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Pide una respuesta a Anthropic y la emite fragmento a fragmento.

        Args:
            messages: Conversación a enviar.

        Yields:
            ``StreamChunk`` con texto; ante un error, un último chunk con
            ``error`` y el stream termina.
        """
        # El try envuelve el ``async with`` y el ``async for`` completos,
        # porque el error puede aparecer al conectar o a mitad del stream.
        try:
            # --- Sobre ``async with`` (context manager asíncrono) ----------
            # ``messages.stream(...)`` abre una conexión HTTP que queda
            # abierta mientras llegan los eventos. ``async with`` garantiza
            # que al salir del bloque (termine bien, con error o porque el
            # consumidor dejó de iterar) la conexión se CIERRE. Es "async"
            # porque abrir y cerrar la conexión también implica esperar a la
            # red, y eso no debe bloquear el event loop.
            #
            # Diferencia entre SDKs: OpenAI usa ``create(stream=True)`` y
            # devuelve chunks crudos; Anthropic ofrece este helper que además
            # expone ``text_stream``, un iterador que ya trae solo el texto.
            # ---------------------------------------------------------------
            async with self._client.messages.stream(**self._build_params(messages)) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield StreamChunk(text=text)
        except Exception as exc:  # noqa: BLE001
            error = map_anthropic_error(exc)
            logger.error("Anthropic stream falló (%s): %s", error.type, error.message)
            yield StreamChunk(error=error)
