"""Demo del cliente unificado: modo normal, streaming, validación y resiliencia.

Ejecutar con::

    python main.py

Este script habla SOLO con ``AsyncLLMManager`` (la fachada). No importa ni
conoce ``OpenAIClient`` ni ``AnthropicClient``: eso demuestra que los
proveedores son intercambiables.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, ChatMessage, ModelConfig

QUESTION = "¿Qué es la entropía?"
PROVIDERS = ["openai", "anthropic"]

# Key falsa para la prueba de resiliencia. Tiene formato plausible pero no es
# válida, así que la API responde 401 (o, sin internet, falla la conexión).
FAKE_API_KEY = "sk-invalid-key-for-resilience-test"


def build_messages() -> list[ChatMessage]:
    """Arma la conversación de ejemplo.

    Returns:
        Lista con un mensaje ``system`` y la pregunta del usuario.
    """
    return [
        ChatMessage(role="system", content="Respondé en español, en un párrafo breve."),
        ChatMessage(role="user", content=QUESTION),
    ]


def print_title(title: str) -> None:
    """Imprime un separador con título para ordenar la salida.

    Args:
        title: Texto del título.
    """
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


async def run_generate(manager: AsyncLLMManager) -> None:
    """Hace la pregunta en modo normal e imprime respuesta y tokens.

    Args:
        manager: Manager ya configurado.
    """
    print(f"\n--- [{manager.config.provider}] Modo normal ---")
    # ``await`` espera el resultado sin bloquear el event loop.
    response = await manager.generate(build_messages())

    # El manager nunca lanza errores de la API: revisamos ``success``.
    if not response.success and response.error is not None:
        err = response.error
        print(f"[ERROR] tipo={err.type} retryable={err.retryable}")
        print(f"        {err.message}")
        return

    print(response.content)
    if response.usage is not None:
        print(
            f"\n(tokens -> input: {response.usage.input_tokens}, "
            f"output: {response.usage.output_tokens})"
        )


async def run_stream(manager: AsyncLLMManager) -> None:
    """Hace la pregunta en modo streaming, imprimiendo cada fragmento.

    Args:
        manager: Manager ya configurado.
    """
    print(f"\n--- [{manager.config.provider}] Modo streaming ---")
    # ``async for`` consume el generador asíncrono: en cada vuelta espera
    # (sin bloquear) el siguiente fragmento que llega por la red.
    async for chunk in manager.stream(build_messages()):
        if chunk.error is not None:
            print(f"\n[ERROR en stream] tipo={chunk.error.type} retryable={chunk.error.retryable}")
            print(f"        {chunk.error.message}")
            break
        # ``end=""`` evita el salto de línea; ``flush=True`` fuerza a que se
        # muestre YA (sin esto, Python acumula el texto en un buffer y el
        # efecto de "escritura en vivo" se pierde).
        print(chunk.text, end="", flush=True)
    print()


async def run_providers() -> None:
    """Corre generate y stream para cada proveedor con API key configurada."""
    for provider in PROVIDERS:
        try:
            manager = AsyncLLMManager.from_env(provider=provider)
        except ValueError as exc:
            # Falta la key o la config es inválida: lo informamos y seguimos
            # con el siguiente proveedor en vez de cortar el programa.
            print(f"\n[{provider}] Se omite: {exc}")
            continue

        print_title(f"Proveedor: {provider} | modelo: {manager.config.model}")
        await run_generate(manager)
        await run_stream(manager)


async def run_resilience_test() -> None:
    """Fuerza una API key inválida y muestra que el error se maneja sin romper."""
    print_title("Prueba de resiliencia: API key inválida")
    for provider in PROVIDERS:
        # Pasamos la key falsa explícitamente: pisa la del .env.
        manager = AsyncLLMManager.from_env(provider=provider, api_key=FAKE_API_KEY)
        await run_generate(manager)
        await run_stream(manager)
    print("\nEl programa siguió funcionando a pesar de los errores.")


def run_validation_demo() -> None:
    """Muestra cómo Pydantic rechaza datos inválidos ANTES de llamar a la API."""
    print_title("Validación con Pydantic")

    # 1) temperature 1.5 es válida para OpenAI pero no para Anthropic.
    try:
        ModelConfig(provider="anthropic", model="claude-haiku-5-5", temperature=1.5)
    except ValidationError as exc:
        print(f"[OK] Config rechazada:\n{exc.errors()[0]['msg']}")

    # 2) Un mensaje con contenido vacío.
    try:
        ChatMessage(role="user", content="")
    except ValidationError as exc:
        print(f"[OK] Mensaje rechazado: {exc.errors()[0]['msg']}")


async def main() -> None:
    """Orquesta todas las demos."""
    run_validation_demo()
    await run_providers()
    await run_resilience_test()


if __name__ == "__main__":
    # ``load_dotenv`` copia las variables del archivo .env a ``os.environ``.
    load_dotenv()

    # La librería solo EMITE logs; la aplicación decide si se muestran y con
    # qué formato. Con WARNING (default) se ven los logs de error de los
    # clientes (van a stderr, separados de lo que imprimimos con print).
    # LOG_LEVEL=INFO muestra más detalle; LOG_LEVEL=CRITICAL los oculta.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ``asyncio.run`` crea el event loop, ejecuta la corrutina ``main()``
    # hasta que termina y cierra el loop. Es el puente entre el código
    # síncrono (el script) y el mundo async.
    asyncio.run(main())
