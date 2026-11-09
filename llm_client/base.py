"""Interfaz común (clase abstracta) que todo cliente de LLM debe cumplir.

¿Por qué una clase abstracta?
    ``BaseLLMClient`` no hace nada por sí misma: solo declara QUÉ métodos debe
    tener un cliente (``generate`` y ``stream``) y con qué firma. Cada
    proveedor (OpenAI, Anthropic, ...) implementa el CÓMO.

    Gracias a esto, el Manager y ``main.py`` pueden trabajar con "cualquier
    ``BaseLLMClient``" sin saber cuál es: los proveedores son intercambiables.
    Si una subclase olvida implementar un método abstracto, Python lanza
    ``TypeError`` al intentar instanciarla, así que el error aparece enseguida.

¿Por qué dos métodos separados y no uno con un flag ``stream=True``?
    Una función que contiene ``yield`` SIEMPRE es un generador: al llamarla no
    ejecuta nada, devuelve un objeto que se recorre con ``async for``. No puede
    "a veces devolver un valor y a veces streamear". Separar ``generate``
    (devuelve un ``ModelResponse`` completo) de ``stream`` (generador de
    ``StreamChunk``) deja el tipo de retorno de cada uno claro y predecible.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk

# Cada módulo usa su propio logger con el nombre del módulo. La librería NO
# configura cómo se muestran los logs (eso lo decide la aplicación, en main.py).
logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Clase base abstracta para clientes de LLM asíncronos.

    Attributes:
        config: Configuración validada del modelo.
    """

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        """Guarda la configuración común a todos los clientes.

        Args:
            config: Configuración del modelo (ya validada por Pydantic).
            api_key: API key del proveedor. Cada subclase la usa para crear
                su cliente del SDK.
        """
        self.config = config
        self._api_key = api_key

    # --- Sobre async/await -------------------------------------------------
    # ``async def`` define una "corrutina": una función que puede pausarse.
    # Dentro, ``await`` dice "esto tarda (p. ej. esperar la respuesta HTTP);
    # mientras tanto, que el event loop haga otra cosa".
    #
    # El event loop es el "director" de asyncio: tiene una lista de tareas y
    # va ejecutando la que esté lista. Si una función hace algo bloqueante
    # (por ejemplo, usar el SDK síncrono o ``time.sleep``), el loop entero se
    # congela y NINGUNA otra tarea avanza. Por eso usamos solo los clientes
    # async de los SDKs (``AsyncOpenAI`` / ``AsyncAnthropic``).
    # -----------------------------------------------------------------------

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (sin streaming).

        Args:
            messages: Conversación a enviar al modelo.

        Returns:
            Un ``ModelResponse``. Nunca lanza excepciones de la API: si algo
            falla, devuelve ``success=False`` con el ``error`` cargado.
        """

    # --- Sobre generadores asíncronos -------------------------------------
    # Las implementaciones de ``stream`` usan ``yield`` dentro de un
    # ``async def``: eso las convierte en GENERADORES ASÍNCRONOS. Cada
    # ``yield`` entrega un fragmento al que consume y pausa la función hasta
    # que pida el siguiente. Se recorren con ``async for`` (no con ``for``)
    # porque entre fragmento y fragmento hay que esperar a la red con await.
    #
    # Nota: acá el método abstracto no tiene ``yield``, así que lo declaramos
    # como ``def`` normal que devuelve ``AsyncIterator``. Las subclases lo
    # implementan con ``async def`` + ``yield`` y cumplen la misma firma de uso:
    # ``async for chunk in client.stream(messages): ...``
    # -----------------------------------------------------------------------

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Genera la respuesta fragmento a fragmento.

        Args:
            messages: Conversación a enviar al modelo.

        Yields:
            ``StreamChunk`` con texto. Si ocurre un error (incluso a mitad del
            stream), se emite un último ``StreamChunk`` con ``error`` y el
            stream termina. Nunca lanza excepciones de la API.
        """
