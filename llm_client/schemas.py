"""Modelos de datos (schemas) del cliente unificado, definidos con Pydantic v2.

Este módulo es el "contrato" del proyecto: define qué forma tienen los datos
que entran (mensajes, configuración) y los que salen (respuestas, fragmentos
de streaming y errores), sin importar el proveedor.

¿Por qué Pydantic?
    Una clase que hereda de ``BaseModel`` valida sus datos al construirse.
    Si algo no cumple las reglas (un rol inexistente, una temperatura fuera de
    rango), Pydantic lanza ``ValidationError`` con un mensaje claro *antes* de
    gastar una llamada a la API. Es mejor fallar temprano y localmente que
    recibir un 400 del servidor.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ``Literal`` restringe un campo a un conjunto cerrado de valores. Definirlo
# una sola vez como alias evita repetir la lista en varios modelos.
ProviderName = Literal["openai", "anthropic"]


class ChatMessage(BaseModel):
    """Un mensaje de la conversación.

    Attributes:
        role: Quién habla. ``system`` da instrucciones generales al modelo,
            ``user`` es la persona y ``assistant`` es el modelo.
        content: Texto del mensaje. No puede estar vacío.
    """

    role: Literal["system", "user", "assistant"]
    # ``Field(min_length=1)`` hace que Pydantic rechace el string vacío "".
    # Un mensaje vacío no aporta nada y algunas APIs lo rechazan con error.
    content: str = Field(min_length=1, description="Texto del mensaje.")


class ModelConfig(BaseModel):
    """Configuración de un modelo y de cómo llamarlo.

    Attributes:
        provider: Proveedor a usar (``openai`` o ``anthropic``).
        model: Nombre del modelo en la API del proveedor.
        temperature: Aleatoriedad de la respuesta (0 = determinista).
        max_tokens: Límite de tokens que el modelo puede generar.
        top_p: Muestreo por núcleo (opcional). Alternativa a temperature.
        timeout: Segundos máximos de espera por request (opcional).
        max_retries: Reintentos automáticos que hace el SDK ante errores
            transitorios (rate limit, conexión, 5xx).
    """

    provider: ProviderName
    model: str = Field(min_length=1, description="Nombre del modelo.")
    # ``ge`` = greater or equal (>=) y ``le`` = less or equal (<=).
    # El rango 0..2 es el más amplio (OpenAI). La restricción más estricta de
    # Anthropic se valida abajo, en ``check_provider_limits``.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # ``gt`` = greater than (>). Pedir 0 tokens no tiene sentido.
    max_tokens: int = Field(default=1024, gt=0)
    # ``float | None = None`` significa "opcional": si no se pasa, queda None y
    # el cliente no lo envía a la API (así se usa el default del proveedor).
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0)

    # ``model_validator(mode="after")`` corre DESPUÉS de validar cada campo por
    # separado. Lo usamos cuando una regla depende de VARIOS campos a la vez:
    # acá, el rango válido de ``temperature`` depende de ``provider``.
    @model_validator(mode="after")
    def check_provider_limits(self) -> "ModelConfig":
        """Valida reglas que dependen del proveedor.

        Returns:
            La misma instancia, si es válida.

        Raises:
            ValueError: Si ``provider`` es ``anthropic`` y ``temperature`` > 1.
                Pydantic lo convierte en un ``ValidationError``.
        """
        # Diferencia entre SDKs: OpenAI acepta temperature 0..2, Anthropic
        # solo 0..1. Validarlo acá da un error claro antes de llamar a la API.
        if self.provider == "anthropic" and self.temperature > 1.0:
            raise ValueError(
                f"Anthropic solo acepta temperature entre 0 y 1 "
                f"(recibido: {self.temperature})."
            )
        return self


# Tipos de error normalizados. Cada SDK tiene sus propias excepciones; los
# clientes las traducen a uno de estos valores para que el código que usa la
# librería no tenga que conocer los detalles de cada proveedor.
ErrorType = Literal[
    "rate_limit",
    "connection",
    "timeout",
    "authentication",
    "api_error",
    "unknown",
]


class LLMError(BaseModel):
    """Error normalizado, independiente del proveedor.

    Attributes:
        type: Categoría del error (ver ``ErrorType``).
        message: Descripción legible del error.
        provider: Proveedor donde ocurrió.
        retryable: True si tiene sentido reintentar más tarde (el problema es
            transitorio), False si reintentar daría el mismo error.
    """

    type: ErrorType
    message: str
    provider: ProviderName
    retryable: bool


class TokenUsage(BaseModel):
    """Tokens consumidos en una llamada.

    Attributes:
        input_tokens: Tokens del prompt (lo que enviamos).
        output_tokens: Tokens generados por el modelo.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Resultado de una llamada no-streaming (``generate``).

    En lugar de lanzar excepciones, los clientes SIEMPRE devuelven un
    ``ModelResponse``. Quien llama revisa ``success`` para saber si salió bien.

    Attributes:
        content: Texto generado, o None si hubo error.
        provider: Proveedor usado.
        model: Modelo usado.
        usage: Tokens consumidos (si la API los informó).
        success: True si la llamada funcionó.
        error: Detalle del error si ``success`` es False.
    """

    content: str | None = None
    provider: ProviderName
    model: str
    usage: TokenUsage | None = None
    success: bool
    error: LLMError | None = None


class StreamChunk(BaseModel):
    """Un fragmento emitido durante el streaming.

    Cada chunk trae texto O un error. Si trae error, es el último del stream.

    Attributes:
        text: Fragmento de texto generado.
        error: Error ocurrido durante el stream (corta el stream).
    """

    text: str | None = None
    error: LLMError | None = None
