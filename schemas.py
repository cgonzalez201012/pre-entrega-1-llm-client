"""Esquemas Pydantic del cliente unificado de LLMs.

Todo lo que entra y sale de la capa de abstraccion pasa por estos modelos,
asi nunca se manejan "diccionarios anidados" sueltos y los errores de
configuracion (temperatura fuera de rango, mensaje vacio, etc.) se detectan
antes de gastar una sola llamada a la API.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"  # bonus: no lo exige la consigna, pero tiene capa gratuita


# Rango de temperatura que acepta cada API. La consigna pide 0-2 (limite general,
# `le=2` en ModelConfig); ademas afinamos por proveedor: Anthropic solo llega a 1.
TEMPERATURE_LIMITS: dict[Provider, float] = {
    Provider.OPENAI: 2.0,
    Provider.ANTHROPIC: 1.0,
    Provider.GEMINI: 2.0,
}

# Nombre de la variable de entorno donde vive la clave de cada proveedor.
API_KEY_ENV_VARS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
}


class ChatMessage(BaseModel):
    """Un mensaje de la conversacion (formato comun a todos los proveedores)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Configuracion de la llamada al modelo, validada."""

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=64_000)
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def _check_temperature_for_provider(self) -> ModelConfig:
        limit = TEMPERATURE_LIMITS[self.provider]
        if self.temperature > limit:
            raise ValueError(
                f"temperature={self.temperature} supera el maximo de "
                f"{self.provider.value} ({limit})"
            )
        return self


class ProviderSecrets(BaseModel):
    """Claves de API. SecretStr evita que aparezcan en logs o tracebacks."""

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    def for_provider(self, provider: Provider) -> SecretStr:
        key = {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GEMINI: self.google_api_key,
        }[provider]
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"Falta la variable de entorno {API_KEY_ENV_VARS[provider]}"
            )
        return key


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(BaseModel):
    """Error controlado: lo que devolvemos en vez de dejar que explote."""

    kind: Literal[
        "auth", "rate_limit", "timeout", "connection", "bad_request", "api", "unknown"
    ]
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    """Respuesta normalizada. Igual para OpenAI y Anthropic."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento emitido por el streaming.

    Normalmente trae solo `delta` (texto). Si algo falla, el ultimo chunk trae
    `error` y el generador termina limpio (sin excepcion).
    """

    delta: str = ""
    finish_reason: str | None = None
    error: LLMError | None = None
