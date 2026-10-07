"""AsyncLLMManager: elige el proveedor por configuracion (Factory Pattern)."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator
from types import TracebackType

from base import BaseLLMClient
from schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    ProviderSecrets,
    StreamChunk,
)


def _build_openai(api_key: str, config: ModelConfig) -> BaseLLMClient:
    # Import diferido: si solo usas Anthropic, no se carga el SDK de OpenAI.
    from openai_client import OpenAIClient

    return OpenAIClient(api_key, config)


def _build_anthropic(api_key: str, config: ModelConfig) -> BaseLLMClient:
    from anthropic_client import AnthropicClient

    return AnthropicClient(api_key, config)


def _build_gemini(api_key: str, config: ModelConfig) -> BaseLLMClient:
    from gemini_client import GeminiClient

    return GeminiClient(api_key, config)


# Registro de proveedores: agregar uno nuevo = una linea aca + una clase.
_FACTORIES = {
    Provider.OPENAI: _build_openai,
    Provider.ANTHROPIC: _build_anthropic,
    Provider.GEMINI: _build_gemini,
}


class AsyncLLMManager:
    """Punto de entrada unico para hablar con cualquier LLM.

    >>> async with AsyncLLMManager.from_env() as llm:
    ...     resp = await llm.generate("Hola")
    """

    def __init__(self, config: ModelConfig, secrets: ProviderSecrets) -> None:
        api_key = secrets.for_provider(config.provider).get_secret_value()
        self.config = config
        self._client: BaseLLMClient = _FACTORIES[config.provider](api_key, config)

    # -- Construccion desde variables de entorno -------------------------
    @classmethod
    def from_env(cls, provider: str | None = None) -> AsyncLLMManager:
        """Lee LLM_PROVIDER, LLM_MODEL, etc. del entorno (cargar .env antes)."""
        prov = Provider((provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower())
        default_models = {
            Provider.OPENAI: "gpt-4o-mini",
            Provider.ANTHROPIC: "claude-haiku-4-5",
            Provider.GEMINI: "gemini-flash-latest",
        }
        env_model = os.environ.get("LLM_MODEL")
        # Si se fuerza otro proveedor por parametro, no reusar el modelo de otro.
        use_env_model = env_model and provider is None
        config = ModelConfig(
            provider=prov,
            model=env_model if use_env_model else default_models[prov],
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
            timeout_s=float(os.environ.get("LLM_TIMEOUT_S", "30")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        )
        secrets = ProviderSecrets(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
        return cls(config, secrets)

    # -- API publica -----------------------------------------------------
    @staticmethod
    def _normalize(messages: str | list[ChatMessage]) -> list[ChatMessage]:
        if isinstance(messages, str):
            return [ChatMessage(role="user", content=messages)]
        return messages

    async def generate(self, messages: str | list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Si falla, devuelve ``ModelResponse`` con ``error``."""
        return await self._client.generate(self._normalize(messages))

    def generate_stream(
        self, messages: str | list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Streaming token a token: generador asincrono que devuelve TEXTO.

        Es el metodo que pide la consigna. Si algo falla, el ultimo fragmento
        es un aviso de error (no se lanza excepcion).
        """
        return self._client.generate_stream(self._normalize(messages))

    def stream(self, messages: str | list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Streaming estructurado: igual que ``generate_stream`` pero cada
        fragmento es un ``StreamChunk`` (con ``finish_reason`` y ``error``)."""
        return self._client.stream(self._normalize(messages))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
