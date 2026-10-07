"""Cliente OpenAI (usa AsyncOpenAI, el SDK oficial asincrono)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import (
    ChatMessage,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        # max_retries=0: los reintentos los maneja la clase base, no el SDK.
        self._client = AsyncOpenAI(
            api_key=api_key, max_retries=0, timeout=config.timeout_s
        )

    @staticmethod
    def _to_api_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        # OpenAI acepta el rol "system" dentro de la lista de mensajes.
        return [{"role": m.role, "content": m.content} for m in messages]

    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        # await: nunca la version sincrona dentro de una funcion async.
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._to_api_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        choice = response.choices[0]  # diferencia clave con Anthropic
        usage = response.usage
        return ModelResponse(
            provider=Provider.OPENAI,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    async def _stream_raw(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._to_api_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for event in stream:
            if not event.choices:  # algunos eventos (usage) vienen sin choices
                continue
            choice = event.choices[0]
            delta = choice.delta.content or ""
            if delta or choice.finish_reason:
                yield StreamChunk(delta=delta, finish_reason=choice.finish_reason)

    def _map_error(self, exc: Exception) -> LLMError:
        msg = str(exc)
        # Orden importa: APITimeoutError es subclase de APIConnectionError.
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return LLMError(kind="auth", message="API key invalida o sin permisos")
        if isinstance(exc, openai.RateLimitError):
            return LLMError(kind="rate_limit", message=msg, retryable=True)
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(kind="timeout", message=msg, retryable=True)
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(kind="connection", message=msg, retryable=True)
        if isinstance(exc, openai.BadRequestError):
            return LLMError(kind="bad_request", message=msg)
        if isinstance(exc, openai.APIStatusError):
            return LLMError(kind="api", message=msg, retryable=exc.status_code >= 500)
        return LLMError(kind="unknown", message=f"{type(exc).__name__}: {msg}")

    async def aclose(self) -> None:
        await self._client.close()
