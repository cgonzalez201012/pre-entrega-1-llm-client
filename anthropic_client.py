"""Cliente Anthropic (usa AsyncAnthropic, el SDK oficial asincrono)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

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


logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        # max_retries=0: los reintentos los maneja la clase base, no el SDK.
        self._client = AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=config.timeout_s
        )
        self._sdk_accepts_temperature = (
            "temperature" in inspect.signature(self._client.messages.create).parameters
        )
        if not self._sdk_accepts_temperature:
            logger.warning(
                "El SDK de anthropic instalado no admite 'temperature'; se ignora "
                "(temperature=%s). Se valida igual en ModelConfig.",
                config.temperature,
            )

    def _request_kwargs(self, messages: list[ChatMessage]) -> dict:
        """Diferencias con OpenAI que abstraemos aca:

        * el mensaje "system" va en un parametro aparte, no en la lista;
        * ``max_tokens`` es obligatorio.
        """
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        kwargs: dict = {
            "model": self.config.model,
            "messages": chat,
            "max_tokens": self.config.max_tokens,
        }
        # Algunas versiones recientes del SDK ya no aceptan ``temperature``.
        # Lo enviamos solo si el SDK instalado lo soporta (si no, se ignora).
        if self._sdk_accepts_temperature:
            kwargs["temperature"] = self.config.temperature
        if system:
            kwargs["system"] = system
        return kwargs

    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self._client.messages.create(**self._request_kwargs(messages))
        # content es una LISTA de bloques; nos quedamos con los de texto.
        text = "".join(b.text for b in response.content if b.type == "text")
        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=response.model,
            content=text,
            finish_reason=response.stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def _stream_raw(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        async with self._client.messages.stream(
            **self._request_kwargs(messages)
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)
            final = await stream.get_final_message()
        yield StreamChunk(finish_reason=final.stop_reason)

    def _map_error(self, exc: Exception) -> LLMError:
        msg = str(exc)
        # Orden importa: APITimeoutError es subclase de APIConnectionError.
        if isinstance(
            exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError
        ):
            return LLMError(kind="auth", message="API key invalida o sin permisos")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(kind="rate_limit", message=msg, retryable=True)
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(kind="timeout", message=msg, retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(kind="connection", message=msg, retryable=True)
        if isinstance(exc, anthropic.BadRequestError):
            return LLMError(kind="bad_request", message=msg)
        if isinstance(exc, anthropic.APIStatusError):
            # 5xx y 529 (overloaded) son transitorios.
            return LLMError(kind="api", message=msg, retryable=exc.status_code >= 500)
        return LLMError(kind="unknown", message=f"{type(exc).__name__}: {msg}")

    async def aclose(self) -> None:
        await self._client.close()
