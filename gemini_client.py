"""Cliente Gemini (bonus, opcional). Usa el SDK oficial ``google-genai`` en su
variante asincrona (``client.aio``).

Diferencias que abstraemos aca (ver la tabla del notebook del profesor):

* el system prompt va en ``system_instruction`` (en la configuracion), no en
  la lista de mensajes;
* el rol del asistente se llama ``"model"`` y no ``"assistant"``;
* el texto esta en ``response.text``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

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


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        # timeout en milisegundos; los reintentos los maneja la clase base.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(config.timeout_s * 1000)),
        )

    def _split_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[types.Content], str | None]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        return contents, (system or None)

    def _gen_config(self, system: str | None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system,
            # No usamos function calling: se desactiva para evitar avisos del SDK.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        contents, system = self._split_messages(messages)
        response = await self._client.aio.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=self._gen_config(system),
        )
        candidate = response.candidates[0] if response.candidates else None
        finish = getattr(candidate, "finish_reason", None)
        usage = response.usage_metadata
        return ModelResponse(
            provider=Provider.GEMINI,
            model=self.config.model,
            content=response.text or "",
            finish_reason=getattr(finish, "name", finish),
            usage=Usage(
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            ),
        )

    async def _stream_raw(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        contents, system = self._split_messages(messages)
        stream = await self._client.aio.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=self._gen_config(system),
        )
        async for chunk in stream:
            if chunk.text:
                yield StreamChunk(delta=chunk.text)

    def _map_error(self, exc: Exception) -> LLMError:
        msg = str(exc)
        if isinstance(exc, errors.APIError):
            code = exc.code
            if code in (401, 403) or "api key not valid" in msg.lower():
                # Gemini responde 400 (no 401) cuando la clave es invalida.
                return LLMError(kind="auth", message="API key invalida o sin permisos")
            if code == 429:
                return LLMError(kind="rate_limit", message=msg, retryable=True)
            if code >= 500:
                return LLMError(kind="api", message=msg, retryable=True)
            return LLMError(kind="bad_request", message=msg)
        # Errores de red: google-genai usa httpx por debajo.
        name = type(exc).__name__
        if "Timeout" in name:
            return LLMError(kind="timeout", message=msg, retryable=True)
        if type(exc).__module__.startswith("httpx") or "Connect" in name:
            return LLMError(kind="connection", message=msg, retryable=True)
        return LLMError(kind="unknown", message=f"{name}: {msg}")

    async def aclose(self) -> None:
        await self._client.aio.aclose()
