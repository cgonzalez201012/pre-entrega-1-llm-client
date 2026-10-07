"""Interfaz comun (clase base abstracta) para todos los proveedores.

Patron "metodo plantilla": esta clase implementa UNA vez la logica de
timeout, reintentos con backoff y conversion de excepciones a errores
controlados. Cada proveedor solo aporta tres cosas pequenas:

* ``_call``       -> la llamada al SDK (respuesta completa)
* ``_stream_raw`` -> la llamada al SDK en modo streaming
* ``_map_error``  -> como clasificar las excepciones de ese SDK
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Contrato que cumple cada proveedor.

    La logica de negocio solo conoce esta clase: no sabe si abajo hay OpenAI
    o Anthropic.
    """

    def __init__(self, api_key: str, config: ModelConfig) -> None:
        self._api_key = api_key
        self.config = config

    # ------------------------------------------------------------------
    # Lo que implementa cada proveedor
    # ------------------------------------------------------------------
    @abstractmethod
    async def _call(self, messages: list[ChatMessage]) -> ModelResponse:
        """Llamada al SDK (async, no bloqueante). Puede lanzar excepciones del SDK."""

    @abstractmethod
    def _stream_raw(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Streaming con el SDK (generador async). Puede lanzar excepciones del SDK."""

    @abstractmethod
    def _map_error(self, exc: Exception) -> LLMError:
        """Traduce una excepcion del SDK a un LLMError controlado."""

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra el cliente HTTP subyacente."""

    # ------------------------------------------------------------------
    # API publica (identica para todos los proveedores)
    # ------------------------------------------------------------------
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Nunca lanza: los fallos vuelven en ``.error``."""
        last_error: LLMError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Timeout total por intento (asyncio.timeout: Python 3.11+).
                async with asyncio.timeout(self.config.timeout_s):
                    return await self._call(messages)
            except asyncio.CancelledError:
                raise  # nunca tragarse una cancelacion
            except Exception as exc:  # noqa: BLE001 - se clasifica abajo
                last_error = self._classify(exc)

            if not last_error.retryable or attempt == self.config.max_retries:
                break
            await self._sleep_before_retry(attempt, last_error)

        assert last_error is not None
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            error=last_error,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Streaming de tokens (generador asincrono).

        Solo se reintenta si el fallo ocurre ANTES de emitir el primer token;
        despues de eso reintentar duplicaria texto ya mostrado al usuario.
        Si falla, el ultimo chunk trae ``error`` y el generador termina limpio.
        """
        for attempt in range(self.config.max_retries + 1):
            started = False
            try:
                async for chunk in self._stream_raw(messages):
                    started = started or bool(chunk.delta)
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                error = self._classify(exc)
                if started or not error.retryable or attempt == self.config.max_retries:
                    yield StreamChunk(error=error)
                    return
                await self._sleep_before_retry(attempt, error)

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Streaming en su forma mas simple: generador asincrono de TEXTO.

        Es lo que pide la consigna ("devolver fragmentos de texto conforme
        lleguen"). Por debajo usa ``stream()``; si algo falla, en vez de lanzar
        una excepcion emite un ultimo fragmento con el aviso del error.
        """
        async for chunk in self.stream(messages):
            if chunk.error:
                yield f"\n[⚠️ Error durante el streaming: {chunk.error.message}]"
                return
            if chunk.delta:
                yield chunk.delta

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _classify(self, exc: Exception) -> LLMError:
        if isinstance(exc, TimeoutError):  # asyncio.timeout vencio
            return LLMError(
                kind="timeout",
                message=f"El modelo tardo mas de {self.config.timeout_s}s",
                retryable=True,
            )
        return self._map_error(exc)

    async def _sleep_before_retry(self, attempt: int, error: LLMError) -> None:
        # Backoff exponencial con jitter: ~0.5s, 1s, 2s...
        delay = 0.5 * (2**attempt) + random.uniform(0, 0.25)
        logger.warning(
            "Error transitorio (%s). Reintento %d/%d en %.1fs",
            error.kind,
            attempt + 1,
            self.config.max_retries,
            delay,
        )
        await asyncio.sleep(delay)
