"""Tests sin API keys: se simulan los SDKs para verificar nuestra logica."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import anthropic
import pytest

try:  # los SDKs mas nuevos usan httpx2; los anteriores, httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
from pydantic import ValidationError

import base
from anthropic_client import AnthropicClient
from manager import AsyncLLMManager
from openai_client import OpenAIClient
from schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    ProviderSecrets,
)

MSGS = [ChatMessage(role="user", content="hola")]


def make_config(provider=Provider.ANTHROPIC, **kw) -> ModelConfig:
    return ModelConfig(provider=provider, model="m", max_retries=kw.pop("max_retries", 2), **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Los reintentos no deben hacer esperar a los tests."""
    async def no_wait(self, attempt, error):
        return None
    monkeypatch.setattr(base.BaseLLMClient, "_sleep_before_retry", no_wait)


def api_error(cls, status: int):
    req = httpx.Request("POST", "https://example.test")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


# ---------------------------------------------------------------- schemas
def test_temperature_range_por_proveedor():
    make_config(Provider.OPENAI, temperature=1.8)  # valido en OpenAI
    with pytest.raises(ValidationError):
        make_config(Provider.ANTHROPIC, temperature=1.8)  # Anthropic llega a 1
    with pytest.raises(ValidationError):
        make_config(Provider.OPENAI, temperature=2.5)
    with pytest.raises(ValidationError):
        make_config(temperature=-0.1)


def test_mensaje_vacio_y_rol_invalido():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="x")


def test_secretstr_no_se_filtra():
    secrets = ProviderSecrets(anthropic_api_key="sk-ant-SUPERSECRETA")
    assert "SUPERSECRETA" not in repr(secrets)
    assert "SUPERSECRETA" not in str(secrets.model_dump())


# ---------------------------------------------------------------- factory
def test_factory_elige_clase_segun_config():
    secrets = ProviderSecrets(openai_api_key="a", anthropic_api_key="b")
    m1 = AsyncLLMManager(make_config(Provider.ANTHROPIC), secrets)
    m2 = AsyncLLMManager(make_config(Provider.OPENAI), secrets)
    assert isinstance(m1._client, AnthropicClient)
    assert isinstance(m2._client, OpenAIClient)


def test_factory_falla_claro_si_falta_la_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AsyncLLMManager(make_config(), ProviderSecrets())


# ------------------------------------------------- generate (Anthropic)
def fake_message(text="Respuesta"):
    return SimpleNamespace(
        model="m",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
    )


async def test_generate_normaliza_respuesta_anthropic():
    c = AnthropicClient("k", make_config())

    async def fake_create(**kw):
        assert kw["system"] == "sé breve"  # el system va aparte
        assert all(m["role"] != "system" for m in kw["messages"])
        return fake_message("Hola!")

    c._client.messages.create = fake_create
    msgs = [ChatMessage(role="system", content="sé breve"), *MSGS]
    r = await c.generate(msgs)
    assert r.ok and r.content == "Hola!" and r.usage.output_tokens == 5


async def test_rate_limit_se_reintenta_y_se_recupera():
    c = AnthropicClient("k", make_config(max_retries=2))
    calls = 0

    async def flaky(**kw):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise api_error(anthropic.RateLimitError, 429)
        return fake_message("ok")

    c._client.messages.create = flaky
    r = await c.generate(MSGS)
    assert r.ok and calls == 3


async def test_rate_limit_agotado_devuelve_error_no_excepcion():
    c = AnthropicClient("k", make_config(max_retries=1))

    async def always_429(**kw):
        raise api_error(anthropic.RateLimitError, 429)

    c._client.messages.create = always_429
    r = await c.generate(MSGS)  # no debe lanzar
    assert not r.ok and r.error.kind == "rate_limit" and r.error.retryable


async def test_auth_error_no_se_reintenta():
    c = AnthropicClient("k", make_config(max_retries=3))
    calls = 0

    async def bad_key(**kw):
        nonlocal calls
        calls += 1
        raise api_error(anthropic.AuthenticationError, 401)

    c._client.messages.create = bad_key
    r = await c.generate(MSGS)
    assert calls == 1 and r.error.kind == "auth" and not r.error.retryable


async def test_timeout_devuelve_error_controlado():
    c = AnthropicClient("k", make_config(max_retries=0, timeout_s=0.05))

    async def slow(**kw):
        await asyncio.get_running_loop().create_future()  # nunca termina

    c._client.messages.create = slow
    r = await c.generate(MSGS)
    assert r.error.kind == "timeout"


async def test_connection_error_es_transitorio():
    c = AnthropicClient("k", make_config(max_retries=0))

    async def down(**kw):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))

    c._client.messages.create = down
    r = await c.generate(MSGS)
    assert r.error.kind == "connection" and r.error.retryable


# ------------------------------------------------- streaming (Anthropic)
class FakeStream:
    def __init__(self, tokens, fail_after=None, exc=None):
        self.tokens, self.fail_after, self.exc = tokens, fail_after, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def text_stream(self):
        async def gen():
            for i, t in enumerate(self.tokens):
                if self.fail_after is not None and i == self.fail_after:
                    raise self.exc
                yield t
        return gen()

    async def get_final_message(self):
        return SimpleNamespace(stop_reason="end_turn")


async def collect(agen):
    return [chunk async for chunk in agen]


async def test_streaming_emite_fragmentos_en_orden():
    c = AnthropicClient("k", make_config())
    c._client.messages.stream = lambda **kw: FakeStream(["La ", "entropía ", "mide..."])
    chunks = await collect(c.stream(MSGS))
    assert "".join(ch.delta for ch in chunks) == "La entropía mide..."
    assert chunks[-1].finish_reason == "end_turn" and not any(ch.error for ch in chunks)


async def test_streaming_reintenta_si_falla_antes_del_primer_token():
    c = AnthropicClient("k", make_config(max_retries=2))
    attempts = 0

    def factory(**kw):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return FakeStream(["x"], fail_after=0, exc=api_error(anthropic.RateLimitError, 429))
        return FakeStream(["ok"])

    c._client.messages.stream = factory
    chunks = await collect(c.stream(MSGS))
    assert attempts == 2 and "".join(ch.delta for ch in chunks) == "ok"


async def test_streaming_no_reintenta_a_mitad_y_cierra_con_error():
    c = AnthropicClient("k", make_config(max_retries=2))
    attempts = 0

    def factory(**kw):
        nonlocal attempts
        attempts += 1
        return FakeStream(["a", "b", "c"], fail_after=2, exc=api_error(anthropic.RateLimitError, 429))

    c._client.messages.stream = factory
    chunks = await collect(c.stream(MSGS))  # no lanza
    assert attempts == 1  # no duplica texto ya emitido
    assert "".join(ch.delta for ch in chunks) == "ab"
    assert chunks[-1].error and chunks[-1].error.kind == "rate_limit"


# ---------------------------------------------------------- concurrencia
async def test_llamadas_concurrentes_no_se_bloquean():
    """5 llamadas de 0.2s en paralelo deben tardar ~0.2s, no ~1s."""
    c = AnthropicClient("k", make_config())

    async def slow_ok(**kw):
        await asyncio.sleep(0.2)  # simula la latencia de red (cede el control)
        return fake_message("ok")

    c._client.messages.create = slow_ok
    t0 = time.perf_counter()
    results = await asyncio.gather(*(c.generate(MSGS) for _ in range(5)))
    assert time.perf_counter() - t0 < 0.7
    assert all(isinstance(r, ModelResponse) and r.ok for r in results)


# ------------------------------------------------------- OpenAI (simulado)
async def test_openai_normaliza_choices_y_streaming():
    c = OpenAIClient("k", make_config(Provider.OPENAI))

    async def fake_create(**kw):
        if kw.get("stream"):
            async def events():
                for t, fin in (("Hola ", None), ("mundo", None), ("", "stop")):
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=t), finish_reason=fin)]
                    )
                yield SimpleNamespace(choices=[])  # evento sin choices: se ignora
            return events()
        return SimpleNamespace(
            model="m",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hola"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=4),
        )

    c._client.chat.completions.create = fake_create
    r = await c.generate(MSGS)
    assert r.ok and r.content == "Hola" and r.usage.input_tokens == 2
    chunks = await collect(c.stream(MSGS))
    assert "".join(ch.delta for ch in chunks) == "Hola mundo"
    assert chunks[-1].finish_reason == "stop"


def test_anthropic_envia_temperature_solo_si_el_sdk_la_admite():
    c = AnthropicClient("k", make_config(temperature=0.3))
    kwargs = c._request_kwargs(MSGS)
    assert ("temperature" in kwargs) == c._sdk_accepts_temperature
    c._sdk_accepts_temperature = True
    assert c._request_kwargs(MSGS)["temperature"] == 0.3
    c._sdk_accepts_temperature = False
    assert "temperature" not in c._request_kwargs(MSGS)


# ------------------------------------------------ generate_stream (texto)
async def test_generate_stream_devuelve_texto_plano():
    c = AnthropicClient("k", make_config())
    c._client.messages.stream = lambda **kw: FakeStream(["Hola ", "mundo"])
    fragmentos = [f async for f in c.generate_stream(MSGS)]
    assert fragmentos == ["Hola ", "mundo"]
    assert all(isinstance(f, str) for f in fragmentos)


async def test_generate_stream_ante_error_emite_aviso_y_no_lanza():
    c = AnthropicClient("k", make_config(max_retries=0))
    c._client.messages.stream = lambda **kw: FakeStream(
        ["a", "b"], fail_after=1, exc=api_error(anthropic.RateLimitError, 429)
    )
    fragmentos = [f async for f in c.generate_stream(MSGS)]  # no lanza
    assert fragmentos[0] == "a"
    assert "Error durante el streaming" in fragmentos[-1]


async def test_manager_expone_generate_stream_y_acepta_str():
    secrets = ProviderSecrets(anthropic_api_key="k")
    mgr = AsyncLLMManager(make_config(), secrets)
    mgr._client._client.messages.stream = lambda **kw: FakeStream(["ok"])
    assert [f async for f in mgr.generate_stream("hola")] == ["ok"]


# ------------------------------------------------------------- Gemini
from gemini_client import GeminiClient  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402


def test_gemini_convierte_roles_y_separa_el_system():
    c = GeminiClient("k", make_config(Provider.GEMINI))
    msgs = [
        ChatMessage(role="system", content="sé breve"),
        ChatMessage(role="user", content="hola"),
        ChatMessage(role="assistant", content="buenas"),
    ]
    contents, system = c._split_messages(msgs)
    assert system == "sé breve"
    assert [x.role for x in contents] == ["user", "model"]  # assistant -> model


async def test_gemini_normaliza_respuesta_y_streaming():
    c = GeminiClient("k", make_config(Provider.GEMINI))

    async def fake_generate(**kw):
        assert kw["config"].system_instruction == "sé breve"
        return SimpleNamespace(
            text="Hola!",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
            usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=2),
        )

    async def fake_stream(**kw):
        async def gen():
            for t in ("Ho", "la", None, ""):
                yield SimpleNamespace(text=t)
        return gen()

    c._client.aio.models.generate_content = fake_generate
    c._client.aio.models.generate_content_stream = fake_stream
    msgs = [ChatMessage(role="system", content="sé breve"), *MSGS]
    r = await c.generate(msgs)
    assert r.ok and r.content == "Hola!" and r.finish_reason == "STOP"
    assert r.usage.input_tokens == 3 and r.usage.output_tokens == 2
    assert [f async for f in c.generate_stream(msgs)] == ["Ho", "la"]


@pytest.mark.parametrize(
    "code, msg, kind, retryable",
    [
        (429, "quota", "rate_limit", True),
        (503, "overloaded", "api", True),
        (403, "forbidden", "auth", False),
        (400, "API key not valid. Please pass a valid API key.", "auth", False),
        (400, "bad prompt", "bad_request", False),
    ],
)
def test_gemini_mapea_errores(code, msg, kind, retryable):
    c = GeminiClient("k", make_config(Provider.GEMINI))
    exc = genai_errors.APIError(code, {"error": {"message": msg, "status": "X"}})
    err = c._map_error(exc)
    assert err.kind == kind and err.retryable is retryable


def test_factory_crea_gemini_y_pide_google_api_key():
    mgr = AsyncLLMManager(
        make_config(Provider.GEMINI), ProviderSecrets(google_api_key="k")
    )
    assert isinstance(mgr._client, GeminiClient)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        AsyncLLMManager(make_config(Provider.GEMINI), ProviderSecrets())
