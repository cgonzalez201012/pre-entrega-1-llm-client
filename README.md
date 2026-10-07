# Unified Async LLM Client

Capa de abstracción asíncrona para hablar con **OpenAI** y **Anthropic** (y **Gemini** como bonus) a través de una única interfaz.
Pre-entrega del Módulo 1 · AI Engineering (Coderhouse).

## Qué incluye

| Archivo | Responsabilidad |
|---|---|
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig`, `ModelResponse`, `StreamChunk`, `LLMError`, `ProviderSecrets` (claves como `SecretStr`). |
| `base.py` | `BaseLLMClient` (clase base abstracta): timeout, reintentos con backoff exponencial, `generate()`, `generate_stream()` y `stream()`, que **nunca lanzan**. |
| `openai_client.py` | `OpenAIClient` sobre `AsyncOpenAI`. |
| `anthropic_client.py` | `AnthropicClient` sobre `AsyncAnthropic` (separa el `system`, extrae texto de `content[]`). |
| `gemini_client.py` | `GeminiClient` (bonus) sobre `google-genai` (`client.aio`). |
| `manager.py` | `AsyncLLMManager`: Factory que elige el proveedor según configuración. |
| `main.py` | Script de prueba: "¿Qué es la entropía?" en modo normal y streaming. |
| `tests/` | 28 tests con SDKs simulados (no requieren API keys). |
| `docs/` | Guía de estudio, cumplimiento de la rúbrica, comparación con el ejemplo del profesor y preguntas de defensa. |

## Requisitos

* Python **3.12+**
* Una API key de al menos un proveedor: Anthropic, OpenAI o Google (Gemini, gratis en <https://aistudio.google.com/apikey>)

## Puesta en marcha

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # y completá tus claves
python main.py
```

### Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `anthropic`) | `anthropic`, `openai` o `gemini`. |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Clave de Anthropic. |
| `OPENAI_API_KEY` | Si usás OpenAI | Clave de OpenAI. |
| `GOOGLE_API_KEY` | Si usás Gemini | Clave de Google AI Studio. |
| `LLM_MODEL` | No | Modelo (default: `claude-haiku-4-5` / `gpt-4o-mini` / `gemini-flash-latest`). |
| `LLM_TEMPERATURE` | No (0.7) | Rango 0–2 (Anthropic: 0–1). |
| `LLM_MAX_TOKENS` | No (1024) | Máximo de tokens de salida. |
| `LLM_TIMEOUT_S` | No (30) | Timeout por llamada. |
| `LLM_MAX_RETRIES` | No (2) | Reintentos ante errores transitorios. |

> Las claves viven solo en `.env` (ignorado por git). Nunca se escriben en el código ni aparecen en logs (`SecretStr`).

### Probar un proveedor u otro

```bash
python main.py                      # usa LLM_PROVIDER del .env
python main.py --provider openai
python main.py --provider anthropic
python main.py --provider gemini
python main.py --resiliencia        # además prueba una API key inválida a propósito
```

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from manager import AsyncLLMManager

async def main():
    load_dotenv()
    async with AsyncLLMManager.from_env("anthropic") as llm:
        # Respuesta completa
        resp = await llm.generate("¿Qué es la entropía?")
        print(resp.content if resp.ok else resp.error)

        # Streaming token a token (generador asíncrono de texto)
        async for fragmento in llm.generate_stream("Explicame asyncio en 3 líneas"):
            print(fragmento, end="", flush=True)

asyncio.run(main())
```

Cambiar de proveedor es cambiar **una variable de configuración**; el resto del código no se toca.

## Decisiones de diseño

* **Factory + clase base abstracta**: la lógica de negocio solo conoce `AsyncLLMManager`; sumar un proveedor es una clase nueva y una línea en `_FACTORIES`.
* **Método plantilla**: timeout, reintentos y clasificación de errores se implementan **una sola vez** en `BaseLLMClient`; cada proveedor solo aporta la llamada al SDK, el streaming y el mapeo de sus excepciones.
* **Totalmente asíncrono**: `AsyncOpenAI` / `AsyncAnthropic` / `client.aio` con `await` y `async for`; nunca se usa el cliente síncrono dentro de una función `async`. `asyncio.timeout()` corta llamadas colgadas.
* **Validación (Pydantic)**: mensajes no vacíos con rol restringido; `temperature` entre 0 y 2 (y hasta 1 en Anthropic, el máximo que acepta su API); `max_tokens`, `timeout_s` y `max_retries` con rangos.
* **Streaming**: `generate_stream()` es un generador asíncrono que hace `yield` de fragmentos de texto conforme llegan; `stream()` entrega lo mismo como `StreamChunk` (con `finish_reason` y `error`).
* **Errores controlados, sin crash**: `generate()` devuelve un `ModelResponse` con `error` (`auth`, `rate_limit`, `timeout`, `connection`, `bad_request`, `api`); en streaming el error llega como último fragmento.
  * Se **reintentan** (backoff exponencial + jitter) solo los errores transitorios: 429, timeout, red y 5xx.
  * **No** se reintentan los permanentes (clave inválida, request mal formado).
  * En streaming solo se reintenta si el fallo ocurre **antes del primer token**, para no duplicar texto ya mostrado.

## Tests

```bash
pytest
```

28 tests: validación, Factory, normalización de respuestas, streaming, reintentos, errores sin reintento, timeout y concurrencia, con SDKs simulados (los tres proveedores).

## Notas de compatibilidad

* **`temperature` en Anthropic:** algunas versiones recientes del SDK de `anthropic` ya no aceptan ese parámetro. El cliente detecta si el SDK instalado lo soporta; si no, lo omite y registra una advertencia (el valor igual se valida en `ModelConfig`).
* **Gemini (bonus):** el rol del asistente es `model` y el system prompt va en `system_instruction`; el cliente lo convierte automáticamente. En los modelos 2.5, `max_tokens` incluye los tokens de razonamiento interno, por eso el valor por defecto es 1024.

## Documentación adicional

* [`docs/01-guia-de-lo-que-hicimos.md`](docs/01-guia-de-lo-que-hicimos.md): qué se construyó y por qué, paso a paso.
* [`docs/02-cumplimiento-de-la-rubrica.md`](docs/02-cumplimiento-de-la-rubrica.md): cada criterio de la rúbrica con su evidencia.
* [`docs/03-comparacion-con-el-ejemplo-del-profesor.md`](docs/03-comparacion-con-el-ejemplo-del-profesor.md).
* [`docs/04-preguntas-para-defender-la-entrega.md`](docs/04-preguntas-para-defender-la-entrega.md).
