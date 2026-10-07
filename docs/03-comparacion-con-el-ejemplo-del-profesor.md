# Comparación con el notebook de ejemplo del profesor

Fuente: notebook de Colab **"Pista - Pre-entrega 1: Cliente de LLM robusto y asíncrono"** (compartido por el profesor). Una copia fiel está guardada en `ejemplos-del-profesor/pre-entrega-1_pista.ipynb`, **fuera de este repo** (es material de referencia, no parte de la entrega).

## Qué es el ejemplo

Un notebook de 34 celdas con la misma estructura que pide la consigna: `schemas` con Pydantic → `BaseLLMClient` abstracta → un cliente por proveedor → `AsyncLLMManager` (Factory) → pruebas en modo normal, streaming y resiliencia (API key inválida a propósito). Incluye **Gemini** como tercer proveedor y una tabla comparativa de los tres SDKs.

Tabla del profesor (resumen):

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| SDK async | `AsyncOpenAI` | `AsyncAnthropic` | `genai.Client(...).aio` |
| Dónde está el texto | `choices[0].message.content` | `content[0].text` | `response.text` |
| Streaming | `stream=True` + `async for` | `messages.stream(...)` (context manager) | `generate_content_stream` + `async for` |
| Rol "system" | mensaje con `role="system"` | parámetro `system` aparte | `system_instruction` en la config |
| Rol del asistente | `"assistant"` | `"assistant"` | `"model"` |
| ¿Tarjeta? | Sí | Sí | No (Free Tier) |

## Equivalencias entre el ejemplo y nuestro repo

| Ejemplo del profesor | Nuestro repo | Comentario |
|---|---|---|
| `LLMConfig` (provider, model, claves, temperature 0–2, max_tokens > 0) | `ModelConfig` + `ProviderSecrets` | Separamos configuración de claves; mismos rangos. |
| `ChatMessage` con validador de rol | `ChatMessage` con `Literal` + contenido no vacío | Misma idea. |
| `ModelResponse(provider, model, content, error: str)` | `ModelResponse(..., usage, finish_reason, error: LLMError)` | Nuestro error es un objeto con `kind` y `retryable`. |
| `BaseLLMClient.generate` / `generate_stream` | `BaseLLMClient.generate` / `generate_stream` / `stream` | **Mismos nombres** que el profesor para `generate` y `generate_stream`. |
| `generate_stream` devuelve `str`; en error emite `"[⚠️ Error durante el streaming: ...]"` | `generate_stream` hace lo mismo; además existe `stream()` estructurado | Idéntico comportamiento visible. |
| `OpenAIClient`, `AnthropicClient`, `GeminiClient` | Los tres, con la misma separación de rol/`system` | Gemini: `assistant`→`model`, `system_instruction`. |
| `AsyncLLMManager._crear_cliente` con `if/elif` | `_FACTORIES` (diccionario) | Mismo patrón Factory; el diccionario evita encadenar `if`. |
| Prueba de API key inválida | `python main.py --resiliencia` | Misma prueba, ahora reutilizable. |

## Qué agrega nuestro repo

* **Reintentos con backoff exponencial + jitter** para errores transitorios; no reintenta errores permanentes.
* **Timeout** por llamada (`asyncio.timeout`) y del SDK.
* **Clasificación de errores** (`auth`, `rate_limit`, `timeout`, `connection`, `bad_request`, `api`) en vez de un texto libre.
* **Mensaje `system` separado en Anthropic.** El ejemplo envía todos los mensajes (incluido `system`) dentro de `messages`, lo que la API de Anthropic rechaza si se usa un system prompt.
* **Validación de temperatura por proveedor** (Anthropic ≤ 1).
* **Compatibilidad con SDKs nuevos** (`temperature` condicional; ver más abajo).
* **Tests** automáticos con SDKs simulados, y **README + docs**.
* **Estructura en módulos** (`schemas.py`, `base.py`, clientes, `manager.py`, `main.py`) en lugar de un notebook.

## Puntos del ejemplo a no copiar tal cual

1. **Modelo de Anthropic:** usa `claude-3-5-sonnet-20241022`, que probablemente ya esté retirado. Usar un modelo vigente por variable de entorno (`LLM_MODEL`).
2. **`temperature` en Anthropic:** lo pasa siempre. Con la versión más reciente del SDK (`anthropic` 1.x) `messages.create()` ya no acepta ese argumento y lanzaría `TypeError`. Nuestro cliente detecta la firma del SDK y lo omite si no existe.
3. **Sin reintentos ni timeout:** un 429 o una red caída devuelve el error al primer intento.
4. **Salidas guardadas incompletas:** en el notebook el streaming de OpenAI aparece vacío y el de Gemini sale cortado ("La entropía / orden o aleatoriedad de"). Con Gemini 2.5 el límite `max_output_tokens` también cuenta los tokens de "razonamiento" interno, así que con `max_tokens=200` la respuesta puede truncarse. Por eso nuestro valor por defecto es 1024.
5. **Es un notebook ("Pista"), no un repo:** la entrega pide repositorio con `main.py`, `.env.example` y README.

## Cómo usar el ejemplo para estudiar

1. Leer primero la tabla comparativa de las celdas 2: resume las diferencias entre SDKs que todo el módulo busca abstraer.
2. Seguir el orden del notebook (schemas → base → clientes → manager → pruebas) y compararlo con `docs/01-guia-de-lo-que-hicimos.md`.
3. Ejecutarlo en Colab con tus propias claves para ver las salidas reales (en Colab las claves se piden con `getpass`, no quedan en el notebook).
