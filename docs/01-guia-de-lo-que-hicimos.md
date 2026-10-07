# Guía: qué construimos y por qué

Documento de estudio de la **Pre-entrega 1** (Módulo 1 · AI Engineering). Está pensado para que puedas explicar cada decisión con tus palabras, no solo mostrar que "anda".

---

## 1. El problema que resolvemos

Una app que habla con un LLM suele empezar así: se instancia el cliente de OpenAI directo en la lógica de negocio. Tres problemas (los nombra el material del curso):

1. **Acoplamiento:** cambiar de proveedor obliga a reescribir código.
2. **Difícil de testear:** si el cliente está "enterrado" en la lógica, es complicado simularlo.
3. **Inconsistencia:** cada SDK devuelve la respuesta con una forma distinta (`choices[0].message.content` en OpenAI, `content[0].text` en Anthropic, `response.text` en Gemini).

**Solución:** una capa intermedia. El resto de la aplicación habla con **un solo objeto** (`AsyncLLMManager`) y no sabe qué proveedor hay detrás.

Y además, todo **asíncrono**: en IA el tiempo se gasta *esperando* la red, no calculando. Mientras un pedido espera, el programa puede atender otros.

---

## 2. Mapa de archivos

| Archivo | Rol | Analogía |
|---|---|---|
| `schemas.py` | Define la forma de los datos (Pydantic): mensajes, configuración, respuesta, errores. | El "formulario" que todos deben completar igual. |
| `base.py` | Clase abstracta `BaseLLMClient`: timeout, reintentos, errores controlados, `generate()`, `stream()`, `generate_stream()`. | El **reglamento** que cumple todo proveedor. |
| `openai_client.py` | Traduce el reglamento al SDK de OpenAI. | Un empleado que habla "OpenAI". |
| `anthropic_client.py` | Idem para Anthropic. | Un empleado que habla "Anthropic". |
| `gemini_client.py` | Idem para Gemini (bonus). | Un empleado que habla "Gemini". |
| `manager.py` | `AsyncLLMManager`: elige qué empleado usar según la configuración (**Factory**). | El **recepcionista**. |
| `main.py` | Script de prueba. | La demo. |
| `tests/` | 28 tests con SDKs simulados (no gastan plata ni requieren keys). | El control de calidad. |

---

## 3. El viaje de una llamada (`generate`)

```
main.py
  └─ llm.generate("¿Qué es la entropía?")                 AsyncLLMManager
       ├─ convierte el str en [ChatMessage(role="user", ...)]   (Pydantic valida)
       └─ self._client.generate(messages)                  BaseLLMClient (base.py)
            └─ for intento in 0..max_retries:
                 ├─ async with asyncio.timeout(timeout_s):       ← corta si se cuelga
                 │     return await self._call(messages)         ← SDK del proveedor (await)
                 └─ si falla → _classify(exc) → LLMError
                      ├─ ¿transitorio? (429, red, timeout, 5xx) → espera (backoff) y reintenta
                      └─ ¿permanente? (auth, bad_request) → corta YA
            └─ si se agotan los intentos → ModelResponse(error=LLMError)   ← nunca una excepción
```

Ideas clave:

* **`await`** en cada llamada de red = el event loop queda libre para otras tareas (si usáramos la versión síncrona, congelaríamos todo: el "bloqueo del loop").
* **`asyncio.timeout`** (Python 3.11+) = ninguna llamada queda abierta indefinidamente.
* **Backoff exponencial con jitter**: espera ~0,5 s, ~1 s, ~2 s (+ un poco de azar). Evita martillar a un servicio saturado y evita que muchos clientes reintenten todos a la vez.
* **Reintentar solo lo transitorio.** Una API key inválida no se arregla reintentando: solo gastaría tiempo.

---

## 4. El viaje del streaming

Dos formas de consumirlo, ambas generadores asincrónicos (`async for`):

| Método | Devuelve | Cuándo usarlo |
|---|---|---|
| `generate_stream(msgs)` | `str` (texto plano) | **Es el que pide la consigna.** Simple: `async for texto in llm.generate_stream(...)`. |
| `stream(msgs)` | `StreamChunk` (`delta`, `finish_reason`, `error`) | Cuando querés metadatos o detectar el error con un campo en vez de un texto. |

Cómo se produce cada fragmento (ejemplo Anthropic):

```python
async with self._client.messages.stream(...) as stream:   # context manager del SDK
    async for text in stream.text_stream:                 # llega token a token por la red
        yield StreamChunk(delta=text)                     # yield = lo entrega YA, sin esperar el resto
```

**Por qué importa:** mide el *Time To First Token* (TTFT). Con streaming el usuario ve la primera palabra en fracciones de segundo, aunque la respuesta completa tarde varios segundos.

**Regla de reintento en streaming:** solo se reintenta si el fallo ocurre **antes del primer token**. Si ya mostramos texto y reintentáramos, el usuario vería el texto duplicado. En ese caso se cierra con un aviso de error.

---

## 5. Patrones de diseño usados (con nombre, para poder explicarlos)

* **Factory Pattern** (`manager.py`): el sistema decide en tiempo de ejecución qué cliente crear según la configuración. Sumar un proveedor = una clase + una línea en `_FACTORIES`.
* **Clase base abstracta** (`ABC` + `@abstractmethod`): fija el contrato que todo proveedor debe cumplir.
* **Método plantilla (Template Method)**: `base.py` implementa **una sola vez** timeout + reintentos + manejo de errores; cada proveedor solo aporta tres piezas chicas: `_call` (la llamada al SDK), `_stream_raw` (el streaming del SDK) y `_map_error` (cómo clasificar sus excepciones). Resultado: no hay código de reintentos copiado tres veces.
* **Import diferido**: `manager.py` importa el SDK de cada proveedor recién cuando se lo necesita. Si solo usás Anthropic, no hace falta tener instalado `openai`.
* **Object con `async with`**: `AsyncLLMManager` cierra los clientes HTTP al terminar (`aclose`).

---

## 6. Validación con Pydantic (`schemas.py`)

| Modelo | Qué valida |
|---|---|
| `ChatMessage` | `role` solo `system/user/assistant`; `content` no vacío. |
| `ModelConfig` | `temperature` entre 0 y 2 (consigna); `max_tokens ≥ 1`; `timeout_s > 0`; `max_retries` 0–5. Además **por proveedor**: Anthropic no admite más de 1. |
| `ProviderSecrets` | Claves como `SecretStr`: si imprimís el objeto o hay un error, la clave **no aparece**. |
| `ModelResponse` | Forma única de respuesta para todos los proveedores (`content`, `usage`, `finish_reason`, `error`, y `.ok`). |
| `LLMError` | Error controlado: `kind`, `message`, `retryable`. |

La validación ocurre **antes** de llamar a la API: un valor inválido falla al instante, sin gastar una llamada.

---

## 7. Errores controlados (en vez de crashes)

La consigna: *"no dejes que un error de API Key inválida o límite de cuota rompa el loop principal"*.

* `generate()` **nunca lanza**: devuelve un `ModelResponse` con `error` completo (`auth`, `rate_limit`, `timeout`, `connection`, `bad_request`, `api`).
* `stream()` / `generate_stream()` **nunca lanzan**: terminan con un error (como `StreamChunk.error` o como un último texto de aviso).
* Lo único que se deja pasar es `asyncio.CancelledError`, a propósito: tragarse una cancelación rompe el cierre ordenado del programa.

Prueba real que hicimos: ejecutar `main.py` con una API key inválida contra la API de Anthropic → devuelve `auth: API key invalida o sin permisos`, sin crash y sin reintentos inútiles.

---

## 8. Problemas reales que aparecieron (y qué aprendimos)

Estos son buenos ejemplos para contar en una defensa oral: muestran criterio, no solo código que funciona.

1. **El SDK de Anthropic cambió.** Con la versión más reciente, `messages.create()` ya **no acepta `temperature`**. El ejemplo del profesor, que lo pasa siempre, fallaría con `TypeError`. Solución: el cliente detecta con `inspect.signature` si el SDK instalado lo soporta; si no, lo omite y avisa. *Lección: los SDKs de IA cambian rápido; no asumir.*
2. **Un test que "pasaba" pero no probaba nada.** El primer test de concurrencia parcheaba `asyncio.sleep` de forma global, así que la espera "real" también era falsa. Lo detectamos al revisarlo y lo corregimos parcheando solo la espera entre reintentos. Ahora demuestra de verdad que 5 llamadas simultáneas de 0,2 s tardan ~0,2 s y no ~1 s. *Lección: desconfiar de un test que pasa a la primera.*
3. **`httpx2`.** Los SDKs más nuevos usan `httpx2` en lugar de `httpx`. Los tests se adaptan a ambos.
4. **Gemini no se pudo probar en vivo** desde el entorno donde se construyó (la red bloqueaba `googleapis.com`). Se verificó con SDK simulado. *Lección: decir con claridad qué se probó en real y qué no.* Conviene que lo pruebes vos con una key gratuita de Google.

---

## 9. Cómo correrlo

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # completá tu clave
python main.py                  # proveedor del .env
python main.py --provider gemini
python main.py --resiliencia    # además prueba una key inválida a propósito
pytest                          # 28 tests, sin keys
```

---

## 10. Glosario rápido

* **Corrutina (`async def`)**: función que puede pausarse con `await` sin bloquear.
* **Event loop**: el "gestor del restaurante" que reparte el trabajo entre corrutinas.
* **I/O-bound**: tareas que esperan red/disco (llamadas a LLM). Asyncio ayuda. En CPU-bound (cálculo pesado) no.
* **Generador asíncrono**: función con `yield` dentro de `async def`; se consume con `async for`.
* **TTFT**: tiempo hasta el primer token.
* **Rate limit (429)**: el proveedor limita cuántas llamadas por unidad de tiempo.
* **Backoff exponencial**: esperar cada vez más entre reintentos.
* **Jitter**: un poco de aleatoriedad en esa espera.
* **`SecretStr`**: tipo de Pydantic que oculta el valor al imprimirlo.
* **Factory**: patrón que decide qué objeto crear según la configuración.
