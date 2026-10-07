# Preguntas para defender la entrega (con respuestas)

Preguntas que un profesor o revisor podría hacer, con respuestas cortas para practicar con tus palabras.

**1. ¿Por qué usar `asyncio` si llamar a un LLM es solo una función?**
Porque casi todo el tiempo se va *esperando* la red. Con `await`, mientras una llamada espera, el event loop atiende otras (otros usuarios u otras llamadas). No acelera cálculo (CPU-bound), acelera la espera (I/O-bound). Está demostrado en el test de concurrencia: 5 llamadas de 0,2 s tardan ~0,2 s y no ~1 s.

**2. ¿Qué pasa si llamo al cliente síncrono dentro de una función `async`?**
Bloquea el event loop: todo el programa queda congelado mientras el LLM "piensa". Por eso usamos `AsyncOpenAI`/`AsyncAnthropic` con `await`. Si hubiera que usar una librería síncrona, se envuelve en `asyncio.to_thread()`.

**3. ¿Qué patrón usaste para cambiar de proveedor y por qué?**
Factory Pattern más una clase base abstracta. El código de negocio solo conoce `AsyncLLMManager`; cambiar de OpenAI a Anthropic es cambiar una variable (`LLM_PROVIDER`). Evita el acoplamiento y facilita testear.

**4. ¿Por qué `generate()` no lanza excepciones?**
Porque un error de API key o de cuota no debe romper el loop principal de la aplicación. Se devuelve un `ModelResponse` con un `LLMError` (con `kind` y `retryable`) para que quien llama decida. Lo único que se deja pasar es `CancelledError`, para no romper el cierre ordenado del programa.

**5. ¿Qué errores reintentás y cuáles no?**
Reintento los **transitorios**: 429 (rate limit), caídas de red, timeouts y 5xx. No reintento los **permanentes**: clave inválida o request mal formado, porque solo gastarían tiempo y dinero. El espacio entre reintentos crece (backoff exponencial) con un poco de azar (jitter).

**6. ¿Por qué el streaming solo se reintenta antes del primer token?**
Si ya se mostró texto al usuario y se reintenta, el texto se duplicaría. En ese caso se cierra con un aviso de error.

**7. ¿Cuál es la diferencia entre `generate_stream()` y `stream()`?**
`generate_stream()` devuelve texto plano (lo que pide la consigna). `stream()` devuelve `StreamChunk` con `delta`, `finish_reason` y `error`, útil para quien necesita metadatos. La primera está construida sobre la segunda.

**8. ¿Qué es el Time To First Token y por qué importa?**
Es el tiempo hasta que aparece el primer token. Con streaming el usuario ve contenido enseguida aunque la respuesta total tarde; es una de las métricas de experiencia más importantes en productos de IA.

**9. ¿Para qué sirve Pydantic acá?**
Para validar **antes** de gastar una llamada: roles válidos, mensajes no vacíos, temperatura en rango, `max_tokens` positivo. Y `SecretStr` evita que las API keys aparezcan en logs o errores. También da una forma única de respuesta (`ModelResponse`) para los tres proveedores.

**10. ¿Por qué el rango de temperatura es distinto en Anthropic?**
La consigna pide 0–2, que es el rango de OpenAI. La API de Anthropic solo acepta hasta 1; si lo dejáramos pasar, fallaría recién al llamar. Con el refinamiento por proveedor, se detecta antes de gastar la llamada.

**11. ¿Qué diferencias entre los SDKs tuviste que abstraer?**
Dónde está el texto (`choices[0].message.content` vs `content[0].text` vs `response.text`), el mensaje `system` (en la lista en OpenAI; parámetro aparte en Anthropic; `system_instruction` en Gemini), el rol del asistente (`model` en Gemini), la forma de hacer streaming (`stream=True` vs context manager) y que `max_tokens` es obligatorio en Anthropic.

**12. ¿Qué encontraste de inesperado?**
Que la versión más nueva del SDK de Anthropic ya no acepta `temperature` en `create()`. Lo resolví detectando la firma del SDK instalado y omitiendo el argumento (con una advertencia). Y que mi primer test de concurrencia no probaba nada, porque parcheaba `asyncio.sleep` globalmente; lo corregí.

**13. ¿Cómo probaste el código sin gastar créditos?**
Con tests que simulan los SDKs (`pytest`, 28 casos): validación, factory, normalización, streaming, reintentos, timeout, errores y concurrencia. Además probé en vivo contra la API de Anthropic con una clave inválida a propósito para ver el error controlado real.

**14. ¿Qué mejorarías para producción?**
Un límite de concurrencia con `asyncio.Semaphore` para respetar rate limits cuando se disparan muchas llamadas, `TaskGroup` para cancelación ordenada, logging estructurado y métricas (latencia, tokens, costo), y trazas con herramientas de observabilidad (los módulos siguientes del curso).

**15. ¿Por qué hay un `GeminiClient` si la consigna no lo pide?**
Es un bonus que demuestra que la abstracción realmente funciona: sumar un proveedor fue crear una clase y una línea en el registro, sin tocar el resto. Además Gemini tiene capa gratuita, lo que permite probar la intercambiabilidad sin tarjeta.
