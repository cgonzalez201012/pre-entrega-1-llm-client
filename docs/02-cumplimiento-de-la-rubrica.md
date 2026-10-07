# Cumplimiento de la rúbrica y de la consigna

Aprobación: **70 / 100**. Esta tabla cruza cada criterio con la evidencia concreta en el repo y cómo demostrarla.

## Rúbrica (4 criterios)

### 1. Abstracción y gestión de proveedores · 35 %

| Se pide | Dónde está | Cómo demostrarlo |
|---|---|---|
| Clase `AsyncLLMManager` que cargue OpenAI **o** Anthropic según configuración | `manager.py` → `AsyncLLMManager`, `from_env()`, `_FACTORIES` | `python main.py --provider anthropic` y `--provider openai`: mismo código, distinto proveedor. |
| Interfaz común | `base.py` → `BaseLLMClient` (ABC) | Los 3 clientes heredan de ella y devuelven el mismo `ModelResponse`. |
| Intercambiabilidad | `openai_client.py`, `anthropic_client.py` (+ `gemini_client.py` bonus) | Test `test_factory_elige_clase_segun_config`. |
| Diferencias entre SDKs absorbidas | `anthropic_client.py`: `system` separado, `content[]` → texto; OpenAI: `choices[0]` | Test `test_generate_normaliza_respuesta_anthropic`. |
| Falta de clave detectada con mensaje claro | `schemas.py` → `ProviderSecrets.for_provider` | Test `test_factory_falla_claro_si_falta_la_key`. |

### 2. Implementación de streaming asíncrono · 25 %

| Se pide | Dónde está | Cómo demostrarlo |
|---|---|---|
| Método que devuelve un **generador asíncrono** de tokens | `base.py` → `generate_stream()` (texto) y `stream()` (estructurado); `manager.py` los expone | `async for t in llm.generate_stream(...)` en `main.py`. |
| Uso de `yield` dentro de un `async for` sobre el stream del SDK | `*_client.py` → `_stream_raw()` | Leer el método en cualquiera de los 3 clientes. |
| Fragmentos de texto conforme llegan | `main.py` imprime con `flush=True` y mide TTFT | Test `test_streaming_emite_fragmentos_en_orden`, `test_generate_stream_devuelve_texto_plano`. |
| Robusto ante fallos a mitad de stream | `base.py` → `stream()` | Test `test_streaming_no_reintenta_a_mitad_y_cierra_con_error`. |

### 3. Validación de esquemas con Pydantic · 20 %

| Se pide | Dónde está | Cómo demostrarlo |
|---|---|---|
| `ChatMessage` (role, content) | `schemas.py` | Test `test_mensaje_vacio_y_rol_invalido`. |
| `ModelResponse` | `schemas.py` | Todas las respuestas pasan por este modelo. |
| Parámetros del modelo validados (**temperatura 0–2**, `max_tokens`, etc.) | `schemas.py` → `ModelConfig` | Test `test_temperature_range_por_proveedor`. |
| Claves protegidas | `ProviderSecrets` con `SecretStr` | Test `test_secretstr_no_se_filtra`. |

> **Nota (decisión consciente):** el rango general es 0–2 como pide la consigna. Además se refina por proveedor porque la API de Anthropic solo acepta hasta 1; así el error se detecta antes de gastar una llamada. Si un corrector prueba `temperature=5`, falla en todos los proveedores; con `1.5` falla en Anthropic y pasa en OpenAI/Gemini.

### 4. Resiliencia y manejo de errores · 20 %

| Se pide | Dónde está | Cómo demostrarlo |
|---|---|---|
| Capturar errores de red | `_map_error` en cada cliente → `kind="connection"` | Test `test_connection_error_es_transitorio`. |
| Capturar **rate limit** (429) | `_map_error` → `kind="rate_limit"`, reintentable | Tests `test_rate_limit_se_reintenta_y_se_recupera`, `test_rate_limit_agotado_devuelve_error_no_excepcion`. |
| Devolver error controlado, **sin crash** | `generate()` nunca lanza | `python main.py --resiliencia` (key inválida → `auth`, el programa sigue). |
| No reintentar errores permanentes | `retryable=False` en `auth`/`bad_request` | Test `test_auth_error_no_se_reintenta`. |
| Timeouts | `asyncio.timeout` en `generate()` + timeout del SDK | Test `test_timeout_devuelve_error_controlado`. |

---

## Checklist de la consigna ("Qué entregás" + "Pasos")

| Ítem de la consigna | Estado | Archivo |
|---|---|---|
| Repo con clientes async (OpenAI + Anthropic) | ✅ | `openai_client.py`, `anthropic_client.py` |
| `schemas.py` (Pydantic) | ✅ | `schemas.py` |
| `.env.example` | ✅ | `.env.example` |
| `main.py` que pruebe modo normal **y** streaming con "¿Qué es la entropía?" | ✅ | `main.py` |
| `README.md` (cómo correrlo + variables de entorno) | ✅ | `README.md` |
| Entorno virtual Python 3.12 + `openai`, `anthropic`, `pydantic`, `python-dotenv` | ✅ | `requirements.txt` |
| Esquema Pydantic con temperatura 0–2, `max_tokens`, etc. | ✅ | `ModelConfig` |
| Clase `AsyncLLMManager` que carga OpenAI/Anthropic según configuración | ✅ | `manager.py` |
| Generación con streaming usando `yield` | ✅ | `generate_stream()` |
| Capturar excepciones de red y rate limit → error controlado | ✅ | `base.py` + `_map_error` |
| `BaseLLMClient` abstracta con `async def generate()` | ✅ | `base.py` |
| `ChatMessage` y `ModelResponse` en `schemas.py` | ✅ | `schemas.py` |
| Todas las llamadas `async/await`, sin bloquear el loop | ✅ | SDKs `AsyncOpenAI`/`AsyncAnthropic`/`client.aio` |
| Sin API keys en el repo | ✅ | `.env` ignorado por `.gitignore`; solo `.env.example` |
| (Bonus) tercer proveedor, tests, docs | ✅ | `gemini_client.py`, `tests/`, `docs/` |

## Antes de subir a GitHub (checklist final)

- [ ] Correr `python main.py` con tu clave y confirmar que ambos modos responden.
- [ ] Correr `python main.py --resiliencia`.
- [ ] Correr `pytest` (debe dar 28 passed).
- [ ] Verificar que **no existe** un archivo `.env` dentro de lo que subís (`git status` no debe listarlo).
- [ ] Hacer un `git clone` limpio en otra carpeta y seguir el README para ver que funciona.
- [ ] (Opcional) Pegar en el README una captura de la salida real de `main.py`.
