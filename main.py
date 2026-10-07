"""Script de validacion: pregunta corta en modo normal y en streaming.

Uso:
    python main.py                       # proveedor segun LLM_PROVIDER del .env
    python main.py --provider openai     # forzar proveedor (openai | anthropic | gemini)
    python main.py --resiliencia         # ademas, prueba una API key invalida a proposito
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider, ProviderSecrets

PREGUNTA = "¿Qué es la entropía?"
MENSAJES = [
    ChatMessage(role="system", content="Sos un profesor claro y conciso. Respondé en español."),
    ChatMessage(role="user", content=PREGUNTA),
]


async def demo_normal(llm: AsyncLLMManager) -> None:
    print("\n=== Modo normal ===")
    t0 = time.perf_counter()
    resp = await llm.generate(MENSAJES)
    if not resp.ok:
        print(f"[ERROR controlado] {resp.error.kind}: {resp.error.message}")
        return
    print(resp.content)
    print(
        f"\n[{resp.provider.value} | {resp.model} | "
        f"{resp.usage.input_tokens} in / {resp.usage.output_tokens} out | "
        f"{time.perf_counter() - t0:.2f}s | fin: {resp.finish_reason}]"
    )


async def demo_streaming(llm: AsyncLLMManager) -> None:
    print("\n=== Modo streaming ===")
    t0 = time.perf_counter()
    ttft: float | None = None  # Time To First Token
    # generate_stream devuelve texto plano, token a token, apenas llega de la red.
    async for fragmento in llm.generate_stream(MENSAJES):
        if ttft is None and not fragmento.startswith("\n[⚠️"):
            ttft = time.perf_counter() - t0
        print(fragmento, end="", flush=True)
    total = time.perf_counter() - t0
    if ttft is not None:
        print(f"\n\n[TTFT: {ttft:.2f}s | total: {total:.2f}s]")


# Nombre del campo de ProviderSecrets que corresponde a cada proveedor.
_SECRET_FIELD = {
    Provider.OPENAI: "openai_api_key",
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.GEMINI: "google_api_key",
}


async def demo_resiliencia(provider: Provider) -> None:
    """API key invalida a proposito: el programa NO debe romperse."""
    print("\n=== Prueba de resiliencia (API key invalida) ===")
    config = ModelConfig(provider=provider, model="modelo-de-prueba", max_retries=1)
    secrets = ProviderSecrets(**{_SECRET_FIELD[provider]: "clave-invalida-a-proposito"})
    async with AsyncLLMManager(config, secrets) as llm:
        resp = await llm.generate("hola")
    print("¿El programa siguió vivo? Sí")
    print(f"Error capturado (sin crash): {resp.error.kind} -> {resp.error.message}")


async def main(provider: str | None, resiliencia: bool) -> int:
    load_dotenv()  # carga el .env (las claves nunca van en el codigo)
    try:
        llm = AsyncLLMManager.from_env(provider)
    except ValueError as exc:  # clave faltante, temperatura invalida, etc.
        print(f"[Configuracion invalida] {exc}")
        return 1

    async with llm:
        print(f"Proveedor: {llm.config.provider.value} | modelo: {llm.config.model}")
        await demo_normal(llm)
        await demo_streaming(llm)
    if resiliencia:
        await demo_resiliencia(llm.config.provider)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=[p.value for p in Provider], default=None)
    parser.add_argument("--resiliencia", action="store_true", help="prueba con API key invalida")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.provider, args.resiliencia)))
