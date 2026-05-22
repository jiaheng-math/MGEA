from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect LightRAG/OpenAI compatibility and test the configured OpenAI-compatible gateway."
    )
    parser.add_argument("--config", required=True, help="Study YAML config containing lightrag_* settings.")
    parser.add_argument("--prompt", default='Return JSON: {"high_level_keywords": ["Paris"], "low_level_keywords": ["France"]}')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    print_runtime_info()
    print_lightrag_info()
    asyncio.run(test_gateway(config=config, prompt=str(args.prompt)))


def print_runtime_info() -> None:
    print("=== Python ===")
    print(sys.version)
    print("executable:", sys.executable)
    print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))
    try:
        import openai

        print("openai version:", getattr(openai, "__version__", "unknown"))
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "dummy")
        chat_completions = getattr(client.chat, "completions", None)
        beta_chat_completions = getattr(getattr(client, "beta", None), "chat", None)
        beta_chat_completions = getattr(beta_chat_completions, "completions", None)
        print("client.chat.completions has parse:", hasattr(chat_completions, "parse"))
        print("client.beta.chat.completions has parse:", hasattr(beta_chat_completions, "parse"))
    except Exception as exc:
        print("openai inspection failed:", type(exc).__name__, str(exc))


def print_lightrag_info() -> None:
    print("\n=== LightRAG ===")
    try:
        import lightrag
        from lightrag import QueryParam
        from lightrag.llm.openai import openai_complete_if_cache

        print("lightrag file:", getattr(lightrag, "__file__", "unknown"))
        print("lightrag version:", getattr(lightrag, "__version__", "unknown"))
        print("QueryParam signature:", inspect.signature(QueryParam))
        print("openai_complete_if_cache signature:", inspect.signature(openai_complete_if_cache))
        source = inspect.getsource(openai_complete_if_cache)
        print("openai_complete_if_cache source head:")
        print("\n".join(source.splitlines()[:80]))
    except Exception as exc:
        print("lightrag inspection failed:", type(exc).__name__, str(exc))


async def test_gateway(*, config: dict[str, Any], prompt: str) -> None:
    print("\n=== Gateway ===")
    from openai import AsyncOpenAI

    model = str(config.get("lightrag_llm_model", "gpt-4.1"))
    base_url = str(config.get("lightrag_llm_base_url", "")) or None
    api_key = str(config.get("lightrag_llm_api_key", "")) or os.environ.get("OPENAI_API_KEY")
    print("model:", model)
    print("base_url:", base_url)
    print("api_key set:", bool(api_key))

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": prompt}]
    await try_call(
        "plain chat.completions.create",
        client.chat.completions.create(model=model, messages=messages),
    )
    await try_call(
        "json_object chat.completions.create",
        client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        ),
    )


async def try_call(label: str, awaitable) -> None:
    print(f"\n--- {label} ---")
    try:
        response = await awaitable
        print("response type:", type(response).__name__)
        content = extract_content(response)
        print("content:", content[:1000])
        try:
            parsed = json.loads(content)
            print("json parse ok:", parsed)
        except Exception as exc:
            print("json parse failed:", type(exc).__name__, str(exc))
    except Exception as exc:
        print("call failed:", type(exc).__name__, str(exc))


def extract_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    if isinstance(response, dict):
        return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return str(response)


if __name__ == "__main__":
    main()
