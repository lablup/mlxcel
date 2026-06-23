"""Managed-mode quickstart: spawn a local server and generate text.

Run with a model available locally or downloadable from Hugging Face::

    python python/examples/quickstart.py
"""

from __future__ import annotations

import mlxcel

MODEL = "mlx-community/Qwen3-4B-4bit"


def main() -> None:
    # Managed mode: mlxcel spawns and supervises a local `mlxcel serve` process,
    # waits until it is ready, and shuts it down on exit.
    with mlxcel.LLM(MODEL) as llm:
        print("resolved model id:", llm.model)
        print("available models:", llm.models())

        text = llm.generate("def fib(n):", max_tokens=128, temperature=0.7)
        print("\ncompletion:\n", text)

        reply = llm.chat(
            [{"role": "user", "content": "Give me one fact about Apple Silicon."}],
            max_tokens=64,
        )
        print("\nchat reply:\n", reply)

        ids = llm.tokenize("hello world")
        print("\ntokens:", ids)
        print("round-trip:", llm.detokenize(ids))


if __name__ == "__main__":
    main()
