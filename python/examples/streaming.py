"""Streaming example: print completion and chat deltas as they arrive.

python python/examples/streaming.py
"""

from __future__ import annotations

import sys

import mlxcel

MODEL = "mlx-community/Qwen3-4B-4bit"


def main() -> None:
    with mlxcel.LLM(MODEL) as llm:
        print("completion stream:")
        for delta in llm.stream("Write a haiku about autumn.", max_tokens=64):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print("\n")

        print("chat stream:")
        messages = [{"role": "user", "content": "List three uses for a Raspberry Pi."}]
        for delta in llm.chat_stream(messages, max_tokens=128):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
