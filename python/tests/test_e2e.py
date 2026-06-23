"""End-to-end test against a real ``mlxcel serve`` process.

Skipped unless ``MLXCEL_BIN`` points at a built mlxcel binary. Set
``MLXCEL_E2E_MODEL`` to choose the model (defaults to a small 4-bit checkpoint).
The first run may download weights from Hugging Face, so the startup timeout is
generous.

Run explicitly with::

    MLXCEL_BIN=/path/to/mlxcel pytest python/tests/test_e2e.py -m e2e
"""

from __future__ import annotations

import os

import pytest

import mlxcel

pytestmark = pytest.mark.e2e

MLXCEL_BIN = os.environ.get("MLXCEL_BIN")
MODEL = os.environ.get("MLXCEL_E2E_MODEL", "mlx-community/Qwen3-0.6B-4bit")

skip_no_bin = pytest.mark.skipif(
    not MLXCEL_BIN, reason="MLXCEL_BIN not set; skipping real-binary e2e test"
)


@skip_no_bin
def test_managed_generate_returns_text() -> None:
    with mlxcel.LLM(MODEL, binary=MLXCEL_BIN, startup_timeout=900.0) as llm:
        assert llm.model
        text = llm.generate("The capital of France is", max_tokens=16, temperature=0.0)
        assert isinstance(text, str)
        assert text.strip() != ""


@skip_no_bin
def test_managed_chat_and_stream() -> None:
    with mlxcel.LLM(MODEL, binary=MLXCEL_BIN, startup_timeout=900.0) as llm:
        reply = llm.chat([{"role": "user", "content": "Say hello."}], max_tokens=16)
        assert reply.strip() != ""
        streamed = "".join(llm.stream("Count: 1 2 3", max_tokens=16))
        assert isinstance(streamed, str)
