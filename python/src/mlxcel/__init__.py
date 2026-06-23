"""mlxcel: a thin Python client over the mlxcel OpenAI-compatible server.

Drive a local or remote ``mlxcel serve`` process with minimal boilerplate. The
primary entry point is :class:`LLM` (synchronous) and its async twin
:class:`AsyncLLM`. Both manage a local server subprocess (managed mode) or
connect to a running one (connect mode), auto-discover the served model id, and
expose the raw OpenAI client for the full API surface.

    import mlxcel

    with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
        print(llm.generate("def fib(n):", max_tokens=128))
"""

from __future__ import annotations

from ._async_client import AsyncLLM
from ._client import LLM
from .errors import MlxcelError, MlxcelServerError, MlxcelTimeoutError

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "AsyncLLM",
    "MlxcelError",
    "MlxcelServerError",
    "MlxcelTimeoutError",
    "__version__",
]
