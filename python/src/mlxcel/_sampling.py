"""Map Python keyword arguments to OpenAI request fields.

The OpenAI Python SDK accepts a fixed set of sampling parameters on
``completions.create`` and ``chat.completions.create``. mlxcel and the
underlying llama-server expose a few extra knobs (``top_k``, ``min_p``,
``repetition_penalty``, DRY sampling) that are not part of the OpenAI schema;
those are forwarded through the SDK's ``extra_body`` escape hatch so the server
receives them verbatim in the JSON request body.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Sampling kwargs accepted as top-level fields by BOTH the completions and the
# chat-completions endpoints.
_OPENAI_FIELDS: Tuple[str, ...] = (
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "n",
    "logprobs",
    "top_logprobs",
)

# Fields the OpenAI SDK accepts only on ``chat.completions.create``. For the
# plain ``completions.create`` endpoint the SDK raises ``TypeError`` on these,
# so they are routed through ``extra_body`` instead (the mlxcel/llguidance
# server reads constrained-decoding settings from the raw request body).
_CHAT_ONLY_FIELDS: Tuple[str, ...] = ("response_format",)

# Server-specific sampling knobs forwarded via ``extra_body``. These are not in
# the OpenAI schema; the server reads them from the raw request body.
_EXTRA_BODY_FIELDS: Tuple[str, ...] = (
    "top_k",
    "min_p",
    "repetition_penalty",
    "repeat_penalty",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
)


def build_params(kwargs: Dict[str, Any], *, chat: bool = True) -> Dict[str, Any]:
    """Split caller kwargs into OpenAI request fields plus an ``extra_body`` payload.

    Known OpenAI fields are passed through as top-level request parameters.
    Recognized server-specific knobs are collected into ``extra_body``. An
    explicit ``extra_body=`` mapping supplied by the caller is merged in and
    takes precedence, so advanced users can send arbitrary server fields. Any
    remaining unknown kwargs are also routed into ``extra_body`` rather than
    silently dropped, since the OpenAI SDK rejects unexpected top-level keys.

    Chat-only fields (see :data:`_CHAT_ONLY_FIELDS`, e.g. ``response_format``)
    are valid top-level parameters for ``chat.completions.create`` but not for
    ``completions.create``. When ``chat`` is False they are routed through
    ``extra_body`` so the server still receives them while the SDK does not
    raise ``TypeError`` on an unexpected keyword.

    Args:
        kwargs: Raw keyword arguments from a generate/chat call.
        chat: True when building params for ``chat.completions.create``; False
            when building for the plain ``completions.create`` endpoint.

    Returns:
        A new dict suitable for splatting into the OpenAI ``create`` call.
    """
    params: Dict[str, Any] = {}
    extra_body: Dict[str, Any] = {}

    # A caller-provided extra_body is applied last so it wins on conflicts.
    caller_extra_body = kwargs.pop("extra_body", None)

    for key, value in kwargs.items():
        if value is None:
            continue
        if key in _OPENAI_FIELDS:
            params[key] = value
        elif key in _CHAT_ONLY_FIELDS:
            # Top-level for chat; the completions endpoint only accepts it in
            # the raw body, so route it through extra_body there.
            if chat:
                params[key] = value
            else:
                extra_body[key] = value
        else:
            # Recognized server knob or an unknown key: send it in the body so
            # the server can interpret it instead of the SDK rejecting it.
            extra_body[key] = value

    if caller_extra_body:
        if not isinstance(caller_extra_body, dict):
            raise TypeError("extra_body must be a mapping")
        extra_body.update(caller_extra_body)

    if extra_body:
        params["extra_body"] = extra_body

    return params


__all__ = ["build_params"]
