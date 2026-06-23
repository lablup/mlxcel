"""Structured output: constrain decoding to a JSON schema via response_format.

The server's llguidance-backed constrained decoding honors ``response_format``.
This example asks for a JSON object matching a schema and parses the result.

    python python/examples/structured_output.py
"""

from __future__ import annotations

import json

import mlxcel

MODEL = "mlx-community/Qwen3-4B-4bit"

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
            "additionalProperties": False,
        },
    },
}


def main() -> None:
    with mlxcel.LLM(MODEL) as llm:
        reply = llm.chat(
            [
                {
                    "role": "user",
                    "content": "Invent a fictional person. Reply as JSON with name, age, city.",
                }
            ],
            max_tokens=128,
            temperature=0.7,
            response_format=SCHEMA,
        )
        print("raw reply:", reply)
        parsed = json.loads(reply)
        print("parsed:", parsed)

        # The same call via the raw OpenAI client (escape hatch):
        oai = llm.openai_client
        completion = oai.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": "Another fictional person as JSON."}],
            response_format=SCHEMA,
            max_tokens=128,
        )
        print("via openai_client:", completion.choices[0].message.content)


if __name__ == "__main__":
    main()
