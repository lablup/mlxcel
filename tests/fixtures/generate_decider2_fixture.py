#!/usr/bin/env python3
"""Generate the Decider-2b prompt-id and parity fixture from mlx-vlm (#2359).

Run under the GPU lock from the repository root:

    /usr/bin/lockf -k <gpu lock> env PYTHONPATH=references/mlx-vlm \
        uv run --python 3.12 --with mlx --with transformers --with numpy ... \
        python tests/fixtures/generate_decider2_fixture.py \
        models/decider-2b tests/fixtures/decision/decider2_parity.json

The output pins the exact prompt token ids upstream's ``Decider2._prompt``
builds (a 3-option and a 12-option choice, the score options, one isolated
score row, and an annotated 8-item list state) and the full ``predict``
answers for two requests. ``src/decision/decider2_tests.rs`` reads it.
"""

import json
import re
import sys

from mlx_vlm.decision import predict
from mlx_vlm.models.decider2.decider2 import Decider2, _annotate, _render_question
from mlx_vlm.utils import load

model_dir, out_path = sys.argv[1], sys.argv[2]
model, processor = load(model_dir)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
d = Decider2(model, tok)

TICKET = "Please refund my duplicate charge"
choice3 = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": ["billing", "technical", "sales"],
}
choice12 = {
    "type": "choice",
    "instructions": "Which product area is this about?",
    "criteria": {
        "accounts": "Sign-in, passwords and profiles",
        "billing": None,
        "cards": "",
        "deposits": {"limit": 5000, "currency": "EUR"},
        "exports": ["csv", "pdf"],
        "fraud": "Unrecognised transactions",
        "loans": 3,
        "mobile": "The iOS and Android apps",
        "notifications": True,
        "savings": "Interest and goals",
        "transfers": "Domestic and international payments",
        "wire": "Zahlungen über SWIFT",
    },
}
score = {
    "type": "score",
    "instructions": "How urgent is this ticket?",
    "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
}
boolean = {
    "type": "bool",
    "instructions": "Does the customer ask for money back?",
    "criteria": {"false": "No refund is requested", "true": "A refund is requested"},
}
list_state = [
    {"id": 1, "text": "Card declined at checkout", "tags": ["card", "payment"]},
    {"id": 2, "text": "Café receipt missing", "amount": 4.5},
    "free text entry",
    7,
    None,
    {"id": 3, "nested": [1, 2, 3, 4, 5, 6, 7, 8]},
    {"_index": "kept first", "id": 4},
    1e-05,
]


def prompt_case(name, state, spec, level=None):
    q = _render_question(spec)
    context = (
        state
        if isinstance(state, str)
        else json.dumps(_annotate(state), ensure_ascii=False)
    )
    if level is None:
        instruction, options = q["instruction"], q["options"]
    else:
        text = re.sub(r"^\s*-?\d+\s*:\s*", "", q["legend"][level])
        instruction = f"{q['instruction']}\nProposed answer: {text}\nDoes the proposed answer fit?"
        options = ["no", "yes"]
    return {
        "name": name,
        "state": state,
        "question": spec,
        "level": level,
        "context": context,
        "ids": d._prompt(context, instruction, options, 32768),
    }


prompts = [
    prompt_case("choice3", TICKET, choice3),
    prompt_case("choice12", TICKET, choice12),
    prompt_case("score_options", TICKET, score),
    prompt_case("score_isolated_level2", TICKET, score, level=2),
    prompt_case("annotated_list_state", list_state, boolean),
]

requests = [
    {
        "name": "ticket",
        "state": TICKET,
        "questions": {"department": choice3, "urgency": score, "refund": boolean},
    },
    {
        "name": "structured",
        "state": {"customer": "Ana", "messages": list_state},
        "questions": {"area": choice12, "refund": boolean},
    },
]
for request in requests:
    request["expected"] = predict(
        model, processor, request["state"], request["questions"]
    )

fixture = {
    "generator": "mlx-vlm d998a620 mlx_vlm.models.decider2 (Decider2._prompt and mlx_vlm.decision.predict)",
    "checkpoint": "nativ-community/decider-2b",
    "label_token_ids": list(d.label_token_ids),
    "pad_token_id": tok.pad_token_id,
    "prompts": prompts,
    "requests": requests,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(json.dumps([r["expected"] for r in requests], ensure_ascii=False, indent=1))
