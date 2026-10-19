"""Regenerate privacy_filter_mlx_vlm.json, the mlx-vlm parity fixture for
OpenAI Privacy Filter.

Usage (any Python with mlx-vlm >= 0.7.8 installed; mlx[cpu] is enough):

    python -I generate_privacy_filter_fixture.py <openai-privacy-filter-bf16 dir> <output.json>

The fixture records, from ``mlx_vlm.privacy_filter.load_privacy_filter`` on
``mlx-community/openai-privacy-filter-bf16``:

- ``log_probs``: the per-token ``[tokens, labels]`` log-probabilities of one
  sentence (``TokenClassifier.scores``), compared within a tolerance;
- ``cases``: for each sentence the token count and the exact result under
  ``decode="viterbi"`` and ``decode="argmax"``, plus the ``--replacement`` and
  ``--keep`` variants of the first sentence, compared exactly;
- ``viterbi_synthetic``: a random lattice decoded by ``ViterbiDecoder`` under
  zero and non-zero transition biases, pinning the decoder port on its own.

Every span case must clear ``MIN_MARGIN`` between the top-1 and top-2
log-probability at every token, so an exact comparison cannot hinge on a
near-tie that bf16 rounding on another device breaks the other way. The
``log_probs`` matrix is compared in probability space: tail labels (below
-18 nats) move by several nats between a bf16 and an f32 forward of the same
weights, so a log-domain tolerance there would measure rounding, not the port.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from mlx_vlm.privacy_filter import VITERBI_BIAS_KEYS, ViterbiDecoder, load_privacy_filter

CASES = [
    # The issue's acceptance sentence.
    "Alice Smith can be reached at alice@example.com.",
    # Phone and date next to a name.
    "Call Maria Garcia on +1 617-555-0142 before 2024-05-17 to confirm.",
    # An account number among ordinary digits.
    "Wire the deposit to account 4111 1111 1111 1111 and quote ticket 8842.",
    # CJK text before every span keeps char offsets honest (one char, many
    # bytes, often several tokens).
    "日本語のメモ: 山田太郎さんの電話番号は 03-1234-5678、メールは taro.yamada@example.jp です。",
    # A secret and a URL.
    "The API key is sk-live-9f8e7d6c5b4a3210 and the docs are at "
    "https://internal.example.org/wiki/onboarding.",
    # Umlauts before the spans, and a sentence where Viterbi and argmax
    # disagree (argmax leaves an unpaired boundary the span builder repairs).
    "Grüße von Jürgen Weiß, erreichbar unter juergen.weiss@example.de oder +49 30 1234567.",
]

# No long case: on a 6816-token text of repeated sentences the model itself
# sits at 50/50 between `O` and an entity tag on 144 tokens (6 exact ties),
# so exact span parity there would hinge on rounding. Chunked attention and
# routing past 4096 / 2048 tokens are checked against an unchunked forward
# of the same weights in `privacy_filter_checkpoint_tests.rs` instead.

# bf16 on CUDA against bf16 on mlx CPU flipped an argmax at a 0.19 margin
# (the German sentence "Café Müller: Jürgen Weiß wohnt in der Hauptstraße 12,
# ..." the first draft used), and the oracle's own f32 forward moves its
# mass-bearing log-probabilities by up to 0.14 against its bf16 run.
MIN_MARGIN = 0.5


def mlx_vlm_revision() -> str:
    dist = importlib.metadata.distribution("mlx-vlm")
    direct = dist.read_text("direct_url.json")
    commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    return f"mlx-vlm {dist.version}" + (f" (git {commit})" if commit else "")


def margin(scores: np.ndarray) -> float:
    top2 = np.sort(scores, axis=-1)[:, -2:]
    return float((top2[:, 1] - top2[:, 0]).min())


def span_case(classifier, text: str, **kwargs) -> dict:
    ids = classifier._tokenize(text)[0]
    scores = classifier.scores(ids)
    gap = margin(scores)
    if gap < MIN_MARGIN:
        raise SystemExit(f"near-tie (margin {gap:.4f}) in {text[:60]!r}; pick another case")
    case = {
        "text": text,
        "tokens": len(ids),
        "min_margin": round(gap, 4),
        "options": {key: value for key, value in kwargs.items()},
    }
    for decode in ("viterbi", "argmax"):
        result = classifier(text, decode=decode, **kwargs).to_dict()
        case[decode] = {"spans": result["spans"], "redacted_text": result["redacted_text"]}
    return case


def synthetic_viterbi() -> dict:
    labels = ["O", "B-x", "I-x", "E-x", "S-x", "B-y", "I-y", "E-y", "S-y"]
    rng = np.random.default_rng(2387)
    emissions = np.round(rng.normal(size=(40, len(labels))).astype(np.float32), 4)
    bias_sets = [
        {key: 0.0 for key in VITERBI_BIAS_KEYS},
        {
            "transition_bias_background_stay": 0.75,
            "transition_bias_background_to_start": -0.5,
            "transition_bias_inside_to_continue": 1.25,
            "transition_bias_inside_to_end": -0.25,
            "transition_bias_end_to_background": 0.5,
            "transition_bias_end_to_start": -1.0,
        },
    ]
    return {
        "labels": labels,
        "emissions": emissions.tolist(),
        "cases": [
            {"biases": biases, "path": ViterbiDecoder(labels, biases).decode(emissions)}
            for biases in bias_sets
        ],
    }


def main() -> None:
    model_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    classifier = load_privacy_filter(str(model_dir))
    first_ids = classifier._tokenize(CASES[0])[0]
    log_probs = classifier.scores(first_ids)
    cases = [span_case(classifier, text) for text in CASES]
    cases.append(span_case(classifier, CASES[0], replacement="[REDACTED]"))
    cases.append(span_case(classifier, CASES[0], keep_labels=["private_email"]))
    fixture = {
        "oracle": mlx_vlm_revision(),
        "model": "mlx-community/openai-privacy-filter-bf16",
        "operating_point": "default",
        "labels": list(classifier.labels),
        "log_probs": {
            "text": CASES[0],
            "tokens": [int(i) for i in first_ids],
            "values": np.round(log_probs, 5).tolist(),
        },
        "cases": cases,
        "viterbi_synthetic": synthetic_viterbi(),
    }
    output.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    main()
