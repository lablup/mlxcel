"""Regenerate rampart_mlx_vlm_spans.json, the mlx-vlm span-parity fixture.

Usage (any Python with mlx-vlm >= 0.7.8 installed; mlx[cpu] is enough):

    python -I generate_rampart_spans.py <rampart-mlx-fp16 dir> <output.json>

The fixture records, for each case, the exact ``TokenClassifier.__call__``
result of ``mlx_vlm.token_classification.load_token_classifier`` on
``nativ-community/rampart-mlx-fp16``. ``src/token_classification/
real_checkpoint_tests.rs`` compares mlxcel's spans against it label for label
and offset for offset.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from mlx_vlm.token_classification import load_token_classifier

KEEP = ["CITY", "STATE", "ZIP_CODE"]

CASES = [
    # The README example (redacted_text is the issue's acceptance string).
    "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net",
    # Same-entity words split by punctuation must merge into one phone span.
    "Call Maria Garcia on 617-555-0142.",
    # Multi-byte characters before every span keep char offsets honest.
    "Café Müller: Jürgen Weiß wohnt in der Hauptstraße 12, 10115 Berlin, "
    "E-Mail juergen.weiss@example.de, Tel. 030 1234567.",
]

# Longer than one 512-token window (706 tokens). The prefix length puts the
# window boundary inside "Cyberdyne", whose first piece scores O and whose
# second scores B-STREET_NAME, so word grouping across windows is pinned too.
LONG_CASE = "Please review. " * 10 + " ".join(
    f"Note {index}. {CASES[0]}" for index in range(26)
)

# Every token's top-1 / top-2 log-probability gap must clear this, so the
# fixture holds no near-tie that f16 rounding on another device could flip.
# A repeated-template text tried first had an exact O / B-GIVEN_NAME tie
# that the CUDA run broke the other way.
MIN_MARGIN = 0.05


def mlx_vlm_revision() -> str:
    dist = importlib.metadata.distribution("mlx-vlm")
    direct = dist.read_text("direct_url.json")
    commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    return f"mlx-vlm {dist.version}" + (f" (git {commit})" if commit else "")


def main() -> None:
    model_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    classifier = load_token_classifier(str(model_dir))
    cases = []
    for text in [*CASES, LONG_CASE]:
        result = classifier(text, keep_labels=KEEP).to_dict()
        ids = classifier._tokenize(text)[0]
        top2 = np.sort(classifier.scores(ids), axis=-1)[:, -2:]
        margin = float((top2[:, 1] - top2[:, 0]).min())
        if margin < MIN_MARGIN:
            raise SystemExit(f"near-tie (margin {margin:.4f}) in {text[:60]!r}; pick another case")
        result["tokens"] = len(ids)
        result["min_margin"] = round(margin, 4)
        cases.append(result)
    fixture = {
        "oracle": mlx_vlm_revision(),
        "model": "nativ-community/rampart-mlx-fp16",
        "keep_labels": KEEP,
        "cases": cases,
    }
    output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
