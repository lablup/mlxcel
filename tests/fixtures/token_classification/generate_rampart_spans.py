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

# Longer than one 512-token window, so the fixture also pins the windowing.
LONG_CASE = " ".join(
    f"Ticket {index}: please call Maria Garcia on 617-555-01{index:02d} or write to "
    f"maria.garcia{index}@example.com about 1984 Cyberdyne Ave, Los Angeles CA 90012."
    for index in range(24)
)


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
        result["tokens"] = len(classifier._tokenize(text)[0])
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
