#!/usr/bin/env python3
"""Single definition of "is this checkpoint a VLM?", shared by both harnesses.

scripts/bench_mlxlm.py imports is_vlm(); scripts/bench_decode.sh shells out to
this file's CLI. They must agree: a checkpoint the Python baseline sweep skips
but the mlxcel sweep measures (or the reverse) produces a VLM comparison table
whose two halves cover different model sets.

Exit status in CLI mode: 0 if the path is a VLM checkpoint, 1 otherwise.
"""

import json
import sys
from pathlib import Path

# Families whose config.json carries no vision_config / image_processor_type
# but which are VLMs nonetheless.
VLM_ARCH_SUBSTR = (
    "Llava", "PaliGemma", "Qwen2VL", "Qwen2_5_VL", "Qwen3VL",
    "Idefics", "Pixtral", "Bunny", "Phi3V", "Phi35V", "AyaVision",
    "Gemma3ForConditional", "Gemma4ForConditional", "Mllama",
    "Mistral3", "Llama4", "MolmoForCausalLM", "Molmo", "InternVL",
    "GotOcr", "Smolvlm", "Florence", "Kimi",
)


def is_vlm(model_path) -> bool:
    """Detect a VLM checkpoint from config.json contents or preprocessor presence."""
    model_path = Path(model_path)
    cfg = model_path / "config.json"
    if not cfg.exists():
        return False
    try:
        with open(cfg) as f:
            data = json.load(f)
    except Exception:
        return False
    if "vision_config" in data or "image_processor_type" in data:
        return True
    archs = data.get("architectures", []) or []
    for a in archs:
        if any(sub in a for sub in VLM_ARCH_SUBSTR):
            return True
    # An image preprocessor with no text-only counterpart means a vision tower.
    if (model_path / "preprocessor_config.json").exists():
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: vlm_detect.py <checkpoint-dir>")
    sys.exit(0 if is_vlm(sys.argv[1]) else 1)
