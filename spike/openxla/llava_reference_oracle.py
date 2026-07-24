#!/usr/bin/env python3
"""Pinned LLaVA reference capture and first-divergence comparator.

Weights and generated captures stay outside Git. The maintained code records
the independent Hugging Face stages needed to diagnose a compiler-backend
divergence without accepting mlxcel output as its own oracle.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_REPO = "llava-hf/llava-interleave-qwen-0.5b-hf"
SOURCE_REVISION = "1090956dd1c79bc93ae98dcf395590369435ec91"
CONVERTED_REPO = "mlx-community/llava-interleave-qwen-0.5b-bf16"
CONVERTED_REVISION = "ba7385935f69c5417bfbe29c3809858a98afc22f"
SOURCE_ARTIFACTS = {
    "added_tokens.json": "f86f2a952b4888d195b8d77e3d56ad96864ecc696cca2155d0d4ac933fcfd55f",
    "chat_template.json": "9d98326321da3c0514e31c907a2119f5efc323fc685d9794536c6098ca897852",
    "config.json": "9b9b24a2b7a08b950c0c338f234d5cdfc5ebf526f360759f98a937b0f9374219",
    "generation_config.json": "0b640df36b033e96856a803d834155919eafae4a427631ea046b67e12643157e",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "ec7b02696781afdb1f27871fdffe0f71ef030932d10fbd759bc59392669605f7",
    "preprocessor_config.json": "6239b91cf50d40f36b4390ccc604a985c2388b95c5c77fa48c0658183bc5102c",
    "processor_config.json": "e8ff88c7591da9738760aec6ca01c8aadc9cf514f30a2a88ca3994d8c1c4a52e",
    "special_tokens_map.json": "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b",
    "tokenizer.json": "d26f54ac5bcc30ba15d418234e89d2ca44caf0bd57ce14749612a74f436738ef",
    "tokenizer_config.json": "5504fd209f6064bcba7a82b875f0af1927cd12fa7da2a6fbda0609070ce8d253",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "9c16795365b84aa42b7792ddcab2b954cc4e09e1a6307dddbcbafc2fac692a62"
)
CONVERTED_ARTIFACTS = {
    "added_tokens.json": "f86f2a952b4888d195b8d77e3d56ad96864ecc696cca2155d0d4ac933fcfd55f",
    "chat_template.json": "b7e65242c4107a669b40a2a6f62e5f4306ef328fa0d00c6bc0117df44f411603",
    "config.json": "77cafd419e5c4218e1b1a45b3bd4b603873a9e7a316c62e8d2d5391f40d93d1b",
    "generation_config.json": "0b640df36b033e96856a803d834155919eafae4a427631ea046b67e12643157e",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "43919c1ea46e00c6063204515169bb9635d0c2c6b3a07f975e20e9ea23c33d4c",
    "model.safetensors.index.json": "0bf1ff182c7dbdede0e53341fa3f0c65019f86ac909ec9d388aca065a713191e",
    "preprocessor_config.json": "6239b91cf50d40f36b4390ccc604a985c2388b95c5c77fa48c0658183bc5102c",
    "processor_config.json": "2634ec0e0c3a222fa9439131f6e4a43a02ab8b50d25cf77d1f7766eb3efdcf2a",
    "special_tokens_map.json": "1710dcae506cfff57fc8e63e1bff58f1d8c6aa2e2a8af56b65b33ed808a4c644",
    "tokenizer.json": "32e8f623d8dce60b5a93496ec810434ef744287ac041cf2c6032743a3578baa5",
    "tokenizer_config.json": "9d874e0d02b5a74d6e4863b38979e749d3738c2b03c1324675a4f1578730a802",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
CONVERTED_ARTIFACT_MANIFEST_SHA256 = (
    "912650461f7abfce6fd3962711387c7aad485df239afb3c83c75126476f2050c"
)
FIXTURE_PATH = "tests/fixtures/test_image.png"
FIXTURE_SHA256 = "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261"
CASE_IMAGE_TRANSFORMS = {
    "image_text": ("identity",),
    "two_images": ("identity", "swap_red_blue"),
    "no_image": (),
}
IMAGE_SIZE = 384
IMAGE_TOKENS = 729
VISION_HIDDEN_SIZE = 1152
VISION_INTERMEDIATE_SIZE = 4304
TEXT_HIDDEN_SIZE = 1024
VOCAB_SIZE = 152000
TEXT_LAYERS = 24
CASE_SEQUENCE_LENGTHS = {
    "image_text": 743,
    "two_images": 1473,
    "no_image": 14,
}
PINNED_KV_SELECTION = {
    "position": "last_effective_prompt",
    "kv_head": 0,
    "width": 8,
    "layers": TEXT_LAYERS,
}
PINNED_GENERATION = {"mode": "greedy", "max_new_tokens": 4}
FORBIDDEN_RUNTIME_ALTERNATE_NAMES = {
    "chat_template.jinja",
    "tokenizer.model",
    "tokenizer.jsonl",
    "tiktoken.model",
}
COMPUTE_DTYPES = {
    "processor": "float32",
    "prompt_embedding_lookup": "bfloat16",
    "vision_projector": "float32",
    "multimodal_merge": "bfloat16 then widened to float32",
    "text_decoder": "float32",
}

VISION_HIDDEN_STATE_STAGES = tuple(
    f"vision_hidden_state_{index:02}" for index in range(27)
)
VISION_BLOCK0_STAGES = (
    "vision_block0_layer_norm1",
    "vision_block0_q_proj",
    "vision_block0_k_proj",
    "vision_block0_v_proj",
    "vision_block0_attention_context",
    "vision_block0_attention_output",
    "vision_block0_attention_residual",
    "vision_block0_layer_norm2",
    "vision_block0_mlp_fc1",
    "vision_block0_mlp_activation",
    "vision_block0_mlp_fc2",
    "vision_block0_output",
)
STAGE_ORDER = (
    "processor_pixel_values",
    "expanded_token_ids",
    "positions",
    "attention_mask",
) + VISION_HIDDEN_STATE_STAGES[:1] + VISION_BLOCK0_STAGES + VISION_HIDDEN_STATE_STAGES[1:] + (
    "selected_vision_features",
    "projected_image_features",
    "merged_embeddings",
    "first_prefill_logits",
    "selected_kv",
    "greedy_tokens",
)

# The processor is mathematically exact. Both implementations preserve the
# checkpoint's BF16 prompt lookup and merge destination, then widen the merged
# result at the IREE boundary. Vision, projector, and decoder arithmetic are F32.
TOLERANCES = {
    "float32": {
        "processor_pixel_values": {"atol": 1.0e-6, "rtol": 1.0e-6},
        "selected_vision_features": {"atol": 4.0e-3, "rtol": 1.0e-3},
        "projected_image_features": {"atol": 1.0e-3, "rtol": 1.0e-3},
        "merged_embeddings": {"atol": 4.0e-3, "rtol": 1.0e-3},
        "first_prefill_logits": {"atol": 3.0e-2, "rtol": 3.0e-3},
        "selected_kv": {"atol": 3.0e-3, "rtol": 3.0e-3},
    },
    "bfloat16": {
        "processor_pixel_values": {"atol": 1.0e-6, "rtol": 1.0e-6},
        "selected_vision_features": {"atol": 8.0e-2, "rtol": 4.0e-2},
        "projected_image_features": {"atol": 8.0e-2, "rtol": 4.0e-2},
        "merged_embeddings": {"atol": 8.0e-2, "rtol": 4.0e-2},
        "first_prefill_logits": {"atol": 1.5e-1, "rtol": 5.0e-2},
        "selected_kv": {"atol": 1.5e-1, "rtol": 5.0e-2},
    },
}
for stage in VISION_BLOCK0_STAGES:
    TOLERANCES["float32"][stage] = {"atol": 2.0e-4, "rtol": 2.0e-4}
    TOLERANCES["bfloat16"][stage] = TOLERANCES["bfloat16"][
        "selected_vision_features"
    ]
for index, stage in enumerate(VISION_HIDDEN_STATE_STAGES):
    # Residual depth accumulates otherwise-independent F32 reduction order.
    # The 12% per-layer budget starts at the strict block boundary and reaches
    # 0.0038 at layer 26; logits and KV retain their tighter output budgets.
    budget = 2.0e-4 * (1.12**index)
    TOLERANCES["float32"][stage] = {"atol": budget, "rtol": budget}
    TOLERANCES["bfloat16"][stage] = TOLERANCES["bfloat16"][
        "selected_vision_features"
    ]

STAGE_POLICY_DTYPES = {"merged_embeddings": "bfloat16"}
INTEGER_STAGES = {
    "expanded_token_ids",
    "positions",
    "attention_mask",
    "greedy_tokens",
}
CASE_REQUIRED_STAGES = {
    "image_text": frozenset(STAGE_ORDER),
    "two_images": frozenset(STAGE_ORDER),
    "no_image": frozenset(
        {
            "expanded_token_ids",
            "positions",
            "attention_mask",
            "merged_embeddings",
            "first_prefill_logits",
            "selected_kv",
            "greedy_tokens",
        }
    ),
}
EXPECTED_NEGATIVE_CASES = {
    "malformed_placeholder": {
        "passed": True,
        "outcome": "rejected",
        "category": "placeholder_count_mismatch",
    },
    "context_overflow": {
        "passed": True,
        "outcome": "rejected",
        "category": "context_capacity_exceeded",
    },
}


class ContractError(ValueError):
    """A capture failed the closed reference-manifest contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"error: missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"error: {label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def canonical_artifact_sha(artifacts: dict[str, str]) -> str:
    payload = "".join(
        f"{filename}={artifacts[filename]}\n" for filename in sorted(artifacts)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def is_runtime_alternate(filename: str, allowed: set[str]) -> bool:
    if filename in allowed:
        return False
    return (
        filename in FORBIDDEN_RUNTIME_ALTERNATE_NAMES
        or filename.endswith(".tiktoken")
        or filename.endswith(".safetensors")
        or filename.endswith(".safetensors.index.json")
        or filename.endswith(".index.json")
    )


def reject_runtime_alternates(
    root: Path, expected_artifacts: dict[str, str], label: str
) -> None:
    allowed = set(expected_artifacts)
    try:
        extras = sorted(
            entry.name
            for entry in root.iterdir()
            if (entry.is_file() or entry.is_symlink())
            and is_runtime_alternate(entry.name, allowed)
        )
    except OSError as error:
        raise SystemExit(f"error: cannot inspect {label} snapshot {root}: {error}") from error
    if extras:
        raise SystemExit(
            f"error: {label} snapshot has unpinned runtime alternate(s): {extras}"
        )


def require_artifact_manifest(
    root: Path,
    expected_artifacts: dict[str, str],
    expected_manifest_sha: str,
    label: str,
) -> dict[str, Any]:
    reject_runtime_alternates(root, expected_artifacts, label)
    for filename, expected_sha in expected_artifacts.items():
        require_sha(root / filename, expected_sha, f"{label} {filename}")
    actual_manifest_sha = canonical_artifact_sha(expected_artifacts)
    if actual_manifest_sha != expected_manifest_sha:
        raise SystemExit(
            f"error: internal {label} canonical manifest mismatch: "
            f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    return {
        "canonical_sha256": actual_manifest_sha,
        "files": expected_artifacts,
    }


def transformed_image(image: Any, transform: str) -> Any:
    from PIL import Image

    if transform == "identity":
        return image.copy()
    if transform == "swap_red_blue":
        red, green, blue = image.split()
        transformed = Image.merge("RGB", (blue, green, red))
        if transformed.size != image.size or transformed.tobytes() == image.tobytes():
            raise AssertionError("swap_red_blue must preserve size and change RGB bytes")
        return transformed
    raise AssertionError(f"unsupported pinned image transform {transform!r}")


def expected_stage_shapes(case_name: str) -> dict[str, list[int]]:
    image_count = len(CASE_IMAGE_TRANSFORMS[case_name])
    sequence_len = CASE_SEQUENCE_LENGTHS[case_name]
    result = {
        "expanded_token_ids": [1, sequence_len],
        "positions": [1, sequence_len],
        "attention_mask": [1, sequence_len],
        "merged_embeddings": [1, sequence_len, TEXT_HIDDEN_SIZE],
        "first_prefill_logits": [VOCAB_SIZE],
        "selected_kv": [
            TEXT_LAYERS,
            2,
            PINNED_KV_SELECTION["width"],
        ],
        "greedy_tokens": [PINNED_GENERATION["max_new_tokens"]],
    }
    if image_count:
        result["processor_pixel_values"] = [
            image_count,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ]
        for stage in VISION_HIDDEN_STATE_STAGES:
            result[stage] = [
                image_count,
                IMAGE_TOKENS,
                VISION_HIDDEN_SIZE,
            ]
        for stage in VISION_BLOCK0_STAGES:
            width = (
                VISION_INTERMEDIATE_SIZE
                if stage
                in {"vision_block0_mlp_fc1", "vision_block0_mlp_activation"}
                else VISION_HIDDEN_SIZE
            )
            result[stage] = [image_count, IMAGE_TOKENS, width]
        result["selected_vision_features"] = [
            image_count,
            IMAGE_TOKENS,
            VISION_HIDDEN_SIZE,
        ]
        result["projected_image_features"] = [
            image_count,
            IMAGE_TOKENS,
            TEXT_HIDDEN_SIZE,
        ]
    return result


def verify_vision_block0_conversion(
    source_model: Path, converted_model: Path
) -> dict[str, Any]:
    """Prove that conversion did not change embedding/block-0 BF16 bits."""
    import torch
    from safetensors import safe_open

    prefixes = (
        "vision_tower.vision_model.embeddings.",
        "vision_tower.vision_model.encoder.layers.0.",
    )
    digest = hashlib.sha256()
    with (
        safe_open(source_model, framework="pt", device="cpu") as source,
        safe_open(converted_model, framework="pt", device="cpu") as converted,
    ):
        keys = sorted(key for key in source.keys() if key.startswith(prefixes))
        if not keys:
            raise SystemExit("error: source checkpoint has no vision block-0 tensors")
        for key in keys:
            if key not in converted.keys():
                raise SystemExit(f"error: converted checkpoint is missing {key}")
            expected = source.get_tensor(key)
            actual = converted.get_tensor(key)
            if key.endswith("embeddings.patch_embedding.weight"):
                expected = expected.permute(0, 2, 3, 1).contiguous()
            if expected.dtype != actual.dtype or expected.shape != actual.shape:
                raise SystemExit(
                    f"error: converted tensor metadata differs for {key}: "
                    f"source={expected.dtype}/{tuple(expected.shape)}, "
                    f"converted={actual.dtype}/{tuple(actual.shape)}"
                )
            if not torch.equal(expected, actual):
                mismatches = int(
                    (expected.view(torch.int16) != actual.view(torch.int16))
                    .sum()
                    .item()
                )
                raise SystemExit(
                    f"error: converted tensor is not bit-exact for {key}: "
                    f"{mismatches} BF16 values differ"
                )
            digest.update(key.encode())
            digest.update(expected.view(torch.uint8).numpy().tobytes())
    return {
        "scope": "vision embeddings and encoder block 0",
        "tensor_count": len(keys),
        "bit_exact": True,
        "canonical_sha256": digest.hexdigest(),
        "layout_transform": "patch_embedding OIHW to OHWI",
    }


def rss_peak_kib() -> int:
    # Linux reports KiB; macOS reports bytes. This harness is currently
    # qualified on Linux GB10/CPU and records the platform in the manifest.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def store_array(
    root: Path, case: str, stage: str, value: np.ndarray, arrays: dict[str, Any]
) -> None:
    value = np.ascontiguousarray(value)
    filename = f"{case}.{stage}.bin"
    value.tofile(root / filename)
    arrays[stage] = {
        "file": filename,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def cases(processor: Any, image_path: Path) -> list[dict[str, Any]]:
    definitions = (
        ("image_text", "Describe the image briefly.", 1),
        ("two_images", "Compare the two images briefly.", 2),
        ("no_image", "Reply with one short greeting.", 0),
    )
    result = []
    for name, user_prompt, image_count in definitions:
        image_transforms = CASE_IMAGE_TRANSFORMS[name]
        if len(image_transforms) != image_count:
            raise AssertionError(f"invalid image transform contract for {name}")
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": user_prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = processor.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )
        result.append(
            {
                "name": name,
                "user_prompt": user_prompt,
                "text": text,
                "image_count": image_count,
                "image_transforms": list(image_transforms),
                "image_path": str(image_path),
                "unexpanded_input_ids": tokenized.input_ids[0]
                .to(dtype=np_int32_torch())
                .tolist(),
            }
        )
    return result


def np_int32_torch() -> Any:
    import torch

    return torch.int32


def capture(args: argparse.Namespace) -> int:
    if args.max_new != PINNED_GENERATION["max_new_tokens"]:
        raise SystemExit(
            "error: pinned capture requires "
            f"--max-new {PINNED_GENERATION['max_new_tokens']}"
        )
    if args.kv_width != PINNED_KV_SELECTION["width"]:
        raise SystemExit(
            "error: pinned capture requires "
            f"--kv-width {PINNED_KV_SELECTION['width']}"
        )
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise SystemExit(
            "error: capture requires torch, transformers, numpy, and Pillow in "
            f"the oracle environment: {error}"
        ) from error

    source = args.source_model.resolve()
    converted = args.converted_model.resolve()
    source_artifact_manifest = require_artifact_manifest(
        source,
        SOURCE_ARTIFACTS,
        SOURCE_ARTIFACT_MANIFEST_SHA256,
        "source snapshot",
    )
    converted_artifact_manifest = require_artifact_manifest(
        converted,
        CONVERTED_ARTIFACTS,
        CONVERTED_ARTIFACT_MANIFEST_SHA256,
        "converted snapshot",
    )
    image_path = args.image.resolve()
    require_sha(image_path, FIXTURE_SHA256, "pinned image fixture")
    conversion_equivalence = verify_vision_block0_conversion(
        source / "model.safetensors", converted / "model.safetensors"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    image = Image.open(image_path).convert("RGB")
    load_started = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    # Match the production ownership boundary exactly. Prompt text embedding is
    # looked up from the immutable BF16 table on the host, while vision,
    # projector, and the IREE decoder widen checkpoint values to F32.
    prompt_embedding_weight = (
        model.model.language_model.embed_tokens.weight.detach().clone()
    )
    model.model.vision_tower.float()
    model.model.multi_modal_projector.float()
    model.model.language_model.float()
    model.lm_head.float()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but torch.cuda is unavailable")
    model.to(device)
    model_load_seconds = time.perf_counter() - load_started

    manifest: dict[str, Any] = {
        "schema": 1,
        "producer": "huggingface-transformers",
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "artifact_manifest": source_artifact_manifest,
            "license": "Tongyi Qianwen Research License",
        },
        "converted_checkpoint": {
            "repo": CONVERTED_REPO,
            "revision": CONVERTED_REVISION,
            "artifact_manifest": converted_artifact_manifest,
        },
        "image_fixture": {
            "path": FIXTURE_PATH,
            "sha256": FIXTURE_SHA256,
            "two_image_transform": "swap_red_blue",
        },
        "conversion_equivalence": conversion_equivalence,
        "processor": {
            "image_size": IMAGE_SIZE,
            "resample": "bicubic",
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        "compute_dtypes": COMPUTE_DTYPES,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "kv_selection": {
            "position": "last_effective_prompt",
            "kv_head": 0,
            "width": args.kv_width,
            "layers": int(model.config.text_config.num_hidden_layers),
        },
        "generation": {"mode": "greedy", "max_new_tokens": args.max_new},
        "tolerances": TOLERANCES,
        "stage_policy_dtypes": STAGE_POLICY_DTYPES,
        "timings": {"model_load_seconds": model_load_seconds},
        "host_peak_rss_kib": rss_peak_kib(),
        "cases": [],
    }

    for definition in cases(processor, image_path):
        started = time.perf_counter()
        images = [
            transformed_image(image, transform)
            for transform in definition["image_transforms"]
        ]
        if definition["name"] == "two_images":
            if images[0].tobytes() == images[1].tobytes():
                raise AssertionError("two-image RGB inputs must be byte-distinct")
        inputs = processor(
            text=definition["text"],
            images=images or None,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        if definition["name"] == "two_images":
            pixels = inputs.pixel_values
            if torch.equal(pixels[0], pixels[1]):
                raise AssertionError(
                    "two-image processor tensors must differ after channel swap"
                )
            if torch.equal(pixels, pixels.flip(0)):
                raise AssertionError(
                    "reversing two-image order must change the processor tensor"
                )
        processor_seconds = time.perf_counter() - started
        expanded_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        position_ids = torch.arange(
            expanded_ids.shape[1], dtype=torch.int64, device=device
        ).unsqueeze(0)

        arrays: dict[str, Any] = {}
        if definition["image_count"]:
            store_array(
                args.out,
                definition["name"],
                "processor_pixel_values",
                inputs.pixel_values.cpu().numpy().astype(np.float32),
                arrays,
            )

        with torch.inference_mode():
            text_embeddings = torch.nn.functional.embedding(
                expanded_ids, prompt_embedding_weight
            )
            image_features = None
            selected_vision_features = None
            vision_hidden_states = None
            vision_block0: dict[str, torch.Tensor] = {}
            vision_seconds = 0.0
            if definition["image_count"]:
                layer0 = model.model.vision_tower.encoder.layers[0]
                hooks = []

                def capture_output(name: str):
                    def hook(_module: Any, _inputs: Any, output: Any) -> None:
                        vision_block0[name] = output

                    return hook

                def capture_input(name: str):
                    def hook(_module: Any, inputs: Any) -> None:
                        vision_block0[name] = inputs[0]

                    return hook

                hooks.extend(
                    (
                        layer0.layer_norm1.register_forward_hook(
                            capture_output("vision_block0_layer_norm1")
                        ),
                        layer0.self_attn.q_proj.register_forward_hook(
                            capture_output("vision_block0_q_proj")
                        ),
                        layer0.self_attn.k_proj.register_forward_hook(
                            capture_output("vision_block0_k_proj")
                        ),
                        layer0.self_attn.v_proj.register_forward_hook(
                            capture_output("vision_block0_v_proj")
                        ),
                        layer0.self_attn.out_proj.register_forward_pre_hook(
                            capture_input("vision_block0_attention_context")
                        ),
                        layer0.self_attn.out_proj.register_forward_hook(
                            capture_output("vision_block0_attention_output")
                        ),
                        layer0.layer_norm2.register_forward_hook(
                            capture_output("vision_block0_layer_norm2")
                        ),
                        layer0.mlp.fc1.register_forward_hook(
                            capture_output("vision_block0_mlp_fc1")
                        ),
                        layer0.mlp.activation_fn.register_forward_hook(
                            capture_output("vision_block0_mlp_activation")
                        ),
                        layer0.mlp.fc2.register_forward_hook(
                            capture_output("vision_block0_mlp_fc2")
                        ),
                    )
                )
                vision_started = time.perf_counter()
                image_outputs = model.get_image_features(
                    pixel_values=inputs.pixel_values.to(torch.float32),
                    vision_feature_layer=model.config.vision_feature_layer,
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    return_dict=True,
                )
                for hook in hooks:
                    hook.remove()
                selected_vision_features = image_outputs.hidden_states[
                    model.config.vision_feature_layer
                ]
                vision_hidden_states = image_outputs.hidden_states
                vision_block0["vision_block0_attention_residual"] = (
                    vision_hidden_states[0]
                    + vision_block0["vision_block0_attention_output"]
                )
                vision_block0["vision_block0_output"] = vision_hidden_states[1]
                image_features = torch.cat(image_outputs.pooler_output, dim=0).to(
                    text_embeddings.device
                )
                vision_seconds = time.perf_counter() - vision_started
                merge_image_features = image_features.to(text_embeddings.dtype)
                mask = model.model.get_placeholder_mask(
                    expanded_ids,
                    inputs_embeds=text_embeddings,
                    image_features=merge_image_features,
                )
                merged_embeddings = text_embeddings.masked_scatter(
                    mask, merge_image_features
                ).float()
            else:
                merged_embeddings = text_embeddings.float()

            prefill_started = time.perf_counter()
            language_output = model.model.language_model(
                inputs_embeds=merged_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            first_logits = model.lm_head(language_output.last_hidden_state[:, -1:])
            prefill_seconds = time.perf_counter() - prefill_started
            cache = language_output.past_key_values
            selected = []
            for layer in cache.layers:
                selected.append(
                    torch.stack(
                        (
                            layer.keys[0, 0, -1, : args.kv_width],
                            layer.values[0, 0, -1, : args.kv_width],
                        )
                    )
                )
            selected_kv = torch.stack(selected)

            greedy = [int(first_logits[0, -1].argmax().item())]
            current = torch.tensor([[greedy[0]]], dtype=torch.int64, device=device)
            decode_started = time.perf_counter()
            while len(greedy) < args.max_new:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                decode_output = model.model.language_model(
                    input_ids=current,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                decode_logits = model.lm_head(decode_output.last_hidden_state[:, -1:])
                token = int(decode_logits[0, -1].argmax().item())
                greedy.append(token)
                current = torch.tensor([[token]], dtype=torch.int64, device=device)
                cache = decode_output.past_key_values
            decode_seconds = time.perf_counter() - decode_started

        store_array(
            args.out,
            definition["name"],
            "expanded_token_ids",
            expanded_ids.cpu().numpy().astype(np.int32),
            arrays,
        )
        store_array(
            args.out,
            definition["name"],
            "positions",
            position_ids.cpu().numpy().astype(np.int32),
            arrays,
        )
        store_array(
            args.out,
            definition["name"],
            "attention_mask",
            inputs.attention_mask.cpu().numpy().astype(np.int32),
            arrays,
        )
        if image_features is not None:
            assert selected_vision_features is not None
            assert vision_hidden_states is not None
            for stage in VISION_BLOCK0_STAGES:
                store_array(
                    args.out,
                    definition["name"],
                    stage,
                    vision_block0[stage].float().cpu().numpy().astype(np.float32),
                    arrays,
                )
            for index, hidden_state in enumerate(vision_hidden_states):
                store_array(
                    args.out,
                    definition["name"],
                    f"vision_hidden_state_{index:02}",
                    hidden_state.float().cpu().numpy().astype(np.float32),
                    arrays,
                )
            store_array(
                args.out,
                definition["name"],
                "selected_vision_features",
                selected_vision_features.float().cpu().numpy().astype(np.float32),
                arrays,
            )
            per_image_features = image_features.reshape(
                definition["image_count"], -1, image_features.shape[-1]
            )
            store_array(
                args.out,
                definition["name"],
                "projected_image_features",
                per_image_features.float().cpu().numpy().astype(np.float32),
                arrays,
            )
        store_array(
            args.out,
            definition["name"],
            "merged_embeddings",
            merged_embeddings.cpu().numpy().astype(np.float32),
            arrays,
        )
        store_array(
            args.out,
            definition["name"],
            "first_prefill_logits",
            first_logits[0, -1].cpu().numpy().astype(np.float32),
            arrays,
        )
        store_array(
            args.out,
            definition["name"],
            "selected_kv",
            selected_kv.cpu().numpy().astype(np.float32),
            arrays,
        )
        store_array(
            args.out,
            definition["name"],
            "greedy_tokens",
            np.asarray(greedy, dtype=np.int32),
            arrays,
        )
        manifest["cases"].append(
            {
                **definition,
                "greedy_text": processor.tokenizer.decode(
                    greedy,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "arrays": arrays,
                "timings": {
                    "processor_seconds": processor_seconds,
                    "vision_projector_seconds": vision_seconds,
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                },
            }
        )
        manifest["host_peak_rss_kib"] = max(
            manifest["host_peak_rss_kib"], rss_peak_kib()
        )
        del (
            inputs,
            expanded_ids,
            text_embeddings,
            merged_embeddings,
            language_output,
            first_logits,
            selected_kv,
            cache,
        )
        gc.collect()

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out / "manifest.json")
    return 0


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON value {constant!r} in {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def validate_artifact_contract(
    value: Any,
    expected_artifacts: dict[str, str],
    expected_manifest_sha: str,
    label: str,
) -> None:
    if not isinstance(value, dict) or set(value) != {"canonical_sha256", "files"}:
        raise ContractError(f"{label} artifact_manifest has an invalid schema")
    if value["files"] != expected_artifacts:
        raise ContractError(f"{label} artifact_manifest file hashes differ")
    if value["canonical_sha256"] != expected_manifest_sha:
        raise ContractError(f"{label} artifact_manifest canonical hash differs")
    if canonical_artifact_sha(value["files"]) != expected_manifest_sha:
        raise ContractError(f"{label} artifact_manifest is not canonical")


def validate_capture_manifest(
    root: Path, manifest: dict[str, Any], role: str
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != 1:
        raise ContractError(f"{role} manifest schema must be 1")
    cases_value = manifest.get("cases")
    if not isinstance(cases_value, list):
        raise ContractError(f"{role} manifest cases must be a list")
    cases: dict[str, dict[str, Any]] = {}
    for case in cases_value:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise ContractError(f"{role} manifest has an invalid case")
        name = case["name"]
        if name in cases:
            raise ContractError(f"{role} manifest has duplicate case {name!r}")
        cases[name] = case
    expected_cases = set(CASE_REQUIRED_STAGES)
    if set(cases) != expected_cases:
        missing = sorted(expected_cases - set(cases))
        extra = sorted(set(cases) - expected_cases)
        raise ContractError(
            f"{role} case set differs: missing={missing}, extra={extra}"
        )

    fixture = manifest.get("image_fixture")
    expected_fixture = {
        "path": FIXTURE_PATH,
        "sha256": FIXTURE_SHA256,
        "two_image_transform": "swap_red_blue",
    }
    if fixture != expected_fixture:
        raise ContractError(f"{role} image fixture contract differs")
    if manifest.get("kv_selection") != PINNED_KV_SELECTION:
        raise ContractError(f"{role} kv_selection contract differs")
    if manifest.get("generation") != PINNED_GENERATION:
        raise ContractError(f"{role} generation contract differs")

    converted = manifest.get("converted_checkpoint")
    if not isinstance(converted, dict):
        raise ContractError(f"{role} converted checkpoint metadata is missing")
    validate_artifact_contract(
        converted.get("artifact_manifest"),
        CONVERTED_ARTIFACTS,
        CONVERTED_ARTIFACT_MANIFEST_SHA256,
        f"{role} converted",
    )
    if role == "reference":
        if converted.get("repo") != CONVERTED_REPO:
            raise ContractError("reference converted repo differs")
        if converted.get("revision") != CONVERTED_REVISION:
            raise ContractError("reference converted revision differs")
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ContractError("reference source metadata is missing")
        if source.get("repo") != SOURCE_REPO:
            raise ContractError("reference source repo differs")
        if source.get("revision") != SOURCE_REVISION:
            raise ContractError("reference source revision differs")
        validate_artifact_contract(
            source.get("artifact_manifest"),
            SOURCE_ARTIFACTS,
            SOURCE_ARTIFACT_MANIFEST_SHA256,
            "reference source",
        )
    elif manifest.get("negative_cases") != EXPECTED_NEGATIVE_CASES:
        raise ContractError(
            "actual negative_cases must exactly match the required rejected outcomes"
        )

    root = root.resolve()
    for case_name, case in cases.items():
        expected_shapes = expected_stage_shapes(case_name)
        expected_transforms = list(CASE_IMAGE_TRANSFORMS[case_name])
        if case.get("image_count") != len(expected_transforms):
            raise ContractError(f"{role} {case_name} image_count differs")
        if case.get("image_transforms") != expected_transforms:
            raise ContractError(f"{role} {case_name} image transforms differ")
        arrays = case.get("arrays")
        if not isinstance(arrays, dict):
            raise ContractError(f"{role} {case_name} arrays must be an object")
        required_stages = CASE_REQUIRED_STAGES[case_name]
        if set(arrays) != required_stages:
            missing = sorted(required_stages - set(arrays))
            extra = sorted(set(arrays) - required_stages)
            raise ContractError(
                f"{role} {case_name} stage set differs: "
                f"missing={missing}, extra={extra}"
            )
        for stage, spec in arrays.items():
            if not isinstance(spec, dict) or set(spec) != {"file", "dtype", "shape"}:
                raise ContractError(
                    f"{role} {case_name}/{stage} has an invalid array schema"
                )
            expected_file = f"{case_name}.{stage}.bin"
            if spec["file"] != expected_file:
                raise ContractError(
                    f"{role} {case_name}/{stage} has an invalid array path"
                )
            expected_dtype = "int32" if stage in INTEGER_STAGES else "float32"
            if spec["dtype"] != expected_dtype:
                raise ContractError(
                    f"{role} {case_name}/{stage} dtype must be {expected_dtype}"
                )
            shape = spec["shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension <= 0
                    for dimension in shape
                )
            ):
                raise ContractError(
                    f"{role} {case_name}/{stage} has an invalid shape"
                )
            if shape != expected_shapes[stage]:
                raise ContractError(
                    f"{role} {case_name}/{stage} shape must be "
                    f"{expected_shapes[stage]}, got {shape}"
                )
            element_count = 1
            for dimension in shape:
                element_count *= dimension
                if element_count > 2**63 - 1:
                    raise ContractError(
                        f"{role} {case_name}/{stage} shape is too large"
                    )
            path = root / expected_file
            if path.is_symlink() or not path.is_file():
                raise ContractError(
                    f"{role} {case_name}/{stage} binary is missing or not regular"
                )
            expected_size = element_count * np.dtype(expected_dtype).itemsize
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ContractError(
                    f"{role} {case_name}/{stage} binary size differs: "
                    f"expected={expected_size}, actual={actual_size}"
                )
    return cases


def load_array(root: Path, spec: dict[str, Any]) -> np.ndarray:
    values = np.fromfile(root / spec["file"], dtype=np.dtype(spec["dtype"]))
    return values.reshape(spec["shape"])


def compare_capture_roots(reference_root: Path, actual_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": 1,
        "reference": str(reference_root),
        "actual": str(actual_root),
        "passed": True,
        "first_divergence": None,
        "cases": [],
    }
    try:
        reference_manifest = strict_json_load(reference_root / "manifest.json")
        actual_manifest = strict_json_load(actual_root / "manifest.json")
        reference_cases = validate_capture_manifest(
            reference_root, reference_manifest, "reference"
        )
        actual_cases = validate_capture_manifest(actual_root, actual_manifest, "actual")
    except ContractError as error:
        report.update(
            passed=False,
            first_divergence={"case": "manifest", "stage": "contract"},
            error=str(error),
        )
        return report

    for case_name in CASE_REQUIRED_STAGES:
        case_report: dict[str, Any] = {
            "name": case_name,
            "passed": True,
            "stages": [],
        }
        reference = reference_cases[case_name]
        actual = actual_cases[case_name]
        for stage in STAGE_ORDER:
            if stage not in CASE_REQUIRED_STAGES[case_name]:
                continue
            ref_spec = reference["arrays"].get(stage)
            actual_spec = actual["arrays"].get(stage)
            stage_report: dict[str, Any] = {"stage": stage, "passed": True}
            if ref_spec["dtype"] != actual_spec["dtype"]:
                stage_report.update(passed=False, error="dtype mismatch")
            else:
                ref = load_array(reference_root, ref_spec)
                got = load_array(actual_root, actual_spec)
                stage_report["reference_shape"] = list(ref.shape)
                stage_report["actual_shape"] = list(got.shape)
                if ref.shape != got.shape:
                    stage_report.update(passed=False, error="shape mismatch")
                elif (
                    np.issubdtype(ref.dtype, np.floating)
                    and (
                        not bool(np.isfinite(ref).all())
                        or not bool(np.isfinite(got).all())
                    )
                ):
                    stage_report.update(
                        passed=False, error="non-finite array value"
                    )
                elif np.issubdtype(ref.dtype, np.integer):
                    mismatch = np.flatnonzero(ref.reshape(-1) != got.reshape(-1))
                    stage_report["mismatch_count"] = int(mismatch.size)
                    if mismatch.size:
                        index = int(mismatch[0])
                        stage_report.update(
                            passed=False,
                            first_mismatch_index=index,
                            reference=int(ref.reshape(-1)[index]),
                            actual=int(got.reshape(-1)[index]),
                        )
                else:
                    policy_dtype = STAGE_POLICY_DTYPES.get(
                        stage, actual_spec["dtype"]
                    )
                    if policy_dtype not in TOLERANCES:
                        policy_dtype = "float32"
                    tolerance = TOLERANCES[policy_dtype][stage]
                    delta = np.abs(got.astype(np.float64) - ref.astype(np.float64))
                    limit = tolerance["atol"] + tolerance["rtol"] * np.abs(
                        ref.astype(np.float64)
                    )
                    mismatch = np.flatnonzero(delta.reshape(-1) > limit.reshape(-1))
                    relative = delta / np.maximum(np.abs(ref), 1.0e-12)
                    stage_report.update(
                        tolerance_dtype=policy_dtype,
                        atol=tolerance["atol"],
                        rtol=tolerance["rtol"],
                        max_abs=float(delta.max(initial=0.0)),
                        max_rel=float(relative.max(initial=0.0)),
                        mismatch_count=int(mismatch.size),
                    )
                    if mismatch.size:
                        index = int(mismatch[0])
                        stage_report.update(
                            passed=False,
                            first_mismatch_index=index,
                            reference=float(ref.reshape(-1)[index]),
                            actual=float(got.reshape(-1)[index]),
                        )
            if not stage_report["passed"]:
                case_report["passed"] = False
                report["passed"] = False
                report["first_divergence"] = report["first_divergence"] or {
                    "case": case_name,
                    "stage": stage,
                }
            case_report["stages"].append(stage_report)
        report["cases"].append(case_report)
    return report


def compare(args: argparse.Namespace) -> int:
    report = compare_capture_roots(args.reference, args.actual)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["first_divergence"], sort_keys=True))
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subcommands = root.add_subparsers(dest="command", required=True)
    capture_parser = subcommands.add_parser("capture")
    capture_parser.add_argument("--source-model", type=Path, required=True)
    capture_parser.add_argument("--converted-model", type=Path, required=True)
    capture_parser.add_argument("--image", type=Path, required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--max-new", type=int, default=4)
    capture_parser.add_argument("--kv-width", type=int, default=8)
    capture_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    capture_parser.set_defaults(run=capture)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--actual", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.set_defaults(run=compare)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    if getattr(parsed, "max_new", 1) <= 0 or getattr(parsed, "kv_width", 1) <= 0:
        raise SystemExit("error: --max-new and --kv-width must be positive")
    sys.exit(parsed.run(parsed))
