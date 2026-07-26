#!/usr/bin/env python3
"""Immutable HF-eager Youtu-VL capture and MLX/IREE comparator.

The capture is intentionally independent of mlxcel: it loads only the pinned
Hugging Face checkpoint and its pinned remote code. Generated tensors stay
outside Git. The model-free ``compare`` command verifies every artifact hash,
the complete ordered Youtu-VL diagnostic surface, and the fresh-cache/reuse
lifecycle contract before comparing numeric values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONTRACT = "youtu-vl-hf-eager-oracle-v1"
CHECKPOINT_REPO = "tencent/Youtu-VL-4B-Instruct"
CHECKPOINT_REVISION = "8d30a0e49662a1d628a472b12df264dbcd768753"
CHECKPOINT_ARTIFACTS = {
    "__init__.py": "1ca8666808d0a0fbdaed1dfa81b313f1b3f0ed713d11be2918cce083f90ce11c",
    "chat_template.json": "172d9ec2636e63f13326ae4a6da2053a7c031ce829be1e7859a0b2f16b2bd088",
    "config.json": "8f6115914aa33b7b48c92efb077c0976bec71407db65c4daa39961793ac4009a",
    "configuration_siglip2.py": "dca3068142f1cacf9891fd3e1f896891afa30457e20205d40e643f8bd39892bc",
    "configuration_youtu_vl.py": "c69b40aa372cddaaf27feca16205b82650f0db684e74202335d70bfa791df595",
    "generation_config.json": "8666c5875737011451ad9874fa297adbb5d6c3de47ef647a17c41ccee5af7e90",
    "image_processing_siglip2_fast.py": "56cbb7f814ac582e6ff6abd0f7486ee1e65fef998994c2ebabc4540a079ed904",
    "model-00001-of-00003.safetensors": "107939f0c4aaf5fdc7c5d9f4ad741546e4ebaef32fcd93d36a462f1e5ea04d0b",
    "model-00002-of-00003.safetensors": "f7898bdeb9c5cbc996c0a2789c7b3495c3faa179c9690cb2c01bad65b615250e",
    "model-00003-of-00003.safetensors": "548ebf9f8deea536596d6dffc3fed769a60f15e414e6fb9f735a7bb7fa627340",
    "model.safetensors.index.json": "8617dad368bb394a9de5f24f08b5b0ecdd8a68b2b8588d2baf2cc8f2bc5b6b7c",
    "modeling_siglip2.py": "af9df4e395d7a1c5dcce6ae2cc87155ec1712c0eb025dc4b14674efb6604ac4f",
    "modeling_youtu_vl.py": "6ec2ffee9382f248149462f111a92d58846fca52dfb7f12aa184fd13b8eefda5",
    "preprocessor_config.json": "eb41cbafe9513f75c79c6f7c1d64f0d8f7419f98cc6e097e63b9467aee6dd6d7",
    "processing_youtu_vl.py": "415c83d7763e5aa97867344f2d71178d5523d310410a08c64b335faf57e4a667",
    "special_tokens_map.json": "03d62862d41de30db9e05cb4865de4f36c48ab3032327139fcdfacc5798a4828",
    "tokenizer.json": "41998384e9cea31ab97207e2ed59fed66b5481bf0c85fd04f8c7bbd3f7648a6d",
    "tokenizer_config.json": "da4d16d9ba9de8afb26c3eb28b474450c2dbd2a647b7159fbbd87d61e78f98a6",
}
# Filled from canonical_artifact_sha(CHECKPOINT_ARTIFACTS). Keeping this
# literal makes accidental edits to the maintained identity fail closed.
CHECKPOINT_ARTIFACT_MANIFEST_SHA256 = (
    "4d67dd2750d1ee8d87e68b0b52f5aa6c5b5b1dd85385df643845e393628812c9"
)

FIXTURE_PATH = "tests/fixtures/test_image.png"
FIXTURE_SHA256 = "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261"
PROMPT = "<|image_pad|>\nDescribe the image briefly."
IMAGE_TOKEN_ID = 128_264
VISION_DEPTH = 27
VISION_HIDDEN = 1_152
TEXT_LAYERS = 40
TEXT_HIDDEN = 2_560
VOCAB = 283_386
KV_LORA_RANK = 512
ROPE_WIDTH = 64
KV_WIDTH = KV_LORA_RANK + ROPE_WIDTH
PATCH_SIZE = 16
PATCH_WIDTH = 3 * PATCH_SIZE * PATCH_SIZE
PATCHES = 256
MERGED_TOKENS = PATCHES // 4
VISION_ROPE_WIDTH = 36
MAX_NEW_TOKENS = 4

SELECTED_VISION_LAYERS = (
    (0, "window"),
    (7, "full"),
    (15, "full"),
    (23, "full"),
    (26, "full"),
)
VISION_STAGE_NAMES = (
    "patch_projection",
    *(f"layer.{index}.{kind}" for index, kind in SELECTED_VISION_LAYERS),
    "post_layernorm",
    "merger.window_order",
    "merger.restored_order",
)
FLOAT_STAGES = (
    "resized_normalized_pixels",
    "flattened_patches",
    "patches.window_order",
    "vision_rope.freqs",
    *VISION_STAGE_NAMES,
    "prepared_embeddings",
    "prefill_logits",
    "selected_kv",
    "reuse_prefill_logits",
    "reuse_selected_kv",
)
INTEGER_STAGES = (
    "expanded_input_ids",
    "placeholder_positions",
    "spatial_shapes",
    "window_group_index",
    "reverse_group_index",
    "window_cu_seqlens",
    "full_cu_seqlens",
    "greedy_tokens",
    "reuse_greedy_tokens",
)
REQUIRED_STAGES = FLOAT_STAGES + INTEGER_STAGES
LIFECYCLE = {
    "slot": 0,
    "events": [
        "reset",
        "prefill",
        "greedy_decode",
        "reset_for_reuse",
        "prefill_reuse",
        "greedy_decode_reuse",
    ],
    "reuse_must_match_fresh": True,
}


class ContractError(ValueError):
    """The capture does not satisfy the closed oracle contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_sha(artifacts: dict[str, str]) -> str:
    payload = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_checkpoint(model: Path) -> dict[str, Any]:
    canonical = canonical_artifact_sha(CHECKPOINT_ARTIFACTS)
    if canonical != CHECKPOINT_ARTIFACT_MANIFEST_SHA256:
        raise ContractError(
            "internal checkpoint artifact manifest hash differs: "
            f"{canonical}"
        )
    for filename, expected in CHECKPOINT_ARTIFACTS.items():
        path = model / filename
        if not path.is_file():
            raise ContractError(f"checkpoint artifact is missing: {filename}")
        actual = sha256(path)
        if actual != expected:
            raise ContractError(
                f"checkpoint artifact hash differs for {filename}: "
                f"expected {expected}, got {actual}"
            )
    unexpected_weights = sorted(
        path.name
        for path in model.glob("*.safetensors")
        if path.name not in CHECKPOINT_ARTIFACTS
    )
    if unexpected_weights:
        raise ContractError(
            f"checkpoint has unpinned weight artifact(s): {unexpected_weights}"
        )
    return {
        "repo": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "artifact_manifest": {
            "canonical_sha256": canonical,
            "files": CHECKPOINT_ARTIFACTS,
        },
    }


def require_fixture(image: Path) -> None:
    if sha256(image) != FIXTURE_SHA256:
        raise ContractError(f"fixture hash differs: {image}")


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ContractError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"read {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return parsed


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "manifest.sha256").write_text(sha256(path) + "\n", encoding="ascii")


def ensure_new_output(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractError(
            f"immutable capture output already exists: {path}"
        ) from error


def store_array(
    np: Any,
    root: Path,
    stage: str,
    value: Any,
    dtype: str,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if dtype.startswith("float") and not np.isfinite(array).all():
        raise ContractError(f"{stage} contains a non-finite value")
    filename = f"image_text.{stage}.bin"
    path = root / filename
    array.tofile(path)
    return {
        "file": filename,
        "dtype": dtype,
        "shape": list(array.shape),
        "sha256": sha256(path),
    }


def patch_order(torch: Any, window_index: Any) -> Any:
    offsets = torch.arange(4, device=window_index.device)
    return (window_index[:, None] * 4 + offsets[None, :]).reshape(-1)


def unpatch(torch: Any, patches: Any, height: int, width: int) -> Any:
    rows = patches.reshape(height, width, 3, PATCH_SIZE, PATCH_SIZE)
    return (
        rows.permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(3, height * PATCH_SIZE, width * PATCH_SIZE)
    )


def rotate_half(torch: Any, value: Any) -> Any:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def interleaved_rotary_k(torch: Any, raw: Any, cos: Any, sin: Any) -> Any:
    batch, heads, sequence, width = raw.shape
    reordered = (
        raw.view(batch, heads, sequence, width // 2, 2)
        .transpose(4, 3)
        .reshape(batch, heads, sequence, width)
    )
    return reordered * cos.unsqueeze(1) + rotate_half(
        torch, reordered
    ) * sin.unsqueeze(1)


def greedy_language_capture(
    torch: Any,
    model: Any,
    prepared: Any,
    max_new_tokens: int,
) -> tuple[Any, Any, Any]:
    latent: dict[int, Any] = {}
    compressed: dict[int, Any] = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(
            layer.self_attn.kv_a_layernorm.register_forward_hook(
                lambda _module, _args, output, index=index: latent.__setitem__(
                    index, output.detach()
                )
            )
        )
        handles.append(
            layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
                lambda _module, _args, output, index=index: compressed.__setitem__(
                    index, output.detach()
                )
            )
        )
    try:
        positions = torch.arange(
            prepared.shape[1], device=prepared.device
        ).unsqueeze(0)
        cos, sin = model.model.rotary_emb(prepared, positions)
        outputs = model.model(
            inputs_embeds=prepared,
            use_cache=True,
            cache_position=positions[0],
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :]).float()
        if set(latent) != set(range(TEXT_LAYERS)) or set(compressed) != set(
            range(TEXT_LAYERS)
        ):
            raise ContractError("HF eager MLA hooks did not capture every layer")
        padded_rows = []
        for index in range(TEXT_LAYERS):
            latent_row = latent[index][0, -1, :].float()
            raw_rotary = compressed[index][
                :, -1:, KV_LORA_RANK:
            ].reshape(1, 1, 1, ROPE_WIDTH)
            rotated = interleaved_rotary_k(
                torch, raw_rotary, cos[:, -1:, :], sin[:, -1:, :]
            )[0, 0, 0].float()
            zeros_latent = torch.zeros_like(latent_row)
            zeros_rotary = torch.zeros_like(rotated)
            padded_rows.append(
                torch.stack(
                    (
                        torch.cat((latent_row, zeros_rotary)),
                        torch.cat((zeros_latent, rotated)),
                    )
                )
            )
        selected_kv = torch.stack(padded_rows)
        tokens = []
        current = logits.argmax(dim=-1, keepdim=True)
        tokens.append(int(current.item()))
        cache = outputs.past_key_values
        while len(tokens) < max_new_tokens:
            decoded = model.model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
            )
            step_logits = model.lm_head(
                decoded.last_hidden_state[:, -1, :]
            ).float()
            current = step_logits.argmax(dim=-1, keepdim=True)
            tokens.append(int(current.item()))
            cache = decoded.past_key_values
        return logits[0], selected_kv, torch.tensor(tokens, dtype=torch.int32)
    finally:
        for handle in handles:
            handle.remove()


def capture(args: argparse.Namespace) -> int:
    if args.max_new != MAX_NEW_TOKENS:
        raise ContractError(
            f"pinned capture requires --max-new {MAX_NEW_TOKENS}"
        )
    checkpoint = require_checkpoint(args.model)
    require_fixture(args.image)
    ensure_new_output(args.out)
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as error:
        raise ContractError(
            "capture requires numpy, torch, Pillow, and transformers"
        ) from error

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    model.eval()
    image = Image.open(args.image).convert("RGB")
    unexpanded = processor.tokenizer(
        PROMPT, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    inputs = processor(
        text=PROMPT,
        images=image,
        max_image_patches=PATCHES,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    spatial_shapes = inputs["spatial_shapes"]
    if list(spatial_shapes.shape) != [1, 2] or spatial_shapes.tolist() != [
        [16, 16]
    ]:
        raise ContractError(
            f"pinned fixture grid must be [[16, 16]], got {spatial_shapes.tolist()}"
        )
    flat_patches = pixel_values[0]
    encoder = model.siglip2.vision_model.encoder
    window_index, window_cu = encoder.get_window_index(spatial_shapes)
    window_index = window_index.to(dtype=torch.int64)
    reverse_index = torch.argsort(window_index)
    ordered_patches = flat_patches[patch_order(torch, window_index)]
    rope = encoder.rot_pos_emb(spatial_shapes)
    ordered_rope = rope[patch_order(torch, window_index)]
    full_cu = torch.nn.functional.pad(
        (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(
            dim=0, dtype=torch.int32
        ),
        (1, 0),
    )

    selected_outputs: dict[int, Any] = {}
    handles = []
    for index, _kind in SELECTED_VISION_LAYERS:
        handles.append(
            encoder.layers[index].register_forward_hook(
                lambda _module, _args, output, index=index: selected_outputs.__setitem__(
                    index, output[0].detach()
                )
            )
        )
    try:
        with torch.inference_mode():
            vision = model.siglip2(
                pixel_values.to(dtype=model.siglip2.dtype),
                inputs.get("pixel_attention_mask"),
                spatial_shapes,
            ).last_hidden_state
            restored_merged = model.merger(vision, spatial_shapes)
    finally:
        for handle in handles:
            handle.remove()
    if set(selected_outputs) != {index for index, _ in SELECTED_VISION_LAYERS}:
        raise ContractError("HF eager vision hooks omitted a selected layer")

    window_patch_order = patch_order(torch, window_index)
    patch_projection = model.siglip2.vision_model.embeddings(
        pixel_values.to(dtype=model.siglip2.dtype)
    )[window_patch_order]
    # The encoder restores group order before returning. LayerNorm is pointwise,
    # so reindexing its output recovers the module's window-order seam.
    post_layernorm_window = vision[window_patch_order]
    merger_window = restored_merged[window_index]
    text_embeddings = model.model.embed_tokens(input_ids)
    image_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(
        text_embeddings
    )
    prepared = text_embeddings.masked_scatter(
        image_mask,
        restored_merged.to(
            device=text_embeddings.device, dtype=text_embeddings.dtype
        ).unsqueeze(0),
    )

    with torch.inference_mode():
        fresh_logits, fresh_kv, fresh_tokens = greedy_language_capture(
            torch, model, prepared, args.max_new
        )
        reuse_logits, reuse_kv, reuse_tokens = greedy_language_capture(
            torch, model, prepared, args.max_new
        )
    if not torch.equal(fresh_tokens, reuse_tokens):
        raise ContractError("fresh-cache and reuse token streams differ")
    if not torch.equal(fresh_logits, reuse_logits) or not torch.equal(
        fresh_kv, reuse_kv
    ):
        raise ContractError("fresh-cache and reuse language captures differ")

    arrays: dict[str, Any] = {}

    def store(stage: str, value: Any, dtype: str) -> None:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            value = value.float() if dtype == "float32" else value.int()
            value = value.numpy()
        arrays[stage] = store_array(np, args.out, stage, value, dtype)

    store(
        "resized_normalized_pixels",
        unpatch(torch, flat_patches, 16, 16),
        "float32",
    )
    store("flattened_patches", flat_patches, "float32")
    store("patches.window_order", ordered_patches, "float32")
    store("vision_rope.freqs", ordered_rope, "float32")
    store("patch_projection", patch_projection, "float32")
    for index, kind in SELECTED_VISION_LAYERS:
        store(f"layer.{index}.{kind}", selected_outputs[index], "float32")
    store("post_layernorm", post_layernorm_window, "float32")
    store("merger.window_order", merger_window, "float32")
    store("merger.restored_order", restored_merged, "float32")
    store("prepared_embeddings", prepared, "float32")
    store("prefill_logits", fresh_logits, "float32")
    store("selected_kv", fresh_kv, "float32")
    store("reuse_prefill_logits", reuse_logits, "float32")
    store("reuse_selected_kv", reuse_kv, "float32")
    store("expanded_input_ids", input_ids, "int32")
    store(
        "placeholder_positions",
        (input_ids[0] == IMAGE_TOKEN_ID).nonzero().flatten(),
        "int32",
    )
    store("spatial_shapes", spatial_shapes, "int32")
    store("window_group_index", window_index, "int32")
    store("reverse_group_index", reverse_index, "int32")
    store("window_cu_seqlens", torch.tensor(window_cu), "int32")
    store("full_cu_seqlens", full_cu, "int32")
    store("greedy_tokens", fresh_tokens, "int32")
    store("reuse_greedy_tokens", reuse_tokens, "int32")

    manifest = {
        "schema": SCHEMA_VERSION,
        "contract": CONTRACT,
        "producer": "hf-transformers-eager",
        "checkpoint": checkpoint,
        "fixture": {"path": FIXTURE_PATH, "sha256": FIXTURE_SHA256},
        "architecture": {
            "vision_depth": VISION_DEPTH,
            "vision_hidden": VISION_HIDDEN,
            "selected_vision_layers": [
                {"index": index, "attention": kind}
                for index, kind in SELECTED_VISION_LAYERS
            ],
            "text_layers": TEXT_LAYERS,
            "text_hidden": TEXT_HIDDEN,
            "vocab": VOCAB,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_rope_head_dim": ROPE_WIDTH,
        },
        "generation": {"mode": "greedy", "max_new_tokens": MAX_NEW_TOKENS},
        "lifecycle": LIFECYCLE,
        "case": {
            "name": "image_text",
            "prompt": PROMPT,
            "unexpanded_input_ids": unexpanded[0].tolist(),
            "arrays": arrays,
        },
    }
    write_manifest(args.out, manifest)
    print(args.out / "manifest.json")
    return 0


def expected_shapes(sequence: int) -> dict[str, list[int]]:
    shapes = {
        "resized_normalized_pixels": [3, 256, 256],
        "flattened_patches": [PATCHES, PATCH_WIDTH],
        "patches.window_order": [PATCHES, PATCH_WIDTH],
        "vision_rope.freqs": [PATCHES, VISION_ROPE_WIDTH],
        "patch_projection": [PATCHES, VISION_HIDDEN],
        "post_layernorm": [PATCHES, VISION_HIDDEN],
        "merger.window_order": [MERGED_TOKENS, TEXT_HIDDEN],
        "merger.restored_order": [MERGED_TOKENS, TEXT_HIDDEN],
        "prepared_embeddings": [1, sequence, TEXT_HIDDEN],
        "prefill_logits": [VOCAB],
        "selected_kv": [TEXT_LAYERS, 2, KV_WIDTH],
        "reuse_prefill_logits": [VOCAB],
        "reuse_selected_kv": [TEXT_LAYERS, 2, KV_WIDTH],
        "expanded_input_ids": [sequence],
        "placeholder_positions": [MERGED_TOKENS],
        "spatial_shapes": [1, 2],
        "window_group_index": [MERGED_TOKENS],
        "reverse_group_index": [MERGED_TOKENS],
        "window_cu_seqlens": [2],
        "full_cu_seqlens": [2],
        "greedy_tokens": [MAX_NEW_TOKENS],
        "reuse_greedy_tokens": [MAX_NEW_TOKENS],
    }
    for index, kind in SELECTED_VISION_LAYERS:
        shapes[f"layer.{index}.{kind}"] = [PATCHES, VISION_HIDDEN]
    return shapes


def validate_checkpoint(value: Any) -> None:
    expected = {
        "repo": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "artifact_manifest": {
            "canonical_sha256": CHECKPOINT_ARTIFACT_MANIFEST_SHA256,
            "files": CHECKPOINT_ARTIFACTS,
        },
    }
    if value != expected:
        raise ContractError("checkpoint identity or artifact hashes differ")
    if (
        canonical_artifact_sha(value["artifact_manifest"]["files"])
        != CHECKPOINT_ARTIFACT_MANIFEST_SHA256
    ):
        raise ContractError("checkpoint artifact manifest is not canonical")


def validate_manifest_hash(root: Path) -> None:
    expected = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    if len(expected) != 64 or sha256(root / "manifest.json") != expected:
        raise ContractError("manifest.sha256 differs")


def validate_capture(root: Path, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_manifest_hash(root)
    manifest = strict_json_load(root / "manifest.json")
    required_top = {
        "schema",
        "contract",
        "producer",
        "checkpoint",
        "fixture",
        "architecture",
        "generation",
        "lifecycle",
        "case",
    }
    if set(manifest) != required_top:
        raise ContractError(f"{role} manifest has an invalid top-level schema")
    if manifest["schema"] != SCHEMA_VERSION or manifest["contract"] != CONTRACT:
        raise ContractError(f"{role} schema or contract differs")
    expected_producer = (
        "hf-transformers-eager" if role == "reference" else "mlxcel-xla-diagnostics"
    )
    if manifest["producer"] != expected_producer:
        raise ContractError(f"{role} producer differs")
    validate_checkpoint(manifest["checkpoint"])
    if manifest["fixture"] != {
        "path": FIXTURE_PATH,
        "sha256": FIXTURE_SHA256,
    }:
        raise ContractError(f"{role} fixture identity differs")
    architecture = manifest["architecture"]
    if architecture != {
        "vision_depth": VISION_DEPTH,
        "vision_hidden": VISION_HIDDEN,
        "selected_vision_layers": [
            {"index": index, "attention": kind}
            for index, kind in SELECTED_VISION_LAYERS
        ],
        "text_layers": TEXT_LAYERS,
        "text_hidden": TEXT_HIDDEN,
        "vocab": VOCAB,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": ROPE_WIDTH,
    }:
        raise ContractError(f"{role} architecture contract differs")
    if manifest["generation"] != {
        "mode": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
    }:
        raise ContractError(f"{role} generation contract differs")
    if manifest["lifecycle"] != LIFECYCLE:
        raise ContractError(f"{role} lifecycle contract differs")
    case = manifest["case"]
    if not isinstance(case, dict) or set(case) != {
        "name",
        "prompt",
        "unexpanded_input_ids",
        "arrays",
    }:
        raise ContractError(f"{role} case schema differs")
    if case["name"] != "image_text" or case["prompt"] != PROMPT:
        raise ContractError(f"{role} case identity differs")
    if not isinstance(case["unexpanded_input_ids"], list) or not all(
        isinstance(value, int) for value in case["unexpanded_input_ids"]
    ):
        raise ContractError(f"{role} unexpanded token ids are invalid")
    arrays = case["arrays"]
    if not isinstance(arrays, dict) or set(arrays) != set(REQUIRED_STAGES):
        raise ContractError(f"{role} stage set differs")
    expanded = arrays["expanded_input_ids"]
    if not isinstance(expanded, dict):
        raise ContractError(f"{role} expanded_input_ids spec is invalid")
    shape = expanded.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise ContractError(f"{role} expanded sequence shape is invalid")
    shapes = expected_shapes(shape[0])
    for stage in REQUIRED_STAGES:
        spec = arrays[stage]
        expected_dtype = "int32" if stage in INTEGER_STAGES else "float32"
        if not isinstance(spec, dict) or set(spec) != {
            "file",
            "dtype",
            "shape",
            "sha256",
        }:
            raise ContractError(f"{role} {stage} array spec schema differs")
        if spec["dtype"] != expected_dtype or spec["shape"] != shapes[stage]:
            raise ContractError(f"{role} {stage} dtype or shape differs")
        filename = f"image_text.{stage}.bin"
        if spec["file"] != filename:
            raise ContractError(f"{role} {stage} filename differs")
        path = root / filename
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ContractError(f"{role} {stage} artifact hash differs")
        itemsize = 4
        elements = math.prod(spec["shape"])
        if path.stat().st_size != elements * itemsize:
            raise ContractError(f"{role} {stage} byte length differs")
    expected_files = {
        "manifest.json",
        "manifest.sha256",
        *(spec["file"] for spec in arrays.values()),
    }
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ContractError(f"{role} capture file set differs")
    return manifest, arrays


def load_array(np: Any, root: Path, spec: dict[str, Any]) -> Any:
    return np.fromfile(root / spec["file"], dtype=np.dtype(spec["dtype"])).reshape(
        spec["shape"]
    )


def validate_semantics(np: Any, root: Path, arrays: dict[str, Any], role: str) -> None:
    spatial = load_array(np, root, arrays["spatial_shapes"])
    if spatial.tolist() != [[16, 16]]:
        raise ContractError(f"{role} spatial grid differs")
    expanded = load_array(np, root, arrays["expanded_input_ids"])
    placeholders = load_array(np, root, arrays["placeholder_positions"])
    expected_placeholders = np.flatnonzero(expanded == IMAGE_TOKEN_ID).astype(
        np.int32
    )
    if not np.array_equal(placeholders, expected_placeholders):
        raise ContractError(f"{role} placeholder selection differs")
    if placeholders.size != MERGED_TOKENS:
        raise ContractError(f"{role} placeholder count differs")
    window = load_array(np, root, arrays["window_group_index"])
    reverse = load_array(np, root, arrays["reverse_group_index"])
    if not np.array_equal(np.sort(window), np.arange(MERGED_TOKENS)):
        raise ContractError(f"{role} window_group_index is not a permutation")
    if not np.array_equal(reverse[window], np.arange(MERGED_TOKENS)):
        raise ContractError(f"{role} reverse_group_index is not the inverse")
    for stage in ("window_cu_seqlens", "full_cu_seqlens"):
        boundaries = load_array(np, root, arrays[stage])
        if (
            boundaries[0] != 0
            or boundaries[-1] != PATCHES
            or np.any(boundaries[1:] <= boundaries[:-1])
        ):
            raise ContractError(f"{role} {stage} boundaries differ")
    for fresh, reuse in (
        ("prefill_logits", "reuse_prefill_logits"),
        ("selected_kv", "reuse_selected_kv"),
        ("greedy_tokens", "reuse_greedy_tokens"),
    ):
        if not np.array_equal(
            load_array(np, root, arrays[fresh]),
            load_array(np, root, arrays[reuse]),
        ):
            raise ContractError(f"{role} lifecycle reuse differs for {fresh}")


def stage_tolerance(stage: str) -> tuple[float, float]:
    if stage in INTEGER_STAGES:
        return 0.0, 0.0
    if stage in {
        "resized_normalized_pixels",
        "flattened_patches",
        "patches.window_order",
        "vision_rope.freqs",
    }:
        return 1e-5, 1e-5
    return 2e-2, 2e-2


def compare_capture_roots(reference: Path, actual: Path) -> dict[str, Any]:
    try:
        import numpy as np

        reference_manifest, reference_arrays = validate_capture(
            reference, "reference"
        )
        actual_manifest, actual_arrays = validate_capture(actual, "actual")
        if (
            reference_manifest["case"]["unexpanded_input_ids"]
            != actual_manifest["case"]["unexpanded_input_ids"]
        ):
            raise ContractError("unexpanded tokenizer ids differ")
        validate_semantics(np, reference, reference_arrays, "reference")
        validate_semantics(np, actual, actual_arrays, "actual")
    except (ContractError, ImportError, OSError) as error:
        return {
            "passed": False,
            "first_divergence": {"stage": "contract"},
            "error": str(error),
        }

    stages = []
    first_divergence = None
    for stage in REQUIRED_STAGES:
        expected = load_array(np, reference, reference_arrays[stage])
        observed = load_array(np, actual, actual_arrays[stage])
        atol, rtol = stage_tolerance(stage)
        finite = (
            np.isfinite(expected).all() and np.isfinite(observed).all()
            if stage in FLOAT_STAGES
            else True
        )
        absolute = np.abs(
            observed.astype(np.float64) - expected.astype(np.float64)
        )
        threshold = atol + rtol * np.abs(expected.astype(np.float64))
        matches = bool(finite and np.all(absolute <= threshold))
        mismatch = (
            int(np.flatnonzero(absolute > threshold)[0])
            if finite and np.any(absolute > threshold)
            else None
        )
        entry = {
            "stage": stage,
            "passed": matches,
            "max_absolute": float(absolute.max(initial=0.0)),
            "atol": atol,
            "rtol": rtol,
        }
        if mismatch is not None:
            entry["first_mismatch_flat_index"] = mismatch
        stages.append(entry)
        if not matches and first_divergence is None:
            first_divergence = {"stage": stage, "flat_index": mismatch}

    return {
        "passed": first_divergence is None,
        "first_divergence": first_divergence,
        "stages": stages,
    }


def compare(args: argparse.Namespace) -> int:
    report = compare_capture_roots(args.reference, args.actual)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.report)
    return 0 if report["passed"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--model", type=Path, required=True)
    capture_parser.add_argument("--image", type=Path, required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS)
    capture_parser.set_defaults(run=capture)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--actual", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.set_defaults(run=compare)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.run(args))
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
