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
SOURCE_MODEL_SHA256 = "ec7b02696781afdb1f27871fdffe0f71ef030932d10fbd759bc59392669605f7"
SOURCE_TOKENIZER_SHA256 = "d26f54ac5bcc30ba15d418234e89d2ca44caf0bd57ce14749612a74f436738ef"
CONVERTED_REPO = "mlx-community/llava-interleave-qwen-0.5b-bf16"
CONVERTED_REVISION = "ba7385935f69c5417bfbe29c3809858a98afc22f"
CONVERTED_MODEL_SHA256 = "43919c1ea46e00c6063204515169bb9635d0c2c6b3a07f975e20e9ea23c33d4c"
CONVERTED_TOKENIZER_SHA256 = "32e8f623d8dce60b5a93496ec810434ef744287ac041cf2c6032743a3578baa5"
IMAGE_SIZE = 384
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
    require_sha(source / "model.safetensors", SOURCE_MODEL_SHA256, "source weights")
    require_sha(source / "tokenizer.json", SOURCE_TOKENIZER_SHA256, "source tokenizer")
    require_sha(
        converted / "model.safetensors", CONVERTED_MODEL_SHA256, "converted weights"
    )
    require_sha(
        converted / "tokenizer.json",
        CONVERTED_TOKENIZER_SHA256,
        "converted tokenizer",
    )
    conversion_equivalence = verify_vision_block0_conversion(
        source / "model.safetensors", converted / "model.safetensors"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    image = Image.open(args.image).convert("RGB")
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
            "model_sha256": SOURCE_MODEL_SHA256,
            "tokenizer_sha256": SOURCE_TOKENIZER_SHA256,
            "license": "Tongyi Qianwen Research License",
        },
        "converted_checkpoint": {
            "repo": CONVERTED_REPO,
            "revision": CONVERTED_REVISION,
            "model_sha256": CONVERTED_MODEL_SHA256,
            "tokenizer_sha256": CONVERTED_TOKENIZER_SHA256,
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

    for definition in cases(processor, args.image.resolve()):
        started = time.perf_counter()
        images = [image.copy() for _ in range(definition["image_count"])]
        inputs = processor(
            text=definition["text"],
            images=images or None,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
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


def load_array(root: Path, spec: dict[str, Any]) -> np.ndarray:
    dtype = spec["dtype"]
    path = root / spec["file"]
    if dtype == "bfloat16":
        raw = np.fromfile(path, dtype=np.uint16)
        values = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.fromfile(path, dtype=np.dtype(dtype))
    return values.reshape(spec["shape"])


def compare(args: argparse.Namespace) -> int:
    reference_manifest = json.loads(
        (args.reference / "manifest.json").read_text(encoding="utf-8")
    )
    actual_manifest = json.loads(
        (args.actual / "manifest.json").read_text(encoding="utf-8")
    )
    reference_cases = {case["name"]: case for case in reference_manifest["cases"]}
    actual_cases = {case["name"]: case for case in actual_manifest["cases"]}
    report: dict[str, Any] = {
        "schema": 1,
        "reference": str(args.reference),
        "actual": str(args.actual),
        "passed": True,
        "first_divergence": None,
        "cases": [],
    }
    for case_name in ("image_text", "two_images", "no_image"):
        case_report: dict[str, Any] = {
            "name": case_name,
            "passed": True,
            "stages": [],
        }
        if case_name not in reference_cases or case_name not in actual_cases:
            case_report["passed"] = False
            case_report["error"] = "case missing from one manifest"
            report["passed"] = False
            report["first_divergence"] = report["first_divergence"] or {
                "case": case_name,
                "stage": "case",
            }
            report["cases"].append(case_report)
            continue
        reference = reference_cases[case_name]
        actual = actual_cases[case_name]
        for stage in STAGE_ORDER:
            ref_spec = reference["arrays"].get(stage)
            actual_spec = actual["arrays"].get(stage)
            if ref_spec is None and actual_spec is None:
                continue
            stage_report: dict[str, Any] = {"stage": stage, "passed": True}
            if ref_spec is None or actual_spec is None:
                stage_report.update(
                    passed=False, error="stage missing from one capture"
                )
            else:
                ref = load_array(args.reference, ref_spec)
                got = load_array(args.actual, actual_spec)
                stage_report["reference_shape"] = list(ref.shape)
                stage_report["actual_shape"] = list(got.shape)
                if ref.shape != got.shape:
                    stage_report.update(passed=False, error="shape mismatch")
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
