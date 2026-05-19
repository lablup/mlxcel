#!/usr/bin/env python3
"""mlx-lm / mlx-vlm baseline sweep matching mlxcel's bench_decode.sh schema.

Run modes:
  ./scripts/bench_mlxlm.py all               # text sweep, all models
  ./scripts/bench_mlxlm.py all --vlm         # VLM sweep, all models
  ./scripts/bench_mlxlm.py models/<dir>      # single model

Output CSV: benchmarks/pylm_<hw>_YYYY-MM-DD[ _vlm].csv

Each model runs in a fresh Python subprocess for memory isolation and
to survive crashes. Numbers come from mlx-lm / mlx-vlm's own
`prompt_tps` / `generation_tps` fields on the final stream_generate
response after a warmup pass.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

TEXT_PROMPT = "Hello, how are you today?"
VLM_PROMPT = "What is in this image?"
VLM_IMAGE = "tests/fixtures/test_image.png"
MAX_TOKENS = 100
WARMUP_TOKENS = 4
MODELS_DIR = Path("./models")
BENCHMARKS_DIR = Path("./benchmarks")
# Memory budget: 85% of 128 GB. Override with PYLM_BENCH_MAX_GB env var to
# enforce a tighter cap (e.g. PYLM_BENCH_MAX_GB=65 to skip very large MoE
# models on a 128 GB host where 85% × 128 GB ≈ 108 GB still admits them).
_env_max_gb = os.environ.get("PYLM_BENCH_MAX_GB")
if _env_max_gb:
    MEMORY_LIMIT_BYTES = int(float(_env_max_gb) * 1024 * 1024 * 1024)
else:
    MEMORY_LIMIT_BYTES = int(128 * 1024 * 1024 * 1024 * 0.85)
# Thermal cooldown thresholds
BIG_MODEL_GB = 20

CSV_HEADER = (
    "model,model_path,prompt_tokens,generated_tokens,"
    "prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,"
    "date,hardware,mlx_version,build_type,max_tokens,prompt"
)


def detect_hardware():
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        chip = "unknown"
    mapping = {
        "M1 Ultra": ("m1ultra", "Apple_M1_Ultra_128GB"),
        "M1 Max": ("m1max", "Apple_M1_Max"),
        "M2 Ultra": ("m2ultra", "Apple_M2_Ultra"),
        "M3 Ultra": ("m3ultra", "Apple_M3_Ultra"),
        "M3 Max": ("m3max", "Apple_M3_Max"),
        "M4 Max": ("m4max", "Apple_M4_Max"),
        "M5 Ultra": ("m5ultra", "Apple_M5_Ultra"),
        "M5 Max": ("m5max", "Apple_M5_Max_128GB"),
    }
    for key, val in mapping.items():
        if key in chip:
            return val
    return ("unknown", chip.replace(" ", "_"))


def estimate_model_size(model_path: Path) -> int:
    """Sum of safetensors file sizes (bytes). Tolerates broken symlinks."""
    total = 0
    for f in model_path.glob("*.safetensors"):
        try:
            total += f.stat().st_size
        except (FileNotFoundError, OSError):
            continue
    return total


def is_vlm(model_path: Path) -> bool:
    """Detect VLM by config.json contents or preprocessor presence."""
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
    VLM_ARCH_SUBSTR = (
        "Llava", "PaliGemma", "Qwen2VL", "Qwen2_5_VL", "Qwen3VL",
        "Idefics", "Pixtral", "Bunny", "Phi3V", "Phi35V", "AyaVision",
        "Gemma3ForConditional", "Gemma4ForConditional", "Mllama",
        "Mistral3", "Llama4", "MolmoForCausalLM", "Molmo", "InternVL",
        "GotOcr", "Smolvlm", "Florence", "Kimi",
    )
    for a in archs:
        if any(sub in a for sub in VLM_ARCH_SUBSTR):
            return True
    # Has image preprocessor → likely VLM
    if (model_path / "preprocessor_config.json").exists():
        return True
    return False


# ---------------------------------------------------------------------------
# Single-model benchmark subprocess code
# ---------------------------------------------------------------------------

SUBPROCESS_TEXT_CODE = """
import json, sys, time
from mlx_lm import load, stream_generate

model_path = sys.argv[1]
prompt = sys.argv[2]
warmup_tokens = int(sys.argv[3])
max_tokens = int(sys.argv[4])

try:
    model, tokenizer = load(model_path, tokenizer_config={'trust_remote_code': True})
except Exception as e:
    print('ERROR load:', repr(e), file=sys.stderr)
    sys.exit(2)

# Warmup
try:
    for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=warmup_tokens):
        pass
except Exception as e:
    print('ERROR warmup:', repr(e), file=sys.stderr)
    sys.exit(3)

# Measured pass
last = None
try:
    for r in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        last = r
except Exception as e:
    print('ERROR bench:', repr(e), file=sys.stderr)
    sys.exit(4)

if last is None:
    print('ERROR no_output', file=sys.stderr)
    sys.exit(5)

result = {
    'prompt_tokens': getattr(last, 'prompt_tokens', None),
    'prefill_tps': getattr(last, 'prompt_tps', None),
    'gen_tokens': getattr(last, 'generation_tokens', None),
    'decode_tps': getattr(last, 'generation_tps', None),
    'finish_reason': getattr(last, 'finish_reason', None),
}
print('RESULT', json.dumps(result))
"""

SUBPROCESS_VLM_CODE = """
import json, sys, time
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = sys.argv[1]
prompt_text = sys.argv[2]
warmup_tokens = int(sys.argv[3])
max_tokens = int(sys.argv[4])
image_path = sys.argv[5]

try:
    model, processor = load(model_path)
except Exception as e:
    print('ERROR load:', repr(e), file=sys.stderr)
    sys.exit(2)

try:
    config = load_config(model_path)
    formatted_prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
except Exception as e:
    # Fallback: use raw prompt
    formatted_prompt = prompt_text

# Warmup
try:
    for _ in stream_generate(model, processor, prompt=formatted_prompt, image=image_path, max_tokens=warmup_tokens):
        pass
except Exception as e:
    print('ERROR warmup:', repr(e), file=sys.stderr)
    sys.exit(3)

last = None
try:
    for r in stream_generate(model, processor, prompt=formatted_prompt, image=image_path, max_tokens=max_tokens):
        last = r
except Exception as e:
    print('ERROR bench:', repr(e), file=sys.stderr)
    sys.exit(4)

if last is None:
    print('ERROR no_output', file=sys.stderr)
    sys.exit(5)

# mlx-vlm GenerationResult vs mlx-lm GenerationResponse — slightly different fields
result = {
    'prompt_tokens': getattr(last, 'prompt_tokens', None),
    'prefill_tps': getattr(last, 'prompt_tps', None),
    'gen_tokens': getattr(last, 'generation_tokens', None),
    'decode_tps': getattr(last, 'generation_tps', None),
}
print('RESULT', json.dumps(result))
"""


def bench_one(model_path: Path, vlm: bool, vlm_image: str, max_tokens: int,
              warmup_tokens: int, timeout: int):
    """Returns (status, fields_dict_or_None)."""
    if vlm:
        prompt = VLM_PROMPT
        code = SUBPROCESS_VLM_CODE
        args = [
            "python3", "-c", code,
            str(model_path), prompt, str(warmup_tokens), str(max_tokens), vlm_image,
        ]
    else:
        prompt = TEXT_PROMPT
        code = SUBPROCESS_TEXT_CODE
        args = [
            "python3", "-c", code,
            str(model_path), prompt, str(warmup_tokens), str(max_tokens),
        ]

    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("FAIL:timeout", None)

    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                try:
                    data = json.loads(line[len("RESULT "):])
                    return ("OK", data)
                except json.JSONDecodeError:
                    pass
        return ("FAIL:no_result", None)

    # Map exit codes to FAIL types matching bench_decode.sh
    code_to_status = {2: "FAIL:warmup", 3: "FAIL:warmup", 4: "FAIL:bench", 5: "FAIL:no_output"}
    return (code_to_status.get(proc.returncode, "FAIL:exit"), None)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help='"all" for sweep, or models/<dir> for single')
    ap.add_argument("--vlm", action="store_true", help="VLM mode (use image prompt)")
    ap.add_argument("--image", default=VLM_IMAGE, help=f"VLM image (default: {VLM_IMAGE})")
    ap.add_argument("--cooldown", type=int, default=30, help="Seconds between models (default: 30)")
    ap.add_argument("--big-cooldown", type=int, default=30, help="Seconds after big model (default: 30)")
    ap.add_argument("--big-threshold-gb", type=float, default=BIG_MODEL_GB)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--warmup-tokens", type=int, default=WARMUP_TOKENS)
    ap.add_argument("--timeout", type=int, default=600, help="Per-model seconds (default: 600)")
    ap.add_argument("--suffix", default="", help="Optional CSV filename suffix")
    args = ap.parse_args()

    hw_short, hw_full = detect_hardware()
    today = date.today().isoformat()
    # Get versions
    try:
        import mlx_lm, mlx_vlm
        if args.vlm:
            mlx_version = f"mlx-vlm-{mlx_vlm.__version__}"
        else:
            mlx_version = f"mlx-lm-{mlx_lm.__version__}"
    except Exception:
        mlx_version = "unknown"

    # Filename: pylm_<hw>_<date>[_vlm][_<suffix>].csv
    parts = ["pylm", hw_short, today]
    if args.vlm:
        parts.insert(2, "vlm")
    name = "_".join(parts)
    if args.suffix:
        name = f"{name}_{args.suffix}"
    if args.model != "all":
        single = Path(args.model).name
        name = f"{name}_single_{single}"
    csv_path = BENCHMARKS_DIR / f"{name}.csv"
    BENCHMARKS_DIR.mkdir(exist_ok=True)

    # Discover models
    if args.model == "all":
        model_dirs = sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())
    else:
        p = Path(args.model)
        if not p.is_dir():
            sys.exit(f"Error: {p} is not a directory")
        model_dirs = [p]

    # VLM image check
    if args.vlm and not Path(args.image).exists():
        sys.exit(f"Error: VLM image not found at {args.image}")

    big_threshold_bytes = int(args.big_threshold_gb * 1024**3)

    # Resume: skip models already in the CSV
    already_done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0]:
                    already_done.add(row[0])
        print(f">>> Resume: {len(already_done)} models already in {csv_path}", file=sys.stderr)
    else:
        with open(csv_path, "w") as f:
            f.write(CSV_HEADER + "\n")

    def append_row(line: str):
        with open(csv_path, "a") as f:
            f.write(line + "\n")

    for i, mdir in enumerate(model_dirs, 1):
        if mdir.name in already_done:
            print(f">>> [{i}/{len(model_dirs)}] [skip-done] {mdir.name}", file=sys.stderr)
            continue
        name = mdir.name
        size = estimate_model_size(mdir)
        size_gb = size / 1024**3

        # OOM guard (same as bench_decode.sh)
        if size > MEMORY_LIMIT_BYTES:
            print(f">>> [skip]   {name} ({size_gb:.1f} GB > {MEMORY_LIMIT_BYTES/1024**3:.1f} GB limit)", file=sys.stderr)
            prompt = VLM_PROMPT if args.vlm else TEXT_PROMPT
            append_row(f'{name},./{mdir}/,,,,,,,{today},{hw_full},{mlx_version},python,{args.max_tokens},"{prompt}",SKIP:oom_estimate')
            continue

        print(f">>> [{i}/{len(model_dirs)}] [warmup+bench] {name} ({size_gb:.1f} GB) ...", file=sys.stderr)
        t0 = time.perf_counter()
        status, fields = bench_one(
            mdir, vlm=args.vlm, vlm_image=args.image,
            max_tokens=args.max_tokens, warmup_tokens=args.warmup_tokens,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - t0
        prompt = VLM_PROMPT if args.vlm else TEXT_PROMPT

        if status != "OK":
            print(f"    {status} ({elapsed:.1f}s)", file=sys.stderr)
            append_row(f'{name},./{mdir}/,,,,,,,{today},{hw_full},{mlx_version},python,{args.max_tokens},"{prompt}",{status}')
        else:
            f = fields
            pt = f.get("prompt_tokens") or ""
            gt = f.get("gen_tokens") or ""
            pre_tps = f.get("prefill_tps")
            dec_tps = f.get("decode_tps")
            pre_ms = ""
            dec_ms = ""
            if pre_tps and pt:
                try:
                    pre_ms = f"{(int(pt) / float(pre_tps)) * 1000:.2f}"
                except Exception:
                    pass
            if dec_tps and gt:
                try:
                    dec_ms = f"{(int(gt) / float(dec_tps)) * 1000:.2f}"
                except Exception:
                    pass
            pre_tps_s = f"{pre_tps:.2f}" if pre_tps else ""
            dec_tps_s = f"{dec_tps:.2f}" if dec_tps else ""
            print(f"    decode: {dec_tps_s} tok/s ({elapsed:.1f}s)", file=sys.stderr)
            append_row(
                f'{name},./{mdir}/,{pt},{gt},{pre_ms},{pre_tps_s},{dec_ms},{dec_tps_s},'
                f'{today},{hw_full},{mlx_version},python,{args.max_tokens},"{prompt}"'
            )

        # Cooldown
        if i < len(model_dirs):
            cd = args.big_cooldown if size > big_threshold_bytes else args.cooldown
            if cd > 0:
                print(f"    cooldown: {cd}s", file=sys.stderr)
                time.sleep(cd)

    print(f"\nResults saved to: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
