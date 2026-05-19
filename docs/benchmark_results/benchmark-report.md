# Benchmark Report - 2026-05-19

This report summarizes mlxcel performance on Apple Silicon using the
2026-05-19 benchmark campaign. It covers two 128 GB Apple Silicon systems:

- Mac Studio M1 Ultra
- MacBook Pro M5 Max

The same report also compares mlxcel against the Python reference stacks:
`mlx-lm` for text models and `mlx-vlm` for vision-language models.

The short version: mlxcel is already at practical decode parity with the
Python MLX stacks on most comparable models, and it completes many models that
the Python baseline did not complete in this sweep. The strongest public story
is same-host decode throughput and model coverage. Short-prompt prefill is
reported separately because it is more prompt-shape-sensitive and, for VLMs,
includes image encoder and projector work.

## Headline Results

| Host | Mode | Baseline | Comparable pairs | Prefill median vs baseline | Decode median vs baseline | Decode result |
|------|------|----------|-----------------:|---------------------------:|--------------------------:|---------------|
| M5 Max | Text | mlx-lm | 67 | 0.66x | **98%** | 59/67 at >=90% parity |
| M1 Ultra | Text | mlx-lm | 74 | 0.58x | **96%** | 56/74 at >=90% parity |
| M5 Max | VLM | mlx-vlm | 17 | 0.63x | **99%** | 14/17 at >=90% parity |
| M1 Ultra | VLM | mlx-vlm | 17 | 0.93x | **98%** | 11/17 at >=90% parity |

Ratios above use successful runs only. Baseline rows use exact CSV model
identifiers and compare mlxcel and the Python stack on the same host. Decode
parity is the primary comparison point because it measures steady-state
generation after the KV cache is built.

## Why This Matters

mlxcel is not only running close to the Python reference implementations on
the common path; it is also running models that the Python baseline did not
complete in this sweep. With a generated-token threshold of 5 tokens:

| Host | Mode | mlxcel successful, Python baseline unavailable or failed | Comparable successful pairs |
|------|------|---------------------------------------------------------:|----------------------------:|
| M5 Max | Text | 25 | 66 |
| M1 Ultra | Text | 22 | 73 |
| M5 Max | VLM | 12 | 17 |
| M1 Ultra | VLM | 20 | 17 |

This is the useful public framing: mlxcel is a Rust inference engine with
direct MLX bindings that is already near Python decode performance for many
models, while providing broad model coverage across text, MoE, hybrid SSM,
and VLM families.

## Representative Text Models

All values are tokens per second. Prefill and decode are shown separately.
The final column compares mlxcel decode throughput against mlx-lm on the same
host.

| Host | Model | Class | mlxcel prefill | mlxcel decode | mlx-lm decode | vs mlx-lm |
|------|-------|-------|---------------:|--------------:|--------------:|----------:|
| M5 Max | smollm-135m-4bit | Small dense | 900.78 | 883.99 | 711.54 | **124%** |
| M5 Max | qwen2.5-7b-4bit | Dense 7B | 299.05 | 126.63 | 123.59 | **102%** |
| M5 Max | gpt-oss-120b-4bit | Large MoE | 23.71 | 113.34 | 110.35 | **103%** |
| M5 Max | solar-open-100b-4bit | Large MoE | 200.85 | 65.59 | 66.30 | 99% |
| M5 Max | qwen3.5-35b-a3b-4bit | Hybrid MoE | 36.32 | 145.49 | 152.96 | 95% |
| M5 Max | nemotron-h-30b-4bit | Hybrid SSM/MoE | 31.22 | 171.76 | 178.80 | 96% |
| M1 Ultra | phi-3.5-moe-4bit | MoE | 9.16 | 76.13 | 69.28 | **110%** |
| M1 Ultra | minicpm3-4b-4bit | MLA | 87.60 | 79.08 | 73.26 | **108%** |
| M1 Ultra | qwen2.5-0.5b-4bit | Small dense | 452.68 | 329.45 | 315.48 | **104%** |
| M1 Ultra | gpt-oss-120b-4bit | Large MoE | 3.79 | 59.62 | 57.58 | **104%** |
| M1 Ultra | command-r7b-4bit | Dense 7B | 29.10 | 111.42 | 107.75 | **103%** |
| M1 Ultra | solar-open-100b-4bit | Large MoE | 11.19 | 36.20 | 35.69 | **101%** |

The table shows the central performance claim without cross-hardware ranking:
mlxcel is already close to mlx-lm on the same machine for dense 7B models,
newer hybrid MoE families, and very large MoE models such as GPT-OSS 120B.

## Representative VLM Models

All values are tokens per second. VLM prefill includes vision-side work when
the run uses image input, so decode is the cleaner apples-to-apples baseline
comparison.

| Host | Model | Class | mlxcel prefill | mlxcel decode | mlx-vlm decode | vs mlx-vlm |
|------|-------|-------|---------------:|--------------:|---------------:|-----------:|
| M5 Max | qwen3.5-0.8b-4bit | Hybrid GatedDeltaNet VLM | 463.75 | 477.51 | 410.96 | **116%** |
| M5 Max | qwen3.5-35b-a3b-4bit | Hybrid MoE VLM | 38.18 | 149.57 | 128.80 | **116%** |
| M5 Max | gemma-4-e2b-it-4bit | Gemma 4 VLM | 2534.10 | 215.86 | 201.70 | **107%** |
| M5 Max | molmo2-4b | Molmo2 vision encoder | 1605.72 | 64.35 | 66.80 | 96% |
| M1 Ultra | llava-interleave-qwen-0.5b-bf16 | SigLIP + Qwen2 | 3334.88 | 263.41 | 225.15 | **117%** |
| M1 Ultra | aya-vision-8b | SigLIP + Cohere2 | 349.35 | 111.11 | 103.74 | **107%** |
| M1 Ultra | molmo2-4b | Molmo2 vision encoder | 576.46 | 59.54 | 60.87 | 98% |
| M1 Ultra | phi-3.5-vision-4bit | CLIP + HD tiling | 793.61 | 92.64 | 92.53 | **100%** |
| M1 Ultra | pixtral-12b-4bit | Pixtral ViT + Mistral | 442.19 | 60.29 | - | - |

The VLM data is most compelling as a coverage and decode-throughput story:
mlxcel runs many VLM paths in the same benchmark matrix, and its comparable
decode results sit near mlx-vlm median parity on both Apple Silicon hosts.

## Where mlxcel Looks Strongest

- **Steady-state decode:** M5 Max text decode is within 2% of mlx-lm median
  parity and VLM decode is within 1% of mlx-vlm median parity. M1 Ultra shows
  the same pattern, with 96% text median parity and 98% VLM median parity.
- **Large MoE practicality:** GPT-OSS 120B reaches 113.34 tok/s on M5 Max and
  is slightly faster than mlx-lm on the same host in this sweep. Solar-Open
  100B reaches 65.59 tok/s and is effectively at mlx-lm parity.
- **Model coverage:** mlxcel completes many runs where the Python baseline
  fails or is unavailable in the same benchmark matrix. This is especially
  visible for VLM wrappers, Gemma 4 variants, ERNIE, Hunyuan, ExaOne4, and
  several newer hybrid/MoE families.

## Reading the Numbers Correctly

- **Decode tok/s** is the headline metric. It measures autoregressive token
  generation after prefill and is the fairest cross-runtime comparison.
- **Prefill tok/s** is shown because it affects time-to-first-token, but this
  campaign uses short prompts. Long-context prefill should be benchmarked
  separately before making public claims about long-prompt throughput.
- **VLM prefill** may include image preprocessing, vision encoder, and
  projector overhead. Compare VLM prefill numbers only when the image path and
  processor setup are the same.
- **Baseline failures are sweep-specific.** A `FAIL` in this report means the
  referenced checkout and local model directory did not complete this run. It
  should not be read as a permanent limitation of mlx-lm or mlx-vlm.

## Benchmark Method

| Item | M1 Ultra | M5 Max |
|------|----------|--------|
| Hardware | Mac Studio M1 Ultra, 128 GB RAM | MacBook Pro M5 Max, 128 GB RAM |
| OS | macOS 26.4 | macOS 26.4 |
| mlxcel | 0.0.28 | 0.0.28 |
| MLX pin | `84961223` via mlxcel-core | `84961223` via mlxcel-core |
| Text prompt | `Hello, how are you today?` | `Hello, how are you today?` |
| VLM prompt | `What is in this image?` | `What is in this image?` |
| Max generated tokens | 100 | 100 |
| Oversize policy | >65 GB skipped for Python baseline | same benchmark campaign policy |

The M5 Max benchmark campaign ran as one continuous campaign across calendar
midnight. For public reporting it is grouped under 2026-05-19. CSV filenames
still reflect the date of each sub-sweep.

## Source Data

Source-of-truth CSVs:

- `benchmarks/metal_m1ultra_2026-05-19.csv`
- `benchmarks/metal_m1ultra_vlm_2026-05-19.csv`
- `benchmarks/pylm_m1ultra_2026-05-19.csv`
- `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv`
- `benchmarks/metal_m5max_2026-05-19.csv`
- `benchmarks/metal_m5max_vlm_2026-05-19.csv`
- `benchmarks/pylm_m5max_2026-05-18.csv`
- `benchmarks/pylm_m5max_vlm_2026-05-18.csv`

Full per-hardware details:

- [M1 Ultra detailed results](model_tests_m1ultra.md)
- [M5 Max detailed results](model_tests_m5max.md)
- [Rolling benchmark index](model_tests.md)
