# Apple Silicon Benchmark Report - 2026-05-19

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
| M5 Max | Text | mlx-lm | 66 | **2.70x** | **99%** | 58/66 at >=90% parity |
| M1 Ultra | Text | mlx-lm | 73 | **1.76x** | **97%** | 62/73 at >=90% parity |
| M5 Max | VLM | mlx-vlm | 20 | 0.94x | **100%** | 16/20 at >=90% parity |
| M1 Ultra | VLM | mlx-vlm | 17 | **1.33x** | **98%** | 11/17 at >=90% parity |

Ratios above use successful runs only. Baseline rows use exact CSV model
identifiers and compare mlxcel and the Python stack on the same host. Decode
parity is the primary comparison point because it measures steady-state
generation after the KV cache is built.

The mlxcel rows use the `mlxcel-bench-decode` harness: the model is loaded
once, a warmup pass is run, and then the measured 100-token pass is recorded in
the same process. This is the benchmark method used for the current Apple
Silicon results. VLM prefill remains more prompt-shape- and processor-sensitive
than text prefill, so decode remains the primary VLM comparison.

Gemma 3 and Gemma 3n VLM rows are included in the M5 Max VLM table.
The Gemma3n E2B and E4B 4-bit checkpoints run above mlx-vlm decode parity
in this sweep, while the E4B bf16 checkpoint is successful but below parity.

## Why This Matters

mlxcel is not only running close to the Python reference implementations on
the common path; it is also running models that the Python baseline did not
complete in this sweep. With a generated-token threshold of 5 tokens:

| Host | Mode | mlxcel successful, Python baseline unavailable or failed | Comparable successful pairs |
|------|------|---------------------------------------------------------:|----------------------------:|
| M5 Max | Text | 25 | 66 |
| M1 Ultra | Text | 22 | 73 |
| M5 Max | VLM | 12 | 20 |
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
| M5 Max | smollm-135m-4bit | Small dense | 6058.41 | 905.24 | 711.54 | **127%** |
| M5 Max | qwen2.5-7b-4bit | Dense 7B | 917.38 | 126.36 | 123.59 | **102%** |
| M5 Max | gpt-oss-120b-4bit | Large MoE | 334.68 | 114.03 | 110.35 | **103%** |
| M5 Max | solar-open-100b-4bit | Large MoE | 210.91 | 65.36 | 66.30 | 99% |
| M5 Max | qwen3.5-35b-a3b-4bit | Hybrid MoE | 480.89 | 151.63 | 152.96 | 99% |
| M5 Max | nemotron-h-30b-4bit | Hybrid SSM/MoE | 414.31 | 177.18 | 178.80 | 99% |
| M1 Ultra | phi-3.5-moe-4bit | MoE | 107.87 | 76.24 | 69.28 | **110%** |
| M1 Ultra | minicpm3-4b-4bit | MLA | 230.24 | 80.24 | 73.26 | **110%** |
| M1 Ultra | qwen2.5-0.5b-4bit | Small dense | 1094.10 | 355.29 | 315.48 | **113%** |
| M1 Ultra | gpt-oss-120b-4bit | Large MoE | 161.69 | 58.89 | 57.58 | **102%** |
| M1 Ultra | command-r7b-4bit | Dense 7B | 92.20 | 110.22 | 107.75 | **102%** |
| M1 Ultra | solar-open-100b-4bit | Large MoE | 73.74 | 35.88 | 35.69 | **100%** |

The table shows the central performance claim without cross-hardware ranking:
mlxcel is already close to mlx-lm on the same machine for dense 7B models,
newer hybrid MoE families, and very large MoE models such as GPT-OSS 120B.

## Representative VLM Models

All values are tokens per second. VLM prefill includes vision-side work when
the run uses image input, so decode is the cleaner apples-to-apples baseline
comparison.

| Host | Model | Class | mlxcel prefill | mlxcel decode | mlx-vlm decode | vs mlx-vlm |
|------|-------|-------|---------------:|--------------:|---------------:|-----------:|
| M5 Max | qwen3.5-0.8b-4bit | Hybrid GatedDeltaNet VLM | 1294.94 | 505.94 | 410.96 | **123%** |
| M5 Max | qwen3.5-35b-a3b-4bit | Hybrid MoE VLM | 355.32 | 151.34 | 128.80 | **117%** |
| M5 Max | gemma-4-e2b-it-4bit | Gemma 4 VLM | 2787.47 | 217.32 | 201.70 | **108%** |
| M5 Max | gemma3n-e2b-4bit | Gemma 3n VLM | 2893.48 | 151.36 | 124.63 | **121%** |
| M5 Max | molmo2-4b | Molmo2 vision encoder | 2512.31 | 64.01 | 66.80 | 96% |
| M1 Ultra | llava-interleave-qwen-0.5b-bf16 | SigLIP + Qwen2 | 8589.48 | 269.86 | 225.15 | **120%** |
| M1 Ultra | aya-vision-8b | SigLIP + Cohere2 | 591.80 | 109.36 | 103.74 | **105%** |
| M1 Ultra | molmo2-4b | Molmo2 vision encoder | 1011.99 | 59.36 | 60.87 | 98% |
| M1 Ultra | phi-3.5-vision-4bit | CLIP + HD tiling | 1164.87 | 94.10 | 92.53 | **102%** |
| M1 Ultra | pixtral-12b-4bit | Pixtral ViT + Mistral | 473.22 | 59.17 | - | - |

The VLM data is most compelling as a coverage and decode-throughput story:
mlxcel runs many VLM paths in the same benchmark matrix, and its comparable
decode results sit near mlx-vlm median parity on both Apple Silicon hosts.

## Where mlxcel Looks Strongest

- **Steady-state decode:** M5 Max text decode is within 1% of mlx-lm median
  parity and VLM decode reaches mlx-vlm median parity. M1 Ultra shows
  the same pattern, with 97% text median parity and 98% VLM median parity.
- **Short-prompt text prefill:** mlxcel prefill is 2.70x mlx-lm median on M5
  Max text and 1.76x mlx-lm median on M1 Ultra text. Representative M5 Max
  values include gpt-oss-120b at 334.68 tok/s and nemotron-h-30b at
  414.31 tok/s.
- **Large MoE practicality:** GPT-OSS 120B reaches 114.03 tok/s on M5 Max and
  58.89 tok/s on M1 Ultra, both at or slightly above mlx-lm parity on the same
  host. Solar-Open 100B reaches 65.36 tok/s (M5 Max) / 35.88 tok/s (M1 Ultra),
  also at mlx-lm parity.
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
| Bench harness | `mlxcel-bench-decode`: one process loads the model, runs a 20-token warmup, then a 100-token measured pass | `mlxcel-bench-decode`: one process loads the model, runs warmup, then a 100-token measured pass |
| Text prompt | `Hello, how are you today?` | `Hello, how are you today?` |
| VLM prompt | `What is in this image?` | `What is in this image?` |
| Max generated tokens | 100 (measured), 20 (warmup) | 100 (measured) |
| Oversize policy | >65 GB skipped on both mlxcel and Python baseline | same benchmark campaign policy |

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
- `benchmarks/metal_m5max_vlm_2026-05-20.csv`
- `benchmarks/pylm_m5max_2026-05-18.csv`
- `benchmarks/pylm_m5max_vlm_2026-05-18.csv`

Full per-hardware details:

- [M1 Ultra detailed results](model_tests_m1ultra.md)
- [M5 Max detailed results](model_tests_m5max.md)
- [Rolling benchmark index](model_tests.md)
