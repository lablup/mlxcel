export const en = {
  announcement: {
    tag: "NEW",
    text: "Announcement placeholder",
    date: "TBD",
    href: "#",
    visible: false,
  },
  hero: {
    badge: "mlxcel",
    title_prefix: "LLM Inference,",
    title_suffix: "Native on Apple Silicon,",
    title_accent: "Blazing Fast",
    description:
      "Run 100+ LLM and VLM models natively on your Mac with Metal acceleration. No Python, no containers, just pure Rust speed.",
    description_secondary:
      "From Llama to DeepSeek, Qwen to Gemma, with an OpenAI-compatible API server built in.",
    supporting_note: "Zero Python dependencies",
    download_btn: "Get it for",
    download_latest: "Get the latest release",
    release_notes: "Release notes",
    other_platforms: "See other builds",
    checking: "Checking the latest release...",
    view_releases: "Browse all releases",
    trust_line: "Open source (Apache 2.0) · macOS native · Your hardware, your models",
  },
  highlights: {
    eyebrow: "Why mlxcel",
    title: "The fastest way to run LLMs on your Mac.",
    items: [
      {
        meta: "Performance",
        title: "Rust-powered, Metal-accelerated",
        description:
          "Direct MLX C++ bindings via cxx FFI. No Python overhead, no interpreter startup. Just native Metal compute on Apple Silicon from M1 to M5.",
      },
      {
        meta: "Models",
        title: "100+ model architectures",
        description:
          "Transformer, MoE, SSM, Hybrid. Llama, Qwen, Gemma, DeepSeek, Mixtral, Mamba, RWKV, Jamba and many more. Text and vision models supported.",
      },
      {
        meta: "Compatibility",
        title: "OpenAI-compatible API server",
        description:
          "Drop-in replacement for llama-server. Streaming completions, chat API, and llama.cpp CLI flag compatibility. Connect your existing tools instantly.",
      },
    ],
  },
  story: {
    eyebrow: "Built for Apple Silicon, Written in Rust",
    title: "Every token counts when inference runs on your own hardware.",
    subtitle:
      "mlxcel is for developers and researchers who want maximum throughput from their Mac without wrestling with Python environments or Docker containers.",
    panels: [
      {
        eyebrow: "Native Performance",
        title: "Direct Metal compute, zero overhead.",
        description:
          "mlxcel talks directly to Apple's MLX framework through cxx FFI bindings. No Python interpreter, no GIL, no serialization overhead. Pure Rust orchestration with Metal GPU compute.",
        points: [
          "Hardware-aware paths for M1 through M5, including Neural Accelerator detection on M5.",
          "Automatic bf16-to-f16 conversion for non-quantized models ensures compatibility across all Apple Silicon generations.",
          "4-bit and 8-bit quantized models from HuggingFace MLX Community work directly, no conversion needed.",
        ],
        stat_label: "Model format",
        stat_value: "SafeTensors native",
      },
      {
        eyebrow: "Developer Experience",
        title: "One binary, everything included.",
        description:
          "mlxcel ships as two binaries: a CLI for generation and a server for API access. No virtual environments, no pip install, no dependency conflicts. Build once with cargo, run anywhere on macOS.",
        points: [
          "Full sampling suite: temperature, top-p, top-k, min-p, XTC, repetition penalty, and DRY.",
          "LoRA adapter support and speculative decoding for faster generation.",
          "OpenAI-compatible streaming API with llama-server flag compatibility.",
        ],
        stat_label: "Dependencies",
        stat_value: "Zero Python",
      },
    ],
  },
  showcase: {
    title: "From single prompt to production API server.",
    subtitle:
      "Generate text, serve models, benchmark performance, and integrate with your existing tools.",
    tabs: [
      {
        label: "Generate",
        alt: "mlxcel text generation CLI",
        title: "Interactive text generation from the terminal",
        description:
          "Run any supported model with a single command. Control sampling, repetition penalty, and output format right from the CLI.",
      },
      {
        label: "Server",
        alt: "mlxcel OpenAI-compatible API server",
        title: "OpenAI-compatible API in seconds",
        description:
          "Start an API server that works with any OpenAI SDK client. Streaming, chat completions, and model switching out of the box.",
      },
      {
        label: "Vision",
        alt: "mlxcel vision model inference",
        title: "Vision-language models, same workflow",
        description:
          "Run VLMs like Llava, Qwen-VL, Pixtral, and Paligemma with the same CLI. Pass images alongside prompts for multimodal inference.",
      },
      {
        label: "Benchmark",
        alt: "mlxcel model benchmarking",
        title: "Benchmark every model in your collection",
        description:
          "Run automated benchmarks across all your downloaded models. Track tokens per second, compare architectures, and find the sweet spot.",
      },
      {
        label: "Models",
        alt: "mlxcel supported model list",
        title: "100+ architectures and growing",
        description:
          "Transformer, MoE, SSM, Hybrid architectures. From 0.5B to 200B+ parameters. Continuously expanding model support.",
      },
    ],
  },
  mesh: {
    eyebrow: "Beyond a Single Model",
    title: "A complete inference toolkit for Apple Silicon.",
    subtitle:
      "mlxcel covers the full workflow from model download to production serving, with tools for every step along the way.",
    mesh_points: [
      {
        step: "Step 1",
        title: "Download models",
        description:
          "Grab quantized models directly from HuggingFace MLX Community. SafeTensors format, no conversion required.",
      },
      {
        step: "Step 2",
        title: "Generate or serve",
        description:
          "Use the CLI for interactive generation or spin up an OpenAI-compatible API server for your applications.",
      },
      {
        step: "Step 3",
        title: "Integrate and\nscale up",
        description:
          "Connect via the API server to your existing tools, IDEs, and workflows. Same interface for all 100+ model architectures.",
      },
    ],
    mesh_badges: [
      "HuggingFace native",
      "OpenAI-compatible API",
      "Zero configuration",
    ],
    mesh_card: {
      eyebrow: "Architecture Support",
      title: "Transformer, MoE, SSM,\nHybrid - all in one binary",
      description:
        "From dense transformers to mixture-of-experts, from Mamba to hybrid architectures like Jamba and Nemotron-H. One tool handles them all.",
    },
    integration_card: {
      eyebrow: "Backend.AI Integration",
      title: "Scale beyond a single Mac\nwith Backend.AI",
      description:
        "When local inference isn't enough, connect to Backend.AI for multi-GPU clusters, team sharing, and enterprise-grade infrastructure.",
    },
  },
  downloads: {
    title: "Get mlxcel and start inferencing",
    subtitle: "Latest release for macOS (Apple Silicon).",
    view_full: "View full release notes on GitHub",
  },
  brew: {
    badge: "Prefer Homebrew?",
    title: "Install it in a single line",
    subtitle: "The fastest way to get started on macOS",
    note: "Requires Homebrew. The formula tracks each stable release.",
  },
  enterprise: {
    badge: "Backend.AI + mlxcel",
    title: "Local inference.\nEnterprise scale.",
    description:
      "mlxcel powers local Apple Silicon inference within Backend.AI's enterprise AI platform. Run models on Mac workstations while connecting to centralized GPU clusters for larger workloads.",
    points: [
      {
        title: "Local + cluster hybrid",
        detail: "Mac workstations and GPU servers, unified",
      },
      {
        title: "Model management",
        detail: "Centralized model registry and distribution",
      },
      {
        title: "Team collaboration",
        detail: "Shared inference endpoints across teams",
      },
      {
        title: "Enterprise deployment",
        detail: "On-premise, air-gapped, fully managed",
      },
    ],
    points_label: "Enterprise capabilities",
    note:
      "mlxcel is the local inference engine. Backend.AI provides the orchestration, scaling, and management layer for teams and organizations.",
    cta: "Talk to us",
  },
  footer: {
    rights: "Lablup Inc. All rights reserved.",
    docs: "Docs",
  },
};

export type Dictionary = typeof en;
