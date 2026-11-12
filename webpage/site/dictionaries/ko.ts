import { Dictionary } from "./en";

export const ko: Dictionary = {
  announcement: {
    tag: "NEW",
    text: "공지 플레이스홀더",
    date: "TBD",
    href: "#",
    visible: false,
  },
  hero: {
    badge: "mlxcel",
    title_prefix: "Apple Silicon 위에서,",
    title_suffix: "네이티브 LLM 추론,",
    title_accent: "극한의 속도로",
    description:
      "100개 이상의 LLM/VLM 모델을 Metal 가속으로 Mac에서 바로 실행하세요.\nPython 없이, 컨테이너 없이, 순수 Rust 속도 그대로.",
    description_secondary:
      "Llama부터 DeepSeek, Qwen, Gemma까지.\nOpenAI 호환 API 서버가 내장되어 있습니다.",
    supporting_note: "Python 의존성 제로",
    download_btn: "다운로드: ",
    download_latest: "최신 버전 다운로드",
    release_notes: "릴리스 노트",
    other_platforms: "다른 빌드",
    checking: "버전 확인 중...",
    view_releases: "전체 릴리스 보기",
    trust_line: "오픈소스 (Apache 2.0) · macOS 네이티브 · 내 하드웨어, 내 모델",
  },
  highlights: {
    eyebrow: "왜 mlxcel인가",
    title: "Mac에서 LLM을 실행하는 가장 빠른 방법.",
    items: [
      {
        meta: "성능",
        title: "Rust 기반, Metal 가속",
        description:
          "cxx FFI를 통한 직접적인 MLX C++ 바인딩. Python 오버헤드도, 인터프리터 시작 지연도 없습니다. M1부터 M5까지, 네이티브 Metal 컴퓨팅.",
      },
      {
        meta: "모델",
        title: "100개 이상의 모델 아키텍처",
        description:
          "Transformer, MoE, SSM, Hybrid. Llama, Qwen, Gemma, DeepSeek, Mixtral, Mamba, RWKV, Jamba 등. 텍스트와 비전 모델 모두 지원.",
      },
      {
        meta: "호환성",
        title: "OpenAI 호환 API 서버",
        description:
          "llama-server 대체 가능. 스트리밍 완성, 채팅 API, llama.cpp CLI 플래그 호환. 기존 도구를 바로 연결하세요.",
      },
    ],
  },
  story: {
    eyebrow: "Apple Silicon 전용, Rust로 작성",
    title: "내 하드웨어에서 추론할 때, 모든 토큰이 중요합니다.",
    subtitle:
      "mlxcel은 Python 환경이나 Docker 컨테이너 없이 Mac에서 최대 처리량을 원하는 개발자와 연구자를 위한 도구입니다.",
    panels: [
      {
        eyebrow: "네이티브 성능",
        title: "Metal 직접 연산,\n오버헤드 제로.",
        description:
          "mlxcel은 cxx FFI 바인딩을 통해 Apple의 MLX 프레임워크에 직접 연결됩니다. Python 인터프리터도, GIL도, 직렬화 오버헤드도 없습니다. 순수 Rust 오케스트레이션과 Metal GPU 컴퓨팅.",
        points: [
          "M1부터 M5까지 하드웨어 인식 경로, M5의 Neural Accelerator 감지 포함.",
          "비양자화 모델의 bf16→f16 자동 변환으로 모든 Apple Silicon 세대 호환.",
          "HuggingFace MLX Community의 4bit/8bit 양자화 모델을 변환 없이 바로 사용.",
        ],
        stat_label: "모델 포맷",
        stat_value: "SafeTensors 네이티브",
      },
      {
        eyebrow: "개발자 경험",
        title: "바이너리 하나로,\n모든 것을\n포함합니다.",
        description:
          "mlxcel은 CLI(생성)와 서버(API) 두 개의 바이너리로 제공됩니다. 가상 환경도, pip install도, 의존성 충돌도 없습니다. cargo로 한 번 빌드하면 macOS 어디서든 실행.",
        points: [
          "temperature, top-p, top-k, min-p, XTC, 반복 페널티, DRY 등 완전한 샘플링 지원.",
          "LoRA 어댑터와 추론 가속을 위한 추측 디코딩 지원.",
          "llama-server 플래그 호환 OpenAI 스트리밍 API.",
        ],
        stat_label: "의존성",
        stat_value: "Python 제로",
      },
    ],
  },
  showcase: {
    title: "단일 프롬프트에서 프로덕션 API 서버까지.",
    subtitle:
      "텍스트 생성, 모델 서빙, 성능 벤치마크, 기존 도구와의 통합까지.",
    tabs: [
      {
        label: "생성",
        alt: "mlxcel 텍스트 생성 CLI",
        title: "터미널에서 바로 텍스트 생성",
        description:
          "지원되는 모든 모델을 단일 명령으로 실행합니다. 샘플링, 반복 페널티, 출력 형식을 CLI에서 바로 제어.",
      },
      {
        label: "서버",
        alt: "mlxcel OpenAI 호환 API 서버",
        title: "몇 초 만에 OpenAI 호환 API 서버",
        description:
          "모든 OpenAI SDK 클라이언트와 호환되는 API 서버를 시작합니다. 스트리밍, 채팅 완성, 모델 전환 기본 내장.",
      },
      {
        label: "비전",
        alt: "mlxcel 비전 모델 추론",
        title: "비전-언어 모델도 같은 워크플로",
        description:
          "Llava, Qwen-VL, Pixtral, Paligemma 등의 VLM을 같은 CLI로 실행합니다. 이미지와 프롬프트를 함께 전달하여 멀티모달 추론.",
      },
      {
        label: "벤치마크",
        alt: "mlxcel 모델 벤치마킹",
        title: "모든 모델을 한 번에 벤치마크",
        description:
          "다운로드한 모든 모델에 대해 자동 벤치마크를 실행합니다. 초당 토큰 수 추적, 아키텍처 비교, 최적 포인트 탐색.",
      },
      {
        label: "모델",
        alt: "mlxcel 지원 모델 목록",
        title: "100개 이상의 아키텍처, 계속 확장 중",
        description:
          "Transformer, MoE, SSM, Hybrid 아키텍처. 0.5B부터 200B+ 파라미터까지. 지속적으로 모델 지원 확대.",
      },
    ],
  },
  mesh: {
    eyebrow: "단일 모델을 넘어",
    title: "Apple Silicon을 위한\n완전한 추론 툴킷.",
    subtitle:
      "mlxcel은 모델 다운로드부터 프로덕션 서빙까지 전체 워크플로를 다루며, 각 단계에 맞는 도구를 제공합니다.",
    mesh_points: [
      {
        step: "Step 1",
        title: "모델 다운로드",
        description:
          "HuggingFace MLX Community에서 양자화 모델을 바로 가져옵니다. SafeTensors 포맷, 변환 필요 없음.",
      },
      {
        step: "Step 2",
        title: "생성 또는 서빙",
        description:
          "CLI로 대화형 생성을 하거나, 애플리케이션을 위한 OpenAI 호환 API 서버를 시작하세요.",
      },
      {
        step: "Step 3",
        title: "통합하고\n확장하기",
        description:
          "API 서버를 통해 기존 도구, IDE, 워크플로에 연결합니다. 100개 이상의 모든 모델 아키텍처에 동일한 인터페이스.",
      },
    ],
    mesh_badges: [
      "HuggingFace 네이티브",
      "OpenAI 호환 API",
      "제로 설정",
    ],
    mesh_card: {
      eyebrow: "아키텍처 지원",
      title: "Transformer, MoE, SSM,\nHybrid - 하나의 바이너리로",
      description:
        "밀집 트랜스포머부터 MoE, Mamba부터 Jamba와 Nemotron-H 같은 하이브리드 아키텍처까지. 하나의 도구가 모두 처리합니다.",
    },
    integration_card: {
      eyebrow: "Backend.AI 연동",
      title: "단일 Mac을 넘어\nBackend.AI로 확장",
      description:
        "로컬 추론만으로 부족할 때, Backend.AI에 연결하여 멀티 GPU 클러스터, 팀 공유, 엔터프라이즈급 인프라를 활용하세요.",
    },
  },
  downloads: {
    title: "mlxcel을 받고 추론을 시작하세요",
    subtitle: "macOS (Apple Silicon) 최신 릴리스.",
    view_full: "GitHub에서 전체 릴리스 보기",
  },
  brew: {
    badge: "Homebrew도 지원합니다",
    title: "명령 한 줄이면 설치됩니다",
    subtitle: "macOS에서 가장 빠르게 시작하는 방법",
    note: "Homebrew가 필요합니다. 안정 버전이 나올 때마다 formula도 함께 업데이트됩니다.",
  },
  enterprise: {
    badge: "Backend.AI + mlxcel",
    title: "로컬 추론.\n엔터프라이즈 규모.",
    description:
      "mlxcel은 Backend.AI 엔터프라이즈 AI 플랫폼 내에서 로컬 Apple Silicon 추론을 담당합니다. Mac 워크스테이션에서 모델을 실행하면서 더 큰 워크로드를 위한 중앙 GPU 클러스터에 연결할 수 있습니다.",
    points: [
      {
        title: "로컬 + 클러스터 하이브리드",
        detail: "Mac 워크스테이션과 GPU 서버를 하나로",
      },
      {
        title: "모델 관리",
        detail: "중앙화된 모델 레지스트리와 배포",
      },
      {
        title: "팀 협업",
        detail: "팀 전체 공유 추론 엔드포인트",
      },
      {
        title: "엔터프라이즈 배포",
        detail: "온프레미스, 폐쇄망, 완전 관리형",
      },
    ],
    points_label: "엔터프라이즈 기능",
    note:
      "mlxcel은 로컬 추론 엔진입니다. Backend.AI가 팀과 조직을 위한 오케스트레이션, 확장, 관리 레이어를 제공합니다.",
    cta: "문의하기",
  },
  footer: {
    rights: "Lablup Inc. All rights reserved.",
    docs: "문서",
  },
};
