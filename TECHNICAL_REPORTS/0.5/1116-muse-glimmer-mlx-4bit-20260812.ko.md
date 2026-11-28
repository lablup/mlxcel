# 기술 보고서: PR #1116 - feat(models): support Muse Glimmer MLX 4-bit

**작성일**: 2026-08-12
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

---

## 요약

PR #1116은 기존 dense BF16 Muse 경로를 유지하면서 pinned `mlx-community/Muse-Glimmer-30B-4bit` affine-Q4 변환을 체크포인트 형식 단위로 지원한다. mlx-vlm weight namespace를 정규화하고, root quantization contract를 text stack에 전달하며, text와 vision-fusion layer를 quantization-aware 형태로 로드하되 50-layer vision tower는 dense로 유지한다.

PR 제출 전 보안 검토에서 두 가지 fail-open metadata 경계를 찾아 교정했다. 문자열이 아닌 `quantization.mode`가 mode 비교를 우회할 수 있었고, affine `.biases` sidecar가 빠지면 공용 loader가 block-float 형식으로 추론할 수 있었다. 이제 VLM과 text-only loader 모두 kernel 선택 전에 동일한 pinned affine contract를 강제한다.

---

## 1. 문제 정의

### 1.1 배경

PR #1101은 59.55 GB BF16 체크포인트로 Muse Glimmer baseline을 구축했지만 quantization sidecar는 의도적으로 거부했다. 공개 mlx-community 변환은 약 19.41 GB이며 다른 root namespace와 root-level affine-Q4 metadata를 사용하므로 기존 BF16 전용 경계로 안전하게 해석할 수 없었다.

### 1.2 기존 한계

- mlx-vlm의 `language_model.*` 및 wrapper 없는 vision root가 mlxcel의 canonical Muse namespace와 일치하지 않았다.
- decoder는 `text_config`를 소비하지만 공개 quantization contract는 JSON root에 있었다.
- dense 전용 fusion projection이 양자화된 adapter와 projector를 로드할 수 없었다.
- server route는 Muse recipient channel을 분리했지만 one-shot CLI는 envelope와 내부 `to=self` reasoning을 그대로 표시했다.
- 임의의 quantization metadata를 허용하면 호환되지 않는 native kernel 선택 또는 조용한 오출력 위험이 있었다.

### 1.3 위험 평가

| 위험 | 영향도 | 완화 |
|------|--------|------|
| Alias key가 canonical tensor를 덮어씀 | High | canonical destination이 중복되면 정규화를 거부 |
| 잘못된 metadata가 의도하지 않은 quantization mode를 선택 | High | affine 고정, type/parameter 검증, 필수 sidecar 강제 |
| 양자화된 vision tower가 미검증 경로에 진입 | High | 모델 구성 전에 vision-tower sidecar 거부 |
| 내부 reasoning이 CLI 출력에 노출 | Medium | server recipient parser를 재사용하고 기본적으로 `to=self` 숨김 |

---

## 2. 기술 검토

### 2.1 보안과 정확성

- Root `quantization`, 호환용 `quantization_config`, nested `text_config.quantization`이 서로 일치해야 한다.
- `mode`가 있으면 문자열 `affine`이어야 하며, group size와 bit width는 공용 quantization validator를 통과해야 한다.
- 정규화된 weight alias는 서로 또는 canonical key와 충돌할 수 없다.
- 지원되는 모든 `.scales` tensor에는 `.weight`와 `.biases`가 필요하며, orphan biases, global scales, vision-tower sidecar는 거부한다.
- Text-only loader가 VLM loader와 동일한 weight-map validator를 호출해 경계별 우회를 차단한다.
- 새 dependency, network request, 인증/인가 경로, filesystem write, subprocess, unsafe block은 추가되지 않았다.
- CLI channel rendering은 기존 ATEM/Muse parser에 위임한다. 이미 생성된 출력에 대한 선형 변환이며 server-visible data 범위를 넓히지 않는다.

두 보안 교정 이후 남은 Critical/High 이슈는 없다. 잔여 위험은 아래에 명시한 pinned checkpoint-format contract와 hardware qualification 범위로 제한된다.

### 2.2 성능

| 시나리오 | 결과 |
|----------|------|
| BF16 text decode baseline | 4.25 tok/s |
| Q4 warm text prefill | 12.43 tok/s |
| Q4 warm text decode | 13.34 tok/s |
| Q4 image prefill | 5.80 tok/s, first-use compile 포함 |
| Q4 image decode | 13.15 tok/s |
| 보안 교정 후 exact-source text 실행 | 11.41 prefill / 13.07 decode tok/s |

동일 NVIDIA GB10에서 Q4 decode는 기록된 BF16보다 약 3.1배 빠르다. 실제 이미지 실행은 64개 vision patch token을 삽입했고 단색 orange-red fixture를 정확히 묘사했다.

### 2.3 호환성

- 검증 체크포인트: `mlx-community/Muse-Glimmer-30B-4bit` revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`.
- 검증 하드웨어: Linux/aarch64 NVIDIA GB10, CUDA.
- 보존 경로: PR #1101의 canonical dense BF16 Muse checkpoint.
- 명시적 미지원: video, quantized vision tower, Turbo/INT8 KV, speculative/DFlash, LoRA/adapters, TP/PP, XLA/IREE/OpenXLA, distributed, disaggregated serving.
- 새 package dependency나 wire-format 변경 없음.

---

## 3. 기술적 선택과 이유

### 3.1 Tensor Data를 복사하지 않는 이름 정규화

**선택:** 공개 mlx-vlm root를 canonical Muse root로 변환하면서 weight handle을 collision-checked map으로 이동한다.

**이유:** BF16과 Q4 model construction이 하나의 runtime namespace를 사용하면서 tensor data 복제를 피한다. 충돌 거부는 crafted alias가 canonical weight를 조용히 교체하는 것을 막는다.

### 3.2 Vision Tower Dense 유지

**선택:** Text stack, LM head, vision adapter, vision projection만 quantization-aware layer를 활성화한다.

**이유:** Pinned conversion의 vision tower는 dense다. 그 증거를 넘어 일반화하면 미검증 shape와 kernel이 노출되므로 vision-tower sidecar는 fail-closed 처리한다.

### 3.3 Server와 CLI의 Recipient Parsing 공유

**선택:** 별도 parser를 구현하지 않고 server의 Muse channel renderer를 one-shot CLI에 export한다.

**이유:** 하나의 parser가 동일한 `to=self`/`to=user` 의미를 보존하고, 서로 다른 CLI 구현에서 reasoning 또는 구조 token이 새는 위험을 줄인다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 16 |
| 추가 라인 | 477 |
| 삭제 라인 | 69 |
| 커밋 | 2 |

### 주요 영역

| 영역 | 요약 |
|------|------|
| Loading/config | Root quantization 상속, namespace 정규화, sidecar와 mode 검증 |
| Model/vision | Dense tower를 유지하는 quantization-aware text/fusion layer |
| CLI/tool channel | 공유 Muse recipient rendering과 reasoning suppression |
| Tests | Alias, collision, config 불일치, malformed mode, sidecar, CLI, real-checkpoint coverage |
| Documentation | Pinned revision, 크기, 지원 경계, GB10 throughput |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `4ea5aca2` | feat | support Muse Glimmer MLX 4-bit |
| `9a23ed2e` | fix | enforce Muse affine Q4 contract |

PR #1101의 후속 작업이다.

---

## 5. 검증과 후속 조치

### 통과

- `cargo fmt --all --check`
- Library와 binary 대상 Clippy `-D warnings`
- Muse-filtered library test 100개
- CLI help/metadata regression 2개
- `MLX_CUDA_ARCHITECTURES=121` CUDA release build
- NVIDIA GB10에서 pinned checkpoint text/image 실제 생성
- 최종 exact-source text 실행은 0.480초에 로드됐고 recipient/control-token 노출 없이 `The capital of France is Paris.`만 출력

### 남은 경계

- 머지 전 hosted PR check가 모두 통과해야 한다.
- 이 보고서는 Apple Silicon/Metal을 qualification하지 않는다.
- 추가 quantization format과 미지원 실행 모드는 각각 별도의 evidence-gated follow-up으로 유지한다.
