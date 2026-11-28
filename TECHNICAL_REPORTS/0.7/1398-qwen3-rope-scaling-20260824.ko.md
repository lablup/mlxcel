# 기술 보고서: PR #1398 - Qwen3 RoPE Scaling 전체 경로 적용

**작성일**: 2026-08-24

**작성자**: 신정규

**상태**: 완료

**언어**: Rust, C++, CUDA, Markdown

**위험도**: Medium

## 요약

PR #1398은 dense Qwen3 및 Qwen3-MoE가 파싱한 뒤 버리던 `rope_scaling` block을 실제 계산에 적용한다. 모델마다 plan을 한 번만 resolve하고 normal, batched, pipeline-parallel, tensor-parallel 경로로 전달하며 graph 및 fused Q/K-normalization RoPE launcher에 일관되게 적용한다. Linear scaling은 fused launcher를 계속 사용하고 frequency table이 필요한 scheme은 graph path를 사용한다. 지원하지 않거나 잘못된 scheme은 조용히 unscaled decode하지 않고 checkpoint 기준 one-shot warning을 출력한다.

실 모델 검증은 이 장비에 연결된 NVIDIA GB10과 기존 로컬 4-bit 4B 및 30B-A3B 웨이트로 수행했다. GitHub Actions에서는 큰 checkpoint를 다운로드하지 않았으며, 중단한 일회성 workflow와 run은 취소하고 최종 변경에서 제거했다.

## 1. 문제 정의

`qwen3::ModelArgs`와 `qwen3_moe::ModelArgs`는 모두 `rope_scaling`을 deserialize했지만 모든 attention layer가 여전히 RoPE에 `1.0`을 전달했다. 따라서 `linear`, `llama3` 또는 다른 scaling scheme을 선언한 checkpoint도 정상 load된 뒤 조용히 unscaled position table을 사용했다. 저장소의 현재 unscaled checkpoint에는 잠재 결함이었지만 미래 또는 third-party scaled checkpoint의 long-context 품질을 저하시킬 수 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 공통 resolve | `RopeScalingSpec`과 `RopeScalingKind`를 재사용하고 checkpoint별 diagnostic label로 한 번만 resolve |
| Dense Qwen3 | `rope_scale`과 optional frequency를 regular, batched, fused, graph attention 경로에 전달 |
| Qwen3-MoE | Fused decode 및 graph prefill/decode 경로에 같은 plan 적용 |
| Fused primitive | Position scale이 `fast_rope`와 같은 의미를 가짐을 문서화하고 검증 |
| 분산 경로 | Pipeline-stage construction과 tensor-parallel argument localization에서 resolve된 설정 보존 |
| Diagnostic | 잘못되거나 미지원인 scaling을 조용히 `1.0`으로 처리하지 않고 checkpoint/scheme당 한 번 warning |
| Regression suite | Key precedence, invalid factor, 정확한 warning 횟수, hand-computed rotation, fused/graph parity, batched attention, real loader, pipeline parallelism, tensor parallelism 검증 |

## 3. 기술적 선택과 그 이유

### 3.1 한 번 resolve한 attention plan 전달

각 모델은 layer를 만들기 전에 config를 `Default`, `Linear { scale }`, `Llama3 { freqs }` 중 하나로 resolve한다. Linear factor 8은 MLX position multiplier `0.125`가 된다. Attention block은 동일 plan의 owned handle을 받아 forward 중 config lookup이나 frequency arithmetic 반복을 피한다.

### 3.2 Linear scaling은 fused path 유지

Qwen3 fused QKV/QK-normalization launcher는 이미 scalar RoPE position multiplier를 받는다. 따라서 linear scaling은 `1 / factor`를 직접 전달하면서 optimized decode path를 유지한다. Precomputed frequency table은 이 launcher로 표현할 수 없으므로 `llama3`는 의도적으로 `fast_rope_with_freqs` graph path로 routing한다.

### 3.3 임의 VLM config에는 hard failure 대신 warning

MiniCPM-o는 임의의 전체 config를 `qwen3::ModelArgs`로 deserialize할 수 있다. 지원하지 않는 scaling type을 hard load error로 바꾸면 기존에 동작하는 VLM을 offline시킬 수 있다. 구현은 기존 unscaled fallback을 유지하되 checkpoint 이름과 scheme으로 key한 warning을 정확히 한 번 출력한다. Factor가 없거나 non-finite, zero, negative인 linear 설정도 같은 명시적 결과를 따른다.

### 3.4 Hosted CI는 가볍게 유지하고 큰 모델은 로컬 검증

저장소의 기존 경계를 유지한다. Actions는 약 0.6B 수준의 작은 model fixture를 사용할 수 있고, 그보다 큰 checkpoint가 필요한 테스트는 qualification 장비에 이미 있는 웨이트로 로컬에서 실행한다. 임시 4B/30B hosted workflow는 취소하고 삭제했다. 최종 PR에는 해당 checkpoint를 다운로드하는 workflow가 추가되지 않는다.

## 4. 검증

### 4.1 결정적 코드 gate

- `cargo fmt --all -- --check`: 통과.
- 최신 `main` 통합 후 Qwen3 focused suite: 110 passed, 0 failed, 20 ignored.
- `cargo clippy -p mlxcel --all-targets -- -D warnings`: 통과.
- Checkpoint-keyed warning 수정 후 correctness, security/performance, finalization review에서 미해결 blocker 없음.
- 최종 gate 전에 PR #1395를 통합했다. 해당 Turbo4/MLA 변경은 Qwen3 RoPE와 겹치지 않으며 focused suite는 계속 통과했다.

### 4.2 로컬 NVIDIA 실 모델 검증

하드웨어 및 runtime:

- NVIDIA GB10, driver 580.173.02, CUDA 13.0, compute capability 12.1.
- CUDA release build: `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda --bin mlxcel`.
- 기존 로컬 checkpoint만 사용: `models/mlx/qwen3-4b-4bit`, `models/mlx/qwen3-30b-a3b-4bit`.

Temperature 0, seed 0, 128 generated tokens 조건의 unscaled regression:

- Qwen3 4B: PR과 최신 `main`이 동일한 128-token text 생성.
- Qwen3 30B-A3B: PR과 최신 `main`이 동일한 128-token text 생성.

Scaled dense 검증:

- 로컬 4B checkpoint의 hard-linked 임시 view에서 `config.json`만 분리해 `{"rope_type":"linear","factor":8.0}`을 주입했다. 원본 checkpoint는 변경하지 않았다.
- 비퇴화 prompt는 tokenizer 기준 3,009 tokens로 요구된 2,048-token 경계를 넘었다.
- Upstream의 graph RoPE evaluation과 같은 조건을 위해 graph Q/K-normalization path를 선택했을 때 mlxcel과 mlx-lm 0.31.3은 8-token 비교에서 모두 `The quick. The quick brown fox.`를 생성했다.
- 주입된 block을 무시하는 최신 `main`은 scaled 결과와 달랐다. PR의 기본 fused decode는 같은 prefix 뒤 이슈에 명시된 f16 reduction-order jitter class로 진입했으며, 수치 기반 fused-versus-graph rotation test는 저장소 tolerance에서 통과한다.
- Upstream CUDA oracle은 mlx 0.32.1의 CUDA 12.9 NVRTC와 이에 맞는 공식 CUDA 12.9 runtime header를 사용했다. 임시 oracle 환경의 CUDA-12-NVRTC/CUDA-13-system-header 불일치를 바로잡은 것으로 model weight를 다운로드하거나 수정하지 않았다.

### 4.3 Hosted 검증 경계

시도했던 일회성 large-checkpoint Actions run은 명시적으로 취소했으며 검증 근거로 세지 않는다. 표준 repository CI는 formatting, lint, compilation, metadata, small-fixture coverage를 담당하고 위 GB10 실행이 실제 4B/30B qualification 기록이다.

## 5. 통합 참고사항

- `rope_scaling` map은 의도적으로 map shape을 유지하므로 `type`과 `rope_type`을 모두 가진 config도 정상 parse하며 upstream과 같이 `type`이 우선한다.
- Dense Qwen3와 Qwen3-MoE는 같은 공통 resolve code를 사용하지만 upstream Qwen3-MoE는 아직 unscaled RoPE module을 만들기 때문에 mlx-lm token diff는 dense Qwen3에서만 유효하다.
- Batched, pipeline-parallel, tensor-parallel test는 distributed construction이 resolve된 scale을 `1.0`으로 되돌리지 않음을 검증한다.

## 6. 관련 작업

- Issue #1388: 파싱 후 사용하지 않던 Qwen3 및 Qwen3-MoE RoPE scaling.
- PR #1398: 이 보고서가 다루는 구현, review correction, 로컬 qualification.
- Issue #1340 및 #1355: Gemma 3 및 shared Llama RoPE scaling 선례.
- PR #1395: 최종 검증 전에 통합한 인접 Turbo4/MLA correction.
