# 기술 보고서: PR #1101 - feat(models): add Meta Muse Glimmer VLM support

**작성일**: 2026-08-12
**상태**: 완료
**언어**: Rust, C++, Markdown
**위험도**: Medium

---

## 요약

PR #1101은 고정된 dense BF16 체크포인트를 대상으로 Meta Muse Glimmer 30B를 mlxcel에 일급 모델로 추가한다. mixed-cache text decoder, vision 전처리와 fusion, CLI 및 continuous-batching server 경로, Muse recipient/reasoning channel, bounded ATEM tool call을 통합하고 검증되지 않은 실행 모드는 시작 단계에서 거부한다.

최종 리뷰에서 실제 GB10 검증 후 도움말 문구가 변경됐지만 기존 pending 문구를 계속 기대하던 CLI 테스트 1건을 발견했다. `d4ba28ac`에서 이 계약 테스트를 현재 지원 문구에 맞췄으며, Muse 범위에 남은 차단급 correctness, security, performance 문제는 없다.

---

## 1. 문제 정의

### 1.1 배경

Muse Glimmer는 52-layer text stack, 50-layer vision tower, sliding/full attention 혼합, 체크포인트 고유 multimodal prompt layout, recipient 기반 reasoning channel, ATEM tool call을 결합한 30B VLM이다. 기존 범용 VLM 경로는 이 계약을 표현하지 못했다.

### 1.2 기존 한계

- 모델 타입과 공개 체크포인트의 weight namespace가 인식되지 않았다.
- sliding layer는 2,048-token window를 회전시키고 full-attention layer는 계속 성장하므로 model-owned per-sequence state가 필요했다.
- image marker를 정확한 merged visual-token 수로 확장하고 multi-image 순서를 보존해야 했다.
- streaming API에서 token-position accounting을 잃지 않으면서 reasoning과 ATEM 구조를 visible content에서 제거해야 했다.
- quantization, video, adapter, speculative decoding, TP/PP, XLA, distributed mode는 검증되지 않았으므로 fail-closed 처리가 필요했다.

### 1.3 위험성

| 위험 | 영향도 | 변경 전 발생 가능성 |
|------|--------|---------------------|
| 잘못된 mixed-cache state가 request 간 교차하거나 long prompt를 손상 | High | High |
| placeholder/feature 불일치가 다른 이미지와 조용히 결합 | High | Medium |
| reasoning 또는 tool payload가 streaming content로 노출 | High | Medium |
| 미지원 모드가 시작된 뒤 native kernel에서 늦게 실패 | High | Medium |

---

## 2. 기술적 검토 사항

### 2.1 Correctness와 Security

- weight loader는 pinned index의 tensor 1,436개를 모두 분류하며 unknown root와 quantization sidecar를 거부한다.
- configuration validation은 layer schedule, vision geometry, RoPE 동작, dense BF16 baseline, unsupported capability를 고정한다.
- image preparation은 marker/cardinality와 feature-row 불일치를 text-only로 폴백하지 않고 거부한다.
- model-owned sequence ID가 admission, batching, release, reset, snapshot, restore 전 과정에서 mixed-cache state를 격리한다.
- ATEM parser는 input, call, parameter, name, argument byte에 상한을 두며 route response 생성 전에 tool allowlist를 적용한다.
- streaming filter는 ATEM depth를 추적하고 byte-split delimiter와 malformed EOF를 처리하며 reasoning/tool markup을 visible delta에서 제거한다.

인증이나 영속 데이터 경계는 변경되지 않았고 새 외부 dependency도 추가되지 않았다.

### 2.2 Performance와 Memory

이 baseline은 59,553,253,376-byte BF16 체크포인트를 GB10급 128 GB unified-memory host에서 실행하는 것을 명시적으로 목표로 한다. sliding layer는 2,048-token rotating cache로 제한하고 full-attention layer는 long-context 계약을 위해 계속 성장한다.

2026-08-11 real-checkpoint gate 결과:

| 시나리오 | 결과 |
|----------|------|
| Greedy text decode | 4.25 tokens/s |
| 2,204-token long prompt prefill | 46.47 tokens/s |
| Single/two-image prompt | orange test fixture를 근거 있게 인식 |
| Scheduler | parallelism 1과 2에서 request 답변 격리 |
| Cold two-image concurrency | system `MemAvailable` 최대 59.608 GiB 감소, process `VmHWM` 4.136 GiB |

이 GB10 backend에서는 CUDA allocator counter를 사용할 수 없으므로 allocator의 0 값을 증거로 해석하지 않고 OS-level memory 측정값을 기록했다.

### 2.3 호환성

- 검증됨: Linux/aarch64, NVIDIA GB10, CUDA 13.0, driver 580.173.02.
- 지원 경로: CLI, OpenAI Chat Completions, Responses, Anthropic-compatible API, streaming, text, single image, multi-image, ATEM replay.
- 명시적 미지원: video, quantized weight, Turbo/INT8 KV, DFlash/speculative decoding, LoRA/adapter, TP, PP, XLA/IREE/OpenXLA, distributed/disaggregated serving.
- 이 체크포인트의 Apple Silicon/Metal 검증은 아직 주장하지 않는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 Model-Owned Mixed Cache

**선택:** Muse sequence state를 모델 내부에 두고 scheduler sequence ID로 접근한다.

**이유:** 범용 homogeneous cache로는 체크포인트의 sliding/full-attention 교대 layer를 표현할 수 없다. 모델이 소유하면 정확한 schedule과 release/reset 의미를 보존할 수 있다.

**트레이드오프:** 같은 mixed-state 계약을 보존할 수 있을 때까지 범용 paged/disaggregated cache 경로를 비활성화한다.

### 3.2 엄격한 Multimodal Cardinality

**선택:** 체크포인트 marker마다 계산된 visual-token 수를 정확히 확장하고 marker, grid, feature-row 불일치를 모두 거부한다.

**이유:** 조용한 truncation이나 text-only fallback은 feature를 잘못된 이미지와 연결할 수 있다. generation 전에 실패시켜 multi-image 순서를 감사 가능하게 유지한다.

### 3.3 Bounded ATEM Parsing과 Streaming Suppression

**선택:** 기존 JSON tool-call format을 변형하지 않고 전용 bounded parser와 depth-aware streaming filter를 사용한다.

**이유:** Muse는 attribute가 있는 XML 유사 tag와 recipient channel을 출력한다. 전용 경로가 typed parameter, parallel-call order, allowlist, malformed output, API별 streaming event를 보존한다.

### 3.4 Fail-Closed Baseline

**선택:** 검증되지 않은 모든 feature를 CLI/server 시작 단계에서 거부한다.

**이유:** dense BF16 경로에는 real-checkpoint 증거가 있지만 거부된 모드에는 없다. 조용히 잘못된 출력이나 늦은 kernel 실패보다 빠르고 구체적인 오류가 안전하다.

---

## 4. 구현 개요

```text
Checkpoint config/index
        |
        +--> Muse text decoder --> per-sequence mixed rotating/full KV state
        |
Images --> Muse processor --> 50-layer vision tower --> 2x2 pixel shuffle
                                                     --> adapter/projection
        |                                                     |
Prompt template --> exact patch-marker expansion -------------+
        |
CLI / continuous scheduler --> generation --> recipient/reasoning split
                                           --> bounded ATEM parser/filter
                                           --> Chat / Responses / Anthropic events
```

이 밖에도 model detection/metadata, checkpoint fixture, generation default, startup guard, scheduler admission test, API round-trip test, 문서, CLI 지원 설명을 추가했다.

---

## 5. 학습 포인트

### 5.1 Cache Shape은 Model Architecture의 일부다

sliding/full attention 혼합은 단순한 메모리 최적화가 아니다. state lifetime을 바꾸므로 batching, sequence identity, snapshot, restore, long-context test에서 직접 표현해야 한다.

### 5.2 Stream Filtering은 Position Accounting을 보존해야 한다

structural token을 숨기는 것만으로는 scheduler가 consumed token position을 잃을 수 있다. Muse filter는 조각난 delimiter에서도 suppressed/consumed position을 기록하여 streaming과 non-streaming 결과의 동등성을 유지한다.

### 5.3 Real-Checkpoint Qualification은 Fail-Closed여야 한다

synthetic test는 shape과 invariant를 검증하지만 대형 모델의 준비 상태에는 pinned revision, hardware, memory, throughput, route, long-context, tool, concurrency 증거가 함께 필요하다. qualification 상태가 바뀌면 help text와 documentation contract test를 함께 갱신해야 한다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|----|
| 변경된 파일 수 | 85 |
| 추가된 라인 | 11,437 |
| 삭제된 라인 | 84 |
| 추가된 Rust test attribute | 127 |

### 주요 영역

| 영역 | 주요 내용 |
|------|-----------|
| Model/runtime | Detection, configuration, weight loading, 52-layer decoder, mixed cache |
| Vision | Processor, layout/position logic, 50-layer tower, fusion, ordered scatter |
| Serving | Scheduler admission, expanded usage, startup guard, 세 API family |
| Tools/streaming | Muse recipient, reasoning split, bounded ATEM parsing/replay |
| Documentation | Supported-model contract, model 추가 가이드, GB10 qualification |
| Review correction | stale top-level help assertion을 현재 GB10 qualification 문구로 변경 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `38307107` | feat | add Muse Glimmer 30B support |
| `7fc1bdd0` | fix | harden Muse streaming and slot admission |
| `d4ba28ac` | test | align Muse qualification help assertion |

관련 이슈: #1100.

---

## 7. 검증과 후속 조치

### 통과

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --features cuda -- -D warnings`
- CLI qualification 집중 테스트: 1 passed
- host GPU와 pinned checkpoint를 사용한 Muse 회귀: 95 passed
- ATEM 회귀: 37 passed
- 최종 push 전 PR hosted cheap gate 통과, 최종 push 후 재실행 중
- 기존 real-checkpoint gate에서 text, single image, multi-image, long prompt, ATEM replay, streaming, scheduler parallelism 1/2 검증

### Repository-Wide Gate 상태

authoritative CUDA 명령은 `--no-fail-fast`로 끝까지 완료됐다. `mlxcel` library target은 5,528 passed, 5 failed, 113 ignored였고 이후 workspace target과 doctest에서는 추가 실패가 없었다. 아래 5개는 단독 실행에서도 재현되며 구현/test 경로가 `origin/main`과 동일하다.

- `execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count`
- `models::bailing_moe_linear::tests::chunked_gla_matches_the_sequential_recurrence`
- `models::deepseek_v2::tests::absorbed_mla_attention_matches_the_decompressed_block_step_for_step`
- `models::florence2::florence2_tests::incremental_decode_matches_full_sequence`
- `models::klear::tests::the_prefill_is_causal_without_being_handed_a_mask`

이는 Muse 회귀가 아니라 repository-wide 기존 CUDA gate 실패다. memory-budget 산술과 numerical tolerance를 독립적으로 검토할 수 있도록 별도 집중 작업에서 교정하는 것이 적절하다.

### 남은 Qualification 경계

- Apple Silicon/Metal 검증은 주장하지 않는다.
- 미지원 baseline mode는 의도적으로 차단된 상태를 유지한다.
