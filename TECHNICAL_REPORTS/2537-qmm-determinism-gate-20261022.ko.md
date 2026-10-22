# PR #2537: qmm determinism 게이트가 실행 불가 시 실패하도록 수정

**작성일**: 2026-10-22
**상태**: 완료 (PR open; #1558 참조, 이슈는 열린 상태 유지)
**위험도**: Low
**언어**: Rust

---

## 요약

`tests/cuda_qmm_determinism.rs`를 재작업했다. 이 테스트는 체크포인트가 모든 호스트에 없었고 체크포인트 부재가 조용한 early return이었기 때문에, 아무것도 실행하지 않고 모든 호스트에서 `ok`를 보고했다. sm_70에서 gemma-4-12B로 실행했을 때(#1558) 실패했지만 원인은 커널이 아니었다. Gemma는 K/V를 모델 소유 상태에 두는데, 테스트가 반복 사이에 그 상태를 리셋하지 않았다. 실모델 테스트는 이제 `#[ignore]`이며, 체크포인트가 없으면 `MLXCEL_REQUIRE_MODELS=1`에서 panic하고, 반복마다 `reset_runtime_state`를 호출하며, make 타깃이 있다. 체크포인트가 필요 없는 커널 테스트 4개를 추가했다. sm_70에서는 아무것도 실행하지 못했으므로 #1558은 열어 두고 PR은 `Refs`를 쓴다. 6개 파일, +393/-57, 테스트 파일이 약 320줄.

---

## 1. 문제 정의

### 1.1 배경

#1558은 sm_70(V100)에서 gemma-4-12B의 prefill 로짓이 반복마다 달라진다고 보고했고, 이를 `qmm_naive`의 경쟁 상태로 의심했다.

### 1.2 기존 문제점

- 체크포인트 부재가 조용한 early return이라 게이트가 모든 호스트에서 `ok`였다.
- Gemma 3/4는 `ModelOwnedSequenceState`로 K/V를 보유하고 `make_caches`가 빈 슬라이스를 반환하므로, 테스트의 "fresh caches"는 새 시퀀스가 아니었다. 각 반복이 이전 반복 위에 prefill했다.
- 테스트가 `reset_runtime_state`를 호출하지 않았다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 게이트가 실행되지 않고 통과 | Medium (결정성 회귀를 못 잡음) | 모든 호스트에서 결정론적 |
| 테스트 결함이 커널 경쟁 상태로 오진됨 | Medium (잘못된 방향의 조사) | 보고 시점에 발생 |

---

## 2. 기술적 검토 사항

### 2.1 sm_70 실패는 테스트 결함이다

보고에 쓰인 빌드 `07ea6b1f`(2026-08-31)에서도 이미 같은 상태였다. GB10(sm_121, 드라이버 580.178.04, `MLX_ENABLE_TF32=0`)에서 기존 루프로 보고된 증상이 그대로 재현된다. gemma-4-12b-it-4bit, gemma-4-e2b-it-4bit, gemma-3-4b-it-4bit에서 첫 발산은 prefill이고, 10회 반복에서 스텝당 로짓 10종, 반복 0만 고유하다. `MLXCEL_SDPA_DETERMINISTIC=1`, `MLX_USE_CUDA_GRAPHS=0`, fused 커널 스위치 off, attention chunking off 모두 결과를 바꾸지 못했다. llama-3.2-1b-4bit, qwen3-0.6b-4bit, qwen2.5-0.5b-4bit는 통과했다. 반복 전에 리셋하면 세 Gemma 체크포인트, gemma-4-26b-a4b-it-4bit, cuDNN SDPA를 끈 gemma-4-12b가 모두 통과한다. 보고 자체의 증거(반복 0의 해시가 두 빌드에서 동일, 프로세스당 1회 greedy `generate`는 재현 가능)와도 일치한다.

### 2.2 `qmm_naive`에서 경쟁 상태를 찾지 못함

코드 리딩 결과 mainloop에 경쟁 상태가 없다. 공유 메모리 버퍼는 하나이고 저장 앞뒤에 barrier가 있으며, 레지스터로 prefetch하고, epilogue를 레지스터에서 전역 메모리로 직접 쓰고, `UniversalCopy`(cp.async 없음)를 쓰며, atomic과 split-K가 없다. #910 메커니즘이 작용할 대상이 없다. k-residue 경로는 `transpose=true`에서 도달 불가다. 보고 이후 추가된 Volta `SM70_8x8x4` atom(#1678)은 실행하지 않았다.

### 2.3 게이트가 크게 실패한다

일반 실행은 실모델 테스트를 거짓 `ok`가 아니라 `ignored`로 보고한다. `MLXCEL_REQUIRE_MODELS=1`에서는 `llama-3.2-1b-instruct-4bit`와 `llama-3.2-1b-4bit` 모두 해석되지 않으면 작업 디렉터리와 무관하게 panic한다. 존재하지 않는 `MLXCEL_TEST_QMM_MODEL_DIR` 오버라이드는 변수 유무와 관계없이 항상 실패한다. make 타깃은 `MLX_ENABLE_TF32=0`을 설정한다. `ctor` 핀이 lib 크레이트에서 `cfg(test)`이고 통합 테스트 바이너리에는 없기 때문이다.

### 2.4 체크포인트 없는 커널 테스트

Llama-3.2-1B의 모든 projection은 N % 128 == 0이라 `qmm_sm80`만 탄다. 합성 테스트 4개가 M in {8, 17, 64, 128, 333}에서 나머지 경로를 덮고, 바이트 단위 동일 출력을 요구하며, 가중치를 activation dtype으로 양자화하고 출력 dtype을 assert한다. 경로는 `MLXCEL_TRACE_ARCH` / `MLXCEL_TRACE_QMM_TILE`로 추적했다.

| Arm | 추적된 커널 |
|---|---|
| bf16, N=256 | `qmm_sm80` |
| f16, N=200 | `qmm_naive`, `MmaPath::Sm80`, `half_t` |
| f32, N=256 | `qmm_naive`, `MmaPath::Sm80`, `float`; `make_tiled_mma<Sm80, float>`이 `UniversalFMA`로 폴백하므로 라벨과 달리 스칼라 arm |
| f16, transpose=false, K=200 | `qmm_naive` k-residue |

`MLXCEL_QMM_FORCE_SCALAR`는 sm_80+에서 타일 선택만 바꾼다. MMA 경로는 compute capability만으로 결정된다.

### 2.5 리뷰 중 변경

- 문서 정정: f32 커널 테스트가 `UniversalFMA`를 실행하면서 `MmaPath::Sm80`으로 추적됨, make 타깃의 TF32 이유, `qmm_sm80` 전체 조건, `CONTRIBUTING.md`의 타깃 언급.
- `repo_model_dir_any`가 models 루트로 폴백하는 대신 비어 있지 않은 이름 목록을 assert.

---

## 3. 검증

모두 GB10(sm_121, `MLX_ENABLE_TF32=0`)에서 측정했다.

| 검사 | 결과 |
|---|---|
| 일반 실행 | 4 passed, 1 ignored; 잘못된 오버라이드는 `MLXCEL_REQUIRE_MODELS=1` 유무와 관계없이 실패 |
| llama-3.2-1b-4bit 모델 테스트 (`qmm_sm80`) | 10회 반복, 스텝당 1종 |
| 커널 arm | 20개 shape, 각 10회 호출에서 출력 1종 |
| 기존 루프(리셋 없음), gemma-4-12b, gemma-4-e2b, gemma-3-4b | 스텝당 로짓 10종 |
| 리셋 적용, 위 셋 + gemma-4-26b-a4b + cuDNN SDPA off gemma-4-12b | 통과 |
| 최종 트리, `gpu-lock` 하 `MLXCEL_REQUIRE_MODELS=1 --include-ignored --test-threads=1` (make 타깃과 동일 플래그) | 5 passed |
| `cargo fmt --all -- --check` | Clean |
| `cuda_qmm_determinism`, `sampling_top_k_above_vocab_e2e` 테스트 크레이트의 clippy `-D warnings` | Clean; 부하 높은 호스트에서 `cargo clippy`가 MLX CUDA CMake 빌드를 재실행하여 cargo가 캡처한 rustc 호출에 `clippy-driver`로 실행 |

---

## 4. 미검증 및 후속 항목

- **미실행:** sm_70에서의 모든 동작, Volta `SM70_8x8x4` atom 포함. #1558이 열려 있는 이유다.
- **미실행:** make를 통한 `make verify-test-cuda-qmm-determinism` 호출. 정확한 플래그는 테스트 바이너리에 직접 실행했다.
- **후속:** 타깃은 `nightly-verify.yml`에 연결되지 않았다.

---

## 5. 학습 포인트

- 입력이 없을 때 조용히 반환하는 테스트는 실행되지 않은 검사를 `ok`로 보고한다. 건너뜀을 보이게(`#[ignore]`) 하고, CI에는 입력 부재를 실패로 바꾸는 스위치를 둔다.
- K/V를 모델 소유 상태에 두는 모델은 하네스가 만든 "fresh caches"를 무시한다. 반복 결정성 테스트는 런타임 상태를 명시적으로 리셋해야 하며, 그렇지 않으면 N번째 반복이 0번째와 다른 시퀀스를 측정한다.
