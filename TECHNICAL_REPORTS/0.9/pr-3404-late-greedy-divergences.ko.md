# PR #3404: 후발 XLA vs eager greedy 발산 해결 (Gemma3, Phi4MM 비전, Gemma3n 핀)

**Date**: 2026-11-26
**Status**: PR open (#3391 종결, #932와 에픽 #566의 일부). GB10 CUDA에서 핀된 체크포인트로 프로브 통과, 양쪽 assert 지점에 마진 기록

## 요약

#3365 이후 세 가지 후발 XLA vs eager greedy 발산이 남아 있었습니다. 이 PR은 각 발산 스텝에서 양쪽의 top-2 로짓 마진을 측정하고 항목별로 결정합니다. 결함은 수정하고, 근접 동률은 마진과 함께 기록합니다. 세 항목 모두 그래프 변경 없이 해결됩니다. Gemma3는 기록된 크로스 백엔드 근접 동률(양쪽이 동일한 top-2 후보 집합을 랭킹, eager는 정확히 bf16 ulp 1개로 2258, XLA는 3355), Phi4MM의 비전 경로는 모든 스테이지에서 비트 동일하고 보고된 발산은 현재 main에서 재현되지 않으며, Gemma3n의 prefix-decode top1은 236798로 리핀됩니다(양쪽 프로덕션 CUDA 경로가 동의, 서드파티 mlx-vlm CPU 오라클만 854). 두 마진 프로브가 미래 변경이 깨뜨릴 패리티를 게이트합니다.

## 1. 문제 정의

### 1.1 세 가지 발산

- **Gemma3** `gemma-3-4b-it-4bit`, `test_image.png`, `-n 16 --temp 0`: 첫 15개 토큰이 eager와 일치, 토큰 16은 eager 2258 vs XLA 3355, cuda와 local-task 양쪽에서. 34스테이지 경계 게이트는 통과하므로 발산은 수치 갭이 아니라 랭킹 플립입니다.
- **Phi4MM 비전** `phi-4-multimodal-instruct` (#3380의 LoRA 레이아웃 필요): 8개 토큰 일치 후 XLA는 "white line in the middle", eager는 "on the left side". XLA f32와 f16 비전이 동일 토큰이므로 갭은 디코더 상류에 있었습니다.
- **Gemma3n** 캐노니컬 lib 테스트 `pinned_e2b_intermediates_prefill_decode_and_greedy_match`가 `src/models/gemma3n.rs`에서 실패: "pinned production-CUDA prefix-decode MLX top1 drifted", 핀 854, 관측 236798. 코드 주석은 근접 랭킹 크로스 디바이스 플립이라 했으나 증명되지 않았습니다.

### 1.2 결정 전에 측정하는 이유

#3291과 #3390이 규칙을 확립했습니다: eager의 결과가 레퍼런스이고, 다른 쪽으로 해결된 근접 동률은 라운딩 경계를 움직여 "수정"하는 것이 아니라 기록합니다. 발산 크기에서 dtype 스텝 1개 이내의 마진은 근접 동률이고, 실제 수치 갭은 큰 마진으로 후보 집합 실패를 보입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/loading/vlm_gemma3_margin_tests.rs` (신규) | `pinned_gemma3_greedy_margin_probe`: CLI처럼 eager와 XLA greedy 루프를 구동하고, 스텝별 top-2 마진을 기록하고, 모든 스텝에서 top-2 후보 집합 일치를 단언(실제 수치 갭은 실패). |
| `src/lib/mlxcel-xla/src/batch.rs` | `XlaPreparedPrefillLogits::decode_ragged_logits`, teacher-forced 연속 셈. `all(iree, any(test-support, diagnostics))` 게이트이고 게이트 없는 `decode_ragged_logits_with_modes`에 제로 어댑터 모드 벡터로 위임해 test-support와 diagnostics 빌드 모두 컴파일(리뷰 수정). |
| `src/loading/vlm_phi4mm_margin_tests.rs` (신규) | `pinned_phi4mm_vision_bisect_probe`: 전처리, per-crop 인코더 히든, 풀링·프로젝트 피처에 걸쳐 eager와 XLA의 스테이지별 비트 동일성. |
| `src/models/gemma3n.rs` | Prefix-decode top1 리핀 854에서 236798, 양쪽 assert 지점에 측정 마진 문서화. |

`src/loading/vlm.rs`와 `vlm_gemma.rs`의 모듈 선언 포함 3개 파일. 톨러런스나 임계값 변경 없음.

## 3. 기술적 선택

### 3.1 후보 집합 일치를 근접 동률 게이트로

Gemma3 프로브는 토큰을 핀하는 대신 모든 스텝에서 top-2 후보 집합 일치를 단언합니다. 실제 수치 갭은 top-2에 어떤 후보가 있는지 바꾸지만, 근접 동률은 집합을 유지하고 랭킹만 뒤집습니다. 발산 스텝의 측정 마진: eager top1 2258(36.000)이 3355(35.750)보다 0.250 위, ~36 로짓 크기에서 정확히 bf16 ulp 1개. XLA top1 3355(36.274)가 2258(35.408)보다 0.866 위. 스텝 0-14는 양쪽에서 top-1 일치.

### 3.2 Gemma3n 리핀 근거

양쪽 프로덕션 CUDA 경로가 동의합니다: eager prefix-decode는 ~14 로짓 크기에서 0.3125로 236798을 먼저 랭킹하고 XLA는 0.375로 동의. 서드파티 mlx-vlm 0.7.8 CPU 오라클만 0.75로 854를 랭킹. 플립은 크로스 빌드(MLX 핀이 551039ee 캡처 이후 #3134로 세 번 진행)이고 크로스 디바이스이며 그래프 결함이 아닙니다. 이슈의 리핀 규칙에 따라 핀은 854에서 236798로 이동하고 마진이 옆에 기록됩니다. `mlx_logits`는 236798 유지, `mlx_greedy`는 [236798, 236789] 유지.

### 3.3 셈의 피처 게이팅

`IreeRaggedLlama::decode_ragged_logits`는 `#[cfg(feature = "diagnostics")]`이므로, `any(test-support, diagnostics)`로 게이트된 셈이 이것을 호출하면 test-support 전용 빌드(루트 `test-utils`가 `mlxcel-xla?/test-support`로 전달)를 깨뜨렸고 프로브 자체의 문서화된 레시피도 포함했습니다. 셈은 게이트 없는 `decode_ragged_logits_with_modes`에 `b_max()` 길이의 제로 어댑터 모드 벡터로 위임하며, diagnostics 게이트 래퍼의 본문을 미러링합니다(동일한 OneD와 길이 검증). 두 피처 집합 모두 컴파일됩니다. `cargo check -p mlxcel-xla --features test-support`와 xla-diagnostics 빌드 모두 통과.

## 4. 결과

GB10 CUDA, 핀된 체크포인트, gpu-lock.

| 항목 | 결과 |
|---|---|
| Gemma3 마진 프로브 | 통과(200.68s): 스텝 0-14 top-1 일치, 스텝 15 동일 후보 집합, 근접 동률 기록 |
| Phi4MM 바이섹트 프로브 | 모든 스테이지 max abs 0.0. CLI greedy는 양쪽 백엔드에서 동일 16 토큰 |
| Gemma3n 리핀된 1행 테스트 | 1 통과, 0 실패 |
| `cargo check -p mlxcel-xla --features test-support` | 통과(리뷰 수정) |
| `cargo check --no-default-features --features xla-diagnostics` | 통과 |

## 5. 한계

- **Phi4MM의 보고된 발산은 재현되지 않음.** 비전 경로는 비트 동일하고 CLI는 동일 토큰을 생성합니다. 이전 관측은 #3380의 LoRA 레이아웃 수정이 머지되기 전이었습니다. 프로브가 이제 그 패리티를 게이트합니다.
- **프로브는 핀된 체크포인트 환경 변수 뒤에 `#[ignore]`d**되어 diagnostics 테스트 관례를 따릅니다. CI에서 실행되지 않습니다.
- **기존 호스트 한계.** dev-profile 통합 테스트 `pipeline_ci_multi_stage_real_models`는 이 호스트에서 cuda 피처 조합으로 링크되지 않습니다(AArch64 BL 범위 초과, 이전 continuum-router 발견과 같은 클래스). 테스트 빌드는 `--lib`을 사용합니다.
- **CPU 오라클은 서드파티.** mlx-vlm 0.7.8은 Gemma3n prefix-decode에서 854를 먼저 랭킹합니다. 마진만 기록되고 게이트되지 않습니다.

## 6. 학습 포인트

- 근접 동률 vs 결함을 결정하기 전에 top-2 마진을 측정하십시오. 후보 집합 일치 프로브는 토큰을 핀하지 않고도 랭킹 플립(근접 동률)과 실제 수치 갭(다른 후보)을 구분합니다.
- mlxcel-xla의 test-utils 소비자는 셈의 피처 게이트가 callee보다 좁으면 깨집니다. callee의 게이트를 넓히는 대신 게이트 없는 변형으로 전달하십시오.
- 이전 MLX 핀에서 캡처된 핀 값은 MLX가 진행하면 부패합니다. 드리프트가 귀속 가능하도록 핀 옆에 캡처 커밋을 기록하십시오.

## 관련 커밋

| 해시 | 타입 | 메시지 |
|---|---|---|
| `2fe6dce` | test | gate the Gemma3 and Phi4MM greedy divergences with margin probes |
| `ab0029c` | fix | re-pin the Gemma3n prefix-decode top1 at 236798 |
| `d4a8ec8` | fix | call the ungated ragged-decode variant from the diagnostic shim |
| `9dc0b01` | docs | deduplicate the Gemma3n re-pin comment and align the probe byte order |
| `64cf984` | docs | reflow the Gemma3n re-capture paragraph lead-in |

관련: #3391(이 변경), #3365(발산 발견), #3380(Phi4MM LoRA 레이아웃), #3291과 #3390(근접 동률 규칙), #3134(MLX 핀 진행), #932, #566.

## 검증

- Gemma3 마진 프로브는 gpu-lock 하에서 직접 통과(200.68s), 기록값과 일치하는 마진. Phi4MM CLI greedy도 직접 재검증, 동일 16 토큰.
- `cargo test -p mlxcel --lib --no-run` 컴파일. `cargo fmt --check` 깨끗. `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` 깨끗.
- 구현 리뷰: 셈 게이팅 수정 후 승인. 보안 검토: 승인, CRITICAL/HIGH 없음.
