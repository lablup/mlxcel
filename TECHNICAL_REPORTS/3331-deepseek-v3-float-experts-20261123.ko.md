# PR #3331: 양자화되지 않은 DeepSeek-V3 MoE 체크포인트 로드

**날짜**: 2026-11-23
**상태**: draft PR로 열림 (#3319 종료). 좁은 단위 테스트 셀렉터, clippy, fmt, CI 스크립트 통과. mlx-lm 대비 실제 체크포인트 트레이스 결과는 아래에 기록.
**위험도**: 낮음 (양자화 전문가 경로는 변경 없이 바이트 단위로 동일한 트레이스, float 경로는 신규)

## 요약

`deepseek_v3`의 라우팅 전문가 블록(`src/models/deepseek_v3.rs`의 `SwitchGLU`)은 모든 전문가 plane에 양자화 `.scales`와 `.biases`를 요구했기 때문에, 양자화되지 않은 체크포인트는 첫 MoE 레이어에서 로드에 실패했습니다. bf16 GigaChat3-10B-A1.8B는 `Weight not found: model.layers.1.mlp.switch_mlp.gate_proj.scales`에서 멈췄습니다. 이제 블록은 블록마다 결정되는 enum입니다. 양자화 variant는 기존 구조체 그대로이고, 새 float variant는 공유 `SwitchLinear`를 통해 `gather_mm`을 실행하며 같은 clipped SwiGLU 활성화를 씁니다. bf16 체크포인트는 이제 로드되고, 8192 위치에서 mlx-lm 대비 decided top-1 불일치가 2771 중 0으로, mlx-lm bf16과 f16 사이의 노이즈와 같습니다.

## 1. 문제

sanitize는 이미 존재하는 leaf에 대해 `mlp.experts.{e}.*`를 `mlp.switch_mlp.*`로 쌓았고, dense 레이어와 shared expert는 float 가중치를 받는 `UnifiedLinear`로 로드되었습니다. 라우팅 전문가 블록만 양자화 전용이었습니다. 필수 텐서 9개를 저장하고 `gather_qmm`을 세 번 호출했습니다. 이 디코더로 연결되는 모든 model type(`deepseek_v3`, `kimi_k2`, `joyai_llm_flash`)과 `MoEBlock`의 다른 호출자(Kimi-VL 텍스트 모델, Xing4.0, V3 파이프라인 stage executor)가 같은 공백을 가졌습니다. PR #3306은 GigaChat3를 트레이스하려고 8비트로 양자화해야 했습니다.

## 2. 변경 요약

| 영역 | 변경 |
| --- | --- |
| `SwitchGLU` | 이제 `enum SwitchGLU { Quantized(QuantizedSwitchGLU), Float(FloatSwitchGLU) }`. `from_weights`가 세 plane을 `.scales` 유무로 나눕니다. 전부 없으면 float, 전부 있으면 양자화, 섞이면 거부합니다. |
| `QuantizedSwitchGLU` | 기존 구조체의 이름만 변경. 본문, NVFP4 거부, plane별 비트 폭 계산은 그대로이고, plane 루프는 공유 상수 `SWITCH_GLU_PLANES`를 씁니다. |
| `FloatSwitchGLU` | `switch_layers::SwitchLinear` plane 세 개이며 각각 `Regular`로 로드되어야 합니다. `forward`는 `prepare_gather_indices`로 gather 인덱스를 한 번 만들고 각 plane의 `forward_indexed`를 호출한 뒤 `clipped_swiglu`, 그리고 MLX 기본 lhs로 down projection을 실행합니다. |
| 로드 거부 | 섞인 plane, `.scales` 없이 `.biases`만 있는 plane, dtype이 f16, bf16, f32가 아닌 float plane. |
| 문서 | `forward_indexed`와 `prepare_gather_indices`의 `Used by:`, `docs/supported-models.md`의 DeepSeek 항목, CHANGELOG `Fixed`. |

## 3. 기술적 결정

### 3.1 공유 `SwitchGLU` 대신 블록을 로컬로 유지

공유 `switch_layers::SwitchGLU`에는 clipped 활성화가 없습니다. Kimi-VL과 Moonlight는 f16 활성화를 유한하게 유지하려면 `clip(silu(gate), -100, 100) * up`이 필요하므로(issue #525) 블록은 계열 로컬로 두고 plane만 공유 `SwitchLinear`로 옮겼습니다. 두 번째 래퍼를 만들지 않고 기존 `gather_mm` 호출(`SwitchLinear::Regular`)을 재사용합니다.

### 3.2 `.scales`로 블록마다 결정

variant는 config의 `quantization` 블록이 아니라 가중치로 고릅니다. mlx-lm 내보내기는 일부 모듈만 양자화할 수 있기 때문입니다. 블록마다 고르면 완전히 양자화된 블록은 이전과 정확히 같은 코드에 도달합니다. 섞인 블록은 plane 목록과 함께 거부됩니다. 예: `p: DeepSeek-V3 routed experts mix quantized and float planes (gate_proj quantized, up_proj/down_proj float)`.

### 3.3 float 경로에서 정수 plane 거부 (리뷰)

`.scales`가 없다는 것만으로 float variant를 고르면, scales를 잃은 양자화 체크포인트의 packed `uint32` 가중치가 float plane으로 로드됩니다. 그러면 첫 라우팅 forward에서 `gather_mm`이 cxx 경계를 넘어 예외를 던지고 프로세스가 중단됩니다. main에서는 로드 시점에 깔끔하게 실패했던 경우입니다. 그래서 float 로더는 고아 `.biases`가 있는 plane과 dtype이 f16, bf16, f32가 아닌 가중치를 거부합니다.

### 3.4 정렬 prefill 경로 없음

공유 `SwitchGLU`는 token-expert 쌍이 64개 이상이면 정렬해서 정렬 `gather_mm` 커널을 씁니다. float variant는 이슈의 설계와 양자화 DeepSeek-V3 경로를 따라 정렬하지 않습니다. bf16 트레이스에서 512 토큰 청크 prefill은 M3 Ultra에서 430에서 575 ms입니다. 두 variant에 정렬 경로를 추가하는 일은 후속 작업으로 남깁니다.

## 4. 검증

teacher-forced `examples/logit_trace`, 512 위치 청크 16개(8192 위치), top-8, mlx-lm 1bf8da0(mlx 0.32.4) 대비. 업스트림은 인덱스 61만 제거하므로, mlx-lm에는 인덱스 26의 MTP 레이어를 버리는 shim이 필요합니다. mlxcel은 로드 시 bf16 가중치를 f16으로 변환하므로 레퍼런스를 bf16 원본과 bf16 파라미터를 f16으로 캐스팅한 경우, 두 번 실행했습니다. 두 레퍼런스끼리의 비교가 노이즈 기준입니다.

| 비교 | decided 불일치 (gap >= 2) | top-1 불일치 | 불일치 최대 gap | perplexity ref / cand |
| --- | --- | --- | --- | --- |
| mlx-lm bf16 vs mlx-lm f16 (노이즈 기준) | 0 / 2771 | 324 / 8192 | 0.875 | 14.557 / 14.550 |
| mlx-lm bf16 vs mlxcel | 0 / 2771 | 312 / 8192 | 0.875 | 14.557 / 14.546 |
| mlx-lm f16 vs mlxcel | 0 / 2718 | 130 / 8192 | 0.734 | 14.550 / 14.546 |

- 수정 전 `mlxcel generate -m models/gigachat3-10b-a1.8b-bf16`은 `scales` 오류로 실패했습니다. 수정 후에는 약 45 tok/s, 상주 메모리 19.9 GB로 일관된 텍스트를 생성합니다.
- 양자화 회귀, main과 이 브랜치에서 각각 빌드한 바이너리 비교: `gigachat3-10b-a1.8b-mlx-8bit` 트레이스(8192 위치)와 `moonlight-16b-a3b-instruct-4bit` 트레이스(256 위치 4개)가 바이트 단위로 동일하고, 두 모델의 greedy `generate` 출력도 동일합니다. 8비트 트레이스는 #3306의 mlx-lm 8비트 레퍼런스 대비 여전히 decided 불일치 2774 중 0입니다.
- 단위 테스트: 작은 f32 float 블록을 손으로 계산한 f64 전문가별 clipped SwiGLU와 상대 오차 1e-4 이내로 비교(한 전문가의 gate를 키워 clip에 도달), 3D 입력이 같은 값을 주는지 확인, 섞인 plane 오류, 고아 `.biases` 유무에 따른 packed plane 거부. 셀렉터(test-fast, GPU 락 아래): `deepseek_v3`, `xing4_0`, `coverage_tests`는 `--test-threads=1`, 그리고 `kimi`, `switch_layers`. `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`는 경고 없음.

## 5. 미검증 및 남은 항목

- 양자화 회귀 트레이스와 bf16 트레이스는 dtype과 고아 bias 거부를 추가한 리뷰 커밋 이전에 측정했습니다. 이 커밋은 로드 시점 검사만 바꾸며, 이 커밋으로 bf16 체크포인트를 다시 로드하고 생성했습니다.
- 양자화되지 않은 Kimi-K2, JoyAI, Kimi-VL, Xing4.0 체크포인트는 실행하지 않았습니다. 이들은 `MoEBlock`을 공유하므로 같은 코드 경로로 다뤄집니다.
- 이 블록의 전문가 shape 검사는 #3320입니다.
- 넓은 게이트(워크스페이스 전체 `cargo test`, 이슈의 전체 검증 목록)는 오케스트레이터에 맡깁니다.

## 6. 배운 점

- 런타임이 로드 시 가중치 dtype을 바꾸면 같은 정밀도 레퍼런스 하나로는 부족합니다. 두 dtype으로 레퍼런스를 돌리면 후보를 비교할 노이즈 기준이 생깁니다.
- 키가 없다는 사실로 코드 경로를 고르면, 그 키만 잃은 입력에도 경로가 열립니다. 선택 로직은 받아들이는 입력을 따로 검증해야 합니다.

## 관련 커밋

- `efe57825` fix(models): load unquantized DeepSeek-V3 MoE checkpoints
- 리뷰 후속: 정수 및 고아 bias float plane 거부, 문서와 CHANGELOG
