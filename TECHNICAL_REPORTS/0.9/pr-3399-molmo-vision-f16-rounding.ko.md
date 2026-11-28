# PR #3399: Molmo v1 비전 타워를 eager의 F16 경계에서 라운딩

**Date**: 2026-11-24
**Status**: PR open (#3390 종결, #932와 에픽 #566의 일부). 유닛 스위트 통과, GB10 CUDA에서 실제 Molmo-7B 체크포인트로 eager MLX 대비 검증

## 요약

XLA Molmo v1 비전 그래프는 CLIP 타워를 f32로 계산했지만, eager MLX는 F16 체크포인트의 모든 연산자 뒤에 F16 텐서를 저장합니다. #3365가 레퍼런스 하네스를 수리하면서 `xla_molmo_reference_check`가 #2837 이후 처음으로 비전 스테이지에 도달했고, 14개 스테이지 중 2개가 IREE `local-task`와 `cuda` 양쪽에서 동일하게 실패했습니다(`vision.selected_layer_21` max_abs 2.80, `vision.selected_layer_14` max_abs 2.69, 실패 채널은 duplicate crop에서도 동일). 이 PR은 eager가 F16 텐서를 저장하는 모든 지점에서 그래프를 F16으로 라운딩하되, 수축(matmul), softmax, 노름 통계 내부의 f32 누적은 유지합니다. 게이트는 양쪽 디바이스에서 `VISION` 바운드를 그대로 두고 14/14로 통과합니다.

## 1. 문제 정의

### 1.1 배경

#3365가 레퍼런스 하네스의 이중 해시 버그(#2837)를 고쳐 Molmo 게이트가 비전 스테이지에 도달했습니다. 두 스테이지가 `VISION` 바운드를 각각 4개, 2개 원소 넘었습니다. Molmo는 #870에서 14/14로 퀄리파이되었으므로, 회귀인지 수치 계약 누락인지가 문제였습니다.

### 1.2 CUDA 회귀가 아님

#870 머지 커밋(`3ba57b30`, 다이제스트 패치 불필요, 이중 해시는 #2837에서 들어옴)에서 게이트를 빌드하면 `vision.selected_layer_21`이 이 GB10 호스트에서 이미 실패합니다(4개 원소, max_abs 1.56). #870은 eager on ROCm(`gfx1151`)으로 퀄리파이되었으므로 호스트 차이는 원래 있었고, 하네스가 비전 스테이지에 도달하면서 처음 보인 것입니다. 베이스 자체가 실패하므로 중간 커밋을 추가로 바이섹트하지 않았습니다.

### 1.3 근본 원인: 라운딩 경계 누락

실패 원소는 타워의 아웃라이어 채널 650(|x| 최대 약 200)과 레이어 21의 채널 363, 그리고 68/117/309 행과 duplicate crop 행들에 있습니다. 아웃라이어 채널이 각 LayerNorm의 분산을 지배하므로 f32와 F16의 차이가 모든 블록으로 전파됩니다. 차이는 한 번의 출력 라운딩이 아니라 블록을 거치며 쌓입니다(그 크기에서 F16 스텝 약 90개). eager의 결과가 레퍼런스이고 그 결과에는 저장된 텐서마다 F16 라운딩이 포함되어 있으므로, 그 라운딩을 건너뛰는 그래프는 실수에 가깝고 eager에서는 멀어집니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `emitter/molmo_vision_f16.rs` (신규) | `round_f16`, `linear_f16`, `layer_norm_f16`, `quick_gelu_f16`과 유닛 테스트 4개. |
| `emitter/molmo_vision.rs`, `emitter/mod.rs` | 새 연산자로 타워를 빌드. 디퀀트된 가중치를 라운딩. `MOLMO_VISION_PRECISION`은 F32 유지(수축은 f32 누적). |
| `operator_numeric_contract/families.rs`, `family_tests.rs` | 라운딩하는 연산자에 f32에서 F16 출력 라운딩 선언. attention scores/softmax와 풀 attention 경로는 f32 유지. Molmo2 불변. |
| `src/multimodal/host_preprocessor_molmo_diagnostics_support.rs` | `compare_stage`가 첫 실패 원소를 출력(16개로 상한)해 행과 채널 단위 추적 가능. |
| `numeric_ops.rs`, `docs/gelu-call-sites.md` | 문서 갱신: Molmo는 `quick_gelu_f16` 사용, 공유 헬퍼는 LLaVA와 Qwen2-VL에 유지. |
| `CHANGELOG.md` | 변경 항목, 1회 재컴파일 포함. |

11개 파일, +279/-60. 톨러런스 변경 없음. `VISION`은 atol 0.25, rtol 0.05 유지.

## 3. 기술적 선택

### 3.1 연산별 라운딩 설계

각 규칙은 F16 활성화로 퀀타이즈된 이 체크포인트에 대해 eager MLX가 저장하는 형태를 따릅니다.

- **디퀀트된 가중치**를 F16으로 라운딩. eager의 양자화 matmul이 F16 활성화 타입으로 디퀀트하기 때문입니다.
- **선형 곱과 바이어스 합**은 두 번 라운딩: 곱, 그 다음 `곱 + 바이어스`(eager의 `quantized_matmul` 후 `add`에 대응).
- **픽셀, 패치 임베딩, 위치 합**은 한 번 라운딩.
- **LayerNorm**은 MLX CUDA 커널을 따름: f32 통계, 정규화 값을 F16으로 라운딩, 그 다음 `w*n+b`를 한 번 라운딩(커널의 half FMA). 곱과 합을 따로 라운딩하면 더 나빠졌습니다(레이어 21 max_abs 2.81 대 2.50).
- **quick_gelu**는 F16 계수(`1.7021484375`)로 f32에서 평가하고 세 번 라운딩. `gelu_sigmoid`의 연산별 저장에 대응.
- **attention**은 f32로 계산하고 컨텍스트를 한 번 F16으로 되돌림. scores와 softmax는 f32 유지.
- **두 잔차 합**은 한 번씩 라운딩.

`local-task` 소거 실험: 선형, attention 컨텍스트, 잔차, quick_gelu, 타워 앞 라운딩 중 하나라도 빼면 게이트가 다시 실패합니다(스테이지당 2~12개 원소). 디퀘트 라운딩도 두-라운딩 LayerNorm과 함께 소거했을 때 없으면 실패합니다.

### 3.2 `round_f16`은 좁히기 변환 쌍

Metal에 BF16 타입이 없어 정수 비트 패턴으로 라운딩해야 했던 Qwen3-VL BF16 케이스(#3336)와 달리, F16은 모든 타깃에서 표현 가능합니다. `round_f16`은 stablehlo f32에서 f16으로, 다시 f32로 되돌리는 좁히기 변환 쌍이며, nearest, ties to even으로 반올림하고 MLX의 오버플로우 to inf 동작과 일치합니다. 변환 쌍은 좁히기를 거치며 NaN 페이로드를 보존하므로 NaN 가드가 필요 없습니다.

### 3.3 계약과 다이제스트 변경

`molmo_vision_operators`는 라운딩하는 연산자에 F16 출력 라운딩을 선언하고, 네 개의 scores/softmax id와 `pool.attention-context`는 면제합니다. eager는 `wo` 전에 풀 attention 컨텍스트를 F16으로 되돌리지만, 그래프는 풀 경로 전체를 f32로 유지합니다. 모든 관측 스테이지가 여유를 두고 통과하므로(vision.projector max_abs 0.1875 local-task / 0.094 cuda, 약 200까지의 레퍼런스에 대한 엔벨로프) 이는 계약이 선언한 선택으로 남습니다. 계약은 `production_artifact_contract`로 흐르므로 Molmo 매니페스트 다이제스트가 바뀌고 캐시된 Molmo 비전 아티팩트가 한 번 재컴파일됩니다. Molmo2의 이미터와 계약은 손대지 않았습니다.

## 4. 결과

GB10, `xla_molmo_reference_check`, 핀된 리비전 `5c04b3a4`, `test_image.png`, `VISION` 바운드 유지, 전후 비교.

| 스테이지 max_abs (실패 수) | main local-task | 수정 local-task | 수정 cuda |
|---|---|---|---|
| vision.patch_embedding | 1.34e-3 | 4.88e-4 | 4.88e-4 |
| vision.selected_layer_21 | 2.80 (8) | 2.50 (0) | 1.81 (0) |
| vision.selected_layer_14 | 2.69 (4) | 2.47 (0) | 1.78 (0) |
| vision.projector | 0.334 | 0.188 | 0.094 |
| prepared.sparse_add_visual_rows | 0.334 | 0.188 | 0.094 |
| decoder.prefill_logits | 2.07e-2 | 1.01e-2 | 1.10e-2 |

양쪽 디바이스에서 14/14. 수정 후 최악의 selected-layer 원소는 `local-task`에서 바운드 안쪽 0.12(이전엔 0.69 초과), 프로젝터는 바운드 안쪽 0.22에서 0.004 안쪽에서 0.22 안쪽으로. Molmo CLI greedy는 XLA 백엔드에서 eager와 동일한 16개 토큰 id를 보고합니다.

## 5. 한계

- **그래프에서 풀 경로는 f32 유지.** eager는 풀링 2D attention 경로(풀링 쿼리 평균, `wo` 전 컨텍스트 캐스트, pad-embed 합) 뒤에 F16 텐서를 저장하지만, 그래프는 f32로 유지합니다. 위의 계약 선언 선택이며 재현되는 라운딩이 아닙니다.
- **모든 타깃에 CUDA 커널 라운딩.** LayerNorm과 quick_gelu 라운딩은 MLX CUDA 커널을 따르며 Metal과 CPU 타깃 포함 모든 타깃에 적용됩니다.
- **F16 체크포인트 전용.** 라운딩 지점은 F16 체크포인트에 맞춰졌고, F32 또는 BF16 Molmo 타워에는 자체 경계가 필요합니다.
- **누적 순서 불일치.** IREE의 수축 순서는 eager와 다르며, 이 수정은 라운딩 경계만 맞춥니다.

## 6. 학습 포인트

- #3336과 같은 클래스: 낮은 정밀도 레퍼런스와 맞추려면 라운딩 위치를 맞춰야 합니다. 라운딩을 건너뛰면 레퍼런스에서 멀어지지 가까워지지 않습니다.
- 체크포인트가 낮은 정밀도 활성화로 퀀타이즈된 패밀리는 그래프의 수치 계약이 수축 정밀도뿐 아니라 활성화 dtype을 선언해야 합니다.
- 실패 채널(LayerNorm 분산을 지배하는 아웃라이어 채널)이 수정 전에 클래스를 특정해 주었습니다. 아웃라이어가 많은 채널은 정밀도 차이를 노름을 통해 증폭시킵니다.
- 레퍼런스 하네스 수리(#3365)는 수개월간 조용히 건너뛰어졌던 스테이지의 실패를 드러낼 수 있습니다. 패밀리 퀄리파이케이션은 하네스 도달 가능성 수정을 포함해야 합니다.

## 관련 커밋

| 해시 | 타입 | 메시지 |
|---|---|---|
| `db82c17` | fix | round the Molmo v1 vision tower at eager's F16 boundaries |
| `a246b06` | docs | correct the Molmo pool-path and quick_gelu user comments |
| `cdc649e` | docs | record the Molmo quick_gelu_f16 call site in the gelu inventory |

관련: #3390(이 변경), #3365(하네스 수리), #2837(이중 해시), #870(원래 퀄리파이케이션), #3336(같은 클래스, Qwen3-VL BF16), #932, #566.

## 검증

- `cargo test -p mlxcel-xla --lib -- --test-threads=1`: 540 통과, 0 실패, 21 무시. `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` 깨끗. `cargo fmt --check -p mlxcel-xla` 깨끗.
- 게이트 14/14(`local-task`와 `cuda`, gpu-lock, 대기 시간 제한), 구현 리뷰어가 독립 재실행.
- Molmo CLI greedy 패리티: eager와 XLA에서 동일한 16개 토큰 id.
