# 기술 보고서: PR #2137 - perf(moe): SwitchGLU 프로젝션 간 gather 인덱스 공유

**날짜**: 2026-10-06
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (그래프 및 greedy 출력 게이트는 CUDA에서 검증, Metal 처리량은 미측정)
**언어**: Rust, C++ (브리지 함수 1개)
**위험도**: 낮음 (공유 헬퍼의 그래프 형태 변경이며, 커널이 받는 인덱스 값은 동일)

---

## 요약

이슈 #1713은 decode 단계에서 호스트가 그래프를 빌드하는 비용이 MLX 연산 수에 비례한다는 점을 확인했고, op를 줄일 첫 대상으로 공유 MoE 헬퍼 `SwitchGLU`를 지목했습니다. 실제 비용은 헬퍼 자체의 `expand_dims`보다 MLX 내부에 있었습니다. `lhs_indices` 없이 `gather_qmm` / `gather_mm`을 호출하면 MLX가 호출마다 `arange -> reshape -> broadcast`를 새로 만들고, expert id가 `uint32`가 아니면 캐스트도 추가합니다. `SwitchGLU`는 레이어마다 이런 호출을 세 번 했습니다. PR #2137은 블록당 인덱스 배열을 한 번만 만들어 넘기고, `expand_dims` 쌍을 노드 하나로 합칩니다. 그 결과 decode 그래프에서 MoE 레이어당 노드 4개와 엣지 7개가 줄었고, 시험한 모든 모델에서 greedy 출력이 바이트 단위로 같았습니다.

---

## 1. 문제 정의

### 1.1 배경

이슈의 측정에서 `forward`(호스트 그래프 빌드)는 모델 크기와 관계없이 거의 일정했고, 디바이스 시간만 모델 크기에 따라 늘었습니다. 그래서 작은 모델과 SSM/MoE 하이브리드의 손해가 가장 큽니다. granite-4.0-h-tiny-4bit는 레이어당 231개 노드를 만들어 qwen3-8b의 89개보다 많았고, 초과분은 Broadcast, AsType, ExpandDims, Arange, Full, Reshape, Slice에 몰려 있었습니다.

### 1.2 원인 귀속

MLX 소스(`ops.cpp`)에서 `gather_qmm`과 `gather_mm`은 두 인덱스 인자 모두에 `indices_or_default`를 호출합니다. 이 함수는 인덱스가 주어지면 `astype(indices, uint32)`를, 주어지지 않으면 `reshape(arange(total, uint32), batch_shape)`를 반환하고, 그 결과는 다시 `broadcast_arrays`를 거칩니다. GB10에서 내보낸 decode 그래프도 이를 확인해 주었습니다. granite-tiny의 Arange 120개는 gather 호출 3회 x 40 레이어와 정확히 일치하고, ExpandDims 120개 중 80개는 `SwitchGLU`가 입력에 `expand_dims`를 두 번 적용한 결과입니다.

---

## 2. 기술적 결정

### 2.1 MLX가 만들었을 인덱스를 그대로 전달

`prepare_gather_indices`는 `rhs = astype(ids, uint32)`와 `lhs = broadcast_to(reshape(arange_u32(total), batch), broadcast_shape(batch, ids))`를 만듭니다. MLX의 `astype`, `reshape`, `broadcast_to`는 dtype이나 shape가 이미 같으면 입력을 그대로 돌려주므로, 이 배열을 넘겨도 MLX 내부에서는 노드가 추가되지 않습니다. gate와 up은 lhs를 공유합니다. 비정렬 경로에서 down은 입력 batch가 `[n, 1]`이 아니라 `[n, k]`이므로 MLX 기본 lhs를 그대로 씁니다. 정렬된 prefill 경로에서는 세 프로젝션 모두 lhs 하나를 공유합니다. shape가 broadcast되지 않으면 `lhs`는 `None`이고, 이때 MLX는 이전과 똑같이 검증하고 오류를 냅니다.

### 2.2 브리지에 uint32 arange 추가

첫 버전은 `arange_i32`에 캐스트를 더해 lhs를 만들었기 때문에, 다른 곳에서 노드 하나를 줄일 때마다 AsType 노드가 하나씩 늘었습니다(측정값: granite-tiny에서 AsType 160 -> 200). 새 브리지 함수 `arange_u32(stop)`은 `indices_or_default`와 같은 호출인 `mlx::core::arange(stop, uint32)`를 사용하므로, 레이어당 네 번째 노드도 줄일 수 있었습니다.

### 2.3 범위는 공유 헬퍼로 한정

`SwitchLinear::forward`는 시그니처를 유지하고 `lhs = None`으로 위임합니다. `SwitchLinear`를 직접 호출하는 15개 패밀리(DeepSeek, GptOss, Phixtral, Qwen3Next 등)는 후속 작업에서 `forward_indexed`를 채택할 수 있습니다. Granite에 남은 Full, Slice, AsType 초과분은 SSM과 attention 경로에 있으며, 이슈에서 범위 밖으로 정한 부분입니다.

---

## 3. 검증 (GB10, CUDA)

| 모델 | 노드 | 엣지 |
|---|--:|--:|
| granite-4.0-h-tiny-4bit | 3419 -> 3259 | 9057 -> 8777 |
| qwen3-30b-a3b-4bit, fused MoE 끔 | 2895 -> 2703 | 7575 -> 7239 |
| qwen3-30b-a3b-4bit, 기본값 (fused decode 커널) | 변화 없음 | 변화 없음 |
| granite-4.0-h-350m-4bit (expert 없음) | 변화 없음 | 변화 없음 |

- greedy `--temp 0` 출력은 granite-4.0-h-350m, granite-4.0-h-tiny, qwen3-30b-a3b, qwen3.5-35b-a3b에서 fused MoE를 켠 경우와 끈 경우 모두 base와 new 바이너리가 바이트 단위로 같았습니다. base를 다시 실행한 결과도 첫 실행과 같았고, 사용한 프롬프트는 정렬(prefill) 경로와 비정렬(decode) 경로를 모두 지납니다.
- 새 단위 테스트는 준비된 인덱스를 MLX 기본값과 비교하고, 블록 출력을 호출별 형태와 비트 단위로 비교합니다. 비교 범위는 decode, 작은 prefill, 정렬된 prefill shape이며, int32와 uint32 id 및 Inkling expert-scale 경로를 포함합니다.
- 처리량: null arm을 포함해 3라운드씩 교차 측정했습니다. null arm은 한 라운드 안에서 base와 최대 34%까지 차이가 났고, base/new 중앙값은 그 범위 안에 있었습니다. 기대 이득(granite-tiny에서 토큰당 약 0.05 ms)이 이 호스트의 노이즈보다 작으므로, PR은 처리량 개선을 주장하지 않습니다.

## 4. 미검증 항목

`benchmarks/pylm_m5max_2026-09-06.csv` 대비 M5 Max tok/s 비율과 Metal `forward` ms/token은 Apple Silicon에서만 측정할 수 있습니다.
