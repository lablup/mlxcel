# PR #3405: Inkling 2 speculative verify의 바이트 단위 일치

**날짜**: 2026-11-25
**상태**: Draft PR (refs #3280). Inkling 2와 core 테스트 통과. M3 Ultra(512 GB)에서 `models/inkling-2-mini`, `inkling-2-mini-4bit`, `inkling-2-small`로 검증

## 요약

이 PR 전에는 block-versus-chain exactness probe가 Inkling 2 Mini와 Small에 동봉된 DSpark drafter를 모든 폭에서 거부했습니다. 그래서 `MLXCEL_MTP_ALLOW_INEXACT=1` 없이는 drafter가 켜지지 않았습니다. visit별 비교로 불일치 원인 다섯 가지를 찾았고, 모두 고친 뒤에는 8행 verify 블록이 decode 8단계와 비트 단위로 같습니다. 이는 8000 토큰 컨텍스트까지 확인했습니다. drafter를 켠 출력도 classic decode와 바이트 단위로 같습니다. sliding window와 prefill chunk 경계를 둘 다 넘는 2006 토큰 프롬프트에서도 같았습니다.

핵심은 새 core 모듈 `mlxcel_core::row_invariant`입니다. 행 수와 무관하게 같은 순서로 합산하는 Metal matvec 위에 만들었고, Inkling 2가 첫 사용자입니다. 커널을 쓴 decode는 MLX gemv보다 Mini에서 1.4%, Mini 4-bit에서 0.9% 빠릅니다. drafter 배율은 Mini가 코드 1.60배, 설명문 1.04배이고, Small이 각각 1.54배, 0.90배입니다.

## 1. 문제

temperature 0의 speculative decoding은 classic decode와 같은 텍스트를 내야 합니다. mlxcel은 이를 probe로 강제합니다. 같은 캐시에서 1토큰 단계 K번과 K행 블록 한 번을 실행해 logits 바이트를 비교하고, 다르면 drafter를 거부합니다. Inkling 2는 두 체크포인트 모두 폭 2와 8에서 probe에 실패했습니다. `qmv_wide`를 꺼도 실패했으므로 양자화 expert matvec은 원인이 아니었습니다.

## 2. 진단

`exactness_diag_tests.rs`(ignored, 실제 체크포인트 사용)는 블록의 각 행과 chain의 각 단계에서 모든 layer visit에 들어가는 hidden state를 캡처하고, 행별로 다른 원소 수를 출력합니다. 이 도구로 원인을 하나씩 찾았습니다.

| # | 원인 | 처음 보인 지점 | 수정 |
|---|---|---|---|
| 1 | MLX의 `matmul`은 1행이면 gemv, 여러 행이면 gemm을 쓰고, 두 커널은 합산 순서가 다릅니다. weight가 공유되면 batch 축이 M으로 접히므로 행을 batch로 reshape해도 소용없습니다. | 행 1 이후는 layer 0부터, 행 0은 layer 6 이후 | 모든 dense 곱을 `row_invariant::dense`로 |
| 2 | query별 zero sink 열이 최신 key 뒤에 있었습니다. 단계에서는 인덱스 N+1, 블록에서는 N+K라서 행 0도 SDPA lane 배정이 바뀝니다. | layer 13 SDPA, 원소 1개 차이 | sink 열을 앞에 둠 |
| 3 | 루프 코어의 message read `query @ bank^T`가 블록 길이에 대한 batched matmul입니다. | visit 18 | verify 블록 안에서 위치별로 계산 |
| 4 | sliding layer의 1024 토큰 window를 넘으면, 블록의 행 r은 decode 단계 r이 이미 버린 오래된 key r개를 mask된 상태로 함께 봅니다. | 1024보다 긴 프롬프트 뒤 행 1 이후 | 행마다 그 행의 window 시작점부터 attention |
| 5 | DSpark prefill은 프롬프트를 forward 한 번에 넣었지만, classic decode는 2048 토큰 chunk로 prefill합니다. | 2048보다 긴 프롬프트 | classic과 같은 chunk로 prefill |

원인 4와 5는 32토큰 프롬프트를 쓰는 probe로는 보이지 않았습니다. 4는 window가 4 토큰인 tiny 모델 테스트에서, 5는 긴 프롬프트를 쓴 end-to-end 실행에서 드러났습니다.

조사 중 원인이 아닌 것으로 확인한 것: NVFP4/MXFP8 expert 커널(이미 행 불변), #3359 Mini 커널, short convolution, speculative append scope.

## 3. 설계

### 3.1 `row_invariant_matvec` (`src/lib/mlxcelverse/kernels/row_invariant_matvec.{h,cpp}`)

1~16행의 `x [M, K] @ w [N, K]^T`를 계산합니다. bf16, f16, f32를 지원하고 `K % 4 == 0`이어야 합니다. 행마다 f32 누산기를 따로 두고 각 lane이 `k = 4 * lane + 128 * j`를 고정된 순서로 돌기 때문에, M행 출력이 1행 출력 M개와 같습니다. weight는 행마다가 아니라 블록당 한 번 읽습니다.

변형은 두 가지이고, 둘 다 M이 아니라 N만으로 고릅니다.

- **Wide** (N ≥ 512): 출력 열당 SIMD 그룹 하나. 시도한 tiling 중 가장 빨랐고, 그룹당 2, 4, 8열은 실제 decode에서 더 느렸습니다.
- **Split** (N < 512, router와 controller down projection): SIMD 그룹 8개가 열 하나를 나눠 계산하고 부분합을 그룹 순서대로 더합니다. 변형이 하나뿐이던 첫 버전은 이 좁은 지점들 때문에 decode가 2.5~3.4% 느렸습니다.

### 3.2 `mlxcel_core::row_invariant`

모델이 opt-in하는 방식입니다. decode와 verify를 가리지 않고 모든 forward를 `Scope::enter(Mode { kernel, verify_block })`로 감싸고, 곱은 다음 helper로 만듭니다.

- `dense` / `linear`
- `per_row`: 커널이 받지 않는 양자화 projection용
- `per_position`: 커널 선택이 블록 길이에 따라 달라지는 그 밖의 곱

scope 밖에서는 이 helper들이 일반 `matmul`, `forward`와 같으므로 다른 모델은 바뀌지 않습니다. 이 opt-in 경계는 의도한 것입니다. 커널을 `UnifiedLinear::forward`에 넣으면 모든 모델의 decode 비트와 속도가 한꺼번에 바뀝니다. `kernel`이 꺼져 있거나 Metal이 아니면 verify 블록은 한 행씩 곱합니다. 결과는 정확하지만 weight를 K번 읽어서, 8행 Mini verify가 51.2 ms 대신 80.6 ms 걸립니다.

### 3.3 Inkling 2 연결

- `Inkling2Model`은 build 시점에 `row_kernel = kernels::enabled()`를 저장하고, `hidden_rows`와 `project`에서 scope에 들어갑니다.
- attention projection 7개, router, message controller, dynamic conv projection, LM head가 helper를 씁니다.
- 양자화되지 않은 dense expert는 `gather_mm`도 행 수에 따라 커널을 바꾸므로, verify 블록 안에서 토큰별로 나눕니다.
- sliding window 규칙은 `attention.rs::attend`에 있고, 공통 SDPA 본문은 `sdpa()`로 옮겼습니다.
- text와 VL speculative target은 모두 `Inkling2Model::prefill_speculative`를 거칩니다. chunk마다 평가하고, 마지막 chunk의 logits만 남기고, 캡처는 이어 붙입니다.

### 3.4 빌드 수정

`mlxcel-core/build.rs`의 `rerun-if-changed`에 Inkling 2 커널 `.cpp` 파일이 없어서, 파일을 고쳐도 다시 빌드되지 않았습니다. #3369부터 그런 상태였습니다. 이것을 찾기 전까지 이 PR의 커널 측정 두 차례가 옛 코드로 조용히 실행됐습니다.

## 4. 결과

M3 Ultra, GPU lock 아래 직렬 실행, ABAB 순서, greedy 256토큰. 전체 표는 `docs/benchmark_results/inkling2-verify-exactness-m3ultra-2026-11-24.md`에 있습니다.

| 항목 | 결과 |
|---|---|
| Probe, 폭 2/4/8 | Mini, Small, Mini 4-bit 모두 일치 |
| 긴 프롬프트 뒤 블록 대 chain | 1100, 3000, 8000(Mini), 1100(Small, 4-bit)에서 일치 |
| Decode, 커널 대 MLX gemv | Mini +1.4%, 4-bit +0.9% |
| Mini 8행 verify | 51.2 ms (이전 불일치 경로 52.1 ms) |
| DSpark Mini, 설명문 / 코드 | 1.04배 / 1.60배, 텍스트는 classic과 같음 |
| DSpark Small, 설명문 / 코드 | 0.90배 / 1.54배, 텍스트는 classic과 같음 |
| DSpark Mini, 2006토큰 프롬프트 | 1.00배, 텍스트는 classic과 같음 |
| mlx-vlm 텍스트 trace, Mini 4-bit | top-1 31/1024, decided 위치 0/372 (이전: 33, 0) |

Small은 verify 라운드 하나가 약 120 ms로 decode 한 단계(51 ms)보다 훨씬 비싸고, 그 비용의 대부분은 expert 경로입니다. 설명문 라운드는 2.3토큰을 내는데, 이것으로는 라운드 비용을 메우지 못합니다.

## 5. 테스트

- `row_invariant_tests.rs` (core):
  - 커널 곱의 각 행이 그 행만의 1행 곱과 같은지, 두 변형과 f32·bf16 모두에서 확인
  - 결과값이 f64 내적과 맞는지 확인
  - 계약 밖의 shape를 거부하는지 확인
  - scope 동작 확인
- `the_verify_block_matches_the_chain_on_metal`: tiny 모델이 폭 2/4/8에서 루프 있는 구성과 없는 구성 모두 probe를 통과하는지 확인합니다. 실제 체크포인트와 같은 fused SDPA 커널을 타도록 tiny 모델의 `head_dim`을 8에서 64로 바꿨습니다. 8일 때는 MLX가 어떤 실제 체크포인트도 쓰지 않는 unfused 경로로 빠집니다.
- Inkling 2 lib 테스트 173개 통과.

## 6. 한계와 후속

- **probe 프롬프트 길이.** probe는 여전히 짧은 프롬프트를 씁니다. 긴 컨텍스트에서 생기는 회귀는 probe로 잡을 수 없고, 진단 테스트(`INKLING2_DIAG_PROMPT`)로 직접 확인해야 합니다.
- **서버 prefill.** 서버 classic decode는 `--prefill-chunk-size`로 prefill하지만, DSpark prefill은 `MLXCEL_PREFILL_CHUNK`를 따릅니다. 그래서 기본값이 아닌 chunk 크기로 띄운 서버에서는 chunk보다 긴 프롬프트에 이전 차이가 남습니다.
- **양자화 attention projection** (affine 4-bit 체크포인트)은 verify 블록 안에서 한 행씩 계산합니다. 이 체크포인트들에는 drafter가 동봉돼 있지 않으므로 그 비용은 prompt lookup에서만 생깁니다.
- **일반화.** drafter를 지원하는 다른 모델들은 후속 이슈에서 조사합니다. probe가 어느 모델을 거부하는지, `row_invariant`로 고칠 수 있는지를 봅니다. Small의 verify 비용도 같은 후속 이슈에서 다룹니다.
