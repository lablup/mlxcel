# PR #2185: CUDA에서 Gemma 4 MTP verify를 decode와 바이트 단위로 일치시키기

**날짜**: 2026-10-07
**상태**: CUDA(GB10)에서 구현 및 검증 완료, Metal 검증은 #2158에서 진행 예정
**위험도**: 중간 (두 Gemma 4 페어의 CUDA verify 경로 변경. history-boundary prefill 분할과 `attend` 게이트 리팩터링은 Metal 31B 서빙에도 적용됨)

## 요약

CUDA에서는 Gemma 4 MTP가 한 번도 켜지지 않았다. 시작 시 block-versus-chain 정확성 프로브가 `gemma-4-12b-it-4bit`, `gemma-4-31b-it-4bit`와 각 assistant 조합 모두에서 첫 8토큰 draw부터 실패했고, 그래서 모든 페어링이 classic decode로 내려갔다. `MLXCEL_QMV_MULTIROW` x `MLXCEL_SDPA_VECTOR_LARGE_D` 귀속 행렬, `MLXCEL_COMPILED_QGELU_MLP` 축, 그리고 attention 레이어 내부의 임시 query/key 캡처로 원인 세 가지를 찾았다. 이슈가 가장 의심했던 multirow qmv 커널은 어느 셀도 바꾸지 않았다.

1. `Attention::attend`의 행 단위 verify 게이트가 31B 지오메트리에만 맞춰져 있었다. 그래서 12B는 M=4 블록을 배치로 돌렸고, head_dim 256에서 classic decode가 쓰는 단일 쿼리 `sdpa_vector` 커널을 벗어났다. 이제 `TextConfig::mtp_requires_linear_singleton`이 CUDA에서는 12B 지오메트리도 포함하며(`mtp_requires_linear_singleton_on(cuda)`로 두 백엔드의 답을 모두 테스트할 수 있다), `attend`는 생성 시점에 계산한 플래그를 보고 `verify_attention::attend_verify_rows`로 보낸다.
2. 12B의 8비트 MLP는 decode에서는 컴파일된 GeGLU 그래프를, 다중 토큰 블록에서는 eager 활성화를 탔다. 이제 `compiled_gelu_approx_mlp_forward`는 CUDA에서 8행 미만 입력에도 컴파일 그래프를 쓴다. 이 범위는 matmul 자체가 행 단위로 정확한 qmv 구간이다.
3. MLX CUDA RoPE는 `B=1, L=1`일 때 `rope_single`을, 그 외에는 일반 커널을 고르며, 두 커널의 결과는 float 마지막 비트가 다르다. CUDA의 행 단위 verify(`mtp_row_rope`)는 이제 Q와 K의 `reshape, norm, transpose, RoPE` 체인을 한 행씩 실행한다(`head_rows_like_decode`).

그래도 채팅 출력이 페어마다 256토큰 프롬프트 세 개 중 하나에서 갈라졌다. 프롬프트 캐시가 켜져 있으면 classic 서빙은 채팅 prefill을 history boundary에서 나누는데(#1143), MTP burst는 그 prefill을 건너뛰었다. 이제 스케줄러가 boundary를 계산해(`history_boundary_split`) B=1 MTP 슬라이스와 burst에 넘기고, 행 단위 prefill이 같은 분할로 forward한다(`mtp_prefill_ranges`).

## 설계 메모

- CUDA의 12B 페어는 이 predicate가 이미 걸고 있던 31B 제약을 그대로 물려받는다. B=1 선형 verify, rotating verify buffer, tree round 없음, buffered snapshot 기증 없음, 그리고 긴 buffered 프로브 draw다.
- RoPE 수정은 MLX `rope.cu`를 고치지 않고 decode가 하는 호출을 행마다 재현하는 방식이다. 커널을 고치면 모든 CUDA 모델의 classic decode 수치가 바뀐다.
- 컴파일된 GeGLU 게이트는 CUDA에서 Gemma 1/2/3의 8행 미만 입력(짧은 prefill, suffix chunk)에도 적용된다. 기존 fallback 테스트는 fallback 경로를 계속 검증하도록 16행으로 바꿨다.
- ring cursor가 없는 sliding 레이어는 키가 window 안에 들어올 때만 행 단위 경로를 탄다. 그 이상이면 버퍼 없는 ring은 decode 순서가 아니다. 서빙과 프로브는 항상 버퍼를 켜므로 이 경로는 fallback이다.
- `mtp_prefill_ranges`는 boundary 세그먼트를 한 chunk로, 나머지를 boundary부터 `prefill_chunk_size` 단위로 forward한다. `capture_history_boundary_snapshot` 다음에 classic prefill이 하는 것과 같다. 배치 burst는 boundary를 넘기지 않는다.

## GB10 검증 (CUDA, release, `--features cuda`)

드라이버 580.178.04, 커널 7.0.0-1019-nvidia, MLX pin `81ba1c6a`.

- 기본 draw(8, 8, 8, 1056)로 돌린 프로브: 두 페어, `MULTIROW` x `LARGE_D` 네 셀 모두 통과. 수정 전에는 모든 셀에서 실패했다.
- 채팅, temperature 0, 256토큰, 페어당 프롬프트 세 개, 기본 설정: MTP가 켜진 상태에서 바이트 단위로 일치. boundary 수정 전에는 페어당 3개 중 2개, `--no-cache-prompt`에서는 3개 모두 일치했다.
- `speculative_parity --ignored`: `greedy_parity_mtp_gemma4_31b`, `greedy_parity_mtp_gemma4_unified_12b`, 배치 Gemma 4 테스트 두 개, `greedy_parity_dflash_qwen35_4b` 통과. Qwen 3.5 DFlash는 여전히 #1935 사유로 거절한다.
- lib 테스트: gemma4(250), 스케줄러 burst/slice/prompt-cache 셀렉터(127), 새 core qmv 및 compiled-GeGLU 비트 단위 테스트. 두 crate 모두 clippy `-D warnings`, `cargo fmt --check` 통과.
- 처리량: classic 대 classic null arm을 포함한 interleaved 3라운드, 프롬프트 캐시 끔. 12B MTP는 classic 대비 +106%에서 +122%(13.2에서 28 tok/s), 31B는 +80%에서 +91%(8.3에서 15.8 tok/s), null 범위는 -5.1%에서 +6.2%, acceptance 0.63과 0.64. 각 페어의 모든 arm 출력 바이트가 같았다. 공유 CI 러너가 계속 바빠서 CPU 유휴 게이트는 건너뛰었다.

## 이 호스트에서 검증하지 못한 것

Metal. #2158에 올릴 공유 경로 변경: `attend` 게이트 리팩터링, cursor 없는 sliding fallback, history-boundary prefill 분할, `cuda_is_available()`를 보는 `mtp_requires_linear_singleton`.

## 후속 과제

Qwen 3.5 DFlash의 CUDA 잔차(#1935)도 같은 RoPE 커널 선택일 수 있다. 여기서는 확인하지 않았다. 첫 MTP bonus 토큰은 여전히 M=prompt LM-head projection에서 나오고 classic은 마지막 위치 projection을 쓴다. 측정한 모든 경우에 일치했다. classic decode는 같은 프롬프트를 반복할 때 프롬프트 캐시 상태에 따라 출력이 달라지는데(whole-prompt hit가 cold 요청과 다른 텍스트를 냄), MTP는 buffered snapshot을 채택하지 않으므로 이를 재현하지 않는다.
