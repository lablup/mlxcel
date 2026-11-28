# PR #2203: 행 단위 지오메트리에서 Gemma 4 배치 MTP 검증을 디코드와 바이트 단위로 일치시킴

**날짜**: 2026-10-07
**상태**: CUDA(GB10)에서 구현 및 검증 완료, Metal 검증은 #2158에서 진행 예정
**위험도**: 중간 (공용 Gemma 4 forward에 `mtp_verify`와 행 단위 레이어 플래그로 게이트된 배치 행별 분기 추가, CUDA 31B의 배치 MTP 서빙은 `MLXCEL_ENABLE_MTP_BATCH` 뒤에 있음)

## 요약

PR #2185는 CUDA에서 B=1 선형 Gemma 4 MTP 어댑터를 클래식 디코드와 바이트 단위로 일치시켰다. 배치 어댑터(`Gemma4MtpBatchedTargetAdapter`)는 `sinks.mtp_verify`를 설정하지 않아 검증이 행 단위 경로를 타지 않았고, 서빙은 31B와 CUDA 12B의 모든 B>1 윈도우를 거절했다. main에서 `b1_batched_baseline_probe`가 실패했다(31B 1, 2, 3행이 각각 1, 9, 17번째 토큰에서 불일치).

1. 배치 어댑터는 선형 어댑터와 마찬가지로 검증 forward에만 표시를 하고 prefill에는 하지 않는다.
2. 행 단위 지오메트리에서는 각 행을 자기 캐시에서 클래식 분할(`mtp_prefill_ranges`, `prefill_mtp_chunk_explicit_cache`의 마지막 행 LM head)대로 prefill하고, 행 캐시들을 각 행의 기록을 앞에, 부족분을 0으로 채운 꼬리로 두어 공유 `[B, ...]` 캐시로 쌓는다(`stack_prefilled_rows`). 짧은 행은 유효 끝이 공유 오프셋보다 뒤처진 행이 되며, 이는 분기된 수락이 이미 남기는 배치와 같다. 따라서 길이가 다른 버스트에도 왼쪽 패딩이 필요 없고 모든 행이 단독 실행과 같은 RoPE 프레임을 유지한다.
3. B>1 검증은 각 행을 `[1, K]` 호출로 따로 실행한다: 투영, 행의 논리 오프셋에서의 `head_rows_like_decode` RoPE, 기록 `[0, ve[r])`과 블록만 물리적으로 잘라낸 키에 대한 `attend_verify_rows`, `o_proj`, MLP(`finish_layer`), LM head. 공유 캐시로의 K/V 쓰기만 배치로 남기 때문에 finalize, rollback, 드래프터 슬랩 계약은 바뀌지 않는다. 슬라이딩 레이어는 행 자신의 길이에서의 링 커서 `(cursor - (offset - ve[r])) mod window`를 쓴다.
4. `run_mtp_burst_batched`는 무조건적인 행 단위 거절을 `RowWiseBatchedWindow::decline_reason`으로 바꾸고, 배치 윈도우는 행별 history boundary 분할을 전달한다.

## 설계 메모

- 기각: `b == 1` 게이트를 `[B, K]` 호출로 넓히는 방법. `B * K >= 8`이면 양자화 matmul이 qmv에서 qmm으로 바뀌고 컴파일된 GeGLU 게이트도 같은 행 수를 센다.
- 한계: 버퍼된 슬라이딩 캐시는 공유 오프셋 기준으로 압축된다(`buffered_planned_drop`). 압축이 일어나면 그 오프셋보다 뒤처진 행은 자기 윈도우의 가장 오래된 키를 잃는다. forward는 이를 보고하고(`row_verify_inexact`) 어댑터는 클래식 디코드가 내지 않을 토큰을 내보내는 대신 `DraftFailed`를 반환한다. 서빙 게이트는 윈도우를 그 아래로 유지한다: `max_prompt_len + K * (max_tokens + 1) <= sliding_window + 32`. `K * max_tokens`는 먼저 끝난 행이 라운드마다 최대 K 위치씩 계속 전진하는 경우를 덮는다.
- 게이트: CUDA 전용(Metal 31B도 행 단위지만 측정되지 않았고, 같은 이유로 ROCm 빌드도 제외), B <= 4(측정한 폭), 밀집 FP16 캐시(스태커 요구 사항), 슬라이딩 윈도우 안의 프롬프트. 그 밖의 윈도우는 드래프터 IO 전에 거절되므로 어떤 행도 클라이언트 오류에 도달하지 않는다.
- 결정은 측정에서 나왔다: GB10에서 31B 클래식 배치 디코드는 B에 따라 거의 늘지 않는다(B=1, 2, 4에서 8.3, 8.8, 8.9 tok/s). 그래서 라운드당 행 하나짜리 검증 약 B번의 비용이 드는 행별 검증도 이긴다.

## GB10 검증 (CUDA, release, `--features cuda`)

드라이버 580.178.04, 커널 7.0.0-1019-nvidia, MLX 핀 `81ba1c6a`.

- 전체 `speculative_parity --ignored --test-threads=1`: 9개 통과, 0개 실패. 두 쌍 모두에서 돈 `b1_batched_baseline_probe`와 새 `greedy_parity_mtp_gemma4_batched_matches_classic`(B=2와 B=4, 같은 길이와 다른 길이, 두 쌍, 24개 행 모두 시퀀스별 클래식 greedy와 바이트 일치, near-tie 허용 없음, 모든 경우에 분기 수락 라운드 포함)을 포함한다. main: 8개 중 7개와 실패하는 probe.
- 라이브러리 `gemma4 speculative_burst`: 327개 통과. 밀폐형 행별 테스트는 공유 링 커서 변이와 행 모드 비활성화 변이에서 모두 실패한다.
- 처리량, 폭마다 교차 3라운드, 클래식 null 암, 모든 암에서 출력 바이트 동일: 31B B=2 +75%~+77%, B=4 +53%~+55%, null -2.3%~+3.1%. 배포 바이너리로 돌린 라운드에서도 배치 윈도우가 형성되었고 바이트가 같았다.
- `MLX_CUDA_GRAPH_CACHE_SIZE=100`에서 그래프 캐시: B=1(4000 토큰)과 B=2(8000 토큰) 각각 20배치를 수명 미스 중단 없이 처리했다.
- Clippy `-D warnings`와 `cargo fmt --check` 통과.

## 변경 범위 밖의 발견

기본 서빙에서는 배치 MTP 윈도우가 형성되지 않는다: `--max-batch-prefill`이 1보다 크면(기본값) 대기 중인 요청 둘 이상이 클래식 배치 prefill로 가고, `--ignore-eos`는 모든 행에 토큰 bias를 주는데 윈도우 수집기가 이를 거부하며, seed가 없는 요청은 서로 다른 seed를 받는다. 12B(`Gemma4Unified`)는 `max_batch_size`를 1로 고정하므로 결정과 무관하다. 프로세스 내 parity 테스트는 서버의 2000이 아니라 MLX 기본값 400짜리 그래프 캐시로 돌고 있었고, 이제 서버 기본값을 적용한다.

## 이 호스트에서 검증하지 못한 것

Metal. #2158에 목록을 남겼다: 배치 어댑터의 검증 플래그, 행별 prefill과 스태킹, B>1의 행별 검증과 LM head, `finish_layer` 추출. 서빙 게이트는 Metal을 계속 거절한다.
