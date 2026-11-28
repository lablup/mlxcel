# 기술 보고서: PR #3085 - rotating cache의 첫 fill에서 window만 보관

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 중간

## 요약

`RotatingKVCache`는 다중 토큰 첫 fill에서 프롬프트 토큰을 전부 저장했고, 첫 decode 단계에서야 window 크기로 잘랐다. 그래서 단일 패스 prefill 동안 모든 sliding-window 층이 프롬프트 전체를 들고 있었다. 이제 첫 fill은 가장 최근 `max_size + 32`개 위치만 새 buffer로 저장하고, 공용 deferred-error `async_eval`로 미리 평가를 예약한다. 해당 층의 attention은 여전히 전체 K/V를 받는다. M5 Max에서 28.7K 토큰 기준 단일 패스 피크가 gemma-4-12b는 27.38 GB에서 18.67 GB로, gemma-3-4b는 15.14 GB에서 7.03 GB로 줄었다. greedy 출력은 바이트 단위로 같고 prefill 시간도 그대로다.

## 1. 문제 정의

gemma-4-12b는 window가 1024 토큰인 sliding 층이 40개다. 28.7K 토큰에서 이 층들은 이후 어느 층도 읽지 않는 K/V를 약 8.75 GiB 들고 있었다(#3062). sliding 층에 `RotatingKVCache`를 쓰는 모든 계열에 같은 문제가 있었다.

## 3. 기술적 선택과 그 이유

- **보관 길이는 `max_size`가 아니라 `max_size + 32`다.** 첫 fill 뒤 speculative verify와 padded-prefill rewind는 최대 NA tile 하나(32개 위치)까지 trim한다. 32개를 더 두면 저장된 buffer가 chunked prefill이 이어질 때 남는 것과 정확히 같은 시간순 배치가 된다. 따라서 decode, 이어지는 append, `trim`, `rewind_padded_prefill`, `enable_speculative_buffer`, snapshot 복원, Turbo4Asym sidecar가 전체 fill일 때와 똑같은 값을 돌려준다. Int8, Turbo4, Turbo4Delegated, Turbo3Asym은 FP16 경로를 탄다.
- **서버의 ragged batched-prefill cohort는 전체 fill을 유지한다.** 서버는 짧은 행을 길이 차이만큼 통째로 rewind하는데, 이 차이는 32를 넘을 수 있다. 어떤 행이 32개 넘게 padding될 때만 들어가는 thread-local scope인 `FullFirstFillScope`가 그 forward에 한해 기존처럼 전체를 저장한다. `rewind_padded_prefill`은 이제 잘라낸 위치가 드러나는 rewind를 조용히 window를 망가뜨리는 대신 오류로 거부한다.
- **헬퍼는 복사하지 않고 하나로 둔다.** `materialized_axis_tail(x, axis, n_keep)`은 #3063의 `materialized_state_tail`을 cache의 시간 축(2)으로 일반화한 것이고, deferred-error 처리를 함께 쓴다.

## 7. 변경 요약

| 모델 | prefill | 피크, main | 피크, 브랜치 | 출력 |
|---|---|---|---|---|
| gemma-4-12b-it-4bit | 단일 패스 | 27.38 GB | 18.67 GB | 동일 |
| gemma-4-12b-it-4bit | 2048 청크 | 14.17 GB | 14.17 GB | 동일 |
| gemma-3-4b-it-4bit | 단일 패스 | 15.14 GB | 7.03 GB | 동일 |
| gemma-3-4b-it-4bit | 2048 청크 | 5.25 GB | 5.25 GB | 동일 |

테스트:
- 새 `cache::first_fill_tests` 9개는 잘라 저장한 cache와 전체를 저장한 cache에 같은 연산을 수행하고, FP16, Int8, Turbo4Asym에서 K/V를 정확히 비교한다.
- 메모리 테스트 2개는 잘라 저장한 첫 fill이 프롬프트를 놓아주는지, scope 안의 대조군은 그대로 붙잡는지 확인한다.
- `server::batch`(610개)와 gemma3, gemma4, kv_snapshot, afmoe 모델 테스트가 통과한다.

자세한 내용: `docs/benchmark_results/rotating-first-fill-window-m5max-2026-11-15.md`.

## 8. 후속 조치

- scheduler의 ragged-cohort scope를 모델까지 거쳐 검증하는 unit test는 없고, 실제 checkpoint로 서버나 MTP를 end-to-end로 돌려 보지도 않았다.
