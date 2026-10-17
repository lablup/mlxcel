# PR #2440: 배치 DFlash의 행별 accept 개수를 하나로 맞춤(clamp)

**작성일**: 2026-10-17
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. Qwen 3.5의 행별 RoPE / divergent mask 포팅은 다루지 않음 ("미해결 항목" 참조)
**위험도**: 중간 (Qwen 3.5 배치 DFlash 라운드마다 확정되는 accept 개수가 바뀌고, 행별 zero-tail 롤백이 제거됨)

## 요약

#2353을 닫는다. Qwen 3.5 / 3.6 / 3.8(`qwen3_5`, `qwen3_5_moe`, 및 VL 래퍼)의 배치(B > 1) DFlash 추측 디코딩이 이제 토큰을 내보내기 전에 모든 라운드를 가장 짧은 활성 행의 accept 개수로 잘라낸다. 그래서 emit, bonus 토큰, drafter hidden slice, 캐시 롤백이 모두 같은 개수를 본다. B = 4 출력이 128 토큰 동안 B = 1 실행 4회와 일치한다.

## 문제

Qwen 3.5의 배치 DFlash는 한 라운드에서 행마다 accept한 draft 토큰 수가 다르면 B = 1 결과와 달라졌다. 롤백은 직사각형 attention 캐시를 가장 긴 행에 맞춰 자르고, 짧은 행의 거부된 K/V 꼬리를 0으로 채웠다. 0으로 채운 키도 여전히 attention 대상이다. `q . 0 = 0`이므로 softmax 분모에 `exp(0)` 가중치로 들어가고, 이후 모든 스텝에서 그 행의 attention을 희석한다. 출력은 여전히 자연스러워서 이상을 알아채기 어렵다. GDN 상태 재생과 drafter의 hidden context도 가장 긴 행 기준으로 잡혀 있었다.

#2353에 따르면 Qwen 3.5 verify forward(`qwen3_5.rs:2832`, `:2844`)는 마스크를 공유 캐시 offset에서 만들고, 행별 유효 끝 위치나 행별 RoPE offset을 받지 않는다. B > 1 프로덕션 타깃 두 곳(`src/models/qwen3_5.rs:2905`, `src/vision/qwen3_5_vl.rs:226`)이 이 롤백을 거치며, 서버는 `src/server/batch/dflash_target.rs`를 통해 도달한다.

upstream은 같은 버그를 Blaizzy/mlx-vlm#2113(커밋 `3b8f727d`, "Clamp ragged batched speculative acceptance to fix phantom KV")에서 고쳤다. 토큰을 내보내기 전에 행별 accept를 균일한 개수로 clamp하고, 행마다 다른 배치가 직사각형 캐시 롤백에 도달하면 에러를 낸다.

## 기술적 결정

- 마스킹 대신 clamp. 이슈는 Gemma 4의 마스킹(`mask_stale_key_gap`, `src/lib/mlxcel-core/src/utils.rs:618`) 포팅을 채택하지 않았다. Gemma 4는 행마다 연속된 유효 prefix를 유지하고 각 행의 윈도우를 자기 논리 위치에서 회전시키므로(`src/models/gemma4.rs:1666`의 `DivergentVerifyRows`, `:4201`의 `build_divergent_verify_mask`) 마스킹이 가능하다. 그렇지 않으면 이후 토큰이 공유 offset으로 회전되어 간격이 라운드마다 누적된다. Qwen 3.5는 이를 전혀 추적하지 않는다. clamp가 나중에 측정 가능한 처리량 손실을 일으키면 Gemma 4 레이아웃을 후속 작업으로 포팅할 수 있다.
- clamp는 greedy 디코딩에서 정확하다. 더 긴 행에서 잘려 나가는 마지막 토큰은 target argmax와 같은 accept된 draft 토큰이다.
- 종료된 행은 개수를 낮추지 않지만, 배치에 남아 있으므로 그 개수에 맞춘다.
- 예산 절단은 walk 안에서 일어나므로 예산이 토큰 1개 남은 행이 있으면 균일 accept는 0이 된다. 이 경우 모든 행이 토큰 1개를 내보내고, 다음 라운드의 bonus는 그 토큰이다.
- walk의 실제 개수는 `accept_lens`에 그대로 남아 acceptance 통계가 정직하게 유지된다. 확정된 개수와 clamp된 라운드 수는 따로 보고한다.

## 변경 사항

- `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`: walk 옆의 순수 함수 `clamp_uniform_acceptance`(Blaizzy/mlx-vlm#2113 포팅)가 emit 루프 전에 모든 라운드를 가장 짧은 활성 행의 accept로 자른다. `DFlashBatchedRunOutput`에 `committed_accept_lens`와 `clamped_rounds`가 추가된다. 루프는 롤백 전에 clamp 결과가 단일 accept 개수인지 debug assert로 확인한다.
- `src/models/qwen3_5.rs`: `rollback_speculative_cache`가 꼬리를 0으로 채우는 대신 균일한 slice인지 assert한다(`assert_uniform_batched_accepts`). `zero_per_row_kv_tail` / `rebuild_with_zero_tail`과 그 테스트는 제거되고, conv-state 복구는 모든 행에 하나의 윈도우를 쓴다.
- `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`: `rollback_partial_batched`의 trait 문서가 행별 꼬리 zeroing 대신 균일 accept 계약을 명시한다.
- `src/server/batch/dflash_target.rs`: `run_dflash_batched_on_target`이 walk 개수와 확정 개수를 debug 레벨로 로그한다.
- `src/vision/qwen3_5_vl.rs`: 롤백이 균일한 accept slice를 요구한다는 문서 주석으로 갱신. `src/models/qwen3_5_tests.rs`: zero-tail 테스트를 균일 accept 롤백 테스트로 교체.
- `tests/speculative_parity/qwen35_dflash_batched.rs`(신규), `tests/speculative_parity.rs`: ignored 실제 체크포인트 parity 테스트 `greedy_parity_dflash_qwen35_batched_b4_matches_b1`.
- `CHANGELOG.md`: 수정 항목 추가.

적용 범위: DFlash 배치 루프에 도달하는 것은 `Qwen35Model`과 `Qwen35VLModel`뿐이다. LFM2, LFM2-VL, Muse Glimmer, Laguna는 `supports_batched() == false`를 반환하며 trait의 B > 1 panic을 그대로 유지한다. Gemma 4 배치 MTP는 행별 마스킹이 있는 별도 루프 `MtpBatchedGenerator`를 쓰므로 건드리지 않았다.

## 동작 변화

Qwen 3.5 배치 DFlash 라운드는 행들의 accept가 다를 때 라운드당 확정되는 accept된 draft 수가 줄어들고, 그 대신 출력이 B = 1과 일치한다. `rollback_speculative_cache`는 B > 1에서 행마다 다른 accept slice를 받으면 꼬리를 0으로 채우는 대신 panic한다.

## 검증

GB10, `MLXCEL_SDPA_DETERMINISTIC=1`, gpu-lock 하에서 실행. 체크포인트는 `~/models_archive/mlx/qwen3.5-4b-4bit`(target)와 `~/models_archive/mlx/qwen3.5-4b-dflash`(drafter)이다.

- Parity 테스트, block 16, 128 토큰: B = 1 self-parity 성립. 수정 전 코드에서는 행 0, 1, 2가 토큰 14, 54, 37에서 처음 갈라진다. 수정 후에는 4개 행이 모두 일치한다.
- 해당 프롬프트의 행-라운드당 평균 accept된 draft 수(정확성 확인 실행이며 벤치마크가 아님):

| 실행 | 평균 accept | 라운드 수 |
|---|---|---|
| B = 1, 프롬프트 4개 | 4.12 / 3.23 / 3.41 / 3.00 | 해당 없음 |
| B = 4 수정 전 | 1.08 | 101 |
| B = 4 수정 후, walk | 4.11 | 57 (55 clamp) |
| B = 4 수정 후, 확정 | 1.23 | 57 (55 clamp) |

- 서버: `mlxcel-server --model <target> --model-draft <drafter> --draft-kind dflash --draft-block-size 4 --max-batch-prefill 1 --no-prompt-cache`. temperature 0의 동시 `/v1/chat/completions` 2건이 하나의 B = 2 배치 burst로 실행됐다. 수정 전에는 순차 요청 2건과 결과가 달랐고, 수정 후에는 일치한다(2회 확인, 45 라운드 중 21 라운드 clamp). block 16에서도 일치한다.
- `rollback_speculative_cache_rewinds_uniform_batched_accepts`는 2층 합성 모델(gated-delta와 full attention)을 만든다. B = 2 롤백의 각 행이 해당 행의 B = 1 롤백과 같은 conv 윈도우 및 gated-delta 상태를 갖는지, KV offset이 일치하는지 확인한다.
- 되돌림 검사: clamp 호출을 끄면 합성 round-loop 테스트 3개가 실패한다. assert를 제거하면 `rollback_speculative_cache_panics_on_ragged_batched_accepts`가 실패한다.
- `drafter::dflash`(94개)와 `models::qwen3_5_tests`(ignored 포함)가 통과하고, mlxcel-core 및 mlxcel lib+tests의 clippy `-D warnings`, fmt, contract 검사도 통과한다.

## 미해결 항목

- Qwen 3.5의 행별 RoPE / divergent mask 포팅(#2353에서 범위 밖).
- 기본값 `--max-batch-prefill 4`에서는 함께 대기열에 들어온 길이가 같은 요청 2건이 일반 배치 prefill을 타고 배치 DFlash burst에 도달하지 않는다. 이 수정과는 별개의 문제다.
- 배치 burst는 B = 1 경로가 수행하는 block 대 chain 정확도 probe를 건너뛴다. 이 또한 이 수정과는 별개다.

## 통계

9개 파일 변경, 810줄 추가, 333줄 삭제. 커밋: `0b7ba553` fix(speculative): clamp ragged batched DFlash accepts to one count, `10b8bc29` test(speculative): compare batched Qwen 3.5 rollback with B = 1 rows, `6c6ddb74` docs: note the batched DFlash uniform-accept fix in the changelog.
