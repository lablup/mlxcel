# PR #2182: 링을 고려한 teardown 되감기로 Gemma 3 decode lookahead 복원

**날짜**: 2026-10-07
**상태**: 구현 완료, CUDA(GB10)에서 검증. Metal은 측정하지 않음
**위험도**: 중간 (Gemma 3의 파이프라인 decode를 다시 켜고, 서버 경로의 모든 Gemma 3 sliding-window 캐시에 상태를 추가)

## 요약

PR #2139는 모든 model-owned 계열의 decode lookahead를 껐습니다. teardown 트림이 `CachePool` 캐시에만 닿아 투기적 append가 모델 자체 상태에 남았기 때문입니다. 그 결과 GB10에서 Gemma 3 4B는 약 10 tok/s를 잃었습니다. 커서 트림만으로는 Gemma 3를 처리할 수 없습니다. sliding window가 한 바퀴 돈 뒤에는 decode 쓰기마다 막 윈도를 벗어난 위치의 K/V를 덮어쓰고, 트림은 그 슬롯에 투기적 K/V를 남깁니다.

이 PR은 `RotatingKVCache`에 opt-in undo 로그(`cache/decode_undo.rs`)를, `LanguageModel`에 두 훅(`supports_decode_lookahead_rewind`, `rewind_decode_appends`, `LoadedModel`과 VLM 래퍼에서 위임)을 추가하고, Gemma 3 구현과 스케줄러 연결을 더합니다. 게이트는 이 능력이 있는 model-owned 계열만 받아들이고, 모든 teardown이 되감기를 호출하며, 되감기가 실패하면 요청을 오류로 종료합니다. Closes #2159.

## 설계 메모

- 기록된 쓰기마다 시작 `offset`과 `idx`(윈도 초과 pre-trim 이후, wrap 이전), 슬롯, 그 슬롯이 유효했는지를 남깁니다. `DecodeLookaheadAppendScope` 안에서 일어난 덮어쓰기는 그 슬롯의 `[B, H, 1, D]` K/V 행도 복사합니다. 스케줄러는 prime forward 주변에서만 이 scope에 들어갑니다. teardown이 되돌릴 수 있는 append는 모두 여기서 생기며, 동기 step은 아무것도 복사하지 않고 되감을 수도 없습니다.
- `rewind_decode_writes(n)`은 먼저 검증하고(버퍼링 꺼짐, FP16, 로그 활성, `n <= offset`, 연속된 항목, 모든 덮어쓰기에 행 존재), 그다음 행을 최신 것부터 복원하고 warmup 슬롯을 다시 0으로 채운 뒤 커서를 정확히 맞춥니다. 로그보다 오래된 쓰기는 링이 아직 시간순일 때만 허용합니다. 그 밖의 변경은 로그를 지웁니다.
- 지연 평가되는 행 복사는 쓰기 전 버퍼를 붙잡아 두므로, MLX가 그 버퍼를 쓰기의 `slice_update`에 넘겨주지(donation) 못하고 매 step 윈도 전체를 복사합니다. 그래서 Gemma 3는 forward를 만든 직후, 그 forward가 평가되기 전에 대기 중인 복사를 한 번의 `async_eval`로 예약합니다. 이슈가 기각한 대안(쓰기 전 핸들 보관)도 같은 버퍼를 붙잡으므로 비용이 같습니다.
- Gemma 3는 스케줄러 시퀀스 캐시에서만(깊이 2, 공유 상수 `DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS`), 그리고 모든 sliding 레이어가 FP16일 때만 로그를 남깁니다. CLI fallback 슬롯은 로그가 없습니다.
- `--parallel 1`에서 Gemma 3는 `DenseKvCache`가 아니라 `ModelOwned`로 할당됩니다. 시퀀스별 백엔드 검사는 모델에 능력이 있을 때 이 할당을 받아들입니다.
- 되감기에 실패하면 시퀀스를 직접 `Finished(Error)`로 설정합니다. 같은 tick의 `length`나 `stop` 종료는 `Error`로 전이할 수 없어, 그대로 두면 어긋난 상태를 prompt cache에 기증하게 됩니다.

## GB10 검증 (CUDA, release, `--features cuda`)

- 단위 및 스케줄러 테스트: `cache::` 556, `server::batch` 495, `models::gemma3` 32, `vision::` 545 통과. 새 lockstep 스케줄러 테스트는 wrap 이후 파이프라인과 force-sync 상태를 비트 단위로 비교하며, 행 복원을 끄면 두 테스트 모두 실패합니다.
- 서버, gemma-3-4b-it-4bit, greedy: 짧은 프롬프트, 1906 토큰 프롬프트, 동시 요청 4개가 `--parallel 4`와 `--parallel 1`에서 lookahead와 `MLXCEL_FORCE_SYNC=1` 사이에 바이트 단위로 같습니다. 동시 요청은 긴 blocker prefill 뒤에 줄을 세워 두 arm이 같은 배치 이력을 보게 했습니다. 그렇게 하지 않으면 force-sync 혼자서도 실행마다 두 가지 출력을 냈습니다.
- 처리량, 교차 라운드, null arm(바이트 동일 바이너리), 호스트 CPU 유휴 게이트:

| Gemma 3 4B, 200 토큰 | new | null | #2139 이전 | force-sync | main |
|---|---|---|---|---|---|
| 짧은 프롬프트, tok/s | 87.3 | 87.2 | 87.6 | 76.2 | 75.7 |
| 1906 토큰 프롬프트 이후, tok/s | 72.7 | 72.9 | 77.1 | 67.3 | 67.4 |

- Llama 3.2 1B dense lookahead: main 270.0 대비 270.4 tok/s, 출력 동일.

남겨 둘 측정 이력:

- CPU 게이트 없이(다른 에이전트가 호스트에서 컴파일 중) 측정하면 두 lookahead arm 모두 40~60 tok/s 라운드가 나왔고 force-sync는 그렇지 않았습니다. 게이트를 넣자 사라졌고 null arm은 1% 이내로 유지되었습니다.
- wrap 구간 비용 분리: 행 복사를 flush하지 않으면 64.5 tok/s, flush하면 72.4, 복사 없음(부정확) 76.9.
- 첫 버전은 모든 쓰기에서 행을 복사해 wrap 구간 force-sync를 67.3(main)에서 65.1로 늦췄습니다. scope 도입으로 해소되었습니다(67.3 대 67.4).

## 이 호스트에서 검증하지 못한 것

- 행 복사와 flush의 Metal 처리량 및 수치.
- 브랜치는 `0ee13dfe` 기반입니다. 이후 머지된 #2162(CUDA SDPA 스위치)는 여기서 다시 빌드하지 않았습니다.

## 후속 작업

- wrap 구간은 여전히 부정확한 #2139 이전 파이프라인보다 5.7% 느립니다. 행 복사와 추가 `async_eval`의 비용입니다. donation을 유지하는 순서로 복사를 forward 자체 평가에 합치는 것이 후보입니다.
- Gemma 4, Llama 4, AFMoE, Muse Glimmer, SSM/hybrid 계열, INT8/Turbo4Asym sliding 저장소는 기본 능력(false)으로 동기 decode를 유지합니다.
