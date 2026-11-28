# PR #2140: model-owned 계열을 위한 시퀀스 단위 패딩 트림

**날짜**: 2026-10-06
**상태**: 구현 완료, GB10(CUDA)에서 강제 타일 정렬로 검증. M5 수동 확인은 미완료
**위험도**: 중간 (스케줄러의 모든 prefill 패딩 트림 지점을 변경. 패딩 경로 자체는 M5 또는 `MLXCEL_FORCE_PADDED_PREFILL`에서만 실행)

## 요약

타일 정렬 패딩 prefill은 실제 토큰 뒤에 패딩 위치를 기록하고, 스케줄러는 `CachePool` 항목의 `KVCache`를 트림해 이를 제거했습니다. 자체 `sequence_state_layout()`이 model-owned인 계열은 K/V를 모델 내부에 보관하므로 이 항목이 비어 있어 트림이 아무것도 지우지 못했습니다. 패딩 위치가 남고 `offset`이 패딩 폭만큼 실제 토큰 수를 앞질렀습니다. PR #1752는 임시 조치로 Gemma 3, AFMoE, Llama 4의 패딩 prefill을 껐습니다.

이 PR은 `LanguageModel::trim_sequence_state(seq_id, excess) -> Result<(), String>`을 추가합니다. 새 스케줄러 헬퍼(`scheduler/pad_trim.rs`)는 풀 캐시를 트림하고, 모델의 본래 레이아웃이 model-owned이면 훅을 호출합니다. 네 prefill 지점(batched, full, chunked 첫 청크, continuation)이 모두 이 헬퍼를 쓰며, `Err`이면 요청을 중단합니다. 기본 훅은 non-batching 모델에서는 `trim_internal_caches`에 위임하고(내부 상태가 곧 실행 중인 시퀀스), batching 모델에서는 실패로 닫힙니다.

Gemma 3는 `Cache::rewind_padded_prefill`로 훅과 `trim_internal_caches`를 구현하고 다시 `supports_padded_prefill() == true`를 반환합니다(sliding 레이어가 Turbo4Asym으로 저장되는 경우 제외). `deepseek_v4`, `bailing_moe_linear`, `qwen3_next`, `PipelineServerModel`은 이유를 명시하고 명시적으로 opt-out합니다. Closes #1755.

## 설계 메모

- `RotatingKVCache::trim`은 `offset`/`idx`만 되돌립니다. 윈도보다 긴 패딩 청크 뒤에는 다음 단일 토큰 업데이트가 물리 버퍼 길이 기준으로 앞부분을 잘라 마지막 `max_size` 슬롯을 남기는데, 여기에 패딩 키가 포함됩니다. `RotatingKVCache::rewind_padded_prefill`(`cache/prefill_rewind.rs`)은 물리 버퍼도 `idx`까지 자릅니다. 시간순이 아닌 버퍼(speculative 버퍼링, decode 쓰기로 감긴 링, 물리 길이와 `idx` 불일치)와 Turbo4Asym 저장소는 거부하며, 이때 캐시는 변경되지 않습니다.
- Gemma 3의 텍스트 forward는 호출자의 패딩 마스크를 무시하고 자체 causal 및 sliding-window 마스크를 만들므로 마스크 변경이 필요 없습니다. 패딩 위치는 실제 위치 뒤에 오고 causal attention이 모든 실제 행에서 이를 배제합니다.
- Prompt lookup은 `supports_padded_prefill`과 무관하게 model-owned 레이아웃 검사로 여전히 Gemma 3를 거부합니다.
- `MLXCEL_FORCE_PADDED_PREFILL`은 하드웨어와 무관하게 정렬을 강제합니다(CLI와 서버). 서버도 이제 `MLXCEL_NO_PADDED_PREFILL`을 따릅니다. `LoadedModel`은 이전에 `trim_internal_caches`를 위임하지 않아 어떤 계열의 CLI 내부 트림도 이를 거쳐 실행되지 않았습니다. 이제 두 훅을 모두 위임합니다.
- continuation 청크의 마스크 offset은 batching 모델에서 첫 풀 캐시를 읽고 빈 항목이면 0으로 떨어졌습니다. 이제 prefill 커서로 대체합니다.

## GB10 검증 (CUDA, release 프로파일, `--features cuda`)

이 PR 이전에는 CUDA에서 어떤 model-owned 계열도 패딩 prefill에 도달하지 않았습니다. model-owned 레이아웃과 `supports_batched_prefill()`을 함께 가진 계열은 qwen3_5뿐이며 패딩을 거부하므로 longest-row 패딩 batched 지점에 도달할 수 없고, 단일 시퀀스 지점은 override 없이 하드웨어로만 게이트되어 있었습니다.

- `cache::prefill_rewind` (5개 테스트): 패딩 append 후 rewind가 패딩 없는 append와 바이트 단위로 같습니다. 윈도 안과 밖, continuation 청크, decode 3스텝 이후까지 포함하며, 거부 시 캐시는 그대로입니다.
- `scheduler_model_owned_pad_trim_tests` (4개 테스트, sliding 1층과 global 1층의 소형 Gemma 3): full prefill(dense와 paged 백엔드)과 16토큰 chunked prefill에서 offset, sliding 레이어의 물리 버퍼, global 키, 디코드 토큰이 패딩 없는 실행과 일치합니다. 패딩된 턴이 기증한 스냅샷을 다음 턴이 채택합니다. Gemma 3 rewind를 끄면 앞의 세 테스트가 실패합니다(offset 32 대 13, 32 대 5, 96 대 37).
- 스위트: `server::batch::scheduler` 178개 통과(7개 ignored), `block_reclaim` 23, `lookahead` 11, `speculative` 175, `generate::tests` 36, `cache::rotating` 11, `models::afmoe` 28, `muse_glimmer` 109, `models::gemma3`와 `models::gemma4` 통과. 두 crate의 `-D warnings` clippy와 `cargo fmt --check` 통과.
- 서버, gemma-3-4b-it-4bit, `--prefill-chunk-size 500`, greedy, 48토큰, 176/730/1852토큰 프롬프트:

| 비교 | short (176) | mid (730) | long (1852) |
|---|---|---|---|
| 패딩 없음 p1 대 패딩 p1 | 동일 | 15번째 토큰에서 갈림, top-2 차이 0 | 11번째 토큰에서 갈림, top-2 차이 0 |
| 패딩 없음 p1 대 패딩 없음 p4(동시) | 동일 | 동일 | 11번째 토큰에서 갈림, top-2 차이 0 |
| 패딩 없음 p4 대 패딩 p4(동시) | 동일 | 15번째 토큰에서 갈림, top-2 차이 0 | 동일 |
| 패딩 없음 p1 대 rewind를 끈 패딩 p1 | 1번째 토큰에서 갈림, 차이 1.0 | 1번째 토큰에서 갈림, 차이 1.5, 출력이 `texId` 반복으로 붕괴 | 11번째 토큰에서 갈림 |

차이는 보고되는 logprob 해상도(0.25) 기준입니다. 수정 후의 모든 분기는 top-2 logprob가 같은 위치에서 발생하며, long 프롬프트는 패딩 없는 두 실행 사이에서도 같은 위치에서 갈립니다.

- CLI: gemma-3-4b-it-4bit로 `mlxcel generate`, 강제 패딩과 패딩 없음의 40토큰 출력이 동일합니다.

## 이 호스트에서 검증하지 못한 항목

- 이슈의 M5 수동 확인과 패딩 경로의 Metal 수치.
- AFMoE와 Llama 4는 계속 opt-out 상태입니다(체크포인트로 검증한 rewind가 없음. Llama 4의 `ChunkedKVCache`는 별도 구현이 필요).
- 여러 스위트를 반복 실행할 때 약 4회 중 1회 `cudaStreamEndCapture` 중단이 발생했습니다. 소스 변경이 주석뿐인 gemma4 스위트에서도 발생했으며, 재실행 시 모든 스위트가 통과했습니다.

## 후속 작업

model-owned 계열의 decode lookahead는 계속 꺼져 있습니다(PR #2139). rotating rewind는 시간순 버퍼를 요구하고 감긴 링에 대한 단일 토큰 decode 쓰기는 그렇지 않으므로, 이 훅으로 해당 teardown을 처리할 수 없습니다.
