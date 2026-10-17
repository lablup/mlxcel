# PR #2436: xla-iree 루프에서 확장된 미디어 프롬프트 길이를 보고

**작성일**: 2026-10-17
**상태**: CUDA(GB10)에서 gemma-3-4b-it-4bit 실제 체크포인트를 XLA 백엔드로 구현 및 검증 완료. Phi4MM 오디오는 실제 체크포인트로 실행하지 않음 (「미검증 항목과 후속 조치」 참고)
**위험도**: 중간 (`xla-iree` 루프에서 미디어 본문의 `usage`, count 라우트, `message_start`가 보고하는 프롬프트 토큰 수를 바꾸고, 이 루프에 `Prefill` 이벤트를 새로 추가)

## 요약

이슈 #2420은 Anthropic 스트리밍 `message_start`가 worker의 첫 `GenerateEvent::Prefill`을 기다리도록 만든 #2415(#2337)의 후속이다. `xla-iree` 서빙 루프는 `Prefill`을 보내지 않았고 논리(placeholder) 기준 프롬프트 길이를 보고했으므로, 미디어 본문의 `usage`, `/v1/messages/count_tokens`, `message_start`가 모두 placeholder 기준 토큰 수를 담았다. 이 PR은 루프가 확장된 prepared 길이를 보고하고 `Prefill`을 보내도록 하여 batch scheduler 및 `docs/llama-server-compat.md`와 일치시킨다.

## 문제 정의

- `src/server/batch/xla_worker.rs`의 `dispatch`는 `Token`, `Done`, `Error`만 보냈으므로, prefill 보고를 기다리는 스트리밍 라우트는 보고를 받지 못했다.
- `xla_worker_admission.rs`는 `effective_prefill_len`(`prepared.sequence_len()`) 옆에 `prompt_token_count`(논리 기준 값)를 두었고, 결과에는 논리 기준 값이 실렸다. #895의 분리는 확장된 길이를 내부 처리량과 KV 계산에만 썼다.
- XLA worker는 `count_prompt_only`를 무시하고 count 라우트를 토큰 하나짜리 생성으로 처리하므로, `/v1/messages/count_tokens`도 같은 placeholder 기준 값을 보고했다.
- 기존 이미지 테스트는 요청이 5 토큰으로 확장되는데도 `prompt_tokens == 3`을 단언하여 이전 동작을 고정하고 있었다.
- XLA 백엔드의 gemma-3-4b-it-4bit(main `cadc6366`)에서 이미지 본문은 `count_tokens`, `usage`, `message_start`가 모두 18이었고, MLX 경로는 277이었다.

## 기술적 선택과 그 이유

- `ServeState`는 `effective_prefill_len` 하나만 유지한다. 미디어는 확장된 prepared 길이, 텍스트는 프롬프트 길이다. 별도의 `prompt_token_count` 필드는 제거했으며 이는 #895의 분리를 되돌리는 것이다. 텍스트 요청은 두 길이가 같으므로 달라지지 않는다.
- `ServeState::report_prefill`은 `PrefillStats { prompt_tokens, cached_tokens: 0, processed, first_token: true, prompt_ms }`를 `prefill_reported` 플래그로 보호하여 한 번만 보낸다. 첫 `Token` 전에, 그리고 엔진이 토큰 없이 끝나는 경우(EOS가 보류된 경우)에는 `Done` 전에 호출된다. 또한 batch scheduler의 `sequence.rs`처럼 decode 상태의 첫 토큰 시각을 기록한다.
- `Done`, stop-string 종료(`finalize_stop`), 예산 0인 미디어 종료(`finish_zero_budget`)가 확장된 길이를 보고한다. `max_tokens = 0`은 기존의 텍스트 전용 count가 적용되기 전에 반환되므로, `finish_zero_budget`은 이제 길이를 인자로 받는다.
- 모든 XLA 이미지 계열은 `handle_preprocessed`를 거치므로 새로 들어온 Molmo(#918)와 Youtu-VL(#919) producer도 포함된다. DeepStack(Qwen3-VL)과 Phi4MM 오디오 및 이미지 경로도 마찬가지다.
- 문서는 바꾸지 않았다. `docs/llama-server-compat.md`에는 XLA 전용 서술이 없고, 그 일반 약속이 이제 이 루프에서도 성립한다.
- 이슈에 적힌 대로 disaggregated coordinator는 범위에서 제외한다. `coordinator.rs`는 `Prefill`을 버리지만, router front는 `/v1/chat/completions`와 `/v1/completions`만 서빙하고 미디어를 거부하므로 미디어 본문이나 Anthropic 본문이 도달하지 않는다.

## 변경 요약

| 영역 | 파일 |
|------|------|
| `ServeState` count, `report_prefill`, `Done`과 stop-string 종료 | `src/server/batch/xla_worker.rs` |
| admission 경로, 예산 0 종료 | `src/server/batch/xla_worker_admission.rs` |
| 테스트 | `src/server/batch/xla_worker_tests.rs` |

origin/main 대비 3개 파일, +328 -40. 커밋: `815a6da0` fix(server): report the expanded media prompt length on the xla-iree loop.

## 검증

새로 추가하거나 수정한 테스트는 fake 엔진으로 실제 `admit`, 전처리, `dispatch` 경로를 구동한다.

- 텍스트는 `Prefill(2)`, 이미지는 `Prefill(5)`, Phi4MM 오디오와 이미지는 각각 `Prefill(6)`을 받는다. 모두 해당 `Token`보다 먼저 전송되며 `Done`이 같은 길이를 보고한다.
- 예산 0인 이미지, 예산 0인 오디오, 순수 EOS 종료도 다룬다.
- fake Phi4MM producer가 이제 placeholder를 확장한다.
- 이 모듈의 기존 단언 두 개는 에러 메시지가 바뀌어(`structured`는 `constrained decoding`, `refusing text fallback`은 `image resolution cardinality mismatch`) main에서 이미 실패하고 있었다. 모듈이 통과하도록 갱신했다.

로컬 게이트: `server::batch::xla_worker` 15 통과 (`--release --features cuda,xla-iree`, `--test-threads=1`, gpu-lock 하에서). 프로덕션 파일을 main으로 되돌리면 9개가 실패한다. 새로 추가하거나 수정한 테스트 7개와 오래된 단언 2개다. `cargo clippy --release --features cuda,xla-iree --lib --tests -- -D warnings`, `cargo check --release --features cuda,xla-iree --all-targets`, `cargo fmt --check`, 저장소 검사 스크립트 모두 통과했다.

실제 체크포인트 `~/models/mlx/gemma-3-4b-it-4bit`을 GB10에서 `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_CONTEXT_CAPACITY=512 mlxcel-server`로 실행했다 (기본 용량 256은 Gemma 3 이미지를 거부한다). 같은 본문(`tests/fixtures/test_image.png`와 짧은 프롬프트)을 `count_tokens`, 비스트리밍 `/v1/messages`, 스트리밍 `/v1/messages`로 보냈다.

| 본문 | 빌드 | count_tokens | usage.input_tokens | message_start.input_tokens |
|---|---|---|---|---|
| 이미지 | main cadc6366, XLA | 18 | 18 | 18 |
| 이미지 | 이 PR, XLA | 277 | 277 | 277 |
| 이미지 | 이 PR, MLX 경로 | 277 | 277 | 277 |
| 텍스트 | 두 빌드 모두, XLA | 17 | 17 | 17 |

모든 스트림에서 `message_start`가 첫 이벤트였고 한 번만 전송되었다.

## 미검증 항목과 후속 조치

- Phi4MM 오디오는 로컬에 체크포인트가 없어 실제로 실행하지 않았다. fake producer 테스트가 이를 다룬다.
- disaggregated coordinator는 이 PR의 범위가 아니다.
