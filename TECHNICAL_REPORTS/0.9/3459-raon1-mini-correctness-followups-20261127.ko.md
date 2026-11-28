# PR #3459: Raon-1 mini 정합성·검증 후속 조치 (이슈 #3457)

**작성일**: 2026-11-27

**상태**: PR 열림(#3457을 닫음). M3 Ultra에서 검증. item 7의 CUDA·ROCm 실행은 남아 있다. 35개 파일, +1300 / -67.

**위험도**: 중간(로드 시 검사가 모든 양자화 계열에 적용되고, presence penalty를 설정한 체크포인트의 서버 기본값이 바뀐다)

## 요약

Raon-1 mini PR 네 개(#3441, #3445, #3453, #3455)가 남긴 공백은 아홉 가지다. 읽지 않던 샘플링 기본값, 텐서 shape를 그대로 믿던 로드 경로, recurrent state를 빠뜨린 메모리 추정, 고정되지 않은 reference oracle, 측정하지 않은 norm 반올림 차이, CUDA 반올림 위험, reasoning effort를 조용히 버리던 서버 경로, 길이 제한 없는 오류 echo. 이 PR은 아홉 개를 한 번에 닫는다. 일곱 개는 main에서 실패하는 테스트가 붙은 수정이고, item 6은 측정 후 그대로 두었으며, item 7은 수정은 들어갔지만 이 호스트에서 CUDA/ROCm 실행을 할 수 없었다.

## 1. 문제 정의

- **샘플링 기본값.** Raon의 `generation_config.json`은 `presence_penalty 1.5`를 설정한다. mlxcel은 같은 파일에서 temperature, top_p, top_k는 읽었지만 이 필드는 읽지 않아 서버의 Raon이 0으로 돌았다.
- **shape 신뢰.** 입력 폭이 config와 다른 projection이나, weight보다 행이 적은 양자화 scales가 로드는 되고 첫 forward에서 실패하거나 범위 밖을 읽었다. #3453은 Raon plane에만 scales 검사를 넣었다.
- **KV 추정.** admission 추정은 MLA latent는 셌지만 RDA의 시퀀스별 recurrent·conv state(Raon-1 mini에서 시퀀스당 48.8 MB)는 세지 않았다.
- **Oracle.** `scripts/raon1_mini_oracle_trace.py`는 체크포인트의 Python 코드를 import하면서 digest를 기록만 하고 확인하지 않았다.
- **effort를 무시하는 경로.** pipeline-parallel, 분리형 prefill/decode, XLA 서빙은 시퀀스별 effort를 전달하지 않아 steering 모델이 오류 없이 잘못된 alpha로 돌게 된다.

## 2. 기술적 선택과 그 이유

### 2.1 presence_penalty는 서버 기본값으로만

서버는 `--presence-penalty`가 없을 때(`presence_penalty_was_set`) config 값을 기본값으로 쓰고, 요청이 지정한 값이 여전히 우선한다. `mlxcel generate`는 0.0을 유지한다. penalty를 끌 플래그가 없고, greedy oracle 비교가 0에 의존하기 때문이다. `models/` 아래에서 이 필드를 설정한 체크포인트는 Raon뿐이다(`raon-1-mini`, `-mlx`, `-nvfp4`, `raon-1-nano`).

부작용: penalty가 0이 아니면 `needs_token_history()`가 true가 되고, `speculative_burst.rs`의 `sampling_config_eq`가 이런 요청을 batched speculative window에서 뺀다. 그래서 penalty를 지정하지 않은 동시 Raon 요청은 이제 단일 요청 DFlash2 burst로 초안한다. 출력은 같고, `--presence-penalty 0`이나 요청별 값으로 batching이 돌아온다. #3458의 batched DFlash2 작업과 맞물린다.

### 2.2 effort를 버리는 경로는 전달이 아니라 거부

| 선택지 | 장점 | 단점 |
|---|---|---|
| 각 경로로 effort 전달 | 기능상 완전 | 지금은 죽은 코드: 분리형 경로는 pool 기반 Fp16 계열만 서빙하고 Raon은 XLA 포트가 없다 |
| **시작 시 경로와 모델 이름을 대며 실패** | 작고 명시적, 잘못된 alpha가 조용히 돌지 않는다 | 이후 포트가 guard를 풀어야 한다 |

`src/server/reasoning_steering_startup_guard.rs`가 모델 로드 전 `start_server`에서 돈다. `start_server`는 경로별 단위 테스트가 불가능해 테스트를 guard 옆에 두었다.

### 2.3 fast RMSNorm 유지 (item 6)

Transformers reference 대비 teacher-forced trace, 같은 lock 안 ABAB decode:

| Arm | 4 x 256 결정 불일치 | 871토큰 결정 불일치 | 871토큰 ppl 차이 | Decode tok/s |
|---|---|---|---|---|
| `fast_rms_norm` | 0 / 578 | 1 / 549 | +0.337% | 33.30 |
| `rms_norm_round_once` | 0 / 578 | 0 / 549 | +0.607% | 32.26 |

round-once는 유일한 불일치(#3441이 reference 쪽 bf16 잡음으로 추적한 row 866)를 없애지만 decode가 3.1% 느려 2% 기준을 넘고, 긴 프롬프트 perplexity 차이도 커진다. 어떤 계열의 norm도 바뀌지 않는다. `rms_norm_round_once`는 남겨 둔다.

## 3. 구현 상세

- item 2, 3: core `layers.rs`의 `UnifiedLinear::check_input_width`, `SwitchLinear::check_input_width`, 공유 `check_scales_leading_axes`. scales 검사는 모든 양자화 계열의 `UnifiedLinear::from_weights_with_mode`와 `SwitchLinear::from_stacked_parts`에서 돌고, Raon의 `check_scales_match_weight`와 `validate_quantized_packing`은 이를 위임한다. Raon은 RDA, MLA, MLP, expert, `lm_head` projection마다 폭 검사를 부른다. `KimiK3MLAAttention::check_input_widths`는 있지만 아직 호출되지 않는다.
- item 4: `kv_arch`가 시퀀스당 `rda_layers * (H*D*D*4 + (K-1)*3*H*D*2)`를 `total_bytes`에 더한다. 토큰당 marginal은 그대로다.
- item 5: revision `6691a076`의 `PINNED_CODE_DIGESTS`를 import 전에 확인한다. `--allow-unpinned-code`는 경고하고 trace header에 표시한다.
- item 7: `compute_g_lower_bounded`가 `dt_bias`를 더하기 전에 `a`를 f32로 바꾼다. Metal은 bf16 + f32를 이미 f32로 올리므로 출력이 비트 단위로 같고, mlxcel의 CUDA overlay는 합을 bf16으로 타이핑해 반올림했다.
- item 9: 잘못된 레벨의 400은 최대 32자를 Debug escape해 echo하고 바이트 수 표시를 붙인다.

## 4. 검증

item 2, 3, 4, 5, 9의 새 테스트는 main에서 실패한다. item 1과 8은 API를 추가해 main에서 테스트가 컴파일되지 않는다. 대상 suite(raon, generation_config, kv_arch, reasoning_steering, gated_delta, switch_layers, server startup, kimi_k3, core layers와 rms_norm)가 통과한다. Raon 체크포인트 세 개와 다른 양자화 계열 두 개가 새 검사와 함께 로드·생성된다. 로컬 `make verify` 전체(17,183개 테스트)와 머지 전 저장소 contract target이 통과했다. pr-reviewer와 pr-security-checker는 CRITICAL·HIGH를 찾지 않았다.

## 5. 후속 조치

- CUDA·ROCm에서 item 7: `make test-fast-cuda FILTER=gated_delta`와 Raon tiny-model suite. 접근 가능한 호스트가 없었다.
- penalty가 있는 요청이 batched speculative window를 공유하려면 window에 행별 token history가 필요하다(2.1 참고).
- Kimi K3가 새 입력 폭 검사를 호출해야 한다.
- LOW, 미수정: 문자열이 아닌 `reasoning_effort` kwarg는 잘라내기 전에 전체가 직렬화된다(요청 본문 제한으로 상한). config의 presence_penalty에 상한이 없다.
