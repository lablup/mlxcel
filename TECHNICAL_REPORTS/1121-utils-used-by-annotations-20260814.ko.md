# 기술 보고서: PR #1121 - docs(core): annotate the three tracked shared functions in utils.rs

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust (주석만), Markdown
**위험도**: Low

---

## 요약

PR #1121은 이슈 #1110을 해결한다. `docs/code-guidelines.md`는 공유 함수 `Used by:` 규칙으로 시작하면서 `utils.rs`의 `create_causal_mask`, `softcap`, `repeat_kv`를 추적 대상 컴포넌트로 지목한다. 이 PR은 셋 모두에 주석을 달고, 호출자가 너무 많아 나열이 불가능한 헬퍼를 위한 정책 절을 추가했다.

실제 결함은 이슈가 서술한 것보다 나빴다. 이슈는 `create_causal_mask`에 주석이 없다고 기록했다. 주석은 있었고, 의미가 뒤집힐 정도로 낡아 있었다. 이제는 그 함수를 전혀 호출하지 않는 계열들을 사용자로 나열하고 있었기 때문이다. 잘못된 명단을 교체하는 일은 빠진 명단을 채우는 일보다 더 중요한 수정이다. 주석이 없으면 기여자가 직접 찾아보지만, 틀린 주석은 찾아보지 말라고 말하기 때문이다.

---

## 1. 문제 정의

### 1.1 배경

`Used by:` 관례는 기여자가 공유 헬퍼를 바꾸기 전에 영향 범위를 볼 수 있게 하려고 존재한다. `docs/code-guidelines.md`도 그 목적을 그대로 적는다. 한 모델을 고치다가 다른 모델을 깨뜨리는 일을 막고, 변경 후 재테스트가 필요한 모델을 분명히 하기 위해서다. 이 관례는 `src/` 전반에서 대체로 잘 지켜지고 있고, 같은 파일의 이웃인 `create_causal_mask_with_left_padding`은 `/// Used by: BatchQuantizedKVCache, BatchTurboQuantKVCache`를 달고 있다. 즉 파일 단위 예외가 아니라 누락이었다.

### 1.2 기존 문제점

- **`create_causal_mask`는 주석이 없던 게 아니라 낡은 주석을 달고 있었다.** PR 이전의 줄은 `Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4, MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal attention callers`였다. 이 커밋 시점의 grep 결과 `mixtral.rs`, `phi.rs`, `phi3small.rs`, `starcoder2.rs`, `llama3.rs`, `gemma.rs`, `gemma2.rs`, `cohere.rs`, `glm4.rs`, `olmoe.rs`, `qwen3_moe.rs`는 각각 호출 횟수가 **0**이다. 이 계열들은 `seq_len > 1`에서 `mask: None`을 넘기는 implicit-causal fused-SDPA 경로로 옮겨 갔다. 따라서 주석은 단순히 불완전한 게 아니었다. 이 함수를 바꿔도 영향받지 않는 계열들을 사용자로 지목하고, 실제로 의존하는 hybrid, sliding-window, VLM, MLA decoder들은 빠뜨리고 있었다. 이 주석을 믿은 기여자는 엉뚱한 집합을 재테스트했을 것이다.
- **`repeat_kv`와 `softcap`은 아무 주석도 없었다.** 둘 다 그대로 나열해도 될 만큼 짧아서 판단이 필요 없었고, 순수한 누락이었다.
- **호출자가 40개인 헬퍼에 대해 가이드라인에 답이 없었다.** 이슈가 이를 명시적으로 요구했다. 큰 공유 함수를 다루는 다음 사람이 나열과 요약 사이의 선택을 다시 논의하지 않도록 하기 위해서다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 기여자가 `create_causal_mask`를 바꾸고 낡은 명단에 적힌 계열들(전부 이 함수를 호출하지 않는다)을 재테스트하는 동안, 실제 호출자인 hybrid와 VLM decoder는 검증되지 않고 지나감 | High | Medium |
| 기여자가 가이드라인을 읽고 그 문서가 직접 지목한 예시 함수를 열었는데 주석이 없어 관례가 강제되지 않는다고 결론 내림 | Medium | Medium |
| 앞으로 큰 헬퍼에 40개짜리 이름 명단이 달리고 한 릴리스 만에 낡음 | Medium | Medium |

---

## 2. 기술적 검토 사항

### 2.1 모든 목록을 이 커밋에서 grep으로 도출

어떤 목록도 이슈 본문이나 기존 주석에서 옮기지 않았다. 커밋 `f0bf3a2c` 기준 측정값이다.

| 함수 | 이슈 수치 | 실측 | 명령 |
|---|---|---|---|
| `create_causal_mask` | 46 | `src/models` 아래 비테스트 44개(`diffusion_gemma/tests.rs`와 `phi3small_tests.rs` 포함 시 46), `src` 전체 55개 | `grep -rln '\bcreate_causal_mask(' src --include='*.rs'` |
| `repeat_kv` | 12 | `src/models` 아래 12개, DeepSeek-OCR Qwen2 vision encoder와 Qwen3-Omni MoE speech layers 포함 시 14개 | `grep -rln '\brepeat_kv(' src --include='*.rs'` |
| `softcap` | 1 | 프로덕션 호출자 1개(RecurrentGemma), 그리고 core 단위 테스트 | `grep -rn '\bsoftcap(' src --include='*.rs'` |

`create_causal_mask`의 이슈 수치 46은 테스트 파일 두 개를 포함한 값이다. `src/models` 밖의 호출자 아홉은 `lib.rs`, `layers.rs`, `cache.rs`, tensor-parallel Llama runtime, GLM4 파이프라인 stage executor, disaggregated handoff, 그리고 테스트 파일 셋이다.

`softcap`에는 단순 grep으로는 드러나지 않는 단서가 하나 필요하다. `src/lib/mlxcel-xla/src/emitter/model.rs`가 XLA emitter용으로 자체 private `fn softcap`을 정의하는데, 이는 다른 크레이트의 다른 함수다. grep 결과에는 나오지만 `utils.rs` 헬퍼의 호출자가 아니다. 주석은 진짜 호출자만 센다.

### 2.2 "Not used by" 절반의 검증

각 그룹에 이름이 오른 계열은 파일별 grep으로 최소 한 번 호출함을 확인했고, "not used by" 목록에 오른 계열은 호출 0임을 확인했다. 여기서는 두 번째 검사가 더 중요하다. 원래의 낡은 주석이 사람을 오도하는 것을 막았을 부분이 바로 "not used by" 절이기 때문이다.

### 2.3 범위 통제

`.rs` diff가 주석뿐임을 증명할 수 있다. `src/lib/mlxcel-core/src/utils.rs`에서 추가되고 삭제된 모든 줄이 `^\s*(///|//)`에 일치한다. 함수 본문, 시그니처, 값 중 바뀐 것이 없다. `cargo fmt --check`는 clean이며, 이는 프로젝트 설정에서 `rustfmt`가 주석을 재배치하지 않는 데서 따라온다.

---

## 3. 기술적 선택과 그 이유

### 3.1 44개 이름 명단 대신 규칙 + 대표 사례

| 옵션 | 장점 | 단점 |
|---|---|---|
| 비테스트 호출자 44개를 전부 이름으로 나열 | 오늘 기준으로는 정확 | 다음 릴리스에 틀려지고, 길이 때문에 아무도 고치지 않는다. 원래 주석이 이 상태가 된 경로가 바로 그것이다 |
| 예외만 나열("X, Y를 제외한 모든 decoder") | 짧다 | 예외 집합 자체가 크고 계속 바뀐다. 유지보수 문제를 뒤집을 뿐이다 |
| **선택: 호출자가 왜 목록에 있는지에 대한 규칙, 그룹별 대표, 명시적 "not used by", 목록을 재생성하는 grep** | 구성원이 바뀌어도 규칙은 유지되므로 변화에 견딘다. 정확한 집합은 한 줄 명령으로 재생성 가능 | 문자 그대로 전수는 아니므로, 정확한 집합이 필요한 독자는 grep을 실행해야 한다 |

핵심은 규칙이다. 호출자는 fused SDPA가 causality를 적용하도록 `mask: None`을 남기는 대신 명시적 prefill mask를 직접 만드는 decoder들이다. 이 문장은 모델이 추가되거나 이전되어도 참으로 남는다. 네 그룹(hybrid와 mixed-layer stack, sliding-window와 chunked 계열, VLM decoder, MLA와 custom-attention decoder)마다 대표를 적었고, 주석은 정확한 목록을 재생성하는 grep으로 끝난다.

### 3.2 "Not used by" 절반을 유지

사용자만 나열했다면 더 짧았을 것이다. 이 부정 절반이야말로 이 주석이 대체한 바로 그 실패에 대해 견고함을 만든다. 즉 공유 헬퍼가 몇 릴리스 전에 이미 떠난 계열까지 여전히 덮고 있다고 기여자가 가정하는 실패다. Llama3, Mixtral, Gemma, Gemma2, Cohere, Phi, GLM4, StarCoder2, Qwen3Moe, OLMoE를 명시적 비호출자로 적고 그 이유(`seq_len > 1`에서 `mask: None`을 넘긴다)까지 남기면, 과거의 틀린 명단이 문서화된 정정으로 바뀐다.

### 3.3 정책을 주석뿐 아니라 가이드라인에도 기록

이슈가 직접 요구한 사항이다. `docs/code-guidelines.md`에 "When the caller list is too long to enumerate" 절이 생겼고, 네 부분으로 이뤄진 정책과 함께 `create_causal_mask`를 실제 예시로 담았다. 이 절은 공개 아이템에 옛 형식 예시의 `//` 대신 `///`를 쓴다는 점도 적는다. 그래야 주석이 rustdoc까지 살아남는다. 가이드라인에 한 번 기록해 두면 다음에 호출자 40개짜리 헬퍼가 나와도 같은 선택을 다시 열지 않는다.

### 3.4 주석을 doc 블록 끝에 배치

세 주석 모두 기존 doc 블록의 `# Returns` 절 뒤, 즉 블록 끝에 놓았다. 같은 파일의 이웃 `create_causal_mask_with_left_padding`과 같은 배치다. 이렇게 하면 함수 자체의 설명이 앞에 오고 호출자 명단이 뒤로 가므로, 긴 명단이 실제 문서를 아래로 밀어내지 않는다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +62 |
| 삭제된 라인 | -2 |
| 변경된 실행 라인 | 0 |
| 주석을 단 함수 | 3 |

### 영역별 변경

| 영역 | 파일 | 주요 내용 |
|---|---|---|
| 기존 주석의 정확성 | `src/lib/mlxcel-core/src/utils.rs` | `create_causal_mask`: 비호출자를 나열하던 낡은 명단을 규칙 수준 주석으로 교체. 대표를 포함한 호출자 네 그룹, `src/models` 밖 호출자, 명시적 "not used by", 재생성 grep |
| 신규 주석 | `src/lib/mlxcel-core/src/utils.rs` | `repeat_kv`: 호출자 14개 전체 나열과 대부분의 decoder가 호출하지 않는 이유 |
| 신규 주석 | `src/lib/mlxcel-core/src/utils.rs` | `softcap`: 유일한 프로덕션 호출자로 RecurrentGemma를 지목하고, Gemma 2와 Gemma 3은 fused `compiled_softcap` / `compiled_softcap_sdpa` 커널을 거친다는 사실 기록 |
| 관례 | `docs/code-guidelines.md` | 새 "When the caller list is too long to enumerate" 절에 정책과 `create_causal_mask` 실제 예시, 공개 아이템은 `///`를 쓴다는 주의 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `f0bf3a2c` | docs | docs(core): annotate the three tracked shared functions in utils.rs |

---

## 5. 검증 및 후속 조치

### 통과

- `.rs` diff가 주석뿐임을 증명: 추가되고 삭제된 모든 줄이 `^\s*(///|//)`에 일치.
- `cargo fmt --check` clean.
- `python3 scripts/ci/check_cross_repo_refs.py` clean.
- 주석에 이름이 오른 계열은 파일별 grep으로 최소 1회 호출 확인, "not used by" 목록의 계열은 0회 확인.

### 이슈에 대한 정정

이슈의 표는 `create_causal_mask`를 주석 없음, 모델 파일 호출자 46개로 기록했다. 구현 시점에 두 수치 모두 정정이 필요했다. 이 함수는 주석을 달고 있었고, 46은 테스트 파일 두 개를 포함한 값으로 실제 비테스트 수는 44다. 두 정정 모두 PR이 해야 할 일을 바꾸지는 않지만, 첫 번째는 결함의 성격을 바꾼다. 주석 누락의 해법은 추가하는 것이고, 비호출자를 나열하는 주석의 해법은 교체하면서 왜 그 계열들이 더 이상 사용자가 아닌지를 파일 안에 적는 것이다. "not used by" 절이 하는 일이 정확히 그것이다.

### 후속 후보

- 이 관례를 강제하는 CI 검사가 없다. `make verify-kernel-dtype-keys`와 `scripts/ci/check_kernel_dtype_keys.py`가 강제하는 JIT 커널 dtype-key 규칙과 대비된다. `utils.rs`나 `layers.rs`의 `pub fn` 중 교차 모듈 호출자가 N개를 넘고 `Used by:` 줄이 없는 것을 잡아내는 검사기가 있으면 재발을 막을 수 있다. 이슈도 이것이 이슈 자체보다 큰 작업이라고 적고 있다.
- 이번에 발견한 staleness는 `create_causal_mask`만의 문제가 아니다. `src/` 어디에 있든 오래된 `Used by:` 명단은 같은 방식으로 썩을 수 있고, 나머지를 감사한 적은 없다.
- `docs/code-guidelines.md`는 `layers.rs`의 KVCache, Attention, Normalization도 추적 대상 공유 컴포넌트로 열거한다. 이 PR은 그 목록의 `utils.rs` 절반만 다뤘다.
