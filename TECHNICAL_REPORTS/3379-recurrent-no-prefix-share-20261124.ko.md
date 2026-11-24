# PR #3379: OpenXLA와 MLX에서 순환 모델을 prefix-hash 캐시에서 제외

**날짜**: 2026-11-24
**상태**: PR 열림(#3341을 닫음, epic #564 Phase 1 단위 U5). ROCm gfx1151에서 검증했다. OpenXLA 서버는 IREE `hip`으로 `mamba2-130m`에서, MLX ROCm 서버는 `qwen3.5-0.8b-4bit`와 `granite-4.0-h-tiny-4bit`에서 돌렸다. Falcon-H1과 PLaMo 2는 로컬 체크포인트가 없어 config 감지 단위 테스트로만 확인했다.

## 요약

저장된 토큰 prefix로는 순환 상태를 복원할 수 없으므로, 순환 모델은 prompt-prefix store나 Automatic Prefix Caching(APC)에서 hit를 얻으면 안 된다. 이 PR 이전 OpenXLA에서 이 성질은 우연히만 성립했다. XLA serve worker가 `PromptCacheWarmup`을 버리고 store를 읽지 않았을 뿐, `ModelProvider`의 OpenXLA 분기는 여전히 store를 붙였고 이 성질을 검증하는 것은 없었다. MLX에서는 APC 제외 목록 `HYBRID_SSM_MODEL_TYPES`에 순환 계열 네 개가 빠져 있었다.

이 PR은 OpenXLA 가드(`src/server/xla_prompt_cache_guard.rs`)를 추가한다. 판단은 emitter의 `Config::has_recurrent_state()` 또는 공유 목록 포함 여부로 하고, 두 곳에서 강제한다. `startup::resolve_prompt_cache_store`는 store를 만들지 않고 `prompt_cache.enabled`와 `apc.enabled`를 끄며, provider의 OpenXLA 분기는 다른 호출자가 넘긴 store를 버린다. 가드가 읽거나 파싱할 수 없는 `config.json`은 닫힌 쪽(캐시 끔)으로 실패한다. MLX 목록에는 `falcon_h1`, `plamo2`(+`plamo2vl`), `qwen3_5`(+`qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`), `granitemoehybrid`와 각 아키텍처 조각이 추가된다. 시작 시 감지기는 이제 HF의 따옴표 없는 `Infinity`를 허용한다. 예전에는 Mamba2와 Nemotron-H config가 엄격한 파싱에 실패해 APC가 켜진 채 남았다. 검사의 목적과 정반대 결과다.

MLX에서 사용자에게 보이는 효과는 시작 경고와 `apc_enabled: false`뿐이다. 네 계열은 이미 상태를 KV slice 밖에 두므로 block-hash store에 닿은 적이 없고, 두 로컬 체크포인트 모두 출력이 origin/main과 바이트 단위로 같다.

## 1. 문제 정의

Epic #564는 SSM, hybrid, 순환 아키텍처를 OpenXLA/IREE 백엔드에 들여오며, 수용 조건으로 순환 경로가 prefix-hash 공유에 절대 참여하지 않을 것을 요구한다. 이슈 #3341은 main의 상태를 이렇게 기록했다.

- **OpenXLA, 우연히 성립.** `xla_worker_admission.rs`는 XLA worker에 snapshot 상태가 없으므로 `PromptCacheWarmup`을 버리지만, `ModelProvider`의 OpenXLA 분기(`model_provider.rs`, 당시 938-953행)는 `provider.prompt_cache = prompt_cache_store`를 설정했다. store가 존재했고 `/v1/cache/stats`는 켜져 있다고 보고했으며, hit가 나지 않은 유일한 이유는 XLA 엔진이 그것을 조회하지 않았기 때문이다. 앞으로 store를 XLA 엔진에 연결하는 변경이 들어오면 이 성질은 조용히 깨졌을 것이다.
- **MLX 목록 누락.** `HYBRID_SSM_MODEL_TYPES`(`src/server/prompt_cache/hybrid_ssm.rs`)에 순환 상태를 가진 `falcon_h1`, `plamo2`, `qwen3_5`, `granitemoehybrid`가 없었다. APC는 기본값이 켜짐(`--apc-enabled`)이므로 이 계열들은 명목상 APC가 켜진 채로 돌았다.
- **감지기가 대상 계열에서 열린 쪽으로 실패**(구현 중 발견). `detect_hybrid_ssm_from_path`는 `config.json`을 엄격한 `serde_json`으로 파싱했다. HF는 Mamba2와 Nemotron-H config의 `time_step_limit`에 Python의 따옴표 없는 `Infinity`를 쓰므로 파싱이 실패했고, `resolve_prompt_cache_store`는 `Ok(Some(_))`만 매칭했기 때문에 바로 그 체크포인트들에서 APC가 켜진 채 남았다.
- **모듈 문서의 낡은 불변식.** `hybrid_ssm.rs` 헤더는 Baichuan-M1을 제외한 모든 계열이 `supports_batching() == false`를 반환해 `ModelOwned`로 간다고 적었다. Mamba2, Nemotron-H, Qwen3.5는 이제 MLX에서 batch되므로 실제로 안전을 보장하는 성질을 다시 써야 했다.

이슈가 적은 위험은 두 가지다. MLX 목록 변경은 네 계열의 사용자 가시 캐싱 동작을 바꾸므로 #3047 turn-boundary 상태 checkpoint와의 상호작용을 확인해야 하고, 나중에 추가되는 순환 계열도 목록에 넣어야 한다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `src/server/xla_prompt_cache_guard.rs` (신규, `xla-iree` 전용) | `xla_serve_backend_selected()`(provider가 쓰는 것과 같은 `MLXCEL_BACKEND == "xla"` 검사); `xla_recurrent_family(model_path)`는 로그할 계열명이나 `None`을 돌려주고, 파일을 읽거나 파싱하지 못하면 `UNDETECTED_FAMILY`(`"unreadable config.json"`)로 닫힌 쪽 실패; `disable_prompt_cache_for_xla_recurrent`는 `prompt_cache.enabled`와 `apc.enabled`를 끄고 계열명을 담은 `info!` 한 줄을 남김; `xla_provider_prompt_cache`는 순환 모델에 넘어온 store를 버림 |
| `src/server/startup.rs` | `resolve_prompt_cache_store`가 XLA 백엔드에서 가드를 먼저 호출하고, 크기 계산이나 store 생성 전에 `None`을 반환; 인라인 APC 제외 블록은 `disable_apc_for_hybrid_ssm`로 대체 |
| `src/server/model_provider.rs` | OpenXLA 분기가 `xla_serve_backend_selected()`를 쓰고, `new_with_xla_worker` 전에 `prompt_cache_store`를 `xla_provider_prompt_cache`로 거름 |
| `src/server/prompt_cache/hybrid_ssm.rs` | `model_type` 문자열 8개와 아키텍처 조각 4개(`falconh1`, `plamo2`, `qwen35`, `granitemoehybrid`) 추가; `detect_hybrid_ssm_from_path`가 파싱 전에 `crate::models::sanitize_config_json` 실행; 신규 `disable_apc_for_hybrid_ssm`(추출한 시작 시 제외 로직)와 `recurrent_prefix_sharing_family(config, backend_threads_recurrent_state)`(목록 먼저, 다음에 백엔드의 답, `model_type` 또는 `"recurrent"`를 로그); 모듈 문서가 안전 보장을 `SequenceStateBackend::keeps_state_outside_kv_slice`로 다시 서술 |
| `src/server/prompt_cache/mod.rs` | 두 신규 함수 re-export |
| `src/lib/mlxcel-xla/src/lib.rs` | `config_json_has_recurrent_state(text) -> Result<bool, String>`, 호출하는 emitter와 같이 `#[cfg(any(feature = "iree", test))]`로 gate; 지원하지 않는 계열은 `Err`이고 호출자의 목록이 결정 |
| `src/server/batch/xla_worker_admission.rs` | 바뀌지 않은 `PromptCacheWarmup` drop의 주석이 가드를 가리킴 |
| `src/server/mod.rs` | `xla-iree` 아래 가드 모듈 선언 |
| `src/main.rs`, `src/bin/mlx_server.rs` | `--apc-enabled` 도움말에 순환 계열 이름을 나열 |
| `src/server/xla_prompt_cache_guard_tests.rs` (신규) | 테스트 6개(5절) |
| `src/server/prompt_cache/hybrid_ssm_tests.rs` | 테스트 7개(5절) |
| `src/server/prompt_cache/apc_integration_tests.rs` | `state_checkpoints_do_not_depend_on_apc` |
| `src/lib/mlxcel-xla/src/emitter/recurrent_tests.rs` | `config_json_has_recurrent_state_matches_the_config` |
| `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `CHANGELOG.md` | 넓어진 제외 목록과 OpenXLA no-store 규칙; APC 환경 변수 절에 이 강제 동작 명시; MLX 동작 변경을 설명하는 `Changed` 항목 |

17개 파일, +722 / -38, 커밋 6개와 origin/main merge 1개.

### OpenXLA에서의 판단 흐름

```
resolve_prompt_cache_store (startup, server와 in-process)
  MLXCEL_BACKEND == "xla" ?
    xla_recurrent_family(model_path)
      config.json 읽기 또는 파싱 실패            -> "unreadable config.json"  (닫힌 쪽 실패)
      detect_hybrid_ssm(config)가 목록과 일치    -> family
      config_json_has_recurrent_state == true    -> model_type 또는 "recurrent"
      그 외                                      -> None (공유 가능)
    Some(family): 한 번 로그, prompt_cache.enabled = false, apc.enabled = false, None 반환
  (MLX 경로는 계속: ctx-checkpoints, 크기 계산, disable_apc_for_hybrid_ssm, store 생성)

ModelProvider OpenXLA 분기
  xla_provider_prompt_cache(model_path, store): 같은 판단, Some(store)를 버림
```

startup이 `None`을 반환하므로 `AppState`도 provider도 store를 갖지 않고, `/v1/cache` 경로는 `enabled: false`를 보고한다. router 모드는 원래 `None`을 넘긴다.

### MLX에서 목록 포함이 하는 일

목록의 소비자는 하나, 시작 시 APC 제외(이제 `disable_apc_for_hybrid_ssm`)이며 `apc.enabled = false`만 설정한다. APC는 dense KV store만 읽는다. insert 시 block-hash chain, 부분 일치 lookup, `apc_stats`, `/v1/cache/stats`다. snapshot, #3047 상태 checkpoint, paged KV, batching에는 관여하지 않는다.

네 계열은 이미 상태를 KV slice 밖에 둔다. Falcon-H1, PLaMo 2, Granite 4.0-H는 `ModelOwned`이고 Qwen3.5는 `SchedulerOwned`(#3254 이후의 `LayerState`)다. scheduler의 dense adopt와 donate는 batch 여부와 무관하게 이런 시퀀스를 거절한다. 실제로 이 확장이 바꾸는 것은 시작 경고와 보고되는 `apc_enabled: false`다.

## 3. 기술적 선택과 그 이유

**`has_recurrent_state()` 또는 목록으로 판단하고, 둘 중 하나만 쓰지 않는다.** 이슈는 XLA 판단을 `Config::has_recurrent_state()`로 한정했다. 이것은 emitter가 로드할 수 있는 config(현재 Mamba2)에만 답한다. `config_json_has_recurrent_state`는 emitter가 지원하지 않는 계열에서 `Err`를 돌려준다(테스트는 Jamba를 쓴다). `HYBRID_SSM_MODEL_TYPES`와의 합집합은 이들도 거부하므로 epic의 13개 타입이 지금 모두 덮이고, 나중에 로드 가능해지는 계열도 이미 거부된다. 목록 쪽은 운영자가 두 백엔드에서 보는 것이므로 새 계열을 추가할 단일 지점이기도 하다. 목록만 쓰면 목록에 없는 `model_type`으로 상태를 전달하는 config를 놓친다. 그 나머지 절반은 `backend_recurrent_state_refuses_an_unlisted_family`가 고정한다.

**startup에서 강제하고 provider에서 한 번 더.** startup은 store의 실제 생성자다(server와 in-process 모두 `resolve_prompt_cache_store`를 거친다). 그래서 거기서 두 플래그를 끄는 것이 `/v1/cache/stats`와 `AppState`를 사실대로 만든다. provider 분기 검사는 store와 함께 provider를 직접 만드는 다른 호출자에 대한 방어 심층화다. 이슈가 처음 정한 대로 provider에서만 막았다면 `AppState`는 store를 쥐고, stats 경로는 결코 hit할 수 없는 모델에 대해 캐시가 켜져 있다고 보고했을 것이다.

**OpenXLA에서는 prompt cache 전체를, MLX에서는 APC만 끈다.** MLX는 전체 snapshot과 #3047 turn-boundary checkpoint로 순환 상태를 복원한다. 이들은 정확한 prefix를 키로 쓰고 순환 상태를 더 짧은 prefix로 자르지 않는다. OpenXLA serve 엔진에는 adopt할 snapshot이 없으므로 store는 메모리만 잡고 결코 쓰이지 않을 캐시를 보고할 뿐이다. MLX에서 이 계열들의 store까지 끄면 동작하던 snapshot hit가 사라진다. MLX 서버 검사는 APC를 끈 상태에서 68 토큰 snapshot hit 1회를 보여 준다.

**읽을 수 없는 config에서는 닫힌 쪽으로 실패.** 첫 버전은 `config.json`이 없거나 파싱되지 않으면 "공유 가능"을 돌려줬다. 악용할 수는 없었다. serve 엔진이 같은 텍스트를 같은 파서(`RuntimeConfig::from_json` -> `Config::from_json_str`)로 파싱하므로 그런 모델은 로드되지 않고, XLA worker는 어차피 store를 읽지 않는다. 그래도 보안 리뷰는 이를 닫힌 쪽 실패로 바꿨다. 순환 상태를 배제할 수 없는 가드가 캐시를 켜 둬서는 안 되고, 파싱 오류는 모델 로드가 스스로 보고한다.

**감지기에서 관대한 파싱.** `detect_hybrid_ssm_from_path`는 이제 `serde_json` 전에 `sanitize_config_json`을 돌린다. 모델 로더가 이미 하는 것과 같다. 보안 리뷰는 sanitizer가 목록의 어떤 계열명에도 나타날 수 없는 부분 문자열만 바꾼다는 것을 확인했으므로, 이 재작성이 감지 결과를 바꿀 수 없다.

**`supports_batching()`을 유지하지 않고 불변식을 다시 서술.** 예전 헤더는 안전을 `supports_batching() == false`에 묶었지만, 목록 중 여섯 계열에서 이제 사실이 아니다. 순환 시퀀스를 block-hash store 밖에 두는 실제 성질은 상태 layout이다. `ModelOwned`나 `SchedulerOwned`(`keeps_state_outside_kv_slice`)이면 batching과 무관하게 dense adopt와 donate 경로가 거절한다. Baichuan-M1은 문서화된 dense 예외로 남는다.

**attention 형제와 겹치지 않는 조각.** `qwen35`는 `Qwen3_5ForConditionalGeneration`과 `Qwen3_5MoeForCausalLM`에 일치하고 `Qwen3ForCausalLM`이나 Qwen2.5에는 일치하지 않는다. `falconh1`은 Falcon에 일치하지 않는다. `new_fragments_do_not_match_attention_siblings`가 이를 고정하고, 구현 리뷰는 로컬 체크포인트 중 attention 전용 config에 일치하는 조각이 없음을 확인했다.

**`disable_apc_for_hybrid_ssm` 추출.** startup의 인라인 블록을 이름 있는 함수로 만들어 MLX 제외 로직을 따로 테스트할 수 있게 했다(`disable_apc_turns_apc_off_only_for_listed_families`). 테스트는 APC만 바뀌고 store, snapshot, checkpoint 예산은 설정대로 남는지 확인한다.

## 4. 리뷰 결과

**구현 리뷰**(2bf5a579에서 수정):

| 발견 | 심각도 | 상태 |
|---|---|---|
| startup 연결이 테스트되지 않았다. `resolve_prompt_cache_store`의 가드 호출을 꺼도 모든 가드 테스트가 통과했다 | MEDIUM | 수정: `startup_resolver_builds_no_store_for_an_xla_recurrent_model`이 `MLXCEL_BACKEND=xla`에서 resolver를 돌리고, 호출을 지우면 실패한다 |
| 운영자가 prompt cache를 켜지 않았어도 가드가 "prompt cache and APC disabled"를 로그한다 | LOW | 열림, 표면적 문제 |
| `ModelProvider` OpenXLA 분기의 drop에 직접 테스트가 없다. 실제 XLA worker가 필요하고, startup은 이미 `None`을 넘긴다 | LOW | 열림 |

리뷰어는 각 변경을 로컬에서 되돌려 확인했다. 목록과 sanitizer 변경 없이는 테스트 6개가, 가드를 무력화하면 가드 테스트 3개가 실패한다.

**보안 및 성능 리뷰**(c0566f66에서 수정): CRITICAL, HIGH, MEDIUM 발견 없음.

| 발견 | 심각도 | 상태 |
|---|---|---|
| `xla_recurrent_family`가 없거나 파싱되지 않는 `config.json`에 "공유 가능"을 돌려줘 store가 붙은 채 남았다(열린 쪽 실패, 악용 불가) | LOW | 수정: 닫힌 쪽 실패; `an_unreadable_or_unparseable_config_fails_closed`는 이전 가드에서 실패 |

추가 발견 없이 확인한 항목: 테넌트 간 경로(startup, in-process, router child, `prompt_cache`가 읽기 전용인 runtime settings PATCH, `None`을 보는 `/v1/cache` 경로); 조작된 config(`has_recurrent_state()`는 엔진이 서빙하는 같은 `Config`에서 나오고, 레이어 수는 `MAX_CONFIG_LAYERS`로 제한되며, `conv_kernel < 2`는 `conv_kernel - 1` 전에 거부); 환경 불일치(가드와 OpenXLA 분기가 같은 `xla_serve_backend_selected()`를 호출하고, 테스트 외 코드는 `MLXCEL_BACKEND`를 설정하지 않으며, 불일치의 최악은 MLX에서 캐시가 꺼지는 것). 성능: 감지는 startup과 provider 생성 시에만 돌고 매번 작은 파일 하나를 읽는다.

**마무리**(58ed3023, a95dd225): origin/main을 충돌 없이 merge했고, resolver 테스트가 drop guard로 `MLXCEL_BACKEND`를 복원하게 해 panic이 이후 테스트로 새지 않게 했다. 이 리포트를 쓸 때 브랜치는 이미 origin/main과 최신 상태였으므로 리포트용으로 테스트를 다시 돌리지 않았다.

## 5. 테스트와 검증

모든 결과는 PR 본문과 리뷰 코멘트에서 가져왔고, ROCm gfx1151에서 돌렸다.

추가된 단위 테스트(15개):

- `xla_prompt_cache_guard_tests.rs`(6): `every_epic_recurrent_type_gets_no_store_and_no_apc`(epic의 13개 타입: `mamba`, `mamba2`, `falcon_mamba`, `jamba`, `nemotron_h`, `falcon_h1`, `plamo2`, `qwen3_next`, `qwen3_5`, `kimi_linear`, `rwkv7`, `recurrent_gemma`, `granitemoehybrid`), `an_emitter_loadable_mamba2_is_refused_through_has_recurrent_state`(따옴표 없는 `Infinity`가 있는 작은 HF 형식 Mamba2), `an_attention_model_keeps_its_store_and_apc`, `a_local_hf_mamba2_checkpoint_is_refused`, `startup_resolver_builds_no_store_for_an_xla_recurrent_model`, `an_unreadable_or_unparseable_config_fails_closed`
- `hybrid_ssm_tests.rs`(7): `every_epic_recurrent_type_is_refused_prefix_sharing_on_any_backend`, `backend_recurrent_state_refuses_an_unlisted_family`, `newly_listed_hybrids_are_detected_every_way`(`model_type`, VLM `text_config`, 아키텍처 이름만으로), `new_fragments_do_not_match_attention_siblings`, `detect_from_path_tolerates_bare_infinity`, `disable_apc_turns_apc_off_only_for_listed_families`, `local_qwen35_and_granite_h_checkpoints_turn_apc_off`(없으면 건너뛰고, `MLXCEL_REQUIRE_MODELS=1`이면 실패)
- `apc_integration_tests.rs`(1): `state_checkpoints_do_not_depend_on_apc`. Qwen3.5 형태의 boundary turn 네 개, turn 2 뒤 히스토리를 수정; APC 설정 양쪽 모두 `matched_len` 20인 `CheckpointHit`
- `mlxcel-xla`(1): `config_json_has_recurrent_state_matches_the_config`(Mamba2 `Ok(true)`, Llama `Ok(false)`, Jamba `Err`)

테스트 스위트(`MLXCEL_REQUIRE_MODELS=1`):

| 명령 | 결과 |
|---|---|
| `cargo test --profile test-fast --lib prompt_cache` | 382 통과 |
| 같은 명령에 `--features xla-iree`와 `xla_prompt_cache_guard` 추가 | 386 통과 |
| `mlxcel-xla` `config_json_has_recurrent_state_matches_the_config` | 통과 |
| `cargo fmt --check`; 기본, `--features xla-iree`, `-p mlxcel-xla`(`iree` 유무)에 대한 `cargo clippy --all-targets -D warnings` | 통과 |

종단 간 검사:

| 검사 | 결과 |
|---|---|
| OpenXLA 서버(hip), `mamba2-130m`, `--prompt-cache-enabled=true --apc-enabled=true --metrics`, 68개 프롬프트 토큰 중 마지막 몇 개만 다른 completion 두 개 | prompt-cache hit, miss, snapshot hit 모두 0; `/v1/cache/stats` `enabled: false`; 두 출력 모두 `mlxcel generate`와 같음 |
| MLX ROCm 서버, `qwen3.5-0.8b-4bit`와 `granite-4.0-h-tiny-4bit`, 각각 completion 세 개(prefix를 공유하는 두 개, 이어서 요청 1을 확장하는 하나) | startup이 APC 제외를 로그; stats는 `apc_enabled: false`, dense lookup 0, snapshot hit 1(68 토큰); 모든 출력이 APC가 켜진 origin/main과 바이트 단위로 같음; 요청 1과 2는 `--no-prompt-cache` 및 CLI와 같음; Granite 요청 3도 일치 |
| Qwen3.5 요청 3(snapshot hit) 대 cold run | origin/main과 이 브랜치 모두 약 토큰 9에서 갈라짐: 알려진 two-forward partition 효과이며 이 변경과 무관 |

처리량이나 메모리는 측정하지 않았다. 이 변경은 시작 시 판단이고 요청당 작업을 더하지 않는다.

## 6. 남은 작업과 한계

- **Falcon-H1과 PLaMo 2는 종단 간 테스트가 없다.** 로컬 체크포인트가 없어 config 감지 단위 테스트만 이들을 덮는다.
- **목록은 모델 지원과 함께 늘어나야 한다.** OpenXLA emitter가 로드하지 못하고 `HYBRID_SSM_MODEL_TYPES`에도 없는 미래의 순환 계열은 가드를 통과한다(`has_recurrent_state` 조회가 `Err`이고 `false`로 읽힌다). 그런 모델은 OpenXLA에서 로드되지 않으므로 store가 읽히지는 않지만, MLX APC 제외도 그것을 놓친다. 열거 테스트는 알려진 13개 타입만 지킨다.
- **가드는 `xla-iree`에서만 컴파일된다.** 이 feature가 없는 빌드에는 OpenXLA 백엔드가 없으므로 잃는 것은 없지만, 가드 테스트는 `xla-iree` 빌드에서만 돈다.
- **LOW, 열림:** prompt cache가 원래 꺼져 있어도 가드가 비활성화 로그를 남긴다.
- **LOW, 열림:** `ModelProvider` OpenXLA 분기의 drop에 직접 테스트가 없다.
- **OpenXLA 순환 상태용 캐싱 방식은 없다.** 이슈 범위 밖이다. OpenXLA 순환 모델은 매 요청 prefill을 다시 한다. OpenXLA hybrid(#3344)는 판단의 두 절반 모두를 통해 이 가드를 물려받는다.
- **CUDA와 Metal은 돌리지 않았다.** 가드는 백엔드와 무관한 Rust 코드지만 서버 검사는 ROCm에서만 했다.

## 7. 학습 포인트

- 소비자가 우연히 자원을 무시해서 성립하는 불변식은 불변식이 아니다. XLA worker는 store를 읽지 않았지만 store는 존재했고 켜져 있다고 보고됐다. 단일 생성자에서 자원을 없애고 거기서 검증하면, 앞으로 연결이 바뀌어도 성질이 유지된다.
- 연결 테스트는 연결을 지웠을 때 실패해야 한다. startup 호출을 지워도 가드 단위 테스트는 모두 통과했고, 실제 환경 선택 아래에서 실제 resolver를 돌리는 테스트만이 그것을 고정했다. 각 변경을 되돌려 어떤 테스트가 실패하는지 보는 것이 이런 빈틈을 찾는 싼 방법이다.
- 파싱 오류에서 열린 쪽으로 실패하는 감지기는 자기가 대상으로 하는 입력에서 실패한다. 따옴표 없는 `Infinity`를 담는 것이 바로 HF 순환 config이므로, 엄격한 파싱은 정확히 Mamba2와 Nemotron-H에서 검사를 꺼 버렸다. HF JSON을 보는 안전 감지기는 로더와 같은 관대한 파싱을 쓰고, 파싱 실패가 무엇을 뜻하는지 명시적으로 정해야 한다.
- 모듈의 안전 논증이 대리 지표(`supports_batching() == false`)를 이름으로 들면, 주변 변경 뒤에도 그 지표가 여전히 성립하는지 확인해야 한다. 여기서 실제 성질은 상태 layout이었고, 대리 지표는 여섯 계열에서 조용히 사실이 아니게 됐다.
- 두 백엔드가 상태를 다르게 복원하면, 같은 "prefix를 공유하지 않는다" 규칙도 다른 스위치에 대응한다. 정확한 prefix snapshot이 여전히 상태를 제대로 복원하는 MLX에서는 APC만, 아무것도 상태를 복원할 수 없는 OpenXLA에서는 store 전체를 끈다.
