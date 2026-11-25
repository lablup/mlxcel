# PR #3381: Inkling 2 CLI 보완 (reasoning effort, 배너 개수, 4-bit trace)

**날짜**: 2026-11-24
**상태**: Draft PR 공개 (#3378 종료). 범위를 좁힌 테스트 통과, M3 Ultra (512 GB)에서 Inkling 2 Mini 체크포인트 두 개로 검증

## 요약

이 PR은 #3367이 Inkling 2에 남긴 세 가지 빈틈을 메운다. `mlxcel generate`와 `mlxcel run`이 `--reasoning-effort`를 받고, 이 값은 단발 렌더, `run -p`, 두 채팅 REPL 모두에서 chat template까지 전달된다. KV 배너는 로드된 모델이 실제로 가진 캐시 수를 세므로, Mini가 물리 레이어 24 대신 `48 of 48`을 출력한다. mlx-vlm이 mlx-community 4-bit 체크포인트를 읽게 해 준 remap을 커밋했고, #3367이 생략한 4-bit 이미지 trace도 실행했다. prompt id 600개가 모두 일치하고 결정된 위치의 불일치는 70곳 중 0곳이다.

## 1. 문제

- **Effort.** Inkling 2 template은 `reasoning_effort`(레벨 이름 또는 [0.0, 0.99]의 숫자, 기본 0.9)를 읽는다. 서버는 요청마다 이 값을 받지만 CLI는 kwargs 없이 렌더링해서 레벨을 지정할 방법이 없었다.
- **배너.** `kv_cache_mode_banner_suffix`는 모델을 로드하기 전에 `config.json`의 `num_hidden_layers`를 읽었다. Mini는 물리 레이어 24개를 48번 방문하며 방문마다 캐시를 두므로 `24 of 24`로 잘못 출력했다.
- **4-bit trace.** mlx-vlm 466069f5는 native 레이아웃만 읽는다. #3367은 커밋되지 않은 이름 변경 스크립트로 텍스트 trace만 실행했고 이미지 trace는 실행하지 못했다.

## 2. 설계

### 2.1 Reasoning effort

서버는 요청 필드를 JSON 문자열로 전달하고, 비활성 레벨(`none`, `off`, `0` 등)은 전달하지 않은 채 `enable_thinking=false`로 바꾼다. Inkling 2는 둘 다 반대로 동작해야 한다. `none`은 레벨 이름으로 template에 도달해야 하고(`Thinking effort level: 0`으로 렌더됨), `0.5`는 JSON 숫자여야 한다. 문자열 `"0.5"`는 `Unknown reasoning_effort` 오류를 낸다. 그래서 `server::in_process::reasoning_effort`가 두 가지를 한다. `apply_to_chat_body`는 이식 가능한 `reasoning_effort` 필드를 설정해 서버의 기존 해석이 그대로 돌고 `reasoning_strength`나 `enable_thinking`을 채우게 하고, 동시에 명시적 `chat_template_kwargs.reasoning_effort`를 타입이 있는 값(유한한 숫자로 파싱되면 숫자, 아니면 소문자 문자열)으로 설정한다. 서버의 키별 우선순위 규칙에 따라 명시적 kwarg가 이긴다. `cli_template_kwargs`는 같은 방식으로 합성 요청을 만들어 서버의 `resolve_effective_kwargs`를 호출하므로 단발 렌더와 채팅 경로가 하나의 매핑을 공유하고, CLI는 별도 매핑 표를 두지 않는다.

`CliPromptMedia`가 레벨을 `resolve_cli_prompt`와 `apply_vlm_chat_template`로 전달하고, `CliServerSettings.reasoning_effort`가 모든 채팅 요청 본문으로 전달한다. 유효성은 template 자신의 `raise_exception` 메시지가 판단하며 그대로 출력한다(`Chat template rejected the request: Unknown reasoning_effort: bogus`). 파싱 단계에서는 빈 값만 거부한다.

비활성 레벨은 해당 키를 읽는 template에서 `enable_thinking=false`도 설정한다. 같은 필드에 대해 서버가 하는 동작과 같다. 두 키를 모두 읽지 않는 template은 어떤 레벨에서도 바이트 단위로 같은 프롬프트를 렌더하고, `enable_thinking`을 읽는 template은 의도적으로 달라진다.

### 2.2 배너

`print_kv_cache_mode_banner`는 이제 일반 경로에서 `load_generation_model` 이후에 실행되며, 모델이 `kv_cache_storage_modes()`를 반환하면 그것을 센다. 캐시마다 항목이 하나이고, 유효 모드를 실제로 저장하는 항목만 적용된 것으로 센다. Mini는 FP16 항목 48개를 반환하므로 Fp16은 `48 of 48`, int8 요청은 `0 of 48`로 표시된다. `None`을 반환하는 모델은 기존 경계 레이어 규칙과 함께 `num_layers()`를 쓰므로 dense 모델의 배너는 그대로다. 파이프라인과 OpenXLA 경로는 `LoadedModel`을 로드하지 않으므로 이전처럼 로드 전에 config 값으로 출력한다.

### 2.3 4-bit remap

`scripts/inkling2_mlx_community_remap.py`는 체크포인트를 이슈가 말한 native `model.llm.*` 레이아웃이 아니라 mlx-vlm의 변환 레이아웃(`Model.sanitize`가 그대로 통과시키는 이름)으로 다시 쓴다. 둘 다 같은 모델을 로드하며, 변환 레이아웃은 native 전용 변환을 다시 거치지 않아도 된다. 이름 변경은 `src/models/inkling2/sanitize.rs`가 이미 쓰는 표를 따르고, `gate_up_proj`는 mlx-vlm의 `_split_gate_up`으로 나누며, `weight_scale_2 = global_scale / 2688`로 설정하고, expert별 모드를 `config.json`에 기록한다. 모드는 packed 폭과 scale 폭의 비율로 추정한다(2는 NVFP4, 8은 MXFP8). 체크포인트에는 `processor_config.json`이 없으므로 `models/inkling-2-mini`에서 해당 파일을 링크한다. 두 trace 스크립트는 `--inkling2-mlx-community`로 이 remap을 사용한다.

## 3. 검증

| 검사 | 결과 |
|---|---|
| `none` / `0.5` / `XHigh` / 기본값의 effort (렌더된 prompt id 디코드) | `Thinking effort level: 0` / `0.5` / `0.99` / `0.9` |
| `--reasoning-effort bogus` | `Chat template rejected the request: Unknown reasoning_effort: bogus` |
| `run -p`와 REPL에서 `none`과 `high` 비교 | 응답이 다르므로 플래그가 in-process 채팅 요청에 도달함 |
| Mini 배너 | `KV cache mode: fp16 (applied to 48 of 48 layers)` |
| remap을 거친 4-bit 텍스트 trace (1024 위치) | top-1 차이 33개, 결정된 위치 372곳 중 0곳, perplexity -0.32% |
| 4-bit 이미지 trace, `tests/fixtures/sapiens2/person.jpg` | prompt id 600개 일치, top-1 차이 127곳 중 7곳, 결정된 위치 70곳 중 0곳, perplexity -1.03% |

텍스트 corpus는 #3367의 것(기록되지 않음)과 달라서 결정된 위치 수가 0/494와 다르다. 결정된 위치에서 불일치가 없다는 결과는 같다.

단위 셀렉터: `server::in_process` 27, `cli::` 321, `coverage_tests` 18, `commands::generate` 119, `commands::run` 11, `commands::cli_server` 6, `commands::chat` 19. lib, bin, tests 전체에 대해 `-D warnings` clippy 통과. `tests/test_inkling2_mlx_community_remap.py` 6개 통과.

## 4. 한계와 후속 작업

- `--no-chat-template`에서는 `--reasoning-effort`가 아무 효과가 없다.
- 배너는 생성이 레이어 모드 표를 주입하기 전의 모델 보고를 센다. 저장 보고에 그 표가 필요한 계열은 이 시점에 `None`을 반환하므로 `num_layers()` 값을 그대로 쓴다.
- 배치 디코드는 #3377이다.
