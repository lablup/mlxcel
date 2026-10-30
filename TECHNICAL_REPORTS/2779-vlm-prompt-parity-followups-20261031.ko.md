# PR #2779: #2750 VLM prompt parity 후속 작업 마무리

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Python
**위험도**: 낮음 (OpenXLA backend의 Gemma 3 이미지 prompt가 273 token에서 272 token으로 줄어 그 backend의 Gemma 3 이미지 출력이 바뀜. MLX 경로는 이전과 같은 id를 만들고, 나머지는 trace harness, 문서, 테스트)

## 요약

#2750은 MLX 경로에서 InternVL, MiniCPM-V 4.6, Gemma 3 이미지 prompt를 transformers와 맞췄고, 남은 세 가지 잔여 항목을 issue #2761로 모았다. 이 PR은 세 가지를 모두 처리한다. Gemma 3 newline fusion을 MLX runtime과 OpenXLA host preprocessor가 함께 호출하는 helper 하나(`expand_image_blocks_with_newline_fusion`)로 옮겨, OpenXLA 경로도 이제 273개가 아니라 transformers와 같은 272개 id를 만든다. newline-run table은 이미지 요청마다가 아니라 모델당 한 번만 probe한다. 이미지 prompt의 server-wide `enable_thinking` 기본값은 유지하고 의도된 차이로 문서화했으며, 두 trace arm 모두에 thinking을 끄는 스위치(`MLXCEL_TRACE_ENABLE_THINKING`, `--enable-thinking`)를 추가했다. thinking을 끄면 gemma-4-e2b와 MiniCPM-V 4.6이 reference id와 정확히 같아진다. 또한 실제 MiniCPM-o checkpoint(`mlx-community/MiniCPM-o-4_5-4bit`)를 model store에 추가하고, #2750이 합성 파일로만 검증했던 built-in ChatML template을 이 checkpoint로 smoke했다.

## 1. 문제 정의

### 1.1 backend 둘, Gemma 3 확장 방식 둘

transformers의 `Gemma3Processor`는 tokenize 전에 text에서 `<start_of_image>`를 확장하므로, wrapper의 `\n\n`이 template의 `user\n`과 합쳐져 `\n\n\n` token 하나가 된다. mlxcel은 token 단위로 확장하며, #2750은 확장 후 인접 newline run을 다시 합치는 방식(`merge_image_block_newline_runs`)으로 MLX 경로를 고쳤다. OpenXLA 경로(`host_preprocessor.rs`의 `Gemma3IreeHostPreprocessor::prepare_iree`)는 같은 `\n\n` wrapper로 `apply_image_token_blocks`만 호출하고 merge를 하지 않았으므로 273-token prompt가 그대로 남았다. 두 backend는 모델이 돌기 전부터 prompt가 달랐고, 수정이 한쪽에만 있었으므로 다음 변경에서 둘이 더 벌어지는 것을 막을 장치도 없었다.

MLX 경로는 이미지 요청마다 newline-run table을 다시 만들기도 했다. `newline_run_token_ids`는 `"\n"`을 1, 2, ... 번 반복한 문자열을 하나의 token이 아니게 될 때까지 encode하며, Gemma 3에서는 32번(run 1~31과 실패하는 probe 하나)이다. 이 table은 tokenizer에만 의존하는 모델별 상수인데 요청마다 다시 계산되고 있었다.

### 1.2 thinking 기본값이 parity 버그처럼 보였다

mlxcel은 tokenizer에 think marker가 있으면 `enable_thinking`을 기본으로 켠다(`server::chat_front`의 `apply_thinking_default`, mlx-lm PR #1114와 정렬). transformers와 mlx-vlm은 끈다. 이 때문에 gemma-4-e2b-it-4bit의 이미지 prompt가 mlx-vlm의 271개 대비 278개(system `<|think|>` turn)가 되었고, upstream template을 쓰는 MiniCPM-V 4.6의 차이 일부도 여기서 왔다(transformers는 빈 think block을 닫고 mlxcel은 `<think>`를 연다). #2750은 이 항목을 열어 두었다. 이미지 trace(`examples/logit_trace/image.rs`)는 항상 server 기본값으로 render했으므로 두 family 모두 같은 template 설정으로 reference와 비교할 수 없었고, 길이 차이가 나도 진짜 processor 버그인지 구분할 수 없었다.

### 1.3 MiniCPM-o fallback은 실제 checkpoint를 본 적이 없었다

#2750은 template이 없는 `MiniCPMV46VLM`, `MiniCPMOVLM` checkpoint용 fallback으로 `MINICPM_CHATML_TEMPLATE`을 추가했다. 그렇지 않으면 server가 generic `User: ...\n\nAssistant: ` prompt를 render하고, 이미지 확장이 그것을 ChatML turn으로 한 번 더 감쌌기 때문이다. store에 MiniCPM-o checkpoint가 없었으므로 MiniCPM-o 쪽은 합성 파일을 쓰는 `template_less_minicpmo_server_prompt_matches_the_cli_prompt`로만 검증되었다.

### 1.4 방치했을 때의 위험

| 위험 | 영향 | 가능성 |
|---|---|---|
| 같은 요청에 대해 OpenXLA와 MLX Gemma 3가 다른 prompt를 만듦 | 중간: backend마다 출력이 다르고, cross-backend trace가 모델이 아니라 prompt를 측정함 | 이 PR 이전에는 확정 |
| 요청마다 newline probe | 낮음: Gemma 3 이미지 요청당 짧은 문자열 encode 약 32회 | 확정, 작음 |
| thinking 기본값 차이를 processor 버그로 오해 | 중간: 헛된 조사 또는 기본값을 잘못 "수정" | gemma-4, MiniCPM-V 4.6 trace에서 높음 |
| 실제 파일에서 MiniCPM-o fallback 실패 | 중간: template 없는 MiniCPM-o server에서 framing 중복 | 이 PR 이전에는 알 수 없음 |

## 2. 기술적 선택과 그 이유

### 2.1 두 backend가 호출하는 확장 helper 하나

`gemma3_vlm_prompt.rs`에 `expand_image_blocks_with_newline_fusion(tokens, info, image_count, newline_runs)`가 추가되었다. 이 함수는 `apply_image_token_blocks`를 실행하고, block이 확장되었으며 layout에 wrapper가 있을 때(`has_image_block_wrapper`: `block_prefix_tokens` 또는 `block_suffix_tokens`가 비어 있지 않음) merge를 실행한다. 반환값은 `apply_image_token_blocks`와 같으므로 호출부의 stats 처리는 그대로다. `vlm_runtime.rs`의 MLX `Standard` arm과 새 free function `host_preprocessor::gemma3_iree_prompt_tokens`가 둘 다 이 helper를 호출한다. `prepare_iree`는 그 함수를 호출한 뒤 이전처럼 merge된 길이로 `validate_sequence_capacity`를 실행한다.

대안은 `prepare_iree`에 merge 호출을 복사하는 것으로, diff는 가장 작다. 하지만 두 경로가 갈라진 원인이 바로 그것이다. #2750은 한 호출부를 고쳤고 다른 호출부는 그 테스트 범위 밖에 있었다. 진입점이 하나면 한 backend에만 merge를 적용하고 다른 쪽을 빠뜨릴 수 없다.

OpenXLA token 단계와 그 block layout(`gemma3_iree_token_block_info`)은 `cfg(any(feature = "xla-iree", test))` 아래의 free function이다. `Gemma3IreeHostPreprocessor`는 `xla-iree`에서만 존재하므로, 그 method로 두면 기본 build에서 테스트할 수 없다. free function이면 테스트할 수 있고, `xla_gemma3_prompt_matches_transformers_ids`는 OpenXLA layout이 MLX loader의 layout(`gemma3_info(256)`)과 같다는 것도 assert하므로 `GEMMA3_VLM_NEWLINE_WRAPPER_TOKEN_ID` 상수(108)가 MLX의 `vec![108]`과 묶인다.

graph identity는 바뀌지 않는다. prompt id는 `stable_identity`나 `prepared_gemma3.rs`의 일부가 아니고, sequence capacity 검사는 원래도 확장 뒤에 실행되었다.

### 2.2 backend별 newline table 위치

MLX: `VisionModule`에 `newline_runs: OnceLock<Vec<i32>>`를 두고, 처음 필요한 이미지 요청에서 `cached_newline_runs(cell, encode)`로 채운 뒤 재사용한다. MLX의 vision loader에는 load 시점의 tokenizer가 없지만 모든 요청이 `encode` closure를 가지므로, lazy fill이 자연스러운 접점이다. `VisionModule` literal 10개(loader 9개와 `loaded_model_tests.rs`)에 `OnceLock::new()`가 추가되었다. wrapper가 없는 layout은 cell을 건드리지 않고 빈 slice를 넘긴다.

OpenXLA: `load_gemma3_iree_host_preprocessor`가 `crate::tokenizer::load_tokenizer(model_path)`로 tokenizer를 load해 한 번 probe하고, 그 table을 `Gemma3IreeHostPreprocessor::from_parts`에 넘겨 저장한다. preprocessor는 load 시 한 번 만들어지고 요청 tokenizer를 갖지 않으므로, 여기서 즉시 만드는 것이 lazy cell보다 단순하다. 비용은 load 시 `tokenizer.json` parse 한 번이며, server가 이미 요구하는 경로다.

`Mutex<Option<_>>`가 아니라 `OnceLock`인 이유는 값이 한 번 쓰이고 이후 매 요청 읽히기 때문이다. 동시에 들어온 첫 요청은 probe 한 번 동안만 대기하고, probe 중 panic이 나면 cell이 비어 있으므로 다음 요청이 다시 시도한다.

### 2.3 빈 table은 실패 대신 경고

두 encode closure 모두 tokenizer 오류를 빈 vector로 바꾸므로, probe가 실패하면 table이 비고 merge는 no-op이 되며 prompt는 조용히 273 token으로 돌아간다. 이 PR 이전에는 probe가 요청마다 돌았지만, 이제 빈 결과가 모델 수명 동안 cache되거나(MLX) load 시 저장된다(OpenXLA). security review가 이를 지적했고, 이제 table을 채우는 곳(`cached_newline_runs`와 OpenXLA loader)에서 `warn_if_newline_runs_empty`가 `tracing::warn!`를 한 번 남기므로 요청마다가 아니라 모델당 한 번 경고한다. load나 요청을 실패시키지 않은 이유는, fuse되지 않은 prompt도 reference와 token 하나 차이일 뿐 모델이 답할 수 있는 유효한 prompt이고, 정상 tokenizer는 `\n`을 오류 없이 encode하기 때문이다.

### 2.4 thinking 기본값은 유지하고 harness가 맞추게 한다

검토한 선택지는 세 가지다.

| 선택지 | 장점 | 단점 |
|---|---|---|
| 이미지 요청에서만 `enable_thinking`을 끔 | 기본 상태에서 이미지 prompt가 transformers, mlx-vlm과 같음 | 같은 checkpoint가 이미지 유무에 따라 다르게 render됨. 한 대화 안의 text 요청과 이미지 요청이 어긋남 |
| server 전체에서 끔 | 모든 곳에서 reference와 같음 | mlx-lm #1114 정렬을 되돌리고 thinking 모델의 모든 text 요청을 바꿈 |
| **선택: 기본값 유지, trace 스위치 추가** | 동작 변화 없음. 설정을 맞춘 상태로 parity 측정 가능 | thinking 가능 checkpoint에서 기본 이미지 prompt는 여전히 reference와 다름 |

client는 이미 `chat_template_kwargs.enable_thinking`으로 요청마다 thinking을 끌 수 있으므로, 이 차이는 기능 부족이 아니라 기본값 차이다. `docs/benchmarks.md`는 이를 기존 항목들 옆에 의도된 차이로 기록한다.

`examples/logit_trace`는 `parse_enable_thinking`으로 `MLXCEL_TRACE_ENABLE_THINKING`을 읽는다. `0`과 `1`은 off와 on이고, unset이나 공백은 기본값 유지이며, 그 밖의 값(`yes`, `true`)은 조용히 기본값으로 처리하지 않고 오류로 거부한다. 값은 `load_model_chat_template` 뒤에 `set_default_enable_thinking`으로 적용되고 `# enable_thinking`으로 출력된다. 거부가 전파되도록 `ImageArm::from_env`는 이제 `Result<Option<Self>>`를 반환한다. `scripts/mlx_vlm_image_logit_trace.py`에는 `--enable-thinking {true,false}`가 추가되어 template kwarg로 `apply_chat_template`에 전달되고 함께 출력된다. 엄격한 parsing이 중요한 이유는, 오타가 나면 다른 설정으로 render되었는데도 정상처럼 보이는 trace가 만들어지기 때문이다.

### 2.5 테스트가 직접 만드는 template 없는 MiniCPM-o 사본

issue는 template을 제거한 작업 사본을 수동으로 만들라고 했다. 테스트는 대신 실행할 때마다 사본을 만든다(`minicpm_chat_template_tests.rs`의 `template_less_copy`). `chat_template.jinja`, `chat_template.json`, `tokenizer_config.json`을 제외한 모든 checkpoint 파일을 symlink하고, `tokenizer_config.json`은 `chat_template` key를 뺀 채로 다시 쓴다. store는 배포된 checkpoint를 그대로 유지하고, 테스트에 수동 단계가 필요 없으며, symlink 덕분에 6 GB weight를 복사하지 않는다. Unix가 아닌 target에서는 복사로 대신한다.

## 3. 변경 요약

19개 파일, +480/-47: commit 두 개(fd941589 fix, a36063ac 후속 정리)와 origin/main merge(f2d0e9c5, #2773 포함).

- `src/multimodal/gemma3_vlm_prompt.rs`: `expand_image_blocks_with_newline_fusion`, `cached_newline_runs`, `warn_if_newline_runs_empty`, `has_image_block_wrapper`.
- `src/multimodal/vlm_runtime.rs`: `Standard` arm이 cache된 table로 공용 helper를 호출.
- `src/multimodal/host_preprocessor.rs`: `gemma3_iree_token_block_info`, `gemma3_iree_prompt_tokens`. `Gemma3IreeHostPreprocessor`가 `newline_runs`를 저장하고 `prepare_iree`가 공용 token 단계를 사용.
- `src/loading/vlm_gemma.rs`: OpenXLA loader가 checkpoint tokenizer로 table을 한 번 생성.
- `src/vision/mod.rs`, `src/loading/vlm_*.rs` 6개, `src/loaded_model_tests.rs`: `newline_runs` field와 `OnceLock::new()` 초기화.
- `examples/logit_trace.rs`, `examples/logit_trace/image.rs`: `MLXCEL_TRACE_ENABLE_THINKING`, `parse_enable_thinking`, `# enable_thinking` 줄.
- `scripts/mlx_vlm_image_logit_trace.py`: `--enable-thinking`.
- `docs/benchmarks.md`: MiniCPM-V 4.6(upstream template)와 gemma-4-e2b의 thinking-off 행, MiniCPM-o 행, thinking 기본값 차이, OpenXLA 설명. `docs/environment-variables.md`: 새 변수.
- Model store(저장소 밖): spark-102에 `minicpm-o-4.5-4bit` 추가, manifest 행 추가, 두 host에서 link view 재생성.

추가된 테스트(7개, 그중 2개는 ignored local-checkpoint 테스트):

| 테스트 | 고정하는 것 |
|---|---|
| `shared_expansion_helper_fuses_the_wrapper_newlines` | helper가 transformers의 272개 id를 만듦 |
| `xla_gemma3_prompt_matches_transformers_ids` | OpenXLA token 단계가 272개를 만들고, 그 layout이 MLX loader와 같음 |
| `newline_runs_are_probed_once_per_model` | counting encoder가 두 번 호출되어도 probe는 한 번 |
| `enable_thinking_accepts_zero_one_and_unset`, `enable_thinking_rejects_other_values` | env parser |
| `local_gemma4_template_thinks_by_default_and_not_when_disabled` (ignored) | load된 기본값에서 `<|think|>` system turn이 있고, 기본값을 끄면 없음 |
| `local_template_less_minicpmo_renders_one_chatml_turn` (ignored) | ChatML user, assistant opener가 각각 하나, `User:`/`Assistant:` 없음, server prompt가 CLI prompt와 같음 |

## 4. 결과

main 52e5fd52에 새 테스트만 얹으면 Gemma 3 테스트 세 개는 compile되지 않는다(helper, `cached_newline_runs`, free function이 없음). 새 함수 안에 main의 동작을 되돌려 넣으면 `left: 273, right: 272`와 `left: 8, right: 4`(두 번째 호출이 다시 probe)로 실패한다.

ignored 테스트의 mutation check:

- `set_default_enable_thinking`이 `false`를 저장하도록 강제하면 gemma-4 테스트의 첫 assertion이 실패한다. getter `default_enable_thinking()`만 `false`로 강제하면 render가 바뀌지 않았는데, render가 field를 직접 읽기 때문이다. 그래서 테스트는 이제 getter도 assert한다.
- `builtin_chat_template` arm에서 `MiniCPMOVLM`을 빼면 MiniCPM-o 테스트가 "generic framing leaked into the prompt"로 실패한다.

GB10 smoke(`gpu-lock`, `MLXCEL_REQUIRE_MODELS=1`, `tests/fixtures/test_image_shapes.png`):

| Family | CLI prompt id 수 | server `prompt_tokens` | 출력 |
|---|---|---|---|
| gemma-3-4b-it-4bit | 272 | 272 | 사각형, 원, 삼각형을 설명 |
| gemma-4-e2b-it-4bit | 278 | 278 | 일관된 출력. 64 token 전부 reasoning channel (thinking 기본값 on) |
| minicpm-v-4.6-bf16 (template 없음) | 82 | 82 | 세 도형을 설명 |
| minicpm-o-4.5-4bit, template 없는 사본 | 78 | 78 | 64 token에서는 reasoning만. `-n 400`에서 CLI가 빨간 사각형, 파란 원, 초록 삼각형을 답함 |

thinking 스위치를 쓴 trace `# prompt_tokens`:

- gemma-4-e2b: 기본값 278, `MLXCEL_TRACE_ENABLE_THINKING=0`이면 271로 mlx-vlm 0.7.8과 id가 같다. mlx-vlm에 `--enable-thinking true`를 주면 278이므로 차이 전체가 기본값 때문이었다.
- upstream template을 쓰는 MiniCPM-V 4.6: 기본값 84, thinking off 86으로 transformers(86)와 id가 같다. mlx-vlm은 `(<image>./</image>)` marker의 괄호를 남기고 빈 think block을 지우므로 84에 머문다. 이는 #2750이 이미 문서화한 차이다.

## 5. 트레이드오프와 잔여 항목

- **OpenXLA Gemma 3 경로는 end to end smoke를 하지 않았다.** token 단계는 unit test로 검증되고 `cargo check --features cuda,xla-iree --all-targets`는 통과하지만, 이 PR에서 실제 checkpoint로 OpenXLA Gemma 3 이미지 요청을 실행하지는 않았다.
- **OpenXLA Gemma 3 mask는 여전히 MLX와 다르다.** main에서 이 branch로 merge된 #2773은 MLX Gemma 3 이미지 prefill을 transformers mask로 바꿨지만, `prepared_gemma3.rs`는 여전히 `GEMMA3_VLM_MASK_MODE = "gemma3-vlm-bidirectional-padding-f32-min-v1"`을 쓴다. 이 PR 이후 두 backend는 같은 272개 id를 만들지만 서로 다른 mask로 prefill하므로, Gemma 3 이미지 출력은 backend마다 여전히 다르다. 작성 시점에 이를 다루는 열린 issue는 없었다.
- **빈 newline table은 고정된다.** 이제 한 번 경고하지만, MLX cell은 재시작 없이는 회복되지 않는다. 정상 tokenizer에서는 생기지 않는다.
- **table을 image id와 대조하지 않는다.** 이 PR 이전(#2750)부터 그랬다. newline run을 `<start_of_image>`, `<end_of_image>`, image token id로 encode하는 잘못된 tokenizer라면 merge가 image 구간을 줄일 수 있다. operator가 제공한 checkpoint만 이를 일으킬 수 있다.
- **thinking 가능 checkpoint에서 기본 이미지 prompt는 여전히 reference와 다르다.** 의도된 것이며 `docs/benchmarks.md`에 기록되어 있다.
- **MLX에서 probe 시점이 조금 앞당겨졌다.** 이제 wrapper가 있는 layout이면 확장 전에 실행되므로, 확장할 block이 없는 첫 요청에서도 실행된다. 모델당 한 번이므로 무시할 만하다.

## 6. 학습 포인트

- 두 backend가 같은 변환을 구현하면 같은 단계를 두 호출부에 두지 말고 함수 하나를 주어라. OpenXLA 차이는 #2750이 MLX 호출부를 고쳤고 다른 호출부가 그 테스트 범위 밖에 있었기 때문에 생겼다. 공용 helper와 두 layout을 비교하는 테스트가 있으면 어긋남은 compile 또는 테스트 실패가 된다.
- 소유 타입이 feature 아래에서만 존재하면, feature-gated 로직을 `cfg(any(feature = "...", test))` 아래 free function으로 두어라. 그러면 기본 테스트 build가 무거운 의존성 없이 그 로직을 검증한다.
- tokenizer에서 파생되는 모델별 상수는 cache하고, 실패는 cache를 채우는 순간에 드러나게 하라. cache는 요청별 약한 실패를 고정된 실패로 바꾸므로 경고는 채우는 시점에 있어야 한다.
- prompt 길이를 비교하기 전에 설정 차이와 processor 버그를 분리하라. reference 설정(`enable_thinking` off)에 맞추는 harness 스위치 하나로 설명되지 않던 7 token 차이가 동일한 id로 바뀌었고, 고칠 버그가 없다는 것이 드러났다.
- mutation check는 테스트가 실제로 어느 진입점을 고정하는지 보여 준다. render가 field를 읽기 때문에 getter를 강제해도 아무것도 바뀌지 않았고, 그래서 테스트에 getter assertion이 추가되었다.
- 실제 checkpoint에서 테스트 fixture를 만들 때는 store에 수동 사본을 유지하지 말고 symlink와 다시 쓴 config로 만들어라. fixture가 배포된 checkpoint와 동기화되고 디스크 비용도 없다.

## 7. 관련 항목

- Issue #2761: 이 PR의 issue(#2750의 잔여 항목 세 가지).
- PR #2750 / issue #2547: 이 잔여 항목들이 나온 MLX prompt parity 작업.
- PR #2773 / issue #2760: MLX Gemma 3 이미지 prefill mask. 이 branch에 merge되었고, OpenXLA 쪽은 남아 있다.
- mlx-lm PR #1114: think marker 기반 `enable_thinking` 기본값의 출처.
