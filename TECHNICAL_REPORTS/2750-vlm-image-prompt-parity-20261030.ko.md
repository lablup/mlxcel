# PR #2750: InternVL, MiniCPM-V 4.6, Gemma 3 이미지 prompt를 reference processor와 일치시킴

**날짜**: 2026-10-30 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Python
**위험도**: 중간 (InternVL, MiniCPM-V 4.6, Gemma 3의 모든 이미지 요청이 다른 token 시퀀스를 읽게 되고, template이 없는 MiniCPM-V 4.6 / MiniCPM-o checkpoint는 text-only 요청을 포함해 server prompt가 달라짐)

## 요약

Issue #2547은 mlxcel이 InternVL, MiniCPM-V 4.6, Gemma 3에 대해 만드는 이미지 조건부 prompt가 vision tower가 돌기 전에 이미 reference processor와 다르다는 것을 발견했다. 그래서 이 계열의 cross-runtime logit 비교는 모델이 아니라 prompt를 측정하고 있었다. 이 PR은 각 차이를 token 수준에서 원인별로 분류하고, `mlxcel generate`, `mlxcel-server`, logit trace가 공유하는 `prepare_vlm_embeddings` 경로에서 고친다. InternVL의 image block은 system turn 안에서 마지막 user turn의 시작으로 옮겨지고, MiniCPM-V 4.6은 `MiniCPMV4_6Processor`를 따르는 자체 layout과 template이 없는 checkpoint용 내장 ChatML template을 얻으며, Gemma 3의 newline wrapper는 reference tokenizer가 합치는 방식대로 합쳐진다. 그 결과 prompt는 InternVL(292)과 Gemma 3(272)에서 transformers와 같아졌고, MiniCPM-V 4.6의 server prompt는 CLI prompt(82)와 같아졌다. 같은 prompt를 넣었을 때 float32 mlx-vlm reference 대비 decided position 불일치는 InternVL과 MiniCPM-V 4.6에서 0이 되었다. Gemma 3는 43개 중 5개로 남아, 남은 차이는 forward pass나 reference의 mask에 있다.

## 1. 문제 정의

### 측정된 불일치

#2423의 GELU 문제를 측정하던 중, 이미지 logit trace(`MLXCEL_TRACE_IMAGE`)와 float32 mlx-vlm 0.7.8 reference(`scripts/mlx_vlm_image_logit_trace.py`)가 이미지 `tests/fixtures/test_image_shapes.png`(448x448), prompt `Describe the image.`에 대해 checkpoint 5개 중 3개에서 prompt 길이가 달랐다. internvl3-1b 262 대 268, minicpm-v-4.6-bf16 78 대 76, gemma-3-4b-it-4bit 273 대 274. `docs/benchmarks.md`는 길이 불일치를 tower 이전에 processor가 서로 다르다는 신호로 본다. 남은 perplexity 차이도 컸다(InternVL 1.79 대 1.39, MiniCPM-V 4.6 4.66 대 1.22에 decided position 39개 중 2개 불일치). issue 시점에는 원인이 template인지, image token 수인지, BOS인지 알 수 없었다.

mlx-vlm과 길이만 맞추는 것은 올바른 목표가 아니었다. mlx-vlm 자체가 여러 곳에서 checkpoint 및 transformers와 다르기 때문이다(2.6절). 그래서 이 PR은 checkpoint마다 id 시퀀스 세 개(mlxcel, mlx-vlm, transformers)를 덤프해서 각 차이의 원인을 분류했다.

### InternVL: block이 system turn 안에 있었음

`internvl3-1b`는 image branch가 없는 Qwen2 ChatML template을 제공하므로, 렌더링된 prompt에는 `<IMG_CONTEXT>` placeholder가 없고 mlxcel이 image block을 직접 끼워 넣는다. 기존 규칙은 첫 token 바로 뒤에 넣었는데, ChatML prompt에서 첫 token은 system turn을 여는 `<|im_start|>`이다. 모델은 `<|im_start|>` + 이미지 + `system\nYou are Qwen...`을 읽었다. InternVL의 `model.chat`(`'<image>\n' + question`), transformers InternVL template(`<IMG_CONTEXT>\n`), mlx-vlm(`<image>\n`)은 모두 block을 user content의 시작에 두고 뒤에 newline을 붙인다.

issue의 262에는 측정 도구의 결함도 하나 숨어 있었다. trace는 image branch가 없는 template에 typed `[image, text]` content list를 넘겼고, 렌더링이 실패해 chat turn 없이 질문만 남는 fallback으로 떨어졌다(text token 4개와 258-token block). `mlxcel generate`는 template을 정상적으로 렌더링했으므로, 실제 운영 경로의 prompt는 262 token이 아니라 template token 33개와 잘못 배치된 block이었다.

### MiniCPM-V 4.6: MiniCPM-o 규칙은 맞지 않는 layout이었음

MiniCPM-V 4.6은 MiniCPM-o의 확장 규칙을 거쳤고, 이 규칙은 `<image><unk>...<unk></image>`를 만든다. `MiniCPMV4_6Processor`는 `<image_id>{i}</image_id><image>` + `<|image_pad|>` x N + `</image>`를 만들고(checkpoint의 `preprocessor_config.json`에 `use_image_id: true`), checkpoint의 template은 image item과 질문을 `\n`으로 잇는다. 기존 규칙은 image id prefix와 newline을 빠뜨렸고, filler가 틀렸으며, upstream template을 쓰면 template이 렌더링한 `<|image_pad|>` marker를 text로 prompt에 남겼을 것이다.

template이 없는 checkpoint(model store의 `minicpm-v-4.6-bf16` 사본에는 `chat_template.jinja`가 없다)에서는 server에 framing 버그가 하나 더 있었다. server는 일반 fallback인 `User: ...\n\nAssistant: `로 렌더링했고, image 확장은 ChatML turn을 찾지 못하자 그것을 다시 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`으로 감쌌다. server prompt에는 두 framing이 모두 들어갔고(88 token), CLI에는 하나만 들어갔다.

### Gemma 3: newline token이 하나 더 많았음

transformers의 `Gemma3Processor`는 렌더링된 text에서 각 `<start_of_image>`를 `\n\n<start_of_image>` + soft token + `<end_of_image>\n\n`으로 바꾼 뒤에 tokenize한다. Gemma vocabulary는 newline 연속(최대 31개)마다 token 하나를 가지므로, template의 `user\n`과 wrapper의 `\n\n`이 합쳐져 단일 token `\n\n\n`(109)이 된다. mlxcel은 token stream 위에서 확장하므로(`apply_image_token_blocks`, wrapper token 108 = `\n\n`) `\n`(107)과 `\n\n`(108)이 두 token으로 남았다. 272 대신 273이었다.

## 2. 기술적 선택과 그 이유

### 2.1 InternVL: user turn은 text에서 찾고 token으로 검증

`insert_internvl_image_tokens`는 이제 렌더링된 prompt와 encoder를 받는다. text에서 마지막 `<|im_start|>user\n`을 찾고(`rfind`), opener까지의 text를 encode한 뒤, 그 결과가 호출자 prompt token의 prefix인지 확인한다. prefix이면(`UserTurnSplit::Aligned`) 각각 encode된 `\n`이 붙은 block들을 그 offset에 끼워 넣고, 나머지 요청은 원래 tokenization을 유지한다. placeholder 확장(`<IMG_CONTEXT>`를 내보내는 template)이 여전히 먼저 실행되고 우선한다.

block을 text로 넣은 뒤 prompt 전체를 다시 encode해도 같은 결과가 나왔겠지만, 이미 올바른 text를 다시 tokenize하게 되고 block의 special token이 tokenizer를 왕복해야 한다. token만으로 opener를 찾는 방식은 아래 경우를 놓친다.

질문이 newline으로 시작하면 Qwen2 pre-tokenizer가 opener의 `\n`과 합쳐 버리므로(예: `\n\n\n`이 token 하나) opener가 token 경계가 아니다. `user_turn_split`은 이 경우(마지막 `\n`을 제외한 head token 전부가 prefix)를 감지해 `UserTurnSplit::Fused`를 돌려준다. 호출자는 head token을 유지하고, 마지막 newline을 뺀 block들을 붙인 뒤, 그 뒤에 `"\n" + rest`를 encode한다. reference가 그 자리에서 tokenize하는 text가 바로 이것이다(`\n\nLeading newlines`에 대해 291 token, reference와 동일). 이 경로는 prompt의 나머지를 다시 tokenize하며, fused 경우에만 도달한다.

두 가지 degradation은 의도된 것이다. user-turn opener가 없는 prompt(`--no-chat-template`, raw prompt)는 block을 앞에 붙이는데, 이는 template 없는 `model.chat`의 `'<image>\n' + question`과 같다. head encoding이 어느 쪽으로도 호출자 token과 맞지 않을 때도 token 중간에 block을 넣지 않고 앞에 붙인다. 호출자의 `add_special_tokens` 선택은 여기서 보이지 않으므로, plain encoding을 먼저 시도하고 맞지 않을 때만 special-token encoding을 계산한다(commit 4305d138에서 일반 경로를 encode 한 번으로 줄였다).

placeholder 전용 절반은 `expand_internvl_image_placeholders`로 분리했다. 자체 Harmony user turn을 이미 찾고 있었고 InternVL helper는 in-place placeholder 확장에만 쓰던 LLM-jp-VL은 이제 이 함수를 직접 호출하며, 동작은 바뀌지 않는다.

### 2.2 MiniCPM-V 4.6: MiniCPM-o의 flag가 아니라 별도 module

`src/multimodal/minicpmv46_prompt.rs`는 MiniCPM-o의 `<image>`/`</image>` bound 계산을 재사용하지만 4.6 layout은 따로 가진다. MiniCPM-o는 `<unk>` filler와 image id 없음을 유지하므로, MiniCPM-o 규칙에 parameter를 추가하면 bound 외에는 공유하는 로직이 없는 두 processor의 규칙이 한 함수에 섞인다. 확장은 text에서 수행한 뒤 special token을 켜고 encode하고, 각 `<image>`...`</image>` 구간을 찾아 그 길이를 이미지 feature 수와 비교한다. 다르면 어긋난 embedding을 넣는 대신 error를 돌려준다.

경우는 순서대로 세 가지다. template marker `<|image_pad|>`와 MiniCPM-V inline marker `<image>./</image>`(mlx-vlm과 예전 client가 쓰는 형식)는 나타난 순서대로 이미지 하나씩으로 바뀌고 개수를 검사한다. marker가 없으면 block과 `\n`이 마지막 `<|im_start|>user\n`의 시작에 들어간다. marker도 user turn도 없으면 ChatML user/assistant opener로 turn을 감싼다.

두 marker 종류는 한 번의 pass로 나눠서, 이미 넣은 block의 `<|image_pad|>` filler를 marker로 다시 읽지 않는다. 첫 버전은 marker를 찾을 때마다 남은 prompt 전체에서 두 종류를 다시 검색했는데, 한 종류가 없으면 이는 quadratic이다. security review에서 typed marker 10만 개(1.3 MB)에 24초가 걸렸다. prompt는 사용자가 제어하고, split은 marker 개수 error 전에 batch scheduler에서 실행되며, VLM 요청은 context-length 검사를 건너뛰고, JSON body 한도는 큰 이미지 여러 장에 맞춰져 있으므로 이것은 실제 문제다. 이제 각 종류는 다음 출현 위치의 offset을 저장해 두고 cursor가 그것을 지난 뒤에만 다시 검색한다(commit 6cece708). 수정 전 24초가 걸리던 regression test가 함께 추가되었다.

### 2.3 template 없는 MiniCPM checkpoint용 내장 ChatML template

`builtin_chat_template`은 이제 `ModelType::MiniCPMV46VLM`과 `ModelType::MiniCPMOVLM`에 대해 `MINICPM_CHATML_TEMPLATE`을 돌려준다. 이 함수는 checkpoint가 template을 제공하지 않을 때만 도달하므로, 제공된 template이 여전히 우선한다. server가 ChatML turn을 렌더링하면 확장은 `<|im_start|>user\n`을 찾아 그 turn을 열 뿐 다시 감싸지 않으므로, server와 CLI가 같은 82-token prompt를 만든다.

대안은 확장 코드가 일반 `User:`/`Assistant:` framing을 인식해 벗겨 내도록 하는 것이었다. 그러면 모델별 prompt module이 server의 일반 fallback 문구에 묶인다. 렌더링 쪽을 고치면 server가 그 계열의 template이 만들 turn을 그대로 만들고, 확장에 같은 "turn이 없으면 감싼다" 규칙을 가진 MiniCPM-o도 같은 template으로 해결된다. template은 typed content list의 text 부분만 렌더링하고 image item은 버린다. 이미지는 렌더링 후 token으로 끼워 넣기 때문이다.

이 변경은 template 없는 MiniCPM-V 4.6과 MiniCPM-o checkpoint의 text-only server 요청도 바꾼다. 이제 `User:`/`Assistant:` 대신 ChatML을 받는다.

### 2.4 Gemma 3: token 수준 확장 뒤에 newline 연속을 합침

Gemma 3를 transformers의 text 수준 확장으로 바꾸면 다른 모든 `ImageTokenBlockInfo` 계열의 준비 방식도 바뀐다. 그 대신 `gemma3_vlm_prompt::merge_image_block_newline_runs`가 token stream을 후처리한다. tokenizer에서 newline 연속의 단일 token id를 조사하고(`newline_run_token_ids`, token 하나로 encode되지 않는 첫 길이에서 멈춤. Gemma 3는 1부터 31까지), 각 `<start_of_image>`와 `<end_of_image>` 옆의 wrapper token을 표시한 뒤, wrapper token을 포함하는 newline-run token의 최대 구간을 reference가 그만큼의 newline을 encode하는 방식으로 다시 encode한다(긴 run부터, 그래서 newline 33개는 31-run과 2-run이 된다). wrapper token이 없는 구간은 tokenizer가 만든 것이므로 그대로 둔다.

`vlm_runtime`의 `Standard` arm은 layout에 비어 있지 않은 block prefix나 suffix가 있을 때만 이 함수를 호출하고, 함수는 wrapper token 중 newline run이 없으면 즉시 반환한다. 현재 loader 중 wrapper가 있는 것은 Gemma 3(`block_prefix_tokens: vec![108]`)뿐이므로 다른 계열은 영향도 비용도 없다. Gemma 3는 이미지 요청마다 짧은 문자열 encode 약 32번의 probe 비용을 낸다.

### 2.5 trace 도구가 양쪽 arm에서 같은 prompt를 채점

이미지 trace는 이제 `mlxcel generate`와 같은 방식으로 렌더링하고(template에 image branch가 없으면 text-only 렌더링, 있으면 typed content), `# prompt_tokens` 옆에 `# prompt_ids`를 출력한다. `scripts/mlx_vlm_image_logit_trace.py --mlxcel-prompt`는 그 id를 읽어 mlx-vlm이 전처리한 pixel로 채점하므로, 이 PR이 mlx-vlm과 의도적으로 다른 곳에서도 두 arm은 forward pass만 다르다. image token 수가 mlx-vlm과 다른 prompt는 거부하고(pixel feature를 공유할 수 없으므로), MiniCPM-V의 `image_bound`를 대체된 id에서 다시 계산하며, mlx-vlm 자체의 길이는 `# mlx_vlm_prompt_tokens`로 출력한다.

reference는 Gemma 3와 PaliGemma도 causal로 실행한다. 이들의 mlx-vlm `__call__`은 전부 1인 2D attention mask로 4D mask를 만드는데, 그러면 teacher-forced row마다 continuation 전체를 보게 된다. script는 image embedding을 병합한 뒤 `mask=None`으로 language model을 호출한다.

### 2.6 mlx-vlm과의 의도된 차이

각 항목은 transformers나 checkpoint를 따르며 `docs/benchmarks.md`에 기록되어 있다.

- InternVL template: mlx-vlm은 checkpoint template을 자체 `User: ...\nAssistant:\n`으로 바꾸는데, 이는 checkpoint의 ChatML template도 InternVL의 conversation template도 아니다. mlxcel은 `tokenizer.apply_chat_template`처럼 checkpoint template을 Qwen system turn까지 그대로 쓴다(292, mlx-vlm 268).
- template 없는 MiniCPM-V 4.6: mlx-vlm은 chat turn 없이 `(<image>./</image>)\nQUESTION`을 보내고, mlxcel은 그 계열의 ChatML turn으로 감싼다(82 대 76). upstream template을 쓰면 mlxcel 84, mlx-vlm 84, transformers 86이며, 남은 차이는 marker의 괄호, 빈 think block, mlxcel server 전역 기본값 `enable_thinking=true`이다.
- Gemma 3: mlx-vlm은 이미 `<bos>`로 시작하는 렌더링 결과를 special token을 켠 채 tokenize해서 BOS가 두 개가 된다. transformers와 mlxcel은 하나다. mlx-vlm은 이미지를 질문 뒤에 두고, mlxcel은 model card와 mlxcel의 다른 이미지 경로가 쓰는 순서대로 이미지를 앞에 둔다(#797).

## 3. 변경 요약

파일 16개, +1645/-147, commit 다섯 개(452f90d2, fff22a28, f4063ebd, 6cece708, 4305d138)와 origin/main merge 하나(b769c15d).

| Checkpoint | mlxcel 이전 | mlxcel 이후 | mlx-vlm | transformers |
|---|---|---|---|---|
| internvl3-1b | 262 (trace) | 292 | 268 | 292 |
| minicpm-v-4.6-bf16, template 없음 | CLI 78 / server 88 | 82 / 82 | 76 | 해당 없음 |
| gemma-3-4b-it-4bit | 273 | 272 | 274 | 272 |
| paligemma2-3b-6bit | 1030 | 1030 | 1030 | |
| gemma-4-e2b-it-4bit | 278 | 278 | 271 | |

- `src/multimodal/internvl_prompt.rs`: aligned/fused split을 이용한 user turn 배치, `expand_internvl_image_placeholders` 분리. test는 `internvl_prompt_tests.rs`로 이동.
- `src/multimodal/minicpmv46_prompt.rs`(신규): 4.6 block layout, 선형 시간 두-marker split, 구간 크기 검사.
- `src/multimodal/gemma3_vlm_prompt.rs`(신규): newline-run table과 wrapper 병합.
- `src/multimodal/vlm_runtime.rs`: MiniCPM-V 4.6 arm은 `prepare_minicpmv46_prompt_tokens`를 호출하고, InternVL arm은 prompt와 encoder를 넘기며, `Standard` arm은 `apply_image_token_blocks` 뒤에 newline 병합을 실행.
- `src/multimodal/llmjp_vl_prompt.rs`: `expand_internvl_image_placeholders` 호출.
- `src/server/chat_template.rs`: `MiniCPMV46VLM`과 `MiniCPMOVLM`용 `MINICPM_CHATML_TEMPLATE`.
- `examples/logit_trace.rs`, `examples/logit_trace/image.rs`: `generate`처럼 렌더링, `# prompt_ids` 출력.
- `scripts/mlx_vlm_image_logit_trace.py`: `--mlxcel-prompt`, `# prompt_ids`, causal Gemma 3 / PaliGemma.
- `docs/benchmarks.md`: parity 표, 수정 내용, 의도된 차이, trace 쌍.
- Test(모두 checkpoint 없이 실행, reference pin은 실제 token id 사용): `internvl3_1b_one_tile_prompt_matches_reference_ids`, `internvl3_1b_question_with_leading_newline_keeps_block_in_user_turn`, `fused_opener_newline_with_two_images_keeps_blocks_in_user_turn`, `caller_tokens_encoded_with_special_tokens_still_align`, `tokenizer_disagreement_degrades_to_prepending`; `minicpmv46_untemplated_prompt_matches_reference_ids`, `many_typed_markers_are_split_in_linear_time`, `marker_count_must_match_images`, `span_size_mismatch_is_an_error`; `gemma3_one_image_prompt_matches_transformers_ids`, `a_run_past_the_longest_token_splits_like_the_reference`, `non_gemma_layouts_are_untouched`; `server::minicpm_chat_template_tests`(4개, `a_shipped_minicpm_template_is_not_shadowed_by_the_builtin` 포함).

## 4. Trace 쌍

CPU의 float32 mlx-vlm reference에 대해 응답 60~64개 위치를 teacher-forcing으로 채점했고, decided position은 logit 차이 2 기준이다. 이전은 수정 전 build를 mlx-vlm 자체 prompt와 비교한 것이고, 이후는 수정된 build를 mlxcel prompt로 채점한 reference(`--mlxcel-prompt`)와 비교한 것이다.

| Checkpoint | 이전: top-1, decided, ppl mlxcel 대 ref | 이후 |
|---|---|---|
| internvl3-1b | 8/64, 0/38, 1.7881 대 1.3916 | 0/64, 0/39, 1.5853 대 1.5869 |
| minicpm-v-4.6-bf16 | 17/60, 2/39, 4.6649 대 1.2230 | 2/60, 0/47, 4.2442 대 4.2325 |
| gemma-3-4b-it-4bit | 21/64, 5/40, 4.2819 대 3.9110 | 20/64, 5/43, 4.8500 대 5.1620 |

InternVL과 MiniCPM-V 4.6의 이전 차이는 prompt 때문이었다. 같은 token을 넣으면 decided 불일치가 0이고 perplexity는 0.3% 안에서 일치한다. 응답이 mlx-vlm의 prompt로 생성되었기 때문에 reference 자체의 perplexity도 움직인다. Gemma 3는 같은 prompt에서도 decided position 43개 중 5개가 여전히 다르므로, 그 차이는 prompt에 있지 않다.

## 5. 트레이드오프와 잔여 항목

- **Gemma 3 forward pass 차이.** prompt는 이제 정확하지만 decided position 43개 중 5개가 여전히 다르다. reference는 병합된 embedding에서 Gemma 3를 causal로 실행하는 반면, transformers는 image token끼리 block 안에서 서로 attend하게 한다. 이 mask가 첫 번째 의심 대상이다. PR은 이를 follow-up으로 표시하지만 issue 번호는 적지 않았다.
- **OpenXLA Gemma 3 경로.** host preprocessor(`host_preprocessor.rs`, `GEMMA3_VLM_NEWLINE_WRAPPER_TOKEN_ID`)는 자체 token 수준 wrapper를 유지하고 새 병합을 실행하지 않으므로, XLA backend는 여전히 273-token prompt를 만든다.
- **MiniCPM-o 내장 template은 smoke 미실시.** MiniCPM-o checkpoint를 구할 수 없었고, template은 `template_less_minicpmo_server_prompt_matches_the_cli_prompt`로만 검증된다.
- **여러 turn에 걸친 이미지 배치.** placeholder 없는 InternVL과 marker 없는 MiniCPM-V 4.6에서는 모든 이미지가 마지막 user turn의 시작으로 간다. text-only 렌더링에는 turn별 이미지 위치가 없으므로 앞선 turn에 첨부된 이미지도 마지막 turn으로 옮겨진다. 기존 코드도 같은 한계가 있었고(첫 token 뒤에 넣었다), image branch가 있는 template이라면 각 이미지를 자기 turn에 두었을 것이다.
- **cache와 출력 변화.** 세 계열의 모든 이미지 요청과 template 없는 MiniCPM-V 4.6 / MiniCPM-o server의 모든 요청에서 prompt id가 바뀌므로, 이전 build의 prefix-cache 항목은 맞지 않고 출력도 이전 release와 달라진다.
- **gemma-4-e2b-it-4bit**는 thinking 기본값에 따른 system `<|think|>` turn 때문에 여전히 278 대 mlx-vlm 271이다. 이 PR에서는 다루지 않았다.

## 6. 검증

- 새 pin은 모두 먼저 실패를 확인했다. InternVL 292와 leading-newline 291, Gemma 3 272, MiniCPM-V 4.6 82, server MiniCPM fallback(수정 전 4개 중 3개 실패), 선형 split(수정 전 24초, 수정 후 milliseconds).
- `cargo test --lib _prompt` 450개 통과, `multimodal::` 318개 통과, `server::minicpm_chat_template_tests::` 4개 통과. fmt와 clippy(`--lib --tests`, `logit_trace` example) 깨끗함.
- GB10 smoke: 계열마다 `mlxcel generate --image`와 `mlxcel-server` 요청 하나. server `prompt_tokens`는 InternVL 292(newline으로 시작하는 질문은 291), Gemma 3 272, MiniCPM-V 4.6 82이고 출력은 정상이었다.
- paligemma2-3b-6bit는 1030으로 변함없음, gemma-4-e2b-it-4bit는 전후 모두 278.

검증하지 않은 것: OpenXLA Gemma 3 경로, MiniCPM-o checkpoint, upstream template을 쓴 MiniCPM-V 4.6의 server 경로, 세 계열 모두에서 여러 turn에 이미지가 있는 대화.

## 7. 학습 포인트

- reference는 하나 이상과 비교한다. mlx-vlm이 issue의 기준이었지만 세 계열 모두에서 checkpoint 및 transformers와 달랐다. mlxcel, mlx-vlm, transformers의 id를 나란히 덤프해야 실제 버그와 의도된 차이를 구분할 수 있었다.
- vocabulary가 연속 문자를 합치는 경우 text 수준 확장과 token 수준 확장은 서로 바꿔 쓸 수 없다. tokenize 전에 text를 고쳐 쓰는 processor에서는 삽입한 문자가 이웃 문자와 합쳐진다. token 수준 splice는 그 병합을 명시적으로 재현하거나(Gemma 3 newline), token 경계로 존재하지 않을 수 있는 경계를 찾아야 한다(Qwen2의 opener newline 병합).
- 이미 tokenize된 prompt에 끼워 넣을 때는 렌더링된 text에서 위치를 찾고 token으로 검증한 뒤, 둘이 맞지 않으면 요청 전체를 다시 tokenize하지 말고 안전한 위치로 물러난다.
- 측정 대상보다 측정 도구를 먼저 확인한다. InternVL 차이의 일부는 운영 경로가 아니라 trace 도구 자신의 렌더링 fallback이었다.
- 사용자가 제어하는 prompt에 대한 문자열 scan은 대부분의 한도보다 먼저 실행된다. 무해해 보이던 marker마다의 재검색이 quadratic이었고 1.3 MB body에서 24초가 걸렸다.
