# 기술 보고서: PR #1762 - feat(speculative): Muse Glimmer DFlash assistant drafter on the round loop

**작성일**: 2026-09-10
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 머지 전 리뷰 라운드
**상태**: 완료 (Apple M5 Max / Metal에서 `models/muse-glimmer-30b-4bit`와 `models/muse-glimmer-30b-assistant-bf16`를 짝지어 검증. 공용 머신이라 `--workspace` 게이트는 로컬에서 돌리지 않았고, 좁은 범위 실행과 CI가 그 자리를 대신한다)
**언어**: Rust, Markdown
**위험도**: Medium (세 번째 계열이 DFlash 라운드 루프에 합류하고, 루프의 verify 폭이 처음으로 고정값을 벗어나며, 그동안 모든 `--draft-*` 플래그를 거부하던 계열이 drafter 하나를 받아들이게 된다)

---

## 요약

Meta는 Muse Glimmer 30B용 "assistant" drafter를 공개한다. sliding attention 디코더 다섯 층이 타깃의 residual stream 다섯 개를 `encoder.fc` 투영으로 받아 16토큰 블록 전체를 비인과 forward 한 번에 예측하고, 자기 토큰 테이블과 헤드를 싣는 대신 타깃 것을 빌린다. mlxcel은 이미 Muse Glimmer를 서빙하고 있었고 DFlash 라운드 루프도 갖고 있었으므로, 작업의 형태는 앞선 LFM2 DSpark 변경(#1751)과 같다. drafter 모델, 타깃 어댑터, 디스패치 배선 세 가지다.

같지 않은 지점이 셋 있다.

drafter의 어텐션이 Qwen 모양이 아니다. 헤드별 q/k norm이 붙어 있고, 마스크가 인과적이지도 없지도 않은 절대 위치 기반 양방향 마스크다. 이 마스크 때문에 drafter 자신의 컨텍스트에는 `RotatingKVCache`를 쓸 수 없다. 링이 한 바퀴 돈 뒤 한 행짜리 append는 버퍼를 물리 슬롯 순서로 돌려주는데, 절대 위치로 만든 마스크는 순서를 모르는 행들 위에 얹을 수 없다. 그래서 drafter는 목적에 맞춘 작은 시간순 윈도를 따로 받는다.

verify 폭이 상수를 벗어난다. 공개된 `block_size`는 16이고, M5 Max에서 16행 고정은 자연문 프롬프트에서 classic decode보다 느린 반면 2660토큰짜리 반복 로그에서는 그 폭으로 제안 15개 중 14개가 수락된다. 둘 중 어느 쪽도 정답이 아니므로, `DFlashGenerator`는 이제 drafter가 선언한 depth를 읽고, drafter가 요청 폭을 고집하지 않는 경우 depth와 상한 사이에서 MTP 루프의 처리량 비교기를 돌린다. Qwen 3.5 DFlash는 depth를 선언하지 않고 DSpark는 요청 폭을 선호하므로, 둘 다 기존 동작을 유지한다.

머지 전 리뷰가 잡은 것도 두 번째 변경에서 나온다. 폭이 적응적이 되면 폭 하나만 재는 게이트는 근거를 잃는다. burst는 `exactness_allows(16)`을 물었고 루프는 warm-up을 4에서 보냈다. 리뷰 라운드의 수정 두 건은 모두 새 폭 정책을 끝까지 따라간 결과이고, 그중 하나는 실제 측정 호스트에서 차이를 만들었다.

---

## 1. 체크포인트

### 1.1 config는 평평하고 키 두 개가 없다

이슈 명세는 `num_target_layers 52`와 `vocab_size 202048`을 적어 두었다. 공개된 `meta-models/Muse-Glimmer-30B-assistant`의 `config.json`에는 둘 다 없고, `dflash_config` 블록도 없다. `block_size`, `mask_token_id`, `target_layer_ids`가 최상위에 놓여 있다.

없는 키 둘을 0으로 기본값 처리한 다음 `target_layer_ids[i] < num_target_layers`를 검사하면 실제 체크포인트가 거부된다. 필수로 만들어도 마찬가지다. 그래서 둘은 `Option`이고, 그 키들이 말했을 사실은 어차피 페어링을 확인해야 하는 자리, 곧 바인딩된 타깃에서 측정한다.

- `target_layer_ids.last() < target.num_layers()`가 `num_target_layers` 상한을 대신한다. 이쪽이 더 강한 검사인데, drafter 작성자가 믿은 타깃이 아니라 손에 든 타깃과 비교하기 때문이다.
- `mask_token_id < vocab`은 0으로 채운 hidden을 `target.lm_head_module()`에 통과시켜 결과 폭을 읽어 측정한다. mask id는 타깃의 임베딩 테이블을 인덱싱하므로, 성립해야 하는 수는 타깃 헤드의 폭이다.
- 두 키가 실제로 있는 경우(둘을 써 주는 도구로 재변환한 체크포인트)에는 그것도 함께 비교한다.

`MuseAssistantConfig::from_json`은 중첩된 `dflash_config`를 여전히 들어 올리므로, Qwen 3.5나 DSpark 계열 도구로 재변환한 체크포인트도 로드된다.

### 1.2 `embed_tokens`도 `lm_head`도 없고, 어느 쪽을 빌리는지가 중요하다

텐서 58개는 `encoder.fc`, `encoder.output_norm_enc`, `norm`, 그리고 투영 네 개 옆에 `q_norm` / `k_norm`이 붙은 디코더 다섯 층이다. 토큰 테이블도 헤드도 없다.

어느 테이블을 빌리느냐가 결과를 좌우한다. Muse Glimmer의 `LanguageModel::embed_tokens`는 `embed_norm`을 이미 적용한 lookup을 돌려주고, 그것이 타깃 자신의 forward가 원하는 값이다. drafter는 자기 `input_layernorm`을 돌리므로 원본 테이블을 원한다. `embed_tokens_module()`과 `lm_head_module()`은 원본 테이블과, `output_multiplier`도 `final_logit_softcapping`도 뒤에 붙지 않은 untied 헤드의 공유 핸들을 넘긴다. `embed_norm`을 씌운 lookup을 drafter에 먹이는 쪽은 가정이 아니라 측정으로 걸렀다. 자연문 프롬프트 세 개에서 라운드당 수락이 1.87 / 3.40 / 1.80에서 1.28 / 1.35 / 0.90으로 떨어졌다.

---

## 2. drafter forward

### 2.1 마스크는 절대 위치 위의 양방향 마스크다

draft 블록은 forward 한 번에 예측되므로 모든 제안 슬롯이 다른 모든 슬롯을 볼 수 있다. 제약은 윈도뿐이다.

```
mask[i, j] = |(query_start + i) - (key_start + j)| <= sliding_window
```

`bidirectional_sliding_mask_bool`이 이것을 덧셈 bias가 아니라 불리언 배열로 만드는 데는 이유가 있다. 덧셈 bias는 fused SDPA가 q/k/v를 승격시키는 dtype과 맞아야 하는데, 타깃의 residual stream과 drafter의 dense 가중치가 다른 dtype으로 저장돼 있으면 그 dtype은 query의 dtype이 아니다. 어긋나면 MLX가 던지고 cxx 브리지를 건너오면서 실패가 아니라 abort가 된다. 불리언 마스크에는 dtype이 없다.

모든 query 행은 거리 0의 key 행이기도 하므로 마스크의 어떤 행도 전부 닫히지 않는다. 덧셈 형태였다면 `-inf` 행이 없는지 따로 확인해야 했을 성질이다.

### 2.2 컨텍스트 캐시가 `RotatingKVCache`가 아닌 이유

`RotatingKVCache`는 Muse 타깃이 sliding 레이어에 쓰는 구조이고, 거기서는 맞는 선택이다. 타깃의 마스크는 레이어가 캐시 자신의 회계로 만들기 때문이다. drafter의 마스크는 drafter가 절대 위치로 만들고, 캐시가 행을 시간순으로, 첫 행의 절대 위치와 함께 돌려주기를 요구한다. 한 바퀴 돈 링은 한 행짜리 append에 대해 그럴 수 없고, drafter는 수락 0 라운드마다 한 행을 append한다.

그래서 `MuseAssistantContextCache`는 append 전용이고, 잘리지 않으며, 최대 `window`행을 유지하고, `offset`은 해당 레이어가 소비한 모든 행을 센다. 캐시에 닿기 전에 버려진 행까지 포함한다. `key_start()`는 `offset - len()`이고, 마스크가 필요로 하는 값이 정확히 그것이다. 이 캐시는 롤백되지 않는다. 라운드 루프가 다음 라운드에 커밋된 행만 넘기므로, drafter의 윈도는 구성상 이미 커밋된 접두사다.

### 2.3 투영 한 번, RoPE 한 번

컨텍스트 행과 제안 행은 하나로 이어붙인 입력으로 `k_proj` / `v_proj`를 통과하고, `cache.offset`에서 `fast_rope` 한 번이 컨텍스트를 `offset..offset + S`에, 제안을 그 바로 뒤에 놓는다. query도 `offset + S`에서 같은 처리를 받는다. 둘을 나누면 RoPE 호출이 두 오프셋에서 두 번 필요하고, 위치 산술을 두 자리에서 맞춰 두어야 한다.

캐시에 들어가는 것은 컨텍스트 행뿐이다. 제안 K/V는 가져온 윈도 뒤에 이어붙였다가 버린다. 그래야 `offset`이 verify한 행 수가 아니라 커밋된 행 수와 같아진다.

---

## 3. 타깃 쪽

### 3.1 최종 norm 앞에서 캡처

`forward_speculative`는 `mask = None`인 생성 forward이므로 각 레이어가 자기 causal + sliding 마스크를 만들고, 여기에 `capture_layer_ids`에 이름이 있는 레이어 뒤마다 `h`를 밀어 넣는 동작이 더해진다. 최종 norm 앞에서 캡처하는 것은 명세이자 측정 결과이기도 하다. 한 층 앞에서 캡처하는 방식(HF `hidden_states[i]` 인덱싱)은 수락률을 의미 있게 바꾸지 못했다(1.83 / 3.47 / 1.50 대 1.87 / 3.40 / 1.80).

`prefill_forward_with_capture_layers`는 트레이트 기본값을 유지한다. 같은 훅을 두고 LFM2가 한 선택과 반대다. LFM2가 override한 이유는 그 verify 출력이 prefill에는 쓸모없는 프롬프트 크기의 short-conv 스냅샷을 들고 있기 때문이다. Muse의 verify 출력에는 롤백 상태가 아예 없다. 롤백이 캐시 offset을 읽기 때문이고, 따라서 건너뛸 프롬프트 크기의 무엇이 없다.

### 3.2 롤백은 trim이고, 그것을 trim으로 만들어 주는 것이 버퍼다

Muse의 모든 캐시는 KV 캐시다. full attention 레이어는 `KVCache`, sliding 레이어는 `RotatingKVCache`다. 부분 수락은 산술로 되감는다. `offset -= n`, 링이면 `idx -= n`도 함께다. 데이터는 움직이지 않고 다음 append가 거부된 행을 덮어쓴다.

이것은 거부된 행이 들어오면서 아직 보이는 무언가를 덮어쓰지 않았을 때만 성립한다. 꽉 찬 링에 `bs`행짜리 verify 블록을 append하면 윈도 엔트리 `bs`개가 밀려나고, offset을 되감아도 그 엔트리는 돌아오지 않는다. 그래서 sliding 캐시는 첫 라운드 전에 `enable_speculative_buffer`로 무장한다. `max_size` 너머의 여분 용량이다. 버퍼가 블록만큼 넓기만 하면 append 후 되감기는 손실이 없다. 7.2절은 그렇지 않았던 경우에 대한 것이다.

---

## 4. verify 폭은 고정이 아니라 측정된다

DFlash 라운드 루프는 실행 내내 `block_size` 하나를 유지했다. 학습된 폭이 곧 써야 할 폭인 drafter에게는 맞고, 여기서는 틀리다.

M5 Max 페어링에서 측정한 값이다. 팔이 비교 가능하도록 모두 inexact override 아래에서, 자연문 프롬프트 세 개다.

| verify 행 | tok/s | 라운드당 수락 |
|---|---|---|
| 16 (공개값) | 13.0 / 18.1 / 12.4 | 1.87 / 3.40 / 1.80 |
| 8 | 24.9 / 30.5 / 21.7 | |
| 4 | 33.6 / 41.3 / 31.6 | 별도 측정에서 1.34 |
| 2 | 30.0 / 33.1 / 31.7 | 첫 제안이 라운드의 74 ~ 82퍼센트에서 수락 |

같은 프롬프트의 classic decode가 18.1 tok/s이므로, 공개된 폭은 자연문에서 오히려 느리다. 2660토큰짜리 반복 로그에서는 같은 16행이 제안 15개 중 14개를 수락한다. 폭을 고정하면 둘 중 하나를 포기해야 한다.

`Drafter::configured_block_size`와 `prefer_requested_block_size`는 MTP 루프용으로 이미 있었다. `DFlashGenerator`가 이제 그것을 읽는다. 요청 폭보다 낮은 depth를 선언하면서 요청 폭을 고집하지 않는 drafter는 둘 사이에서 `BlockThroughputController`를 받고, 컨트롤러는 depth에서 warm-up한 뒤 측정 윈도를 번갈아 돌리며 밀리초당 토큰이 더 많은 쪽을 붙든다. Qwen 3.5 DFlash는 depth로 `None`을, DSpark는 선호로 `true`를 돌려주므로 `dflash_round_loop_starts_at_the_configured_depth`가 둘 다 첫 라운드부터 요청 폭으로 draft함을 고정한다.

---

## 5. 서버 계약

PR #1751은 `DFlashTargetModel` 바깥에 계열별 정책을 한 조각 남겨 두었다. batched variant 게이트에 LFM2 arm이 하드코딩돼 있었고, `supports_batched()` 훅이 해법이며 그것이 필요한 다음 계열로 미룬다는 주석이 붙어 있었다. Muse Glimmer가 그 계열이므로 훅이 여기 들어왔고, 게이트는 이제 구체 타입을 되찾은 뒤 정책은 트레이트에서 읽는 match다.

`requires_dspark_drafter() -> bool`이 `required_drafter_family() -> Option<DFlashDrafterFamily>`가 된 것도 같은 이유다. 이제 두 계열이 페어링을 제한하는데, 불리언은 어느 쪽인지 말하지 못한다. 메시지는 넘어온 계열과 요구되는 계열을 두 실행 arm 모두에서 이름으로 밝힌다.

이 게이트는 drafter로 옮길 수 없다. drafter는 타깃을 아키텍처 문자열이 없는 `LanguageModel`로 보므로, 평범한 DFlash drafter의 `validate_target_compat`은 자기가 돌릴 수 없는 Muse 타깃에도 `Ok`를 돌려준다. 아는 쪽은 타깃이다.

---

## 6. 스타트업 가드

`validate_muse_glimmer_unsupported_startup`은 `--draft-model`, `--draft-kind`, `--draft-block-size`를 조건 없이 한꺼번에 거부했다. 이제 drafter가 지정됐는지로 갈린다.

- drafter 없음: `--draft-kind`와 `--draft-block-size`는 여전히 거부한다. 통짜 규칙도 잡던 운영자 실수다.
- drafter 있음: 그 디렉터리가 `muse_glimmer_assistant`(또는 `MuseGlimmerAssistantModel` 아키텍처)를 선언해야 하고, `--draft-kind`는 없거나 `dflash`여야 한다.

양쪽 모두 서버가 무엇을 로드하기 전에 이름을 밝혀 거부하며, 어댑터, KV 모드, TP, PP는 계속 거부한다.

---

## 7. 머지 전 리뷰 라운드

세 건을 브랜치에서 고쳤다. 나머지는 PR 본문에 기록해 두었다.

### 7.1 게이트가 루프가 돌지 않는 폭을 쟀다

burst는 요청 block size로 `exactness_allows(bs)`를 부르고, 판정은 프로세스 안에서 (모델, 폭)별로 메모된다. 폭이 고정일 때는 타당했다. 4절이 그 전제를 없앤다. 루프는 4에서 warm-up하는데 게이트는 16을 쟀다.

형식적인 문제가 아니다. forward 폭이 MLX가 어떤 quantized matmul 커널을 디스패치할지 고르고, 그래서 `docs/benchmarks.md`에 같은 비교가 폭 8에서 불일치 20.6퍼센트, 폭 32에서 0.0퍼센트로 기록돼 있다. 상한에서의 통과는 depth에 대해 아무 말도 하지 않는다.

`probed_verify_widths(block_size)`가 실행이 정착할 수 있는 폭을 좁은 것부터 나열하고, `dflash_exactness_allows_every_width`가 전부를 요구한다. 실행 막바지에 방출 예산이 강제하는 더 좁은 라운드는 의도적으로 목록에 없다. 그 clamp는 적응 폭보다 먼저 있었고 모든 DFlash 계열에 적용되므로, 전부를 함께 다루는 변경의 몫이다.

측정 호스트에서는 폭 4에서도 판정이 거부로 나왔다(로짓 바이트 404096개 중 162404개 불일치, 폭 16에서는 187123개). 게이트가 루프라면 warm-up 뒤에 떠났을 폭에 대한 판정을 내보내고 있었다는 뜻이다. 이 호스트의 서빙 동작은 두 폭 모두 거부이므로 바뀌지 않는다. 16은 통과하고 4는 통과하지 못하는 호스트에서는 올바른 거부와 조용한 발산의 차이가 된다.

### 7.2 speculative 버퍼가 자기가 감쌀 블록보다 좁을 수 있었다

`speculative_buffer_size`는 Gemma 4 MTP 규칙 그대로인 `clamp(block * 8, 32, 128)`이었다. 16행까지는 비율이 넉넉하고 그다음부터 상한이 지배하므로, 129행 이상에서는 버퍼가 블록보다 좁다.

`--draft-block-size`는 들어오는 길 어디에서도 위로 막혀 있지 않다. `resolve_draft_block_size`는 override를 그대로 돌려주면서 구체 generator가 자기 최솟값을 강제한다고 적어 두었는데, 그것은 최솟값에 대한 진술이다. 그래서 `--draft-block-size 200` 실행은 128행짜리 여유에 200행을 append하고, 아직 보이는 윈도 엔트리 72개를 덮어쓰며, 그다음 `RotatingKVCache::trim`이 live 길이로 clamp하면서 복원할 수 없는 행 위로 offset을 되감는다. 시퀀스는 구멍 난 윈도에서 이어지고, 어디에도 오류가 없고 알아챌 테스트도 없다.

규칙은 이제 block size에서 바닥을 친다. 실제 실행이 닿는 모든 폭에서 값이 예전과 같으므로 기존 Gemma 규칙 테스트는 그대로 두고, 불변식은 두 번째 테스트가 진다.

같은 독해에서 작은 항목 둘이 더 나왔다. 부족한 되감기는 이제 release 프로파일이 지워 버리는 `debug_assert` 대신 레이어와 부족분을 이름으로 밝히는 `tracing::error!`를 낸다. 그리고 drafter 슬롯에 residual stream을 캡처하지 못한 verify는 0으로 채운 슬랩을 조용히 대신 넣는 대신 그 사실을 말한다. drafter는 그 슬랩을 residual stream으로 읽는다.

### 7.3 config 차원, 그리고 버릴 행을 투영하던 인코더

`MuseAssistantConfig`의 모든 차원은 `i32`로 MLX에 넘어간다. wrap 지점을 넘긴 `sliding_window`는 음수 extent가 되고, 음수 extent는 MLX가 던지는 slice로 도착해서 로드 오류가 아니라 cxx 브리지를 건넌 abort가 된다. `validate`가 이제 각각을 막고, `encoder.fc` 모양 검사가 계산하는 `len(target_layer_ids) * hidden_size` 곱도 함께 막는다.

별개로, drafter의 `forward`는 컨텍스트를 윈도로 자르는 일을 `encoder.fc`와 `output_norm_enc`를 통과시킨 뒤에 했다. 둘 다 행 단위 연산이므로 자르는 일이 앞으로 왔다. 첫 라운드에 도착하는 행은 프롬프트 전체다. `-c 16384`에서 들어오는 `[1, S, 5 * 6656]` 슬랩은 f16으로 1.1 GB이고, 그중 8분의 7이 drafter dtype으로 캐스팅되고 투영된 뒤 잘려 나가고 있었다.

---

## 8. 실제 체크포인트 측정

`models/muse-glimmer-30b-4bit`와 `models/muse-glimmer-30b-assistant-bf16`, M5 Max, GPU 세대 17, 포트 19343. 서버는 리베이스와 리뷰 수정 뒤에 다시 빌드했고 바이너리보다 새로운 소스 파일은 없다. 텍스트 프롬프트 세 개를 `temperature: 0, max_tokens: 256, logprobs: true`로 보내고 `--model-draft` 없는 같은 서버와 토큰 문자열을 위치별로 비교했으며, 이미지 요청 하나를 더했다.

| 프롬프트 | 프롬프트 토큰 | 출하 기본값 | `MLXCEL_MTP_ALLOW_INEXACT=1` |
|---|---|---|---|
| tides | 66 | IDENTICAL, classic으로 서빙 | 토큰 59에서 DIVERGE. 109 라운드, 375개 중 146개 수락, 평균 1.34, decode 38.4 tok/s |
| 윈도 넘는 decode | 1968 | IDENTICAL, classic으로 서빙 | IDENTICAL. 47 라운드, 213개 중 209개 수락, 평균 4.45, decode 60.5 tok/s |
| prefill에서 wrap | 2660 | IDENTICAL, classic으로 서빙 | IDENTICAL. 47 라운드, 214개 중 209개 수락, 평균 4.45, decode 60.6 tok/s |

같은 프롬프트의 classic arm과 끝에서 끝까지 비교하면 초당 36.7 대 31.6, 33.9 대 22.5, 31.3 대 20.1 토큰이다.

출하 기본값이 거부하는 것은 probe가 거부하기 때문이고, 이는 문서화된 Apple GPU 세대 15 이상 조건이자 이 호스트에서 LFM2 DSpark가 받는 것과 같은 판정이다. override 아래 tides의 발산은 probe가 거부한 바로 그 바이트 동일성을 override한 뒤에 관찰한 것이므로, 페어링에 대한 반증이 아니라 override를 기본값으로 두지 않는 근거다.

이미지 요청은 `multimodal VLM request detected` 한 줄과 함께 거부되고 HTTP 200으로 classic 서빙된다.

---

## 9. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 28 |
| 추가 | +3848 |
| 삭제 | -165 |
| 새 `#[test]` 함수 | 20 |

| 영역 | 요약 |
|------|------|
| Drafter 코어 | `drafter/dflash/muse/` (신규): 선택 키 둘과 차원 상한을 가진 config, 헤드별 q/k norm과 양방향 마스크를 가진 sliding attention, 시간순 컨텍스트 캐시, 타깃 바인딩을 가진 모델, `Drafter` 어댑터와 페어링 게이트. `drafter_kind_by_model_type`과 `load_drafter`의 `muse_glimmer_assistant` 항목, `Drafter::is_muse_assistant` |
| 라운드 루프 | `DFlashGenerator`가 `configured_block_size` / `prefer_requested_block_size`를 읽고, 선언된 depth와 요청 상한 사이에서 `BlockThroughputController`를 돌린다 |
| Muse 타깃 | `muse_glimmer_speculative.rs` (신규): 캡처가 붙은 verify forward, trim 롤백, speculative 버퍼, exactness probe와 다중 폭 게이트, `SpeculativeTarget`. `MuseCache::trim`과 `MuseCache::enable_speculative_buffer`, 두 wrapper의 `embed_tokens_module` / `lm_head_module` |
| 서버 | `DFlashTargetModel::supports_batched`가 하드코딩된 LFM2 batched arm을 대체하고, `required_drafter_family`가 DSpark 불리언을 대체한다. `MuseGlimmerVLM`이 `run_dflash_burst`의 세 arm과 `model_variant_label`에 합류하고, 스타트업 가드는 assistant drafter만 받아들인다 |
| CLI | `resolve_draft_block_size`의 `peek_muse_assistant_configured_block_size`, 오프라인 거부 메시지가 세 번째 drafter 모양을 이름으로 밝힌다 |
| 문서 | `supported-models.md` DFlash 행과 계열 문단, `speculative-acceptance.md` 다중 폭 게이트, README |

검증: `cargo fmt --all -- --check`, 그리고 `--features metal,accelerate` 아래 `-D warnings`로 `cargo clippy --lib --tests`와 `cargo clippy -p mlxcel-core --lib --tests`. 테스트는 `--profile test-fast --features metal,accelerate`로 `-p mlxcel-core drafter::dflash` 86 통과, 루트 크레이트에서 `muse_glimmer` 109, `server::batch::speculative_burst` 67, `server::startup` 80, `models::detection` 63, `cli::speculative_args` 22, `server::batch::dflash_target` 4 통과.

---

## 10. 후속 과제

- `--draft-block-size`는 여전히 위로 막혀 있지 않다. 버퍼 바닥값이 Muse의 조용한 손상 가능성은 없앴지만, 터무니없는 폭은 여전히 터무니없는 verify 블록을 할당하고 MTP arm에서는 행별 exactness probe에까지 닿는다. `resolve_draft_block_size`나 서버 스타트업 검증에 공용 clamp를 두면 모든 DFlash와 MTP 계열이 한 번에 덮인다. #1751 보고서가 열어 둔 것과 같은 항목이다.
- burst의 거부 줄은 요청 상한을 말하고, 그 위의 probe 판정 줄은 실제로 실패한 폭을 말한다. burst 메시지는 모든 DFlash 계열이 공유하고 읽는 사람을 판정 줄로 안내하므로, 계열별로 나누는 값이 갈라짐의 비용보다 크지 않았다.
- `MuseAssistantContextCache::append`는 매 라운드 윈도 전체를 다시 만든다. 읽을 때 회전시키는 링이나, 절대 key 위치를 보고하도록 배운 `RotatingKVCache`면 복사가 사라진다. 타깃 verify forward 한 번의 일부이고 위 측정을 움직이지 않는다.
- batched(B > 1) Muse 윈도와 drafter 아래의 멀티모달 요청은 범위 밖으로 남고, `draft_window_size`는 받아들이되 쓰지 않는다.
- 오늘 거부되는 커널에서 probe를 통과시키는 경로는 이슈 #1289(순서를 보존하는 streamed qmv)다. 그전까지 Muse burst는 세대 15 이상에서 측정된 속도 향상이 아니라 측정된 거부이며, DSpark와 똑같다.
- `ProbeKey`에는 아직 계열 구분자가 없다. 기존 문제이고 MTP와 DSpark arm이 공유하며 이제 세 번째 계열까지 공유하므로, 메모 자체에서 고칠 근거는 전보다 강해졌다.
