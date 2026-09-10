# 기술 보고서: PR #1753 - feat(speculative): GLM-4.7-Flash (glm4_moe_lite) MTP drafter

**작성일**: 2026-09-10
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 리뷰 후속 사이클
**상태**: 완료 (Apple M5 Max / Metal에서 `models/glm-4.7-flash-4bit`를 `models/glm-4.7-flash-bf16`에서 `mlxcel split-mtp`로 뽑아낸 4비트 drafter와 짝지어 검증. 공용 머신이라 `--workspace` 게이트는 로컬에서 돌리지 않았고, 좁은 범위 실행과 CI가 그 자리를 대신한다)
**언어**: Rust, Markdown
**위험도**: Medium (네 번째 계열이 MTP 라운드 루프에 합류하고 `glm4_moe_lite`가 풀 캐시 계열 중 처음으로 모델 소유 캐시 슬롯을 갖게 되며 사용자가 준 경로에 체크포인트 디렉터리를 쓰는 새 서브커맨드가 생긴다)

---

## 요약

`zai-org/GLM-4.7-Flash`는 next-token-prediction 레이어 하나를 47개 디코더 레이어 바로 뒤인 `model.layers.47.*`에 싣고, 커뮤니티 4비트 변환본은 그것을 전부 버린다. 내려받아서 `--draft-model`로 가리킬 물건이 없다는 뜻이라, 이 변경은 drafter를 만들어 낸다. `mlxcel split-mtp`가 nextn 블록을 독립된 `glm4_moe_lite_mtp` 디렉터리로 뽑아내고, `Glm4MoeLiteMtpDraftModel`이 그것을 상태를 가진 drafter로 돌리며, `MtpTarget` 어댑터가 `Glm4MoeLiteModel`을 이미 Gemma 4와 Qwen 3.5와 Inkling을 서빙하던 라운드 루프에 올린다. 라운드 루프도 drafter 트레이트도 타깃 트레이트도 지연 greedy verify 규칙도 블록 대 체인 정확성 게이트도 이미 있었다. 없던 것은 도구 하나, 모델 하나, 어댑터 하나다.

결정 둘은 따로 들고 갈 값어치가 있다. verify forward에는 폭에 민감한 조각이 둘 있는데 방향을 반대로 잡았다. 어텐션은 디코드 스텝이 내는 shape 그대로 SDPA를 부르도록 materialize한 쿼리 행을 하나씩 돌리고 MoE는 일부러 배치인 채로 두어 프로브가 재도록 넘긴다. 그리고 리뷰가 찾아낸 것은 이 브랜치가 새로 넣은 모델 소유 캐시 슬롯이 맨 `KVCache::new()`로 만들어져서 운영자의 `--kv-cache-mode`가 그 슬롯에 닿을 경로가 없었다는 사실이다. `glm4_moe_lite`가 풀 캐시 계열 중 처음으로 그런 슬롯을 갖게 되었고 모델 소유 계열들이 구현하는 모드 훅이 이 계열에는 필요했던 적이 없었기 때문이다.

---

## 1. drafter는 내려받을 수 있는 어느 체크포인트에도 없다

GLM-4.7-Flash는 `num_nextn_predict_layers: 1`을 선언하고 그 레이어를 `model.layers.47.*`에 담는다. 자기 `embed_tokens`, `enorm`, `hnorm`, `eh_proj`, MLA + MoE 디코더 블록 하나, `shared_head.norm`, `shared_head.head`를 가진 DeepSeek-V3 계열 nextn 블록이다. mlx-community 4비트 변환본의 인덱스는 이 키들을 하나도 이름 붙이지 않는다. 원본 체크포인트가 유일한 출처이고 drafter는 타깃 로더가 읽지 않는 별도 디렉터리가 되어야 한다.

타깃 로드 시점에 nextn 레이어를 직접 읽는 방안은 이슈에서 배제했고 그대로 배제한다. 별도 디렉터리는 독립적으로 양자화할 수 있는데 여기서는 그게 실질적인 차이를 만든다. 4비트 타깃이 16 GB이고 뽑아낸 4비트 drafter가 705 MB이니, drafter만 다른 깊이로 쓰고 싶은 운영자가 타깃을 다시 변환할 이유가 없다. 타깃 로더의 레이어 범위도 `0..num_hidden_layers`로 남으므로 drafter 없이 원본 체크포인트를 서빙하면 예전과 똑같이 동작한다.

`ModelArgs`에 `#[serde(default)]`가 붙은 `num_nextn_predict_layers: usize`가 생겼고 로더는 여전히 `0..num_hidden_layers`만 만든다. 이 필드가 있는 이유는 타깃 config의 사본인 drafter config가 여기서 기본 블록 크기를 끌어내기 때문이다.

---

## 2. `split-mtp`

`src/lib/mlxcel-surgery/src/ops/split_mtp.rs`가 변환(`WeightMap` 위의 순수 함수 `split_mtp`)과 디렉터리 드라이버(`split_mtp_dir`)를, `src/commands/split_mtp.rs`가 `mlxcel split-mtp` 서브커맨드를 들고 있다. 둘 다 `#[cfg(feature = "surgery")]` 뒤에 있다.

드라이버는 체크포인트를 통째로 읽지 않는다. `index_names_nextn_layer`는 `model.safetensors.index.json`만 읽어 `model.layers.{num_hidden_layers}.`로 시작하는 키가 있는지 답하며 샤드는 하나도 열지 않고, 이어서 `load_weights_from_dir_index_filtered`가 인덱스가 그 접두사에 대해 이름 붙인 샤드만, 원본 체크포인트 48개 중 3개만 연다. 62 GB짜리 원본이 주소 공간 몇 GB로 끝난다.

### 2.1 이름 변경, 그리고 블록을 로드 가능하게 만드는 재작성 둘

접두사 여섯 개가 블록의 루트 텐서를 레이어 네임스페이스 밖으로 끌어올리고 `*.rotary_emb.inv_freq`는 버려지며 나머지는 전부 `model.mtp_block.` 아래로 간다.

```text
model.layers.N.embed_tokens.weight      -> model.embed_tokens.weight
model.layers.N.enorm.weight             -> model.enorm.weight
model.layers.N.hnorm.weight             -> model.hnorm.weight
model.layers.N.eh_proj.weight           -> model.eh_proj.weight
model.layers.N.shared_head.norm.weight  -> model.shared_head_norm.weight
model.layers.N.shared_head.head.weight  -> lm_head.weight
model.layers.N.<rest>                   -> model.mtp_block.<rest>
```

이름만 바꿔서는 로드되지 않는다. `glm4_moe_lite` 로더가 원본 체크포인트에 없는 텐서 배치 둘을 기대하기 때문이다. 두 재작성 모두 타깃 자신의 sanitizer가 디코더 레이어마다 이미 수행하던 것이다.

첫째는 MLA 분해다. `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` 짜리 `kv_b_proj.weight`를 `[H, qk_nope + v, rank]`로 reshape한 뒤 `[H, kv_lora_rank, qk_nope_head_dim]`인 `embed_q.weight`(nope 절반을 transpose한 것)와 `[H, v_head_dim, kv_lora_rank]`인 `unembed_out.weight`로 쪼개고, `MultiLinear`의 matmul이 view가 아니라 연속 stride를 보도록 각각 복사한다. GLM-4.7-Flash에서는 `[8960, 512]`가 `[20, 512, 192]`와 `[20, 256, 512]`가 된다.

이 코드는 원래 `src/models/glm4_moe_lite_sanitize.rs`에 인라인으로 있었고 그래서 이 diff에서 sanitizer가 `+29 / -112`로 167줄에서 84줄로 줄었다. `mlxcel-surgery`가 의존성 그래프에서 바이너리 크레이트 아래에 있어서 원래 자리에서는 호출할 수 없었기 때문에 `mlxcel_core::mla::decompose_kv_b_proj`로 옮겼다. 지금 이것을 공유하는 호출자는 셋이다. 타깃 sanitizer(디코더 레이어마다 한 번, 라벨 `layer {i}`), surgery op(한 번, 라벨 `MTP block`), 그리고 `kv_b_proj`를 그대로 둔 손수 만든 디렉터리를 위한 drafter 자신의 sanitizer다. `split-mtp` 출력에는 이미 쌍이 들어 있으므로 세 번째 호출자는 `Ok(false)`를 돌려준다. `kv_b_proj_geometry(&ModelArgs)`도 함께 공개되어 있어 두 sanitizer가 어느 head dimension이 무엇인지를 두고 갈라질 수 없다.

둘째는 전문가 스태킹이다. `e in 0..n_routed_experts`에 대한 `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`가 `[64, out, in]`짜리 `mlp.switch_mlp.{proj}.weight`가 된다. 이미 쌓인 형태로 들어온 projection은 다시 만들지 않고 건너뛴다.

### 2.2 예외 둘, 그런데 서로 다른 종류의 예외

dtype 패스는 `mlp.gate.e_score_correction_bias`만 빼고 전부 bf16으로 캐스팅한다. 이 텐서를 float32로 강제하는 이유는 라우터의 선택이 점수를 이 값과 비교하기 때문이고 bf16 왕복 한 번이 어느 전문가가 뽑히는지를 바꾼다.

선택적 affine 양자화 패스가 빼는 텐서는 다르고 이유도 다르다. `quantizable`은 `.weight` 접미사, 랭크 2 이상, 그룹 크기로 나누어떨어지는 마지막 축을 요구하고 라우터 자신이 dense로 남도록 `mlp.gate.weight`를 이름으로 제외한다. 걸리는 것은 전부 packed `.weight`에 `.scales`와 `.biases`를 붙여 쓰고 config가 `quantization`과 `quantization_config` 양쪽에 `{"group_size": 64, "bits": 4, "mode": "affine"}`을 기록한다. 루프는 양자화 전에 뜬 키 스냅샷을 도므로 자기가 쓴 `.scales`를 다시 방문하지 않는다.

만들어진 4비트 drafter는 텐서 54개다. `mlp.gate.weight`가 bf16 `[64, 2048]`로, `e_score_correction_bias`가 float32 `[64]`로 들어 있는 것이 산출물에서 눈으로 확인되는 두 예외다.

쓰이는 `config.json`은 키 넷에 양자화 블록 둘이다. `model_type: "glm4_moe_lite_mtp"`, `block_size`, `tie_word_embeddings: false`(블록이 자기 head를 싣고 있으니 무조건), 그리고 원본 text config에서 `quantization`과 `quantization_config`를 떼어 낸 `text_config`. 4비트 원본의 낡은 블록이 drafter 자신의 상태를 설명하는 일이 없도록 떼어 낸다. 동반 파일 다섯 개는 있으면 복사하고 `tokenizer.json`이나 `tokenizer_config.json`이 없으면 오류가 아니라 경고로 알린다.

### 2.3 도구가 거부하는 것

`--block-size`는 `MAX_BLOCK_SIZE = 16`에서 묶이고 검사는 geometry 읽기보다도 텐서 작업보다도 먼저 일어난다. 기록된 값이 `peek_glm4_moe_lite_mtp_configured_block_size`를 통해 서버의 기본 verify 폭이 된다. 이 계열은 verify 위치마다 쿼리 행 하나를 materialize하므로 폭이 라운드당 그래프 크기와 정확성 프로브의 체인 갈래를 함께 곱한다. 검사 없는 `--block-size 20000`은 멀쩡히 로드되고 나서 스케줄러 tick을 쥔 채 라운드마다 그래프 노드를 대략 백만 개 만드는 디렉터리를 내놓았다. 학습된 깊이를 넘겨 draft하는 것은 이미 수락률을 잃는 일이니 쓸 만한 어떤 폭보다도 한참 위에 천장을 두면 비용이 없다.

`--force` 가드는 이제 `model.safetensors` 하나가 아니라 디렉터리를 체크포인트로 만드는 것 전부를 본다. `existing_checkpoint_marker`가 `model.safetensors`, `model.safetensors.index.json`, `config.json`, `model-*.safetensors` 샤드 중 먼저 발견되는 것을 돌려준다. 샤드로 나뉜 체크포인트에는 `model.safetensors`가 없으므로 그전에는 그런 디렉터리를 `--output`으로 가리켜도 가드가 발화하지 않았다. 그 실행이 그 `config.json`을 drafter의 것으로 갈아치우고 토크나이저 파일 최대 다섯 개를 덮어쓰면서 샤드를 고아로 만들었다. `--force`는 낡은 `model.safetensors.index.json`도 함께 지운다. `collect_shard_paths`가 맨 `model.safetensors`보다 인덱스를 선호하므로 남은 인덱스가 로더를 피해자 쪽 샤드로 보내기 때문이다.

원본과 canonicalize 결과가 같은 `--output`은 아무것도 쓰이기 전에 거부된다. 실패 방식이 조용해서다. 자기 자신을 가리키는 경로에 대한 `std::fs::copy`는 `O_TRUNC`로 목적지를 연 뒤 `Ok(0)`을 돌려주고(검증 호스트에서 확인했다), 그래서 동반 파일이 전부 0바이트가 되면서 오류는 하나도 드러나지 않는다. 원본 아래의 하위 경로는 다른 디렉터리이므로 여전히 허용된다.

이미 양자화된 채로 도착한 nextn 레이어는 `reject_packed_tensors`가 거부한다. 분해와 스태킹 뒤에 돌기 때문에 그 둘이 다루지 않는 packed 텐서만 본다. `stack_experts`는 쌓이지 않은 packed 전문가를 전문가별로 이름을 대며 거부하지만 전문가가 이미 쌓인 채 packed된 변환본은 그 검사를 빠져나가 bf16 패스에 도달했고 거기서 packed uint32 페이로드가 오류 한 줄 없이 float로 다시 쓰였다.

마지막으로 인덱스가 `model.layers.{N}.*` 텐서를 하나도 이름 붙이지 않는 체크포인트는 원인을 지목하는 메시지로 거부된다. MTP 레이어 없이 변환된 것이니 원본 zai-org 체크포인트를 쓰라는 내용이다. 그게 발화하기 전에 출력 디렉터리에는 아무것도 만들어지지 않는다.

### 2.4 `decompose_kv_b_proj`와 `i32::try_from`

분해를 공유 헬퍼로 옮기면서 그것이 surgery 경로에 올라갔는데 여기서 config는 로더가 이미 검증한 체크포인트가 아니라 사용자가 준 JSON 파일이다. geometry 필드 넷은 `as i32`로 변환되고 있었다. `test-fast` 프로파일이 release를 상속하니 오버플로 검사가 꺼져 있고 `as i32` 의미로 계산한 `num_heads * head_dim`은 우연히 `w_shape[0]`과 같은 값으로 랩될 수 있다. 망가진 config를 `reshape`에서 떼어 놓으려고 존재하는 바로 그 shape 검사를 통과해 버린다는 뜻이다. MLX는 잘못된 reshape를 throw로 알리고 `cxx::bridge`(`Result`가 아니라 `UniquePtr<MlxArray>`를 돌려준다)를 건너는 C++ throw는 호출을 실패시키는 대신 프로세스를 abort시킨다. 서버에서는 그 일이 weight sanitize 도중에 일어나 모든 테넌트를 함께 내린다.

이제 변환은 오류에 필드 이름을 담는 `i32::try_from`이고 `qk_nope_head_dim + v_head_dim`은 `checked_add`이며 head-dim 곱은 shape 비교가 쓰기 전에 `checked_mul`로 범위 검사된다. 가드용 곱은 계산하고 버리며, 비교는 다시 계산한다.

---

## 3. drafter

### 3.1 한 스텝

`Glm4MoeLiteMtpDraftModel`은 자기 토큰 테이블, `enorm`, `hnorm`, `eh_proj`, `TransformerBlock` 하나, `shared_head_norm`, 묶이지 않은 `lm_head`를 들고 있다. 타깃에서 빌려 오는 것은 없다. `e = embed_tokens(token_{t+1})`, `h_t`를 타깃의 위치 `t` 은닉이라 할 때:

```text
x           = eh_proj(concat(enorm(e), hnorm(h_t), axis=-1))
x           = block(x, cache)          # cache.offset에서 RoPE, 자기 KVCache
logits      = lm_head(shared_head_norm(x))
hidden_next = x                        # 다음 draft 스텝으로 되먹인다
```

블록은 `TransformerBlock::from_weights_with_prefix(weights, args, "model.mtp_block", is_moe)`로 만들어진다. `TransformerBlock::from_weights`는 이제 `model.layers.{i}`와 `args.is_moe_layer(i)`를 넘겨 여기로 위임한다. 이 배치의 요점이 그것이다. drafter의 absorbed MLA, 선택 전용 보정 편향이 붙은 sigmoid 라우팅, 그룹핑과 `routed_scaling_factor`와 shared expert가 전부 디코더의 코드이지 따로 갈라질 수 있는 두 번째 사본이 아니다.

weight 인벤토리는 닫힌 채로 실패한다. 모든 키가 허용된 접두사 일곱 중 하나로 시작해야 하고 `model.layers.0.*`를 든 디렉터리는 그 이름을 지목당하며 거부되므로 온전한 타깃 체크포인트가 drafter 디렉터리로 오인될 수 없다.

### 3.2 캐시 산술

drafter는 상태를 갖는다. 소비한 타깃 위치마다 항목 하나를 담는 `KVCache` 하나에, 다음 append의 절대 타깃 위치를 좇는 `next_position`과 이번 라운드의 speculative append를 세는 `round_appended`가 붙는다.

`prefill_from_target_hidden`은 프롬프트를 한 칸 왼쪽으로 밀고 첫 보너스를 붙인 뒤 타깃의 프롬프트 은닉과 짝지어 한 번의 forward로 통과시키며, 캐시를 `P`에 남긴다. 산술이 사는 곳은 `accept_verified_tokens`다.

```rust
let keep = accepted.min(self.round_appended.max(0) as usize);
let trim = self.round_appended - keep as i32;
if trim > 0 { self.cache.trim(trim); self.next_position -= trim; }
let mut tokens: Vec<i32> = draft_tokens[keep..accepted].to_vec();
if let Some(&last) = new_tokens.last() { tokens.push(last); }
```

`keep`이 `accepted` 하나가 아니라 `round_appended`로도 묶이는 이유는, seed로 나온 첫 제안의 캐시 항목을 붙인 것이 이번 라운드가 아니라 직전 훅이어서 이번 라운드가 자를 몫이 아니기 때문이다. 재forward는 `draft_tokens[keep..accepted]`와 보너스를 verify 은닉의 `[keep, keep + tokens.len())` 행과 짝짓는데 이것은 drafter 자신의 추측이 아니라 그 위치들에 대한 타깃의 참값이다. `P`에서 출발해 어떤 수락 개수든 캐시는 `P + accepted + 1`에 도착한다. 그래서 다음 `set_shared_kv`가 다시 앵커를 잡는 대신 히스토리를 유지한다.

seed를 발견한 `draft_block`은 첫 제안에 forward를 하나도 쓰지 않으므로, seed가 있는 라운드는 `block_size - 2`번, 없는 라운드는 `block_size - 1`번 forward한다. 기본값 `block_size = 2`에서는 seed가 있는 라운드가 블록을 아예 돌리지 않고 draft한다.

`accept_verified_tokens`는 변경 전에 검증하고 오류 경로에서 런타임 상태를 지운다. 오염된 drafter가 반쯤 적용된 라운드에서 이어 가는 대신 다음 `set_shared_kv`에서 깨끗하게 다시 앵커를 잡는다.

### 3.3 어느 은닉 상태를 먹이는가

이슈가 일부러 열어 둔 지점이다. nextn 블록이 타깃의 post-final-norm 출력을 원하는지 pre-norm residual을 원하는지 결정해 주는 키 이름이 없어서 재서 drafter 모듈 문서에 기록하라고 요구했다.

**post-final-norm이 이겼고 기본값이다.** `models/glm-4.7-flash-4bit`와 4비트 drafter 짝, M5 Max, `block_size = 2`에서 greedy 128토큰 기준으로 post-final-norm 탭은 제안 73개 중 55개를 수락했고(평균 수락 길이 1.753), pre-norm residual은 74개 중 53개였다(1.716). 방출된 128토큰 id 스트림은 두 탭에서 동일했다. 타깃의 verify 패스가 그것을 보장한다. 그러니 탭이 움직이는 것은 수락률뿐이다. `MLXCEL_GLM_MTP_HIDDEN_TAP=pre`가 진 쪽을 다시 재기 위해 남아 있다.

이 측정은 DeepSeek-V3 nextn 관례와도, Qwen 3.5 어댑터의 기존 선택과도 일치한다. 프롬프트 하나에서 나온 0.037토큰 차이를 믿을 수 있는 이유가 그것이다. 숫자는 선택의 근거 전부가 아니라 사전 가정에 대한 확인이다. 두 forward 모두 pre-norm residual을 캡처하고 캡처 경로에서 `apply_final_norm`을 다시 적용하므로, 기본 탭은 forward가 로짓용으로 이미 계산한 마지막 RMSNorm을 한 번 더 계산한다.

### 3.4 drafter를 만드는 곳

`mlxcel-core`는 이 drafter를 만들 수 없다. `glm4_moe_lite`의 `TransformerBlock`을 재사용하는데 그것은 core 위 바이너리 크레이트에 살아서 core에는 이 타입을 부를 이름이 없다. `src/models/drafter_loader.rs`가 오프라인 CLI와 서버의 drafter 슬롯과 speculative 벤치가 모두 지나는 단일 진입점이다. kind를 해석하고 peek한 `model_type`이 맞으면 GLM drafter를 직접 만들고, 나머지는 전부 그대로 core에 위임한다.

core는 여전히 `("glm4_moe_lite_mtp", DrafterKind::Mtp)`를 등록해서 `--draft-kind` 없이도 `--draft-model`이 kind를 자동 판정하게 하고 `GLM4_MOE_LITE_MTP_MODEL_TYPE`은 공개다(바로 옆의 Qwen 상수는 비공개로 남는다). 디스패치가 한 크레이트 위에서 일어나기 때문이다. core 자신의 `load_drafter`는 이 model_type을 `DrafterError::BinaryCrateDrafter`로 이름을 대며 거부하는데 Gemma 4 assistant catch-all보다 앞에 두었다. 그러지 않으면 디렉터리가 엉뚱한 계열을 탓하는 weight 인벤토리 오류를 내는 로더로 떨어진다. 오류 문구가 `mlxcel::models::drafter_loader::load_drafter`를 해결책으로 지목하고 테스트가 그 지목이 유지되는지를 확인한다.

`configured_block_size()`는 `runtime_block_size()`를 보고한다. `min(block_size, num_nextn_predict_layers + 1)`을 2로 바닥 친 값이니 파싱된 필드가 아니라 학습된 깊이다. `prefer_requested_block_size()`가 `true`라서 더 큰 `--draft-block-size`는 여전히 이긴다.

---

## 4. verify forward: 폭에 민감한 조각 둘을 반대로 결정하다

온도 0 계약은 `M = bs` verify 블록이 `bs`번의 단일 토큰 디코드 스텝과 같은 로짓을 낸다는 것이다. 양자화 체크포인트에서 이를 깨뜨릴 수 있는 것이 셋이다. `M = bs`와 `M = 1`에서 MLX가 어느 양자화 matmul 커널을 디스패치하는가, SDPA 커널과 점수 리덕션이 한 행 쿼리와 여러 행 쿼리에서 달라지는 어텐션, 그리고 행 수에 따라 리덕션이 갈리는 MoE다. 첫째는 게이트의 `qmv_wide` 재시도가 처리한다. 나머지 둘은 서로 반대로 결정했다.

**어텐션은 materialize한 쿼리 행을 하나씩 돌린다.** projection(`q_a`/`q_b`, `kv_a`, `embed_q`, `unembed_out`, `o_proj`)은 블록 전체에 대해 한 번씩 돈다. verify가 시간을 버는 자리가 거기이기 때문이다. 이어지는 행별 루프가 행 `i`를 떠서 그 행이 볼 수 있는 `offset + i + 1`개 캐시 항목에 어텐션하므로, 인과성이 슬라이스 범위에서 나오고 덧셈 마스크는 만들어지지 않으며 모든 SDPA와 모든 `q_pe @ k_pe^T` 호출이 단일 토큰 디코드 스텝이 내는 shape 그대로다. Qwen 3.5의 `target_verify`가 하는 것과 같은 분리다.

shape만으로는 부족했다. 배치된 `[1, H, bs, d]` 쿼리의 한 행 슬라이스는 부모의 stride를 유지하고 MLX의 SDPA와 matmul은 커널을 연속성으로 고르므로, 행 view는 쓰이기 전에 `mlxcel_core::contiguous(..., false)`를 통과한다. view인 채로 두었을 때 verify 행들은 데이터에 따라 다른 자리에서 디코드 체인과 갈라졌고(어떤 실제 프롬프트에서는 4번 위치, 다른 프롬프트에서는 14번 위치), 그 어긋남이 캐시를 타고 누적됐다. materialize하는 것은 쿼리 쪽뿐이다. 키와 latent 슬라이스는 인덱스 0에서 시작하니 연속 부모의 앞쪽 접두사라서 이미 연속이다.

이 결함이 프로브 크기도 정했다. `PROBE_BLOCKS_PER_DRAW = 4`가 있는 이유는 draw당 블록 하나로는 갈라짐을 통째로 놓쳤기 때문이다. `M = bs` 블록 여러 개가 캐시를 먹인 뒤에야 드러났다. 프로브는 8토큰 프롬프트로 draw 세 번이고 각 draw가 연속 블록 4개를 걸으며 모든 행을 비트 단위로 비교한다. 판정은 decline 로그 줄과 `qmv_wide` 재시도와 `MLXCEL_MTP_ALLOW_INEXACT`를 소유한 공용 `mtp_exactness_gate`를 통해 `ProbeKey`에 메모이즈된다.

**MoE는 일부러 배치인 채로 둔다.** `SwitchGLU::forward`는 `n_tokens * top_k >= 64`가 되면 gather-sort 경로로 넘어간다. 기본 `block_size = 2`에 `num_experts_per_tok = 4`면 곱이 8이라 verify 블록이 디코드 체인과 같은 리덕션을 타고 손댈 것이 없다. 넓은 `--draft-block-size`는 문턱을 넘는다. 16행에 top-4면 정확히 64다. 이것을 행별로 되돌리면 verify가 번 것을 전부 반납하게 되므로 프로브가 재도록 남겨 두었고 문턱을 넘으면서 갈라지는 폭은 갈라진 스트림을 방출하는 대신 페어링을 decline한다. 이 계열에서 넓은 `--draft-block-size`와 갈라진 스트림 사이에 서 있는 것은 프로브뿐이며 그것은 놓친 것이 아니라 의도한 위치다.

---

## 5. MTP 슬롯과 KV 캐시 모드 표

리뷰가 HIGH로 매긴 발견 사항이고 살아 있는 회귀가 아니라 구조적 구멍이다. 이 문장의 양쪽 절이 모두 중요하다.

`Glm4MoeLiteModel`은 `DenseKvCache` 계열이다. 고전 서빙은 캐시를 스케줄러의 `CachePool`에 두고, 스케줄러가 할당 직후 풀 안에서 해석된 레이어별 모드로 올려 준다. 그래서 이 모델은 `LanguageModel::set_kv_cache_layer_modes`를 구현한 적이 없고 no-op 기본값을 썼다. Gemma 4와 Qwen 3.5는 처음부터 끝까지 모델 소유이고 둘 다 이 훅을 구현한다.

MTP 어댑터는 풀을 쓸 수 없다. 트레이트 메서드가 캐시 인자 없는 `&self`이고 서버의 기본 MTP 경로가 tick마다 어댑터를 다시 만드는 tick 협조 슬라이스이니 캐시는 모델에 살아야 한다. 그래서 이 브랜치가 `Glm4MoeLiteModel`에 `ModelOwnedSequenceState<KVCache>`를 붙였고, `glm4_moe_lite`는 모델 소유 슬롯을 갖는 첫 풀 캐시 계열이 되었다. 그런데 모든 설치가 그 슬롯을 무조건 FP16인 `make_caches()`로 만들었다. 운영자가 해석해 넘긴 모드 표를 실제로 MTP 세션이 도는 캐시로 가져가는 것이 아무것도 없었다. 서버는 요청된 모드를 적용했다고 계속 로그에 남겼다.

수정은 모델 위의 `KvCacheLayerModes` 표, 트레이트 메서드 둘의 구현, 그리고 모든 슬롯 설치가 지나는 새 `make_configured_caches()`다. `make_caches()`는 일부러 무조건 FP16으로 남는다. 그쪽은 풀을 먹이고 풀이 모드를 스스로 적용하기 때문이다. 호출 지점 일곱 개가 옮겨 갔다. `reset_mtp_sequence_state`의 두 갈래, prefill과 verify 훅의 지연 생성 클로저 둘, `reset_runtime_state`, 그리고 **블록 대 체인 프로브의 양쪽 갈래 모두**다. 마지막 것이 건너뛰기 쉬우면서 건너뛰면 틀리는 지점이다. 양자화된 KV 모드는 프로브가 판정하려는 산술의 일부이므로, FP16에서 프로브하면 실제 세션이 돌지 않는 경로를 통과시키게 된다.

오프라인 CLI에는 세 번째 주입 지점이 필요했다. `run_offline_mtp`는 `GenerationConfig`를 만들지 않으니 그 경로의 어떤 것도 알림에 찍힌 모드를 슬롯으로 옮기지 못한다. 이제 서버의 `resolved_kv_cache_layer_modes`가 계산하는 것과 같은 `resolve_layer_modes(kv_cache_mode, num_layers, boundary_v_layers_from_env())`를 계산해 디스패치 전에 주입한다. 이 주입을 `glm4_moe_lite`로 좁힌 것은 의도적이다. 다른 MTP 계열들도 오프라인 경로에 같은 구멍이 있지만 그쪽을 고치면 그들의 오프라인 실행이 재는 대상이 바뀌므로 자기 몫의 실제 체크포인트 검증과 함께 가야 한다.

**오늘 이 불일치에 도달하는 설정은 없다.** `resolve_kv_cache_mode_for_model`(#1350)이 이미 MLA-latent 계열의 모든 양자화 모드를 fp16으로 해석하고 `glm4_moe_lite`도 거기 들어가며, `--kv-cache-mode` 경로와 `--kv-bits` 경로 양쪽에서 그렇다. `--kv-cache-mode int8` 검증 실행이 바깥에서 그것을 확인해 준다. 배너가 `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)`를 보고하고 고전과 MTP 스트림이 64토큰에 걸쳐 id 동일하다. #1350의 치환이 이 구멍이 운영자에게 보이지 않았던 이유이고 구멍을 닫아 두면 이 계열이 언젠가 보정된 양자화 모드를 갖더라도 불변식이 유지된다.

테스트는 일괄 적용으로는 통과할 수 없게 짜여 있다. 주입하는 표가 마지막 레이어만 Fp16으로 두고 나머지를 전부 Int8로 두며 생성자 하나가 아니라 설치 지점 둘을 모두 확인한다.

---

## 6. 이 계열에서 접두사 캐싱과 MTP는 함께 가지 않는다

두 기능은 여기서 상호 배타적이고 문서가 이제 적응이 정상 동작한다고 암시하는 대신 그렇다고 적는다.

Gemma 4와 Qwen 3.5와 Inkling 어댑터는 모델 소유 시퀀스 상태 위에서 도는데 프롬프트 캐시 적응이 접두사를 복원해 둔 자리가 바로 거기라서 접미사만 forward한다(#518). `glm4_moe_lite`는 고전 캐시를 `CachePool`에 두고 MTP 어댑터는 별도의 모델 소유 슬롯에서 돈다. 적응된 접두사는 그 슬롯에서 닿을 수 없고 그런 요청을 MTP로 끝내면 finalizer가 `prompt ++ generated` 전체 키로 기부하는 동안 풀 캐시는 접두사만 들고 있게 된다.

`mtp_adopted_prefix_reusable(model)`이 한 줄짜리 정책이고 burst와 tick 슬라이스 모두 `prefill_start_offset > 0`일 때 drafter를 슬롯에서 꺼내기 전에 이 값을 읽는다. decline은 그저 참을 만한 정도가 아니라 안전하다. dense 갈래에서도 paged 갈래에서도 부분 항목이 프롬프트 캐시 저장소에 도달할 수 없다. 비용은 잃어버린 최적화이고 `docs/supported-models.md`가 이제 그 교환을 직접 이름 붙여 준다. 턴 사이의 접두사 재사용이 디코드 속도 향상보다 중요하면 `--model-draft` 없이 서빙하라고 적혀 있다. MTP로 서빙된 요청은 아무것도 기부하지 않으므로 나중 턴이 적응할 접두사를 찾지 못한다.

---

## 7. 나머지 리뷰 발견 사항과 그에 대한 조치

### 7.1 고친 것

**tick 슬라이스가 park할 때마다, 요청이 끝날 때마다 drafter를 버렸다.** `park_speculative_slice`와 `finalize_speculative_slice`에 `LoadedModel::Glm4MoeLite` 팔이 없어서 둘 다 방어용 `_ => None`으로 떨어졌고, drafter 핸들을 워커 슬롯에 돌려주는 대신 버려서 뽑아낸 drafter를 요청마다 한 번씩 디스크에서 다시 읽었다. 트리 자신의 가드인 `every_mtp_dispatch_site_covers_every_capable_variant`(#1165)는 `mtp_capable_target`의 본문을 읽어 거기 이름 붙은 모든 변종이 디스패치 지점 일곱 곳에 전부 나타나기를 요구하며 두 지점 모두에서 이미 실패하고 있었다. 눈에 띄지 않았던 이유는 이 브랜치가 검증에 쓴 좁은 테스트 필터가 `server::batch::speculative_burst_tests`에 닿지 않아서다. 모든 지점에 `_ =>` 팔이 있으니 빠진 계열은 컴파일 실패가 아니라 조용한 decline이 되고 그 가드가 존재하는 이유가 정확히 이 모양의 결함이다.

**`decompose_kv_b_proj`가 geometry를 `i32::try_from`으로 변환**하고 head-dim 곱을 범위 검사한다. 2.4절에서 다뤘다.

**`split_mtp_dir`이 원본을 가리키는 `--output`을 거부**하고 `split_mtp`가 이미 양자화된 nextn 레이어를 거부한다. 둘 다 2.3절에서 다뤘고 각각 단위 테스트를 달았다. 별칭 테스트는 거부가 발화하는지만이 아니라 이후에도 원본 `config.json`과 `chat_template.jinja`가 온전한지까지 확인한다.

더 작은 교정들은 전부 운영자가 가장자리에 닿기 전에는 알아채지 못했을 것들이다. `num_nextn_predict_layers + 1`이 surgery op와 drafter config 양쪽에서 포화 덧셈이라 `u64::MAX` 모양 필드가 블록 크기 0으로 랩되지 못한다. `block_size`와 `num_nextn_predict_layers`가 둘 다 없는 config는 아무도 쓰지 않은 `block_size`를 탓하는 대신 어느 필드가 없는지를 듣는다. `prefill_from_target_hidden`은 빈 프롬프트 조기 반환보다 먼저 런타임 상태를 지워서 재활용된 drafter가 직전 세션의 seed 토큰과 은닉을 물려받지 못한다(서버가 빈 프롬프트를 상류에서 두 번 거부하므로 도달 가능한 경로를 고친 것이 아니라 그 가드들에 대한 의존을 없앤 것이다). `peek_drafter_model_type`은 core 밖 호출자가 생겼으니 `docs/code-guidelines.md`가 요구하는 `Used by:` 주석을 얻었다.

### 7.2 열어 둔 것

**프로브는 폭 하나만 잰다.** `mtp_exactness_allows(block_size)`는 설정된 폭에서 프로브하는데 이 프로젝트의 벤치마크 지침 자체가 블록 대 체인 불일치가 폭에 따라 크게 달라진다고 적고 있다. MTP 갈래는 Gemma 4 때부터 이 모양을 공유했으므로 회귀는 아니지만 계약은 엄밀히 말해 프로브한 폭에서만 측정되어 있다.

**`ProbeKey`에 계열 구분자가 없다.** 키는 `{block_size, hidden_size, num_hidden_layers}`이고 한 프로세스가 타깃 모델 하나를 서빙한다고 가정하는데 라우터 모드는 그렇지 않다. 기존 사안이고 다른 모든 MTP 갈래와 #1751이 함께 쓰므로 메모 자체를 건드리는 변경의 몫이다.

**`--draft-block-size`는 무경계인데 `split-mtp --block-size`는 16에서 묶인다.** 천장은 체크포인트가 실행에 옮길 수 있는 값을 묶고, 운영자 플래그는 신뢰할 수 없는 입력이 아니니 이 비대칭은 종류로는 맞다. 다만 천장을 만들게 한 행별 verify 비용이 플래그 오타 하나로 도달 가능하다. #1751이 DSpark에 같은 메모를 남겨 두었으니 누가 쓰든 한 변경이 둘을 함께 답한다.

**`rollback_speculative_cache_for_sequence`는 레이어 0의 trim 개수만 돌려주고** 유일한 호출자가 그것을 버린다. 모든 레이어가 보조를 맞춰 전진하니 갈라짐은 불가능해야 하지만 갈라져도 알아챌 것이 없다.

---

## 8. #1751 위로의 리베이스

LFM2 / LFM2.5 DSpark(#1751)가 같은 파일들에 착륙했고 충돌은 전부 speculative 서브시스템에 있었다. 하나같이 고르는 대신 둘 다 남기는 쪽으로 풀렸다.

- 두 블록 크기 peek은 서로 다른 `DrafterKind`를 본다. #1751의 `peek_dspark_configured_block_size`는 `Dflash` 팔에 있고 `DFlashConfig` 전체를 파싱한다. 이 브랜치의 `peek_glm4_moe_lite_mtp_configured_block_size`는 세 번째 `Mtp` peek이고 `model_type`으로 자기를 밝힌다. core가 공통 본문을 `peek_configured_block_size_for(path, expected_model_type)`으로 뽑아내서 무관한 drafter 모양의 같은 이름 필드가 블록 크기 힌트로 읽힐 수 없게 했고 테스트가 양방향 비간섭을 고정한다.
- `DrafterError`는 새 변종 둘을 서로 무관한 인접 팔로 들고 있다. #1751의 `GreedyOnly`와 이 브랜치의 `BinaryCrateDrafter`다.
- `speculative_burst.rs`에서는 import가 갈린다. `load_drafter`는 이제 이 브랜치가 필요로 하는 바이너리 크레이트 래퍼로 해석되고 `sampler_is_greedy`는 #1751이 둔 자리인 `mlxcel_core::drafter::dflash::drafter`에서 계속 온다.
- `model_variant_label`과 burst의 decline 메시지는 두 계열의 팔을 모두 싣는다.

---

## 9. 검증

Apple M5 Max의 실제 체크포인트, 리베이스 뒤 이 트리에서 다시 빌드한 릴리스 바이너리. 타깃 `models/glm-4.7-flash-4bit`(16 GB), drafter `models/glm-4.7-flash-mtp-4bit`(705 MB, 텐서 54개, `block_size: 2`)는 `mlxcel split-mtp --model models/glm-4.7-flash-bf16 --output models/glm-4.7-flash-mtp-4bit --q-bits 4`가 만들었다. 모든 CLI 실행은 `MLXCEL_PRINT_TOKEN_IDS=1` 아래의 `mlxcel generate -m models/glm-4.7-flash-4bit -p "Explain in two sentences why the sky is blue." -n 128 --temp 0 --show-reasoning`이고 id는 `cmp`로 비교했다.

**4갈래 CLI 동일성.** 고전과 MTP, `MLXCEL_FUSED_MOE` 미설정과 `=0`, 전부 `MLXCEL_QMV_WIDE=0` 아래: 128토큰 전부 id 동일, md5 `f22d2a4b`. 리뷰 수정 전과 후와 리베이스 후 세 번 다시 돌렸고 매번 같은 md5였다. 이 계열은 융합 단일 토큰 MoE 커널을 한 번도 디스패치하지 않으므로 `MLXCEL_FUSED_MOE`는 어느 갈래에서도 아무것도 바꾸지 않는다.

**수락률.** 평균 수락 길이 1.753(제안 73개 중 55개)이고 세 번의 실행에서 바뀌지 않았다. pre-norm 탭은 1.716(74개 중 53개)이었고 같은 id를 냈다.

**갈라진 지점 하나는 알려진 커널 효과이지 MTP 경로가 아니다.** 기본 wide `qmv` 커널을 쓴 고전 실행이 near-tie 위치 한 곳에서 다른 모든 갈래와 다른데, 그 갈래는 narrow 고전 실행과도 다르다. 정확성 게이트가 존재하는 이유인 #1199 커널 선택 효과다. 게이트가 그것을 로그에 남긴다. `MTP exactness probe failed under qmv_wide ... passed without it. Disabling qmv_wide for this process`.

**옵트인 실제 체크포인트 게이트.** `MLXCEL_TEST_GLM_MTP_TARGET=models/glm-4.7-flash-4bit cargo test ... real_checkpoint_verify_block_matches_decode_chain`: 128개 위치, 바이트 불일치 0, 최대 절대 차이 0.0, argmax 뒤집힘 0. 이 테스트는 teacher-forced라 갈라짐 때문에 잃는 것이 없고 세대 15 이상에서 게이트가 하는 것과 똑같이 `set_qmv_wide(false)`로 프로세스를 narrow에 고정하며 `docs/benchmarks.md`를 따라 바이트 동일성이 아니라 결정된 위치의 불일치를 단언한다. materialize하지 않은 쿼리 view의 어긋남을 잡아낸 것이 이 테스트다.

**서버.** 포트 19326, 양쪽 다 `MLXCEL_QMV_WIDE=0` 아래: `mlxcel-server -m models/glm-4.7-flash-4bit --model-draft models/glm-4.7-flash-mtp-4bit`와 drafter 없는 같은 서버를 비교. 챗 템플릿을 적용한 프롬프트로 `temperature: 0`, `n_predict: 128`, `return_tokens: true`인 `POST /completion`. 양쪽 다 id 36개 뒤 EOS에서 멈추고 두 id 스트림이 동일하다. 시작 시 `MTP exactness probe passed: verify block is byte-identical to the single-token chain block_size=2`를 남기고, 요청은 tick 슬라이스를 통해 18라운드 16수락, `emitted_per_verify` 1.889로 서빙됐으며 `/v1/internal/mtp-policy`는 `acceptance_rate: 0.889`와 함께 `profiling`을 보고한다.

**양자화 KV.** `--kv-cache-mode int8`: 고전과 MTP가 64토큰에 걸쳐 id 동일하고 배너는 `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)`다. #1350의 MLA-latent 치환이며 5절의 슬롯 불일치가 운영자에게 보이지 않았던 이유다.

**스위트.** `cargo test --profile test-fast`로 `-p mlxcel-surgery`(160), 루트 `--lib models::glm4_moe_lite`(33), `-p mlxcel-core drafter::`(179)와 `mla::`(41), `--bin mlxcel commands::split_mtp`(2). 루트와 두 멤버 크레이트에 대한 `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`와 `cargo clippy --bins` 전부 깨끗하고 `cargo fmt --all -- --check`도 깨끗하다.

**확립되지 않은 것.** 전체 `--workspace` 게이트는 로컬에서 돌리지 않았고(다른 에이전트와 공유하는 머신이라 좁은 범위만), CI가 그것을 덮는다. 배치 `B > 1` 윈도는 설계상 고전으로 decline하며 실행해 보지 않았다. 두 은닉 탭 사이의 수락률 비교는 한 호스트의 프롬프트 하나이고 선택을 지탱하는 것은 DeepSeek-V3 관례와의 일치다. 정확성 계약은 `block_size = 2`와 프로브가 도는 폭에서 측정되었지 여러 폭에 걸쳐 측정되지는 않았다.

---

## 10. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 30 |
| 추가된 라인 | +5263 |
| 삭제된 라인 | -145 |
| 추가된 테스트 | 47 |

| 영역 | 주요 내용 |
|-----|----------|
| Surgery | `ops/split_mtp.rs`(신규, 617줄): 이름 변경, 공유 헬퍼를 통한 `kv_b_proj` 분해, 전문가 스태킹, float32 예외를 가진 bf16 패스, 라우터 예외를 가진 선택적 affine 양자화, drafter `config.json` 작성기. 인덱스만 읽는 nextn 탐지와 샤드 필터 로드. `--block-size`를 16에서 묶고, 원본을 가리키는 출력과 이미 양자화된 nextn 레이어를 거부. 작성기를 위해 `safetensors`가 dev-dependencies에서 dependencies로 이동 |
| Core | `mla/kv_b_split.rs`(신규): `glm4_moe_lite_sanitize.rs`(167줄에서 84줄로)에서 그대로 들어낸 `decompose_kv_b_proj`와 `KvBProjGeometry`. `as i32` 캐스트는 `i32::try_from`과 `checked_mul` 범위 검사로 교체. `GLM4_MOE_LITE_MTP_MODEL_TYPE`과 kind 맵 항목, `DrafterError::BinaryCrateDrafter`, 공개로 바뀐 `peek_drafter_model_type`, 그리고 하나의 본문으로 합쳐진 좁은 블록 크기 peek 둘 |
| Drafter | `glm4_moe_lite_mtp_drafter.rs`와 `_config.rs`(신규): `model.mtp_block`에서 디코더 자신의 `TransformerBlock` 위에 올린 상태 있는 drafter, 닫힌 채 실패하는 weight 인벤토리, trim 후 확장하는 수락 산술, seed 빠른 경로, 학습된 깊이에서 묶인 `runtime_block_size`. `drafter_loader.rs`(신규)가 모든 호출자가 지나는 바이너리 크레이트 진입점 |
| Target | `glm4_moe_lite_mtp_hooks.rs`와 `_target.rs`(신규): 행별 materialize 쿼리를 쓰는 absorbed MLA verify forward, 모델 소유 시퀀스 슬롯, trim 기반 롤백, 블록 대 체인 프로브, post-final-norm 은닉 탭을 가진 `MtpTarget` 어댑터. `Glm4MoeLiteModel`이 `num_nextn_predict_layers`, `forward_with_hidden`, `TransformerBlock::from_weights_with_prefix`, `KvCacheLayerModes` 표와 모드 트레이트 메서드 둘을 얻음 |
| 서버 / CLI | MTP 디스패치 지점 일곱 곳 전부와 tick 슬라이스와 speculative 벤치에 `Glm4MoeLite` 편입. `mtp_adopted_prefix_reusable`이 적응된 프롬프트 캐시 접두사를 두 실행 갈래 모두에서 decline. 오프라인 경로가 디스패치 전에 해석된 KV 모드 표를 주입. `--force` 체크포인트 마커 가드를 가진 `split-mtp` 서브커맨드 |
| 문서 | `supported-models.md` MTP 행과 계열 목록 항목(APC 비결합 포함), `mtp-policy-api.md` 계열 목록, `environment-variables.md`의 `MLXCEL_GLM_MTP_HIDDEN_TAP`과 `MLXCEL_PRINT_TOKEN_IDS` 행, README 불릿과 사용 예시 블록 |

---

## 11. 후속 조치

- `--draft-block-size`는 CLI에서 여전히 무경계인데 `split-mtp --block-size`는 16에서 묶인다. 게이트보다 경고가 어울리고 #1751이 DSpark에 같은 메모를 남겼으니 한 변경이 둘 다 답할 수 있다.
- #1763에서 해결: `MAX_BLOCK_SIZE` 거부 메시지의 14칸짜리 공백 덩어리 두 개는 줄바꿈된 문자열 리터럴이 `\` 연속 표시를 잃고 rustfmt가 줄을 합치면서 생긴 것이었다. 이제 메시지는 한 문장으로 렌더링되고, 테스트는 공백이 두 칸 이상 연속하지 않음을 단언한다.
- #1763에서 해결: `--force`는 이제 출력 디렉터리에 이전에 샤드로 나뉜 체크포인트의 `model-*.safetensors` 샤드가 남아 있으면 삭제 대신 거부하며, 샤드 하나를 이름으로 지목하고 모든 샤드와 인덱스를 그대로 둔다. 샤드가 남아 있지 않을 때는 여전히 낡은 인덱스를 지운다.
- #1763에서 해결: `split-mtp`는 이제 어떤 텐서 작업도 하기 전에 `--q-bits`를 `SUPPORTED_AFFINE_BITS`(`{2, 3, 4, 5, 6, 8}`)로 검증하여 도움말과 강제되는 집합이 일치한다. 공유 로드 시점 검증인 `validate_quantization_params`의 범위 검사(`1..=32`)는 그대로다.
- 이 계열의 배치 `B > 1` MTP는 범위 밖으로 남고, 기본값으로 `num_nextn_predict_layers`보다 깊이 draft하는 것도 그렇다.
- 은닉 탭 수락률 차이(1.753 대 1.716)는 프롬프트 하나에 기대고 있다. 이 페어링이 언젠가 벤치마크 항목을 갖게 되면 모듈 문서에서 다시 끌어오는 대신 거기서 여러 프롬프트에 걸쳐 탭을 재는 편이 낫다.
- 이슈 #1289(순서 보존 스트리밍 `qmv`)가 프로세스를 narrow에 고정하는 대신 wide 커널에서 프로브를 통과시키는 경로다. 그전까지 이 페어링은 `MLXCEL_QMV_WIDE=0` 아래에서만 바이트 동일하고 게이트는 서버 프로세스 안에서 그것을 적용하지만 단독 `mlxcel generate` 비교는 손으로 설정해야 한다.
