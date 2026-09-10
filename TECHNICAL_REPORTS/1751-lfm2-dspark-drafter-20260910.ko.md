# 기술 보고서: PR #1751 - feat(speculative): LFM2 / LFM2.5 DSpark drafter on the DFlash loop

**작성일**: 2026-09-10
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 리뷰 후속 사이클
**상태**: 완료 (Apple M5 Max / Metal에서 `models/lfm2.5-2.6b-bf16`와 `models/lfm2.5-8b-a1b-bf16`를 각자의 공개 drafter와 짝지어 검증. 공용 머신이라 `--workspace` 게이트는 로컬에서 돌리지 않았고, 좁은 범위 실행과 CI가 그 자리를 대신한다)
**언어**: Rust, Markdown
**위험도**: Medium (두 번째 계열이 DFlash 라운드 루프에 합류하고, 서버의 Qwen 전용 DFlash 타깃 계약이 트레이트가 되며, 그동안 아예 로드되지 않던 체크포인트를 위해 LFM2 로더 기본값 두 개가 바뀐다)

---

## 요약

LiquidAI는 LFM2.5 계열용 "DSpark" drafter를 공개한다. DSpark drafter는 타깃의 residual stream 다섯 개를 `fc` 투영으로 받아먹는 5층 DFlash 계열 블록 병렬 트랜스포머에, 블록의 병렬 로짓을 순차 체인으로 바꿔 주는 학습된 rank-256 저랭크 토큰 전이 헤드를 붙인 물건이다. mlxcel은 이미 `lfm2`와 `lfm2_moe`를 서빙하고 있었고 DFlash 라운드 루프, drafter 트레이트, 타깃 트레이트도 이미 갖고 있었다. 그래서 이 변경은 새 서브시스템이 아니라 빠져 있던 네 조각이다: Markov 헤드, DSpark config 필드, LFM2 타깃 어댑터, 그리고 잘라낼 수 없는 상태를 위한 롤백.

마지막 것이 흥미롭다. 지금까지의 speculative 타깃은 부분 수락을 되감을 때 무언가를 짧게 잘라 왔다. 어텐션 KV 캐시는 잘린다. `offset`이 뒤로 가고 다음 append가 덮어쓴다. 짧은 컨볼루션 상태는 그럴 수 없는데, 애초에 무언가의 접두사가 아니기 때문이다. 그것은 해당 레이어 게이트 입력의 마지막 `L_cache - 1` 행이고, `bs` 행짜리 블록이 지나간 뒤에는 `bs - 2`행과 `bs - 1`행을 담고 있다. 블록을 커밋된 `n`행에서 끊는다는 것은 상태가 `concat(previous_state, bx_block[:, :n])`의 마지막 두 행을 담아야 한다는 뜻이고, 그건 손에 든 텐서의 슬라이스가 아니라 다른 텐서다. 그래서 verify forward가 임의의 접두사를 재구성하는 데 필요한 것을 기록해 두고, 롤백은 자르는 대신 다시 계산한다.

두 번째 구조 변경은 서버의 `Qwen35DFlashTarget`, 메서드가 하나뿐이던 비공개 트레이트가 메서드 넷을 가진 `DFlashTargetModel`이 된 것이다. 리팩터링은 의도적으로 보수적이다. 두 번째 계열이 필요로 한 훅마다 Qwen 3.5의 동작을 그대로 재현하는 기본 구현을 달아 두어서, 제네릭 드라이버를 지나는 Qwen 경로의 제어 흐름은 이름만 바뀐 예전 함수다.

---

## 1. 체크포인트가 실제로 말한 것과 mlxcel이 읽은 것

### 1.1 DFlash 로더가 버리던 config 필드 네 개

`DFlashConfig::from_json`에는 `deny_unknown_fields`가 없다. 그래서 DSpark `config.json`은 아무 불평 없이 파싱되면서 그것을 DSpark drafter로 만드는 모든 것을 잃었다:

```
block_size 9
dflash_config: { mask_token_id 125017, target_layer_ids [2, 9, 17, 21, 27], num_target_layers 30 }
markov_rank 256, rope_is_neox_style false, enable_confidence_head true, markov_head_type "vanilla"
```

`markov_rank`, `rope_is_neox_style`, `runtime_block_size`, `enable_confidence_head`가 이제 필드가 되었고, 기본값은 기존 Qwen 3.5 DFlash 체크포인트가 예전과 정확히 같은 값으로 파싱되도록 골랐다(`markov_rank: 0`, `rope_is_neox_style: true`, `runtime_block_size: None`, `enable_confidence_head: false`). `dflash_defaults_keep_neox_rope_and_no_markov_head`가 `json!({})`에 대해 그것을 고정한다.

`num_target_layers`는 손을 한 번 더 대야 했다. Qwen 3.5 체크포인트는 이 값을 최상위에 선언하고, DSpark 쪽은 `dflash_config` 안에 `mask_token_id`, `target_layer_ids`와 나란히 중첩한다. 뒤의 둘은 기존 lift가 이미 처리하고 있었다. 세 번째 키가 없으면 30층짜리 LFM2.5-2.6B 페어링이 조용히 Qwen 기본값 32를 읽고, 그 값을 타깃 레이어 수와 비교하는 페어링 게이트가 정상적인 짝을 거부하게 된다.

### 1.2 `block_size`가 세는 대상이 다르다

가장 오독하기 쉬운 필드다. DFlash의 `block_size`는 verify 행을 센다. 보너스 행 더하기 제안들이다. DSpark의 `block_size`는 제안만 센다. 따라서 공개된 값 9는 verify 폭 9가 아니라 gamma = 9, verify 폭 10을 뜻한다.

`DFlashConfig::verify_width()`는 `is_dspark()`일 때 `block_size + 1`을, 아니면 `block_size`를 돌려준다. `runtime_verify_width()`는 여기에 `runtime_block_size` 상한을 씌우는데, 공개 체크포인트는 이 키를 생략하므로 여덟 행 기본값(제안 7 + 앵커)으로 읽힌다. `--draft-block-size 10`이 학습된 폭을 복원한다.

CLI 쪽 결과는, 평평한 `DEFAULT_DFLASH_BLOCK_SIZE` 16이 DSpark drafter에는 두 겹으로 틀린다는 것이다. 제안 9개짜리 헤드에 제안 15개를 요구하게 된다. `resolve_draft_block_size`에 세 번째 peek인 `peek_dspark_configured_block_size`가 붙었고, 이것은 DSpark drafter가 아닌 모든 것에 `None`을 돌려주므로 Qwen 3.5 DFlash 체크포인트는 예전과 똑같이 평평한 16을 유지한다.

### 1.3 탐지는 이름이 아니라 헤드를 본다

`is_dspark()`는 `architectures` 일치가 아니라 `markov_rank > 0`이다. draft 스텝을 바꾸는 것은 헤드의 존재이고 그 크기를 정하는 것은 rank이므로, 판정하는 필드가 곧 실제로 중요한 필드다. `architectures: ["Lfm2DSparkDraftModel"]`은 여전히 `is_dflash_drafter_config`가 받아들이는데, 이쪽은 "이 디렉터리가 단독 모델이 아니라 drafter인가"라는 별개의 질문이다. 그 검사 덕분에 `mlxcel generate -m <dspark-dir>`가 `Weight not found: model.embed_tokens.weight` 대신 설명을 내고 실패한다.

---

## 2. Draft 스텝

### 2.1 모든 위치가 제안이다

평범한 DFlash draft는 `block_size` 행짜리 `[bonus, mask, ..., mask]`를 만들고 0번 위치를 버린다. 그 행은 타깃이 이미 결정한 것을 다시 말하는 로짓을 가진 발판이다. DSpark draft는 gamma 행짜리 `[anchor, mask * (gamma - 1)]`을 만들고 전부 유지한다. DSpark 레이아웃에서는 앵커 행이 이미 커밋된 컨텍스트보다 한 칸 앞에 앉아 있어서, 0번 위치의 로짓은 마지막 토큰의 재진술이 아니라 다음 토큰에 대한 제안이기 때문이다.

이걸 틀리면 크래시가 나지 않는다. 라운드마다 제안 하나를 조용히 잃고 나머지를 전부 밀어내며, 그 결과는 버그가 아니라 낮은 수락률처럼 읽힌다. 그래서 `dspark_draft_block_uses_all_positions`가 위치별로 표시된 합성 로짓으로 이것을 직접 단언한다.

### 2.2 Markov 체인

5층 forward는 블록에 대해 비인과적이라, `i + 1`번 위치는 `i`번 위치가 무엇이 될지 보지 못한 채 점수를 받는다. 헤드가 그것을 순차적으로 메운다:

```
prev = anchor
for step in 0..gamma:
    step_logits = base_logits[:, step] + markov_w2(markov_w1[prev])
    prev = argmax(step_logits)
    draft[step] = prev
```

`markov_w1`은 `[vocab, rank]` 임베딩 테이블이고 `markov_w2`는 `rank -> vocab` 투영이라, 한 스텝은 gather 한 번과 `[1, 256] x [256, 128000]` 행렬곱 한 번이다. 둘 다 `UnifiedEmbedding` / `UnifiedLinear`로 로드되므로 양자화된 drafter의 `.scales` / `.biases` 형제 텐서도 두 번째 코드 경로 없이 잡힌다.

체인은 하나의 지연 MLX 그래프로 만들어져 한 번만 materialize된다. 각 스텝의 argmax는 디바이스에 머물며 다음 스텝의 gather로 바로 들어가므로, 한 라운드가 gamma번이 아니라 한 번의 호스트 동기화만 쓴다. 순차 체인이라면 스텝마다 왕복해야 한다고 읽는 것이 자연스럽기 때문에 이 점은 명시할 값어치가 있다.

`markov_chain_feeds_previous_token_into_next_step`이 자기 출력을 무시하는 체인을 잡아내는 테스트다. 평평한 base 로짓과 `prev + 1`에 스파이크를 주는 `w2`에서 `anchor+1, anchor+2, ...`를 뱉는 유일한 방법은 argmax를 실제로 앞으로 먹이는 것뿐이다.

### 2.3 RoPE 짝짓기

`rope_is_neox_style: false`는 체크포인트가 차원 `i`를 `i + 1`과 회전시킨다는(GPT-J 방식) 뜻이고, 이는 MLX의 `traditional = true`다. `DFlashAttention`은 세 군데 `fast_rope` 호출에서 `false`를 하드코딩하고 있었다. 이제 `rope_traditional: !config.rope_is_neox_style`을 들고 다니며, 테스트는 구조체 필드가 아니라 그 선택이 출력에 도달하는지를 단언한다. 같은 입력을 두 짝짓기로 통과시켜 어텐션 출력이 달라야 한다고 요구하는 방식이다.

---

## 3. 자를 수 없는 롤백

### 3.1 짧은 컨볼루션 상태가 접두사가 아닌 이유

`Lfm2LayerCache::Conv`는 `[1, L_cache - 1, hidden]`, 즉 패딩된 컨볼루션 입력의 마지막 두 행(공개 체크포인트는 전부 `L_cache = 3`)을 담는다. `ShortConv::forward`는 그것을 덮어쓰고 아무것도 남기지 않았고, 유일하게 존재하던 trim 훅은 상태를 `None`으로 리셋했다. 캐시 축출에는 맞고 부분 수락에는 틀린 동작이다. 시퀀스 중간에서 순환을 0에서 다시 시작시켜 버린다.

커밋된 행이 `n = accepted + 1`일 때 다음 라운드가 필요로 하는 상태는 `concat(previous_state_or_zeros, bx_block[:, :n])`의 마지막 `L_cache - 1` 행이다. 두 피연산자 모두 평범한 forward를 넘기지 못하므로 `forward_with_capture`가 둘 다 기록한다:

```rust
snapshots.push(ConvRollbackSnapshot {
    layer_idx,
    prev_state: conv_state.as_ref().map(|s| mlxcel_core::copy(s)),
    bx_block: mlxcel_core::contiguous(&bx, false),
});
```

캡처는 `bx`가 계산된 뒤, `*conv_state`가 대입되기 전에 놓여 있고, `mlxcel_core::copy`는 핸들 별칭이 아니라 진짜 복사 노드인 `mlx::core::copy`다. 별칭 핸들이었다면 어떤 shape 검사로도 잡히지 않는 방식으로 미묘하게 틀렸을 것이다. 롤백이 방금 블록이 만들어 낸 상태로부터 상태를 재구성하게 되는데, 그것이 바로 되돌리려는 값이기 때문이다.

이전 상태가 `None`일 때의 0 채움은 `bx` 자신의 dtype으로 `[bx.shape[0], L_cache - 1, bx.shape[2]]`이고, 이는 `conv_padding()`이 새 인과 forward를 왼쪽 패딩하는 양 `(l_cache - 1, 0)`과 일치한다.

### 3.2 어텐션 쪽, 그리고 완전 수락일 때

어텐션 레이어는 `KVCache::trim(block_size - n)`을 쓴다. `offset`이 뒤로 가고 버퍼는 그대로 두며 다음 append가 덮어쓴다. 완전 수락은 이 중 아무것도 부르지 않는다. verify forward가 남긴 그대로가 이미 커밋된 상태다.

### 3.3 테스트가 비교하는 것

`conv_rollback_matches_committed_prefix`는 shape를 단언하지 않는다. 토큰 5개를 prefill하고 4행 블록을 verify한 뒤 `accepted = 1`에서 롤백하고, `prompt ++ verify[:2]`를 한 번에 소비한 레퍼런스 모델과 비교한다. 어텐션 offset(7), 컨볼루션 상태(`allclose` 1e-5), 그리고 양쪽에서의 다음 디코드 스텝 로짓(`allclose` 1e-4)이다. 마지막 항목이 이 테스트를 값어치 있게 만든다. 컨볼루션 상태는 상태 비교를 통과할 만큼 가깝고도 토큰 하나를 움직일 수 있기 때문이다.

`conv_rollback_from_fresh_caches_pads_with_zeros`가 `prev_state: None` 분기를 같은 방식으로, 1토큰 prefill을 상대로 덮는다.

---

## 4. 은닉 상태 캡처와 첫 draft

drafter의 `fc`는 `len(target_layer_ids) * hidden_size` 개 특징을 읽는다. 따라서 타깃은 캡처된 레이어마다 `[1, bs, 2048]` 슬랩 하나씩을, 해당 레이어 다음이자 마지막 `embedding_norm` 전에 떠서, 특징 축으로 이어 붙인 `[1, bs, 10240]`으로 돌려줘야 한다. `forward_speculative`는 캡처와 스냅샷 싱크를 꿴 `forward_embeds_with_caches`다. 첫 어텐션 레이어의 offset에 고정된 같은 인과 마스크, 같은 왼쪽 패딩 컨볼루션, 같은 MoE 라우팅이다. 그 동일성이 정확성의 전제이고, 6절은 커널이 그것을 지키는지에 관한 것이다.

두 계열 사이의 정책 차이 하나는 이름을 붙일 만큼 중요했다. Qwen 3.5 DFlash 경로는 prefill이 캡처한 은닉을 마지막 프롬프트 위치로 슬라이스한다. 그쪽 drafter는 컨텍스트 캐시를 보너스 토큰부터 시작하기 때문이다. DSpark drafter는 건네받은 모든 행을 자기 append 전용 컨텍스트 캐시에 붙이며, 첫 제안 전에 프롬프트 전체를 원한다. 대신 한 행만 먹여도 어디에서도 shape 오류가 나지 않는다. 그저 drafter가 한 행짜리 컨텍스트를 보게 되어 수락률이 떨어질 뿐이다. `FirstHiddenRows`가 그 선택을 명시적으로 만들고 `first_hidden_rows_defaults_to_the_qwen_policy`가 두 계열의 답을 고정하므로, 재정의를 잊은 세 번째 계열은 벤치마크가 아니라 테스트에서 잡힌다.

---

## 5. `DFlashTargetModel`

`DFlashGenerator`는 이미 임의의 `SpeculativeTarget`을 받는다. 여전히 Qwen 모양이던 것은 버스트의 서버 쪽이었다. 타깃의 이종 캐시 벡터를 할당하고, 계열별 `VerifyOut`에서 로짓과 은닉 슬랩을 읽고, 첫 draft의 행 정책을 정하는 일이다. `Qwen35DFlashTarget`은 첫 번째를 메서드로 들고 나머지 둘을 바운드에 이름만 적어 두고 있었다.

대체물은 넷을 들고 다닌다. `make_dflash_caches`, `first_hidden_rows`(정책이 로드된 인스턴스가 아니라 drafter 계열에 속하므로 연관 함수다), `enable_speculative_buffers`(다중 행 verify를 위해 캐시를 준비해야 하는 타깃용 no-op 훅), `exactness_allows`다. 동반 트레이트 `DFlashVerifyOutput`이 어느 계열의 출력 타입도 이름 붙이지 않고 드라이버에 로짓과 은닉 슬랩을 준다.

살아 있는 코드 위에서 이 리팩터링을 안전하게 만든 성질이 셋이다.

- 추가된 훅마다 기본값이 Qwen 3.5의 동작이다. `first_hidden_rows`는 `LastPromptPosition`, `enable_speculative_buffers`는 no-op, `exactness_allows`는 `true`, `Drafter::validate_target_compat`는 이미 no-op 기본이었다. `Qwen35Model`은 `make_dflash_caches`만 재정의한다.
- 인라인 코드를 대체한 헬퍼 둘은 비슷한 것이 아니라 동등하다. `concat_captured_hidden`은 예전의 copy 후 fold 루프를 `concatenate_many(refs, -1)`로 쓴 것이고, `first_hidden_for(LastPromptPosition, ..)`은 예전 슬라이스 그대로다.
- 새 트레이트 메서드를 가리는 고유 메서드가 없다. `Qwen35Model`이 가진 것은 `exactness_allows`가 아니라 `mtp_exactness_allows`이므로, 버스트 게이트의 `m.exactness_allows(bs)`는 관대한 트레이트 기본값으로 해석되고 Qwen DFlash 경로는 한 번도 걸지 않던 프로브에 걸리기 시작하지 않는다.

트레이트가 의도적으로 들고 있지 않은 하나는 라운드별 블록 크기 재정의다. `DFlashGenerator`는 실행 전체에 대해 `block_size` 하나를 들고 있으므로, 여기에 훅을 두면 지원되는 것처럼 읽히면서 아무 일도 하지 않는다.

---

## 6. 정확성 게이트와 호스트가 한 말

온도 0 계약은 방출된 스트림이 고전 greedy 디코드와 동일하다는 것이다. 이는 모든 verify 행의 로짓이 그 위치에서의 단일 토큰 디코드 스텝이 낼 값과 같을 때에만 성립하는데, 양자화 체크포인트에서 `M = bs` 행렬곱과 `M = bs` MoE 디스패치는 `M = 1`과 비트 단위로 같다고 보장되지 않는다. 특히 LFM2의 MoE 블록은 `x_flat.shape[0] == 1`일 때에만 융합 단일 행 전문가 커널을 타므로, verify 블록은 디코드 스텝이 타는 경로를 결코 타지 않는다.

전제를 단언하는 대신, 이 변경은 MTP 갈래가 쓰는 블록 대 체인 프로브를 재사용한다. 합성 draw 세 번, 같은 prefill 상태에서 `bs` 토큰 블록과 같은 토큰을 하나씩 디코드한 것을 비트 단위로 비교한다. `mtp_exactness_gate`가 메모이제이션, decline 로그 줄, `qmv_wide` 재시도, 그리고 의미가 바뀌지 않은 `MLXCEL_MTP_ALLOW_INEXACT`를 소유한다. 버스트는 decline 시 자기 이름이 붙은 경고를 하나 더 낸다. 공용 게이트의 판정 줄은 "MTP declined"라고 말하는데, DSpark drafter로 `--draft-model`을 설정한 운영자가 그것을 자기 요청이 고전 디코드로 떨어진 사실과 연결할 이유가 없기 때문이다.

**검증 호스트(Apple M5 Max, GPU 세대 17)에서 프로브는 두 쌍 모두에 대해 decline한다.** 이는 문서화된 세대 15 이상 조건이고 출하 기본값이다. 버스트는 고전 디코드로 되돌아가 베이스라인의 토큰을 서빙한다. 그래서 아래 측정은 `MLXCEL_MTP_ALLOW_INEXACT=1` 아래에서 돌았고, 그 결과는 게이트가 막으려던 바로 그것이기 때문에 정확히 기록할 값어치가 있다.

| 쌍 | 폭 | 평균 수락 길이 | tok/s | 고전 tok/s | 베이스라인 대비 토큰 id |
|----|----|---------------|-------|-----------|----------------------|
| Dense 2.6B | 8 | 3.59 | 203.3 | 72.3 | 256 / 256 |
| Dense 2.6B | 10 | 3.75 | 212.1 | 72.3 | 256 / 256 |
| MoE 8B-A1B | 8 | 2.61 | 115.5 | 104.3 | 153 / 256 |
| MoE 8B-A1B | 10 | 2.88 | - | 104.3 | 153 / 256 |

153번 위치의 MoE 갈라짐은 두 갈래가 다르게 푸는 동점이다. 베이스라인은 logprob -1.0312의 `':'`를, 블록 경로는 -1.0000의 `' formula'`를 고르며, 더 짧은 프롬프트에서는 두 후보가 모두 -1.4375로 읽힌다. 두 폭과 여러 실행에 걸쳐 같은 위치에서 재현되는 반면, 융합 갈래와 비융합 고전 갈래는 서로 일치한다. 그것이 바로 프로브가 decline한 비트 동일성이며, decline을 무시한 뒤에 관찰된 것이고, 그 무시가 기본값이 아닌 이유다.

Greedy 전용 강제는 별개의 게이트이고 이 호스트에서 작동한다. dense 쌍에 대한 `"temperature": 0.7`은 `timings` 없이 고전 디코드로 서빙되며 정확히 한 줄, `DFlash speculative dispatch declined for seq seq-2: the drafter is greedy-only (DSpark) and the request samples with temperature 0.7 / top_k 50; serving it with classic decode`를 남긴다. decline은 drafter가 상주한 뒤, 꺼내지기 전에 결정되므로 drafter는 다음 greedy 요청을 위해 슬롯에 남는다.

---

## 7. 함께 딸려 온 LFM2 로더 수정 둘

둘 다 scope creep이 아니라 전제 조건이다. bf16 원본이 이것들 없이는 로드되지 않거나 올바로 로드되지 않고, 공개 drafter가 짝짓는 대상이 바로 그 bf16 원본이다.

**`rope_parameters` 아래의 `rope_theta`.** LiquidAI 원본은 transformers 5.x 익스포트라 최상위 `rope_theta`를 갖지 않는다. 값은 `rope_parameters` 블록 안에 있다(LFM2.5-2.6B는 1e7, 8B-A1B는 5e6). 예전 필드의 `#[serde(default)]`가 1e6이었으므로 그 체크포인트들은 모든 어텐션 레이어를 틀린 base로 돌리며 그럴듯해 보이는 틀린 출력을 냈다. `ModelArgs::rope_theta()`는 이제 최상위 키를 먼저 읽고(mlx-community 변환본이 그것을 유지하며, 둘 다 있으면 이기는 쪽이다), 다음으로 중첩된 것을, 마지막으로 첫 릴리스 기본값 1e6을 읽는다. 세 분기 모두 테스트된다.

**쌓이지 않은 전문가별 MoE 텐서.** MoE 원본은 `feed_forward.experts.{e}.w1/w2/w3`를 싣는다. 이름 변경 단계가 dense 형태인 `.feed_forward.wN.`만 잡았으므로 전문가 텐서는 `wN` 이름을 유지했고, `experts.{e}.gate_proj`를 찾는 스태킹 탐색이 한 번도 발화하지 않아 로드가 `switch_mlp.gate_proj`에서 실패했다. 이제 이름 변경이 두 형태를 모두 덮는다. mlx-community 4비트 변환본은 미리 쌓인 `switch_mlp.*` 텐서를 싣고 두 형태 어느 쪽과도 맞지 않으므로 건드려지지 않는다.

---

## 8. 리뷰 발견 사항

리뷰는 CRITICAL이나 HIGH를 찾지 못했다. 이 보고서 외에 브랜치에서 바뀐 것은 없다. 아래는 다시 발견되지 않도록 기록해 둔다.

**LFM2 타깃 위의 DSpark 아닌 잘못된 페어링에는 게이트가 없다.** `validate_target_compat`는 drafter가 DSpark가 아니면 즉시 `Ok`를 돌려주고, `DFlashDraftModel::forward`는 자기 `fc` 입력 폭을 검사하지 않는다. 그래서 진짜 Qwen 3.5 DFlash drafter와 짝지어진 LFM2 타깃은 `Result`가 아닌 cxx 심을 통과하는 MLX shape 예외에 도달한다. 이 위험은 종류로는 기존에도 있었지만(4B Qwen 타깃과 27B DFlash drafter가 같은 방식으로 실패한다), LFM2 타깃은 예전에 variant 게이트에서 decline되었고 이제는 그렇지 않다. 좁은 수정은 타깃이 LFM2이고 drafter가 DSpark가 아닐 때 버스트 게이트에서 decline하는 것이고, 넓은 수정은 모든 DFlash drafter에 `fc` 폭 검사를 거는 것인데 후자는 어떤 Qwen 페어링이 받아들여지는지를 바꾸므로 리뷰 편집이 아니라 메인테이너의 판단 사항이다.

**세 번째 계열은 match 팔 하나보다 많이 건드린다.** 모듈 문서는 "`impl` 블록 하나 더하기 버스트 게이트의 match 팔 하나"라고 말한다. 실제 개수는 넷이다. 정확성 게이트, `drive!` 디스패치, LFM2의 B = 1 decline을 하드코딩한 배치 게이트, 그리고 `model_variant_label`이다. 트레이트에 `supports_batched()` 메서드를 두면 셋째를 제자리로 옮길 수 있고, 다음 계열이 들어올 때 함께 하면 좋다.

**프로브는 폭 하나만 측정한다.** `dflash_exactness_allows(block_size)`는 설정된 verify 폭에서 프로브하지만, 라운드 루프는 토큰 예산 끝에서 `bs`를 좁힌다(`bs = block_size_cfg.min(remaining_plus_one)`). 그리고 이 프로젝트의 벤치마크 지침 자체가 블록 대 체인 불일치가 폭에 따라 크게 달라진다고 적고 있다. MTP 갈래도 같은 모양이므로 회귀는 아니지만, 계약은 엄밀히 말해 넓은 폭에서만 측정되어 있다.

**`ProbeKey`에 계열 구분자가 없다.** 키는 `{block_size, hidden_size, num_hidden_layers}`이고, 문서 주석은 한 프로세스가 타깃 모델 하나를 서빙한다고 가정한다. 라우터 모드는 그렇지 않다. 기존 사안이며, 세 번째 계열이 같은 메모에 도달하면서 넓어졌다.

**작동하지 않는 블록 크기 훅.** `configured_block_size()`와 `prefer_requested_block_size()`는 이슈가 요구한 대로 `DFlashDrafter`에 구현되었지만, 읽는 쪽은 MTP 생성기뿐이다. DFlash 라운드 루프는 생성 시 받은 `block_size`를 그대로 쓴다. DSpark의 실효 폭은 `resolve_draft_block_size`에서 온다. drafter가 낮은 수락률에서 물러서지 않는다는 요구는 성립하지만, 공허하게 성립한다.

**벤치마크 수치는 PR 본문에 남았다.** 이슈는 두 폭의 tok/s를 `docs/benchmark_results/`에 기록하라고 했다. PR은 대신 본문에 기록했고, 호스트가 다른 작업을 함께 돌았으며 각 처리량이 표본 하나임을 감안하면 단일 표본 수치를 벤치마크 코퍼스 밖에 두는 것이 이 프로젝트의 벤치마크 규율상 옳은 판단이다. 그 이탈을 여기 적어 두는 것이 요점이다.

---

## 9. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 27 |
| 추가된 라인 | +3080 |
| 삭제된 라인 | -577 |
| 추가된 테스트 | 25 |

| 영역 | 주요 내용 |
|-----|----------|
| Drafter 코어 | `markov.rs`(신규), DSpark config 필드와 `verify_width` / `runtime_verify_width`, `DFlashAttention`에 꿴 RoPE 짝짓기, DSpark draft 스텝과 페어링 게이트, `DrafterError::GreedyOnly`, `Drafter::greedy_only` |
| LFM2 타깃 | `lfm2_speculative.rs`(신규): 캡처를 동반한 verify forward, 컨볼루션 롤백, 정확성 프로브, `SpeculativeTarget`. `rope_parameters`와 전문가별 MoE 이름 변경 수정 |
| 서버 | `dflash_target.rs`(신규): `DFlashTargetModel`, `DFlashVerifyOutput`, `FirstHiddenRows`, 제네릭 드라이버 둘. 버스트 게이트를 LFM2 세 변종으로 확장하되 B = 1로 제한 |
| CLI | `resolve_draft_block_size`의 DSpark 블록 크기 peek. 오프라인 거부 메시지가 두 drafter 형태를 모두 지칭 |
| 문서 | `supported-models.md` DSpark 행, `speculative-acceptance.md`의 greedy 전용 decline, README |

리뷰 시점에 로컬 검증: `cargo check --lib --tests`, `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, 그리고 좁은 테스트 범위 `-p mlxcel-core drafter::dflash`(69), `--lib models::lfm2`(29), `--lib server::batch::speculative_burst`(67), `--lib server::batch::dflash_target`(2), `--lib cli::speculative_args`(22), `--lib models::detection`(60). 전부 통과.

---

## 10. 후속 조치

- 이슈 #1343(Muse Glimmer assistant drafter)이 다음 `DFlashTargetModel` 구현자다. 회전 캐시를 위한 `enable_speculative_buffers`가 필요한데 이미 존재하고 두 드라이버가 호출한다. 회전 캐시용 `rollback_partial`도 필요한데 이미 계열별이다. 그 작업을 하면서 `supports_batched()`를 함께 추가할 것.
- 이슈 #1289(순서 보존 스트리밍 qmv)가 오늘 decline하는 커널에서 프로브를 통과시키는 경로다. 그전까지 DSpark 버스트는 세대 15 이상에서 측정된 속도 향상이 아니라 측정된 decline이다.
- 배치(B > 1) DSpark와 샘플링 수락 규칙은 명시적으로 범위 밖이며 계속 그렇다.
- confidence 헤드는 로드되고 호출되지 않는다. 그 위에 조기 종료 정책은 만들어져 있지 않다.
