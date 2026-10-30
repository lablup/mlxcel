# PR #2748: NemotronNAS와 LongCat의 cache를 모델이 소유하도록 수정

**날짜**: 2026-10-30 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (두 family 모두 이전에는 engine 경로에서 쓸 수 없었으므로 동작하던 기능이 퇴행할 여지는 없다. 다만 두 모델에 지속되는 sequence 단위 상태가 추가되고, LongCat의 MoE router와 n-gram embedding 계산이 바뀐다)

## 요약

PR #2748 (issue #2544를 닫음)은 NemotronNAS와 LongCat-Flash / LongCat-Flash-Ngram이 batching을 지원하지 않는 모델의 계약을 지키도록 고친다. engine은 이런 모델의 `forward`에 빈 cache slice를 넘기므로 모델이 cache를 직접 소유하고 `make_caches()`에서 초기화해야 한다. NemotronNAS는 `Engine::prefill`을 거치면 첫 attention layer에서 index out of bounds로 panic했기 때문에 `mlxcel generate`와 `mlxcel-server` 모두 텍스트를 만들지 못했다. LongCat은 호출마다 cache와 n-gram context를 새로 만들어서, decode의 매 step이 RoPE position 0에서 자기 토큰 하나에만 attention했다. 이제 두 family 모두 상태를 모델의 `RefCell`에 둔다. LongCat이 아예 실행되게 하려면 기존 forward 버그 두 개도 함께 고쳐야 했다. 하나는 매 forward마다 프로세스를 abort시킨 rank-3 router slice이고, 다른 하나는 decode step마다 이웃 토큰 하나를 잃게 만든 n-gram context 절단이다. NemotronNAS는 GB10에서 49B 4-bit checkpoint로 end to end 검증했다. LongCat은 128 GB 호스트에 들어가는 checkpoint가 없어서 tiny config 동치성 테스트만이 gate다.

## 1. 문제 정의

### batching을 지원하지 않는 모델에 대한 engine 계약

`supports_batching() == false`인 모델은 cache vector가 비어 있는 `ModelOwned` pool entry를 받고(`src/lib/mlxcel-core/src/cache.rs`의 `SequenceCacheSet::model_owned`), `Engine::prefill`은 그 빈 slice를 `forward`에 넘긴다. 이런 모델은 cache를 `&self`에 두고 `make_caches()`가 호출될 때 초기화해야 한다. scheduler는 prefill 시점에, `DirectEngine::open_sequence`는 raw completion을 열 때 이를 호출한다(#2494, PR #2542). ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum, Laguna는 이미 이 패턴을 따른다. PR #2542의 security review에서 이를 지키지 않는 family 두 개가 발견되었다.

### NemotronNAS가 빈 slice를 indexing함

`NemotronNASModel::forward`는 attention subblock이 있는 block마다 `caches[cache_idx]`를 indexing했다. mask 코드는 `caches.first()`로 빈 slice를 견뎠지만 layer loop는 그렇지 않아서, `Engine::prefill`을 거치는 모든 prefill이 `nemotron_nas.rs:731`에서 panic했다. #2225가 `mlxcel generate`를 engine 위로 옮긴 뒤로는 server뿐 아니라 CLI도 같은 panic을 겪었다. #1688을 보면 CLI가 cache를 직접 만들어 넘기던 #2225 이전에는 이 family가 텍스트를 생성했다.

### LongCat이 호출 사이에 KV를 유지하지 않음

`LongcatFlashNgramModel::forward`는 호출마다 layer별 `[KVCache; 2]`를 새로 만들고 `ngram_context = None`으로 시작했다. decode step에서 `offset`이 0이므로 RoPE는 position 0에서 돌았고, `update_and_fetch`는 현재 토큰의 latent K/V만 돌려주었으며, n-gram hash도 현재 토큰만 보았다. 한 조각으로 끝나는 prefill은 정확했지만 2048-token prefill chunk보다 긴 prompt는 앞선 조각을 모두 잃었고, decode는 어느 경로에서도 정확한 적이 없었다. struct 하나가 `longcat_flash`와 `longcat_flash_ngram`을 모두 담당하므로 두 model type 모두 영향을 받았다.

### 첫 버그 뒤에 숨어 있던 LongCat 버그 두 개

테스트가 LongCat forward에 도달하자, 기존 테스트가 한 번도 실행하지 않았던 오래된 버그 두 개가 드러났다.

- **Router slice**: `LongcatFlashTopkRouter`는 argpartition 결과 index를 `slice(&indices, &[0, 0], &[shape[0], top_k])`로 잘랐다. router 입력은 `[B, L, hidden]`이므로 index는 rank 3이고, MLX는 `[slice] Invalid number of indices`를 던졌다. cxx bridge를 넘어가는 MLX 예외는 abort이므로 LongCat forward는 매번 프로세스를 죽였다. 입력이 rank 2였다면 같은 호출이 expert 축이 아니라 token 축을 잘랐을 것이다.
- **N-gram 절단**: `NgramEmbedding::forward`는 이전 호출에서 넘어온 tail과 새 토큰을 이어 붙인 뒤, hash를 계산하기 전에 이어 붙인 context 자체를 `n - 1` 토큰으로 잘랐다. 새 토큰이 하나인 decode step에서는 context 길이가 `n - 1`이 되고, `n - 1`만큼의 `shift_right`가 0을 반환해서, 가장 긴 n-gram의 가장 오래된 이웃이 매 step마다 사라졌다.

## 2. 기술적 선택과 그 이유

### batch 지원이 아니라 모델 소유 `RefCell` 상태

issue는 NemotronNAS를 batch 가능하게 만드는 방안을 검토하고 기각했으며, PR도 이를 따른다. `make_caches`는 attention subblock이 있는 block마다 cache 하나를 만들고(49B checkpoint 기준 49개), `num_layers()`는 `num_hidden_layers`(80)이다. `num_layers()`가 server의 KV-mode table과 paged layout을 결정하므로, `supports_batching`만 켜면 잘못된 layout을 기술하게 되고, issue 작성 시점에 개발 호스트에 checkpoint조차 없던 family에 APC와 paged decode까지 켜게 된다. PR은 `supports_batching() == false`를 유지하고, 공용 helper `fresh_caches(&layers)`로 채우는 `caches: RefCell<Vec<KVCache>>`를 추가하며, `docs/CONTINUOUS_BATCHING.md`에 batch 가능한 layout은 후속 작업이라고 적는다.

LongCat은 `RefCell` 안에 private `LongcatState { caches: Vec<[KVCache; 2]>, ngram_context: Option<UniquePtr<MlxArray>> }`를 둔다. KV cache와 n-gram tail을 한 struct에 묶어 두면 한 번의 초기화로 둘 다 지워진다. 하나만 초기화하면 새 요청이 이전 요청의 마지막 토큰을 기준으로 hash를 계산하게 된다. `forward`는 빌린 상태를 분해해서 두 필드를 기존 `internal_forward`에 넘기며, `internal_forward`의 signature는 바뀌지 않았다.

두 family 모두 `reset_runtime_state()`를 구현하고 `make_caches()`에서 이를 호출한다. `make_caches`는 engine과 scheduler가 sequence마다 호출하는 함수이고, `reset_runtime_state`는 `SequenceId` 없이 도는 단일 row 실행을 위한 trait hook이며(#2176) 위임 wrapper들이 그대로 전달한다. `make_caches`가 반환하는 vector는 이전과 같은 모양이지만 아무도 읽지 않는다.

### NemotronNAS mask는 첫 full attention slot 기준

issue는 causal mask 크기를 내부 첫 cache에서 구하자고 제안했다. PR은 그렇게 하지 않는데, 첫 slot이 항상 기록되는 것은 아니기 때문이다. `has_attention()`은 full attention block과 `replace_with_linear` block 모두에서 참이므로, linear replacement block은 한 번도 쓰지 않는 cache slot을 가진다. `caches.first()`를 쓰면 attention이 있는 첫 block이 linear replacement인 checkpoint는 매번 `live_len() == 0`을 읽는다. 첫 prefill chunk는 동작하지만 두 번째 chunk는 `(L, L + offset)` score에 `(L, L)` mask를 만들어 `broadcast_shapes (3,3) vs (1,4,3,6)`로 abort한다. 새 `first_attention_slot()`은 attention이 있는 block들 가운데 첫 `AttentionSubblock::Attention` block의 slot index를 찾아 그곳의 `live_len()`을 읽고, full attention block이 없으면 0을 쓴다. 이 수정을 아래 n-gram 수정과 함께 되돌린 mutant는 `nemotron_nas_linear_first_chunked_prefill_matches_full_forward`에서 같은 broadcast 오류로 abort한다.

### LongCat mask는 `live_len()` 기준, padded prefill 거부

`internal_forward`는 mask 크기를 `caches[0][0].offset`에서 구했다. offset은 단조 증가하지만 `update_and_fetch`는 live window를 반환하므로, PR은 NemotronNAS가 이미 쓰던 #421 방식대로 `live_len()`으로 바꾸고 빈 cache 목록도 허용한다. LongCat의 `supports_padded_prefill()`은 이제 `false`를 반환하고, NemotronNAS는 `false`를 유지하되 이유를 고쳤다. cache를 모델이 지속적으로 소유하면 pad position이 호출자의 trim이 닿지 않는 상태에 남는다(#1755). 기존 NemotronNAS 주석은 이 family에 없는 Mamba recurrent state를 이유로 들었다. NemotronNAS를 hybrid SSM family로 분류한 `scheduler_tests.rs`의 같은 오류도 함께 고쳤다.

### router는 마지막 축에 `slice_axis`로 수정

`slice_axis(&indices, -1, 0, top_k)`는 rank와 무관하게 expert 축에서 `[..., :top_k]`를 표현한다. index 쌍 세 개를 쓰는 rank 고정 `slice`로도 abort는 막을 수 있었겠지만, 원인이 된 rank 가정은 그대로 남았을 것이다.

### 저장하는 n-gram tail만 자름

이제 이어 붙인 context 전체로 hash를 계산하고, 다음 호출을 위해 `context`에 저장하는 사본만 `n - 1` 토큰으로 자른다. 이번 호출 토큰들의 hash는 첫 새 토큰 앞의 이력을 최대 `n - 1` 토큰까지 읽으므로, 이어 붙인 context는 그 이력을 모두 유지해야 한다. hash 계산 뒤 `new_ids`가 이미 마지막 `seq_len` position만 고르므로, context가 길어져도 출력 길이는 변하지 않는다.

## 3. 변경 요약

8개 파일, +770 / -49. commit 두 개(수정 `9630310f`, 중복된 `impl NemotronNASModel` block을 합친 `942761b4`)와 origin/main merge(`48071260`).

| 파일 | 변경 |
|---|---|
| `src/models/nemotron_nas.rs` | `caches: RefCell<Vec<KVCache>>`, `fresh_caches`, `first_attention_slot` 추가. `forward`는 호출자의 slice를 무시. `make_caches`와 `reset_runtime_state`가 초기화. padded prefill과 batching 주석 수정. |
| `src/models/longcat_flash_ngram.rs` | `RefCell` 안의 `LongcatState`. `forward`가 이를 사용하고 `make_caches`와 `reset_runtime_state`가 초기화. mask는 `live_len()` 기준. `supports_padded_prefill() == false`. router `slice_axis` 수정. n-gram context 절단 수정. |
| `src/models/nemotron_nas_engine_tests.rs` (신규) | tiny f32 config 테스트 4개. 세 가지 GQA 폭의 full attention, `no_op` block, `replace_with_linear` block을 섞은 순서와 linear가 맨 앞인 순서 두 가지로 구성. |
| `src/models/longcat_flash_ngram_engine_tests.rs` (신규) | 두 변형을 다루는 테스트 6개. 고정 순서로 넣은 결정적 fixture weight로 `sanitize_weights`를 거쳐 모델을 만든다(#1265). |
| `src/lib/mlxcel-core/src/engine/direct.rs`, `shared_cache_tests.rs` | doc comment의 모델 소유 family 목록에 두 family 추가. |
| `src/server/batch/scheduler_tests.rs` | doc comment에서 NemotronNAS를 recurrent state family로 부르지 않도록 수정. |
| `docs/CONTINUOUS_BATCHING.md` | class D 제외 항목이 두 family가 모델 소유 cache 한 벌을 유지한다고 설명하고, NemotronNAS layout 후속 작업을 명시. |

## 4. 트레이드오프와 잔여 항목

- **LongCat decode가 원래 치러야 할 비용을 치른다.** decode step마다 각 layer의 두 sub-layer 모두에서 cache된 모든 position에 attention하며, KV는 sub-layer마다 토큰당 `kv_lora_rank + qk_rope_head_dim`개 값씩 늘어난다. 이전에는 step마다 key 하나에만 attention했다. context가 길어질수록 tok/s가 떨어지지만, 들어가는 checkpoint가 없어서 측정하지 못했다.
- **LongCat은 실제 weight로 검증되지 않았다.** 560B checkpoint는 128 GB 호스트에 들어가지 않는다. tiny config 테스트는 incremental, chunked, engine 경로가 전체 sequence forward와 일치함을 보이지만, 실제 checkpoint에 고유한 loader나 수치 문제는 잡지 못한다.
- **NemotronNAS batching은 계속 꺼져 있다.** attention block별 cache와 `num_layers()`의 불일치는 문서화만 했고 고치지 않았다.
- **모델 소유 family 전체에 공통이며 범위 밖인 항목** (ExaOne 4, GPT-OSS, Mellum 등): `--kv-cache-mode`와 `--max-kv-size`가 내부 cache에 닿지 않고, 마지막 요청의 KV가 다음 prefill이 초기화할 때까지 메모리에 남는다. 이전 LongCat 코드는 호출이 끝나면 cache를 버렸으므로, LongCat에는 새로 생기는 메모리 상주다.
- **구조상 단일 sequence 전용이다.** `RefCell`은 sequence 하나만 담는다. `supports_batching() == false`가 scheduler를 단일 row로 묶어 두기 때문에만 안전하다. 두 family 중 하나라도 batching을 켜려면 이 상태부터 바꿔야 한다.
- **issue에서 범위 밖으로 둔 항목**: `NASMLP::forward`가 `silu` / `relu` / `gelu` 외의 activation을 모두 `silu`로 처리하는 문제는 #2423에서 다룬다.

## 5. 검증

새 테스트는 engine과 같은 방식으로 모델을 구동한다. `DirectEngine::run`을 거치는 경로와, 빈 slice로 `forward`를 부르는 경로다. 기준은 새 상태에서 돌린 전체 sequence forward(f32 weight, max abs diff 1e-3)와, 매 step 처음부터 다시 계산하는 greedy decode다.

| 테스트 | main 8385f40b | PR |
|---|---|---|
| `nemotron_nas_direct_engine_run_completes` | panic, index out of bounds | ok |
| `nemotron_nas_consecutive_direct_engine_runs_match_recompute` | 같은 panic | ok |
| `nemotron_nas_incremental_logits_match_full_forward` | 같은 panic | ok |
| `nemotron_nas_linear_first_chunked_prefill_matches_full_forward` | 같은 panic | ok |
| `longcat_flash{,_ngram}_incremental_logits_match_full_forward` | abort, `[slice] Invalid number of indices` | ok |
| `longcat_flash{,_ngram}_consecutive_direct_engine_runs_match_recompute` | 같은 abort | ok |
| `longcat_flash_ngram_chunked_engine_prefill_matches_single_piece` | 같은 abort | ok |
| `longcat_flash_refuses_padded_prefill` | assert 실패 | ok |

테스트는 수정마다 따로 실패를 잡는다. router 수정만 적용하면 cache 버그가 단독으로 드러나며, incremental과 full의 max abs diff는 5.65(flash)와 6.76(ngram)이다. n-gram 절단 수정과 linear-first mask 수정만 되돌린 mutant는 `longcat_flash_ngram_*`에서 실패하고(diff 5.14) NemotronNAS linear-first 테스트에서 abort한다.

GB10 로컬 gate(`--profile test-fast --features cuda`, 테스트 binary는 `gpu-lock` 아래에서 실행, `MLX_ENABLE_TF32=0`): `nemotron_nas` 6/6, 최종 head에서 `longcat_flash` 8/8, `hybrid_ssm_workers_never_use_paged_decode_backend`, mlxcel-core `engine::` 71/71, `cargo fmt --all -- --check`, `mlxcel`과 `mlxcel-core`의 `cargo clippy --lib --tests -- -D warnings`. Metal과 ROCm은 로컬에서 돌리지 않았다.

실제 checkpoint: GB10에서 `mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit`(`model_type: nemotron-nas`, 80 layer, 상주 26.1 GB). `--temp 0`으로 돌린 `mlxcel generate`는 Rayleigh scattering을 설명하는 일관된 96-token 답을 7.06 tok/s로 생성했다(1회 실행이며 benchmark가 아니다). `mlxcel-server`에 같은 greedy `/v1/chat/completions` 요청을 두 번 연달아 보내자 byte 단위로 동일하고 정확한 텍스트가 돌아왔으므로, server 경로에서 요청별 초기화가 동작한다. main에서는 두 경로 모두 첫 attention layer에서 panic했다.

## 6. 학습 포인트

- **빈 cache slice는 버그가 아니라 계약이다.** batching을 지원하지 않는 모델의 engine pool entry는 설계상 모델 소유이며 비어 있다. `supports_batching() == false`로 두고도 호출자의 slice를 indexing하는 모델은, CLI가 cache를 직접 만들던 시절에 동작했더라도 engine 경로에서 panic한다. ExaOne 4 패턴을 따르면 된다. 상태를 `RefCell`에 두고, `forward`에서 빌리고, `make_caches`와 `reset_runtime_state`에서 초기화한다.
- **mask 기준 cache는 실제로 기록되는 slot에서 고른다.** 일부 layer가 채우지 않는 cache slot을 가지는 구조(linear replacement, `no_op` attention, live 길이가 다른 sliding-window layer)에서는 `caches.first()`가 sequence 길이를 대신하지 못한다. 실패는 두 번째 prefill chunk에서만 나타나므로, 테스트에 첫 chunk 이후의 chunk가 있어야 한다.
- **실행 여부만이 아니라 계산 결과를 테스트한다.** LongCat의 router abort와 n-gram 절단은 어떤 테스트도 실행한 적 없는 코드에 있었다. incremental 실행과 새 상태의 전체 sequence forward 사이의 동치성 테스트가 LongCat 버그 세 개를 모두 잡았고, 수정별 mutant로 각 테스트가 자기 버그를 잡는다는 것을 확인했다.
- **MLX 예외는 bridge를 넘으면 abort다.** 이 코드베이스에서 MLX op 안의 shape 오류는 복구 가능한 Rust 오류가 아니라 프로세스 종료다. `slice` 호출의 rank 가정은 이름 붙은 축에 대한 `slice_axis`로 바꾸는 편이 낫다.
