# PR #3283: Gemma 3, Gemma 4, AFMoE의 ring 전용 state checkpoint와 --ctx-checkpoints, --checkpoint-min-step

**날짜**: 2026-11-23
**상태**: Draft PR, 머지 전 (#3255 해결, #3259의 일부). 단위 테스트와 GB10에서 `gemma-4-12b-it-4bit` 종단 간 replay로 검증
**위험도**: Medium

## 요약

Gemma 4에서 prompt cache는 멀티턴 대화를 그대로 이어 갈 때만 적중했고, 프롬프트 끝보다 앞을 고치면 전부 미스였습니다. 1,024토큰 sliding ring이 이미 한 바퀴 돌았기 때문에 history boundary snapshot을 잘라 쓸 수 없었고, 직전 사용자 메시지를 수정하는 클라이언트는 매 턴 `cached_tokens=0`을 받았습니다. 이 경우를 위한 llama-server 플래그 `--ctx-checkpoints`와 `--checkpoint-min-step`은 0보다 큰 값이면 시작 단계에서 거부됐습니다.

이 PR은 두 플래그를 prompt cache가 환경 변수 뒤에 이미 갖고 있던 checkpoint 메커니즘(keep-previous-turn, 주기적 prefill checkpoint, 턴 경계 state checkpoint)에 연결합니다. 그리고 Gemma 3, Gemma 4(text, VL, Unified 래퍼), AFMoE에 ring 전용 state checkpoint를 추가합니다. history boundary 시점의 sliding layer ring만 저장해 두었다가, 대화의 가장 최근 whole snapshot 위에 복원하고 full-attention layer는 checkpoint 길이로 잘라 냅니다. 실제 12B 모델에서 마지막 메시지보다 한 턴 앞을 고친 요청은 이제 경계를 정확히 복원합니다(12,853 / 12,853토큰. 스위치를 켠 `main`은 12,288). issue가 제안한 패밀리 기본값(item 5)은 넣지 않았습니다. 가장 흔한 마지막 메시지 수정은 keep-previous-turn만으로 처리되고, ring checkpoint 두 개를 기본으로 두면 더 앞의 수정을 위해 대화당 약 2 GB가 더 들기 때문입니다.

## 1. 문제 정의

### 1.1 배경

issue의 기준 측정은 GB10에서 `gemma-4-12b-it-4bit`(FP16 KV, 약 5.8K토큰 프롬프트)로 했습니다. 그대로 이어지는 요청은 적중했고(5,823토큰 중 5,808 캐시, TTFT 약 4.9초에서 0.4~0.56초로), 프롬프트 끝 이전에서 갈라지면 `snapshot_diverged`로 전부 미스였습니다. 한 바퀴 돈 rotating cache의 whole snapshot은 정확히 그 길이에서만 재사용할 수 있기 때문입니다. 같은 조건에서 Qwen 3는 갈라진 지점까지 재사용했습니다.

`main`에는 복원 지점을 여러 개 두는 장치가 이미 있었지만 모두 환경 변수로만 켤 수 있었습니다. `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`(#2973)은 256 이상 간격의 절대 배수 위치에 whole snapshot을 저장하고, `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`(#3035)은 직전 턴의 복원 원본을 session chain supersede 규칙에서 제외합니다. 턴 경계 state checkpoint(#3047)는 다시 만들 수 없는 상태만 저장하고 attention row는 같은 대화의 더 긴 donor snapshot에서 가져옵니다. state checkpoint hook을 구현한 것은 Qwen 3.5, Qwen3-Next, Qwen4-exp 계열뿐이어서 Gemma에는 없었습니다.

### 1.2 기존 문제

- **플래그 거부.** `src/cli/slot_args.rs`는 0보다 큰 `--ctx-checkpoints N`과 `--checkpoint-min-step M`을 거부하면서, mlxcel이 `capture_history_boundary_snapshot`으로 시퀀스당 snapshot을 최대 하나만 만든다고 안내했습니다. 그 함수는 이미 없었고, 이 설명은 #2973과 #3047 이후 사실이 아니었습니다.
- **sliding window 계열에 checkpoint hook 없음.** 스위치를 모두 켜도 Gemma 3, Gemma 4, AFMoE는 whole snapshot에서만 복원할 수 있었습니다. Gemma 4에서 whole snapshot은 비쌉니다. sliding 토큰당 320 KiB, full-attention 토큰당 16 KiB라서 5.8K토큰에 약 431 MB입니다.
- **대화 단위 한도 없음.** state checkpoint에는 바이트 예산과 LRU 축출만 있었고, 한 대화가 몇 개를 얼마나 촘촘하게 가질지는 제한하지 않았습니다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 앞선 메시지를 고치는 에이전트·데스크톱 클라이언트가 Gemma 4에서 매번 전체 컨텍스트를 다시 prefill(14K토큰에서 약 14.5초) | High | High |
| llama-server 명령줄을 옮겨 온 운영자가 흔한 플래그 두 개 때문에 시작 단계에서 거부됨 | Medium | High |
| ring checkpoint가 잘못된 상태를 복원해 출력이 조용히 달라짐 | High | Low (정확성 테스트와 복원 검증으로 막음) |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/kv_ring_checkpoint.rs` (신규) | 공용 ring 전용 캡처와 복원. `state_checkpoint_from_snapshot`은 각 sliding layer의 rotating 텐서와 스칼라만 남기고, 모든 ring이 아직 돌지 않았으면 `None`을 돌려주며(이때는 whole snapshot을 잘라 쓸 수 있음), KV를 공유하는 layer에는 아무것도 저장하지 않습니다. `restore_sequence_state_from_checkpoint`는 sliding layer를 checkpoint에서, full-attention layer를 donor에서 가져와 `standard_truncatable_to` 이후 잘라 냅니다. 길이가 맞지 않는 ring은 거부합니다. |
| `gemma3.rs`, `gemma4.rs`, `afmoe.rs`, `vision::{mod, gemma4_vl, gemma4_unified}` | 공용 모듈을 호출하는 hook. VL과 Unified 래퍼가 그대로 전달합니다. |
| `src/server/prompt_cache/ctx_checkpoints.rs` (신규) | `CtxCheckpointFlags` 검증, 환경 변수 대비 `resolve_ctx_checkpoint_policy`, `RING_FAMILY_DEFAULT_CTX_CHECKPOINTS = None`. |
| `state_checkpoints.rs` | `enforce_conversation_checkpoint_rules`: 삽입할 때 바이트 예산 LRU보다 먼저 대화당 개수 상한과 최소 간격을 적용합니다. |
| `kv_arch.rs`, `snapshot_sizing.rs` | `sliding_window_ring_bytes_from_config`, `recommend_ring_checkpoint_capacity_from_config`. ring 패밀리 매칭에 `gemma4_unified` 추가. |
| `startup.rs`, `cli_input.rs`, `scheduler/prompt_cache.rs`, `policy.rs` | 시작 시 해석과 로그. scheduler는 프로세스 전역 환경 변수 대신 store에 해석된 interval을 읽습니다. |
| `slot_args.rs`, `cache_args.rs` | 모듈 문서와 help 문구를 새로 썼고, 낡은 `capture_history_boundary_snapshot` 문구를 지웠습니다. |
| 문서와 manifest | `docs/llama-server-compat.md`, `CONTINUOUS_BATCHING.md`, `environment-variables.md`, `CHANGELOG.md`. b10621 manifest 항목 두 개를 `not_applicable`에서 `by_design`으로 옮기고 차이점을 적었습니다. |
| 벤치마크 기록 | `docs/benchmark_results/gemma4-ctx-checkpoints-replay-gb10-2026-11-23.md`, 턴별 CSV와 cache stats CSV. |

36개 파일, +2,256 / -117 줄.

PR 이후 플래그 의미는 다음과 같습니다. `--ctx-checkpoints`를 주지 않으면 지금 동작이 그대로입니다. `N > 0`이면 keep-previous-turn과 해당 패밀리의 state checkpoint를 켜고 대화당 `N`개로 제한합니다. `N == 0`은 `state_checkpoint_capacity_bytes = 0`으로 만들고 keep-previous-turn도 끕니다. 이 경우 hybrid Qwen 계열이 기본으로 저장하던 state checkpoint도 꺼지는데, 이미 `0`을 넘기던 배포에서는 동작이 바뀌므로 info 로그를 한 번 남깁니다. 음수는 거부합니다.

`--checkpoint-min-step M`은 `M > 0`이면 주기적 prefill checkpoint 간격을 `M`으로 정하고, 한 대화의 state checkpoint 두 개가 `M`토큰보다 가깝게 남지 않게 합니다. `0 < M < 256`은 하한을 명시한 메시지로 거부하고, `0`이면 주기적 checkpoint도 간격 규칙도 없습니다. 두 플래그(그리고 clap을 거친 `LLAMA_ARG_*` 변수)는 `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`과 `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`보다 우선합니다. 바이트 예산은 여전히 `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES`가 정하고, `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`은 여전히 전부를 끄며 플래그가 주어졌으면 경고를 남깁니다. 플래그를 주지 않았을 때 llama-server 기본값(32, 8192)을 따르지는 않습니다.

## 3. 기술적 선택과 그 이유

### 3.1 ring 크기 계산이 issue의 `window x bytes`와 다름

issue는 ring checkpoint 크기를 sliding window에 sliding layer의 토큰당 바이트를 곱한 값으로 잡았고, gemma-4-12b에서 320 MiB였습니다. 첫 빌드가 이 공식을 썼더니 예산에 목표인 12개가 아니라 4개만 들어갔고, replay 동안 checkpoint 대부분이 축출됐습니다.

원인은 `RotatingKVCache::update_concat`이 여러 토큰을 한 번에 붙이는 방식에 있습니다. 이전 row 중 최신 `window - 1`개를 남기고 새 piece 전체를 이어 붙이므로, prefill piece 직후 ring에는 `window`가 아니라 `window - 1 + piece`개 row가 들어 있습니다. history boundary는 prefill piece가 끝난 시점에 캡처되니 캡처된 ring도 그 크기입니다. gemma-4-12b의 기본 prefill chunk 2,048토큰에서는 3,071 row x 327,680 bytes = 1,006,305,280 bytes(약 960 MiB)입니다. 이제 `sliding_window_ring_bytes_from_config`가 `piece_tokens`를 받아 sliding layer마다 `window - 1 + piece` row로 계산하고(`piece_tokens = 1`이면 decode 형태 ring), 예산은 이런 항목 12개를 모델 로드 후 남은 메모리의 1/8 이내로 잡습니다. 테스트 호스트에서는 12,075,663,360 bytes였습니다.

| 선택지 | 장점 | 단점 |
|---|---|---|
| `window` row로 계산 (issue) | 이론상 ring 크기와 일치 | 실제 항목을 3배 과소평가해 12개 중 4개만 들어감 |
| 캡처한 ring을 `window` row로 잘라 저장 | checkpoint당 320 MiB | 복원이 설치하는 텐서가 바뀌어 별도 정확성 증명과 재측정이 필요 |
| **선택: `window - 1 + prefill chunk` row로 계산** | 예산이 실제 저장량과 맞고 복원 경로가 그대로 | 14K토큰에서 checkpoint 하나가 whole snapshot과 거의 같은 크기 |

예산은 시작 시점의 `prefill_chunk_size`를 씁니다. 실행 중 chunk를 키우면 항목이 커져 12개보다 적게 들어가지만, 축출이 항목별 실제 `size_bytes`를 쓰므로 바이트 상한은 지켜집니다.

### 3.2 패밀리 매칭의 `gemma4_unified`

`gemma-4-12b-it`의 `model_type`은 `gemma4_unified`(text config는 `gemma4_unified_text`)입니다. ring 패밀리 매칭에는 `gemma3`, `gemma4`, `afmoe`만 있었기 때문에 실제 체크포인트에서 `--ctx-checkpoints`의 state checkpoint 예산이 0으로 해석됐고 아무 효과가 없었습니다. 단위 테스트 fixture는 `gemma4`를 써서 통과했습니다. 커밋 `2efd6a69`가 이 이름과 Unified config 크기 테스트를 추가했습니다.

### 3.3 개수 상한과 간격을 대화 lineage 전체에 적용

`--ctx-checkpoints N`은 세션이 아니라 대화 단위 개수입니다. 여기서 대화는 삽입된 checkpoint와 `SessionlessBucketKey`(모델, LoRA, 템플릿, 미디어 digest, rope 구간)와 세션이 같고, 토큰이 그것의 prefix이거나 그것이 자신의 prefix인 저장된 checkpoint들의 집합입니다. 첫 버전은 삽입된 것의 prefix인 checkpoint만 셌습니다. 요청이 순서와 다르게 끝나면 긴 checkpoint 뒤에 들어온 짧은 checkpoint는 긴 것들을 하나도 보지 못했고, 그래서 한 대화가 `N`개를 넘기거나 `M`보다 가까운 두 지점을 남길 수 있었습니다. 메모리는 바이트 예산이 여전히 막았지만 대화 단위 규칙은 지켜지지 않은 셈입니다. 구현 리뷰가 MEDIUM으로 지적했고 `6343b7aa`가 고쳤습니다.

이제 `enforce_conversation_checkpoint_rules`는 양쪽 prefix 방향으로 lineage를 모아 긴 순서로 정렬하고(길이가 같으면 digest로 정렬), 같은 줄에 남긴 더 긴 checkpoint와 `M`토큰 안에 있는 것을 버린 다음, `N`개를 넘는 가장 짧은 것들을 버립니다. 여기서 '더 새로운' 것은 나중에 삽입된 것이 아니라 경계가 더 긴 것이므로, 결과는 삽입 순서와 무관합니다. 긴 것부터 삽입해 개수 상한과 간격을 확인하는 테스트가 이를 고정합니다.

### 3.4 알려진 한계: 갈라진 branch 사이의 과잉 축출

양방향으로 모으는 데는 대가가 있습니다. 대화가 갈라진 뒤(턴 k+1을 수정해 형제 branch가 생긴 뒤) 더 짧은 공통 조상이 다시 삽입되면(예를 들어 턴 k를 regenerate) 두 branch가 모두 조상의 줄에 들어갑니다. 그러면 상한이 두 branch를 합쳐 가장 긴 `N`개를 남기고, 한쪽 branch는 checkpoint를 전부 잃을 수 있습니다. 더 많이 버릴 뿐 더 많이 남기지는 않으므로 메모리와 격리에는 영향이 없습니다. 보안 리뷰는 '한 대화를 다른 대화의 개수로 세지 않는다'는 doc comment와 어긋난다는 이유로 이를 MEDIUM으로 올렸습니다. PR은 이 동작을 `enforce_conversation_checkpoint_rules`에 문서로 적고, 개수가 상한 안에 머무는지 확인하는 테스트로 고정한 뒤, branch별 규칙은 scheduler가 소유하는 범용 checkpoint 경로인 #3254로 넘겼습니다.

### 3.5 주기적 지점을 `M`의 절대 배수에 둠

llama-server는 직전 checkpoint에서 `M`토큰 뒤에 다음 checkpoint를 둡니다. mlxcel은 `main`의 절대 배수 방식(`policy.rs`의 `checkpoint_points`)을 유지합니다. `M = 4096`이면 앞선 요청과 상관없이 4,096, 8,192, 12,288 위치에서 prefill을 나눕니다. 이력 기준 간격을 쓰면 분할 지점이 요청 도착 시점에 캐시가 갖고 있던 내용에 따라 달라지고, 적중한 요청이 cold 실행과 같은 prefill 분할을 재현하지 못합니다. #3040의 출력 동일성 비교가 바로 이 재현에 기대고 있습니다. 절대 지점은 공통 prefix를 가진 다른 대화와 checkpoint를 공유하게도 해 줍니다. replay에서 세션 B와 C는 첫 턴에 공유된 4,096토큰 prefix를 재사용했습니다(cold 약 5.2초 대비 1.7초). 이 차이는 manifest와 `docs/llama-server-compat.md`에 기록돼 있습니다.

`--checkpoint-min-step`은 서로 다른 두 규칙을 움직입니다. 주기적 간격은 `M`의 배수 위치에 whole snapshot 지점을 두고, 간격 규칙은 같은 lineage에서 `M`보다 가까운 경계 state checkpoint 두 개 중 짧은 쪽을 버립니다. `--ctx-checkpoints 3 --checkpoint-min-step 4096` 측정이 둘을 함께 보여 줍니다. target 5는 주기적 지점에서 8,192토큰을 얻었고, 대화의 턴 간격이 약 1,600토큰이라 state checkpoint는 하나만 남았습니다.

### 3.6 item 5(패밀리 기본값)를 넣지 않음

issue는 측정에서 적중이 확인되면 Gemma 3, Gemma 4, AFMoE가 기본으로 keep-previous-turn과 ring checkpoint 두 개를 갖도록 제안했습니다. `RING_FAMILY_DEFAULT_CTX_CHECKPOINTS`는 `None`으로 남았습니다. 마지막 사용자 메시지 수정(target 7)은 keep-previous-turn만으로 14,481 / 14,481토큰이 모두 처리되고, ring checkpoint 유무와 관계없이 `state_checkpoint_hits = 0`입니다. ring checkpoint가 더해 주는 것은 두 턴 이상 앞의 수정뿐인데, 이 컨텍스트 길이에서 ring checkpoint 하나(960 MiB)가 whole snapshot 하나(14K토큰에서 약 1.0 GB)와 비슷하게 들어서 기본값 두 개는 그 경우를 위해 대화당 약 2 GB를 더 씁니다.

ring을 window 크기로 잘라 캡처하면(320 MiB) 이 계산이 달라지지만, 복원의 정확성 경로를 바꾸므로 별도 동등성 증명이 필요해 후속 작업으로 남겼습니다. 구현 리뷰는 이 계열에 keep-previous-turn만이라도 기본으로 켤지도 열린 결정으로 남겼습니다. 흔한 경우를 처리하는 쪽이 item 5의 그 절반이기 때문입니다.

## 4. 측정

GB10, `gemma-4-12b-it-4bit`(48 layer 중 40개가 window 1,024의 sliding, FP16 KV), `scripts/bench_prompt_cache_replay.py --edit-target n --verify`, `gpu-lock` 아래에서 측정 조건마다 release 서버 하나, 기본 prefill chunk 2,048. 기준은 `4cd537cf`의 `main`, 변경은 `c84ae709`의 브랜치입니다. 세션 A는 열 턴이고 8번째 턴(A8)이 턴 `n`의 사용자 메시지를 다시 씁니다. '필요'는 턴 `n - 1`의 history boundary입니다.

| 조건 | A8 캐시 / 필요 | A8 TTFT (초) | ckpt inserts / hits | ckpt entries / evictions |
|---|---|---|---|---|
| main, 설정 없음, target 7 | 0 / 14481 | 14.83 | 0 / 0 | 0 / 0 |
| main, 환경 변수, target 7 | 14481 / 14481 | 1.70 | 0 / 0 | 0 / 0 |
| main, 환경 변수, target 6 | 12288 / 12853 | 3.79 | 0 / 0 | 0 / 0 |
| main, 환경 변수, target 5 | 8192 / 11307 | 7.70 | 0 / 0 | 0 / 0 |
| main, 환경 변수, target 3 | 4096 / 7924 | 11.12 | 0 / 0 | 0 / 0 |
| PR, 플래그 없음, target 7 | 0 / 14481 | 14.54 | 0 / 0 | 0 / 0 |
| PR, `--ctx-checkpoints 3`, target 7 | 14481 / 14481 | 1.74 | 16 / 0 | 10 / 6 |
| PR, `--ctx-checkpoints 3`, target 6 | 12853 / 12853 | 3.36 | 16 / 1 | 11 / 5 |
| PR, `--ctx-checkpoints 3`, target 5 | 0 / 11307 | 14.73 | 16 / 0 | 12 / 4 |
| PR, `--ctx-checkpoints 3 --checkpoint-min-step 4096`, target 5 | 8192 / 11307 | 7.49 | 16 / 0 | 4 / 12 |
| PR, `--ctx-checkpoints 8`, target 3 | 7924 / 7924 | 8.94 | 16 / 1 | 13 / 3 |

`main`의 환경 변수는 `KEEP_PREVIOUS_TURN=1`과 `CHECKPOINT_INTERVAL=4096`입니다. 플래그가 없으면 PR은 `main`과 같습니다. `N = 3`에서 target 5와 3이 미스인 것은 상한이 A5~A7만 남기고 `--ctx-checkpoints`가 주기적 지점을 켜지 않기 때문입니다. `N = 8`에서는 checkpoint 13개(11.26 GB)가 12.08 GB 예산 안에 들어갔습니다.

acceptance criterion 2를 글자 그대로 읽으면 직전 메시지 수정에서 checkpoint hit를 요구합니다. target 7에서는 keep-previous-turn whole snapshot으로 직전 경계에 도달하므로 그 조건의 hit는 0이고, ring checkpoint가 복원 지점이 되는 것은 target 6과 `N = 8`의 target 3입니다. 모든 조건에서 A8은 cold 재실행과 같았습니다. `--verify` 불일치는 전부 연쇄된 whole snapshot 적중(A2~A5, B2, B3, C2)에서 나왔고, `main`에서도 같은 자리에 나타나며, #3040이 설명하는 bf16 차이입니다. 캡처 비용은 측정되지 않을 만큼 작습니다. 수정하지 않은 턴은 `--ctx-checkpoints 3` 유무와 관계없이 TTFT가 0.1초 안에서 같았는데, checkpoint가 boundary snapshot의 버퍼를 refcount로 공유하기 때문입니다.

ring 복원의 정확성은 기본으로 ignore된 `models::gemma4_ring_checkpoint_tests`가 고정합니다. window 8이 한 바퀴 돈 4-layer fixture에서 boundary 13의 checkpoint와 18토큰 donor로 복원하면 cold 실행과 비트 단위로 같은 값을 디코드하고, 음성 대조(donor의 ring에 checkpoint의 스칼라를 얹은 것)는 logit을 바꿉니다.

## 5. 리뷰 결과

구현 리뷰는 MEDIUM 하나(lineage, `6343b7aa`에서 수정)와 LOW 하나(Gemma 3와 AFMoE 크기 계산 테스트 없음, `392b1021`에서 수정)를 찾았습니다. 바이트 예산 변수만 설정해도 이 계열에 대화 단위 상한 없는 ring checkpoint가 켜진다는 문서 LOW는 `d1661e94`에서 반영했습니다. 보안·성능 리뷰에서 CRITICAL이나 HIGH는 없었습니다. 이 리뷰는 메모리 상한(예산보다 큰 항목은 거부, 축출은 실제 크기 기준, 캡처는 버퍼 공유), 격리(대화 규칙은 sessionless bucket과 세션으로 거르고, donor 규칙은 그대로이며, 멀티모달 요청은 boundary snapshot을 캡처하지 않음), 복원 검증(패밀리, 길이, offset, window, geometry를 확인하고 모든 layer가 성공한 뒤에만 상태를 설치하며, 거부된 복원은 checkpoint를 버리고 없이 재시도), CLI 산술(saturating)을 확인했습니다. 리뷰의 MEDIUM인 fork 과잉 축출은 문서화하고 넘겼습니다(3.4). 남은 LOW 두 건, 시작 시점 chunk 기준의 크기 계산과 write lock 아래의 O(n x L) 스캔(n은 약 12)은 그대로 받아들였습니다.

## 6. 검증하지 않은 부분과 남은 일

- **replay는 lineage 수정 이전 측정입니다.** 벤치마크는 `c84ae709`에서 돌았고, `6343b7aa`가 상한과 간격이 무엇을 축출할지를 바꾼 뒤 다시 돌리지 않았습니다. entries와 evictions 열, 상한 관련 조건은 이전 규칙의 결과입니다. replay는 요청을 순서대로 끝내므로 복원 결과는 그대로일 것으로 보지만 측정하지는 않았습니다.
- **AFMoE**에는 hook과 크기 계산 테스트가 있지만 실제 체크포인트 벤치마크는 없습니다.
- **Gemma 3와 Gemma 4 VL 래퍼**는 종단 간 벤치마크가 없습니다. replay한 것은 Unified 12B text 경로뿐입니다.
- **병렬 CUDA graph 중단.** `gemma3`와 `gemma4` 테스트 필터를 전체 또는 4스레드 폭으로 함께 돌리면 `cudaStreamEndCapture ... previous error during capture`로 간헐적으로 중단됩니다. Makefile이 CUDA gate를 직렬로 돌리는 이유인 병렬 캡처 실패와 같은 현상입니다. 같은 필터 묶음이 전체 폭에서 한 번 통과했고(1,051 통과), 각 테스트는 작은 묶음이나 단독으로 통과하며, 변경 영역을 직렬로 돌린 실행은 483 통과 0 실패였습니다. 이 epic에서는 원격 CI를 취소했으므로 PR을 다루는 CI 실행은 없습니다.
- **열린 결정과 후속 작업.** sliding window 계열에 keep-previous-turn을 기본으로 켤지, item 5를 다시 열게 할 window 크기 ring 캡처, #3254와 함께 할 branch별 상한이 남았습니다. 한 번도 돌지 않은 경계의 ring은 그 경계의 whole snapshot이 밀려난 뒤에도 checkpoint를 저장하지 않고, `M`이 턴 간격보다 크면 간격 규칙 때문에 대화당 state checkpoint가 하나만 남아 `N`이 사실상 1이 됩니다. 둘 다 issue 명세대로입니다.

## 7. 학습 포인트

- **캐시 항목 크기는 자료구조의 명목 용량이 아니라 append 경로가 실제로 저장하는 양으로 계산합니다.** window `W`인 rotating cache는 여러 토큰을 붙인 직후 `W - 1 + piece` row를 갖습니다. 3배 오차는 12개가 아니라 4개만 담는 예산으로 나타났고, 축출 카운터가 있는 실제 모델 replay에서만 드러났습니다.
- **`model_type`으로 패밀리를 고르는 코드는 배포된 config로 테스트해야 합니다.** 표준 이름을 쓰는 fixture는 통과하는데, 배포된 체크포인트는 변형 이름(`gemma4_unified`)을 써서 예산 0을 조용히 받았습니다.
- **prefix tree 위의 그룹 단위 규칙은 양방향을 모두 봐야 합니다.** 조상만 세면 결과가 완료 순서에 따라 달라지고, 조상과 후손을 함께 세면 순서와 무관해지는 대신 공통 조상을 통해 형제 branch가 합쳐집니다. 두 실패는 과잉 보존과 과잉 축출을 맞바꾸는 관계이고, 상한을 절대 넘지 않는 쪽을 고르는 것이 임시로는 안전합니다.
- **기본값은 측정한 뒤에 넣습니다.** issue가 item 5를 적중 확인 조건부로 두었고, replay에서 더 싼 메커니즘(keep-previous-turn)이 흔한 경우를 이미 처리한다는 것이 드러나 비싼 기본값을 보류했습니다. 그 수치는 상수 옆에 기록돼 있습니다.

## 관련 이슈

- #3255: 이 이슈. #3259: 상위 epic.
- #3254: scheduler가 소유하는 범용 checkpoint. branch별 상한이 들어갈 곳입니다.
- #3256: paged sliding block. 이후 Gemma의 sliding layer는 checkpoint 없이 재사용합니다.
- #2973, #3035, #3047: 플래그가 연결되는 주기적 checkpoint, keep-previous-turn, state checkpoint 메커니즘.
- #3040: `--verify`의 출력 동일성 규칙.
