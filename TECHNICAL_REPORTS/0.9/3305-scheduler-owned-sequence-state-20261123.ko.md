# PR #3305: 레이어별 시퀀스 상태를 모델에서 스케줄러로 이동

**작성일**: 2026-11-23
**상태**: 열린 PR, 미병합 (#3254 해결, 에픽 #3259의 일부). clippy, 변경 영역 lib 테스트, core 테스트, 통합 계약 테스트, engine parity, GB10 디코드 벤치마크로 검증
**언어**: Rust, Python (벤치마크 스크립트), Markdown
**위험도**: 높음

## 요약

Gemma 3와 Qwen 3.5 텍스트 모델은 더 이상 시퀀스별 캐시를 모델 래퍼 안에 두지 않는다. 각 시퀀스의 상태는 `CachePool` 엔트리가 소유하는 `Vec<LayerState>`이고, 엔진은 새로 추가된 `LanguageModel::forward_layer_state`로 이 상태를 넘겨 forward를 실행한다. 스냅샷, 복원, 절단, 턴 경계 체크포인트, 링 체크포인트, 패드 트림, 디코드 되감기는 패밀리마다 따로 구현하지 않고 한 곳(`server::prompt_cache::state_snapshot`과 `mlxcel_core::cache::layer_state`)에서 한 번만 구현한다. ADR 0010에 설계와 모든 model-owned 패밀리의 감사 결과를 기록했다. 저장된 스냅샷은 텐서 이름, shape, 스칼라를 그대로 유지하고, 실제 체크포인트에서 응답과 `/v1/cache/stats`가 `main`과 같으며, 측정한 모든 셀에서 디코드 처리량 회귀가 없다. 나머지 model-owned 패밀리는 기존 경로를 유지하며 #3259 아래 여섯 개의 후속 이슈로 나뉜다.

## 1. 문제 정의

### 1.1 배경

프롬프트 캐시 재사용은 스케줄러가 소유하는 dense/paged KV에 대해 한 번 구현되어 있었다(radix-trie 저장소, `adopt` / `adopt_paged`, `DetachedCacheSet::truncate_to`). 레이어마다 저장 형태가 다른 패밀리(full 레이어 옆의 슬라이딩 링, attention 옆의 recurrent 상태)는 캐시를 모델 래퍼 안에 시퀀스 id로 키를 잡아 보관했다. 트레이트가 forward에 `&mut [KVCache]`를 넘기는데, 이런 동질적인 슬라이스로는 `RotatingKVCache`나 GatedDeltaNet 상태를 전달할 수 없기 때문이다. 이 패밀리들은 `SequenceStateLayout::model_owned`를 반환하고, 스케줄러는 이들에 대해 일반 KV 엔트리 재사용을 건너뛰며 `CachePool::detach`는 이들을 거부한다.

### 1.2 기존 문제점

- **모든 기능을 패밀리마다 한 번씩 작성.** 정확한 prefix 복원, 절단(#1145), 턴 경계 체크포인트(#3047), 링 체크포인트(#3255), 패드 트림(#1755), 디코드 되감기(#2159)가 각각 패밀리별 훅을 가졌다. 스냅샷 훅이 없는 패밀리는 prefix 재사용이 전혀 없었다.
- **다음 에픽 항목이 이를 다시 곱한다.** paged 슬라이딩 윈도우(#3256), `--swa-full`(#3257), 양자화 사이드카(#3250)는 모두 시퀀스 상태를 건드리므로 model-owned 패밀리 수만큼 구현이 필요했다.
- **상태의 소유자가 둘.** 이 시퀀스들에도 풀 엔트리는 있었지만 길이 미러 역할만 했다. 실제 상태는 모델의 시퀀스별 맵에 있었고 스케줄러는 모델 훅을 통해서만 접근했다.
- **대화별 체크포인트 상한이 브랜치를 구분하지 않음** (#3283 리뷰에서 이월). `--ctx-checkpoints N`이 다시 삽입된 공통 조상 아래의 모든 브랜치를 합쳐 세었기 때문에, 포크 후 이전 턴을 재생성하면 형제 브랜치의 체크포인트가 축출될 수 있었다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 모델과 맞지 않는 상태(레이어 수 불일치, K/V 불일치, 행보다 큰 offset)가 복원되어 출력이 조용히 어긋남 | 높음 | 낮음 (모든 복원이 설치 전에 검증하고, 불일치 사례를 테스트로 다룸) |
| 풀을 우회한 호출자가 여러 시퀀스를 공유 fallback 슬롯 하나에서 실행 | 높음 | 낮음 (마이그레이션된 래퍼의 by-id 진입점은 `forward_layer_state`를 명시하며 panic) |
| 변경 전에 저장된 스냅샷이 복원되지 않음 | 중간 | 낮음 (generic 경로가 기존 인코더를 사용하고, 텐서 단위로 비교하는 테스트 존재) |
| 추가된 간접 참조로 디코드가 느려짐 | 중간 | 낮음 (회귀 미관측, 부록 B 참고) |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core/src/cache/layer_state.rs` (신규) | `LayerStateKind { Full, Sliding { window }, Chunked { chunk }, Recurrent, None }`가 `models::model_owned::LayerKv`를 대체. `LayerState { Kv, Rotating, Chunked, Recurrent, Empty }`(Gemma 3의 `Cache` enum을 core로 승격)는 내부 캐시로 match하여 `KvAttention`을 구현하고, `rewind_padded_prefill`, `rewind_decode_appends`, `visible_len`을 제공. `RecurrentState`는 이름 붙은 텐서의 순서 있는 맵과 offset. |
| `mlxcel-core/src/cache/layer_state_request.rs` (신규), `generate.rs`, `language_model_ref.rs` | 요청 타입 `LayerStateForward`, 트레이트 메서드 `layer_state_kinds()`, `make_layer_state()`, `layer_state_snapshot_spec()`, `forward_layer_state(request, rows)`. 기본 구현은 마이그레이션되지 않은 패밀리를 기존 경로에 둔다. |
| `mlxcel-core/src/cache.rs` | `SequenceStateBackend::SchedulerOwned`. `SequenceCacheSet`에 `layer_state: Option<Vec<LayerState>>` 추가, paged decode override로 엔트리가 `PagedKvCache`로 할당될 때도 유지. `layer_state_nbytes()`는 `nbytes()`와 별도로 보고. |
| `mlxcel-core/src/engine/{mod, prefill, direct, direct_decode, lookahead, speculative}.rs` | `step`, `prefill`, `score`, `prefill_cohort`, `verify`가 시퀀스에 layer state가 있으면 `forward_layer_state`로 분기하고, 패드 트림, 디코드 되감기, paged 길이 미러를 레이어에 직접 수행. |
| `src/server/prompt_cache/state_snapshot.rs`, `state_snapshot_dispatch.rs` (신규) | `[LayerState]` 위의 전체 스냅샷, 복원, 절단 복원, 절단 가능 여부, 상태 체크포인트, 체크포인트 복원의 단일 구현. dispatch는 scheduler-owned 패밀리에 generic 경로를, 나머지에 모델 훅을 사용. |
| `src/models/hybrid_snapshot.rs` (신규) | `Qwen3NextCache`에서 hybrid(attention + GatedDeltaNet) 인코더를 추출해 Qwen3-Next와 generic Qwen 3.5 경로가 공유. |
| `src/models/gemma3.rs` | forward를 model-owned `Cache`(AFMoE, Kolibri-1, 파이프라인 스테이지, backbone)와 `LayerState`가 모두 구현하는 작은 트레이트에 대해 제네릭으로 변경. 시퀀스별 맵과 링 체크포인트 훅 삭제. bare `forward` 호출자용 id 없는 fallback 슬롯 하나만 유지. |
| `src/models/qwen3_5.rs`, `qwen3_5_layer_state.rs` (신규), `qwen3_5_mtp_target.rs` | Qwen 3.5는 풀 상태를 forward 한 번 동안 `Qwen3NextCache`로 빌려(`LentCaches`) 실행. MTP 타깃 어댑터는 시퀀스 id 대신 빌린 상태를 한 tick 동안 보관. |
| `src/models/pooled_model.rs`, `pooled_model_language_model.rs` (신규) | `Pooled<M>`: 실제 `CachePool` 위의 모델. by-id 호출로 작성된 모델 테스트가 엔진과 같은 진입점을 사용. |
| `src/server/prompt_cache/state_checkpoints.rs` | 브랜치별 체크포인트 상한. |
| 스케줄러 (`scheduler.rs`, `scheduler/prompt_cache.rs`, `speculative_burst.rs`, `speculative_finalize.rs`) | 모델 훅 대신 `state_snapshot::*` 호출. layer state가 없는 Qwen 3.5 시퀀스에 대한 MTP slice 가드. |
| 기타 약 35개 모델 파일 | `LayerKv`에서 `LayerStateKind`로의 기계적 이름 변경. |
| 문서 | ADR 0010, `docs/architecture.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/llama-server-compat.md`, `CHANGELOG.md`, 원시 JSON과 스크립트를 포함한 디코드 벤치마크 기록. |

111개 파일, +18,571 / -1,163 줄. 추가된 줄 중 약 12,600줄은 `docs/benchmark_results/scheduler-owned-state-decode-gb10-2026-11-23/` 아래의 벤치마크 원시 JSON이다. 신규 `#[test]` 함수 33개.

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `1105b1db` | fix | count the checkpoint cap per conversation branch |
| `0dbb8502` | refactor | move per-layer sequence state from models to the scheduler |
| `365ee0b2` | test | pin scheduler-owned round trips for Gemma 3 and Qwen 3.5 |
| `c98af10f` | fix | keep by-id drivers and the MTP adapter off the shared fallback slot |
| `902e6b6b` | docs | record the scheduler-owned state decode benchmark |
| `9151cc8d` | fix | share one pooled driver in the image trace and decline MTP without layer state |
| `03b630ca` | docs | add the Gemma 3 256-token rerun and state what the benchmark leaves open |
| `f3caea80` | fix | refuse ids on Gemma 3 batched entry and keep pooled MRoPE state |
| `9a274d0e` | fix | validate attention tensors on every scheduler-owned restore |
| `00940c54` | update | keep Qwen 3.5 recurrent slot tables in place while lent |
| `4a879e9c` | fix | refuse a restored full layer with no rows behind its offset |
| `3ddbeb0d` | test | pin the Qwen 3.5 MTP slice guard and record the quiet-host Gemma 3 rerun |

## 3. 기술적 선택과 그 이유

### 3.1 `KVCache`를 다형적으로 만들지 않고 별도 forward 추가

**컨텍스트:** 마이그레이션된 패밀리의 상태는 `&mut [KVCache]`로 전달할 수 없다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `KVCache`를 rotating, chunked, recurrent 저장소에 대해 다형적으로 확장 | 기존 forward 시그니처 유지 | ADR 0008/0009의 `KvAttention` dispatch를 `KVCache` 내부에 한 번 더 만들고, recurrent 텐서를 attention 캐시 타입 뒤에 둠 |
| 마이그레이션된 래퍼에 시퀀스별 테이블을 두고 by-id 진입점 유지 | 엔진 변경 없음 | 이름만 바꾼 model-owned 레이아웃이며 한 상태에 소유자가 둘 |
| **선택: `forward_layer_state(request, rows)`, `rows[i]`는 행 `i`의 `&mut [LayerState]`** | 풀이 단일 소유자. 엔진이 패드 트림, 되감기, 길이 미러를 한 번에 처리 | 엔진에 forward 경로가 둘이 되고, 마이그레이션하는 패밀리마다 새 진입점을 구현해야 함 |

**트레이드오프:** 엔진은 매 forward 전에 layer state 존재 여부를 확인한다. 기존 by-id 진입점은 마이그레이션되지 않은 패밀리를 위해 트레이트에 남아 있고, 마이그레이션된 래퍼에서는 id가 있으면 fallback 슬롯에서 조용히 실행하는 대신 panic한다. 리뷰에서 Gemma 3의 batched 진입점이 트레이트 기본 구현을 통해 여전히 도달 가능하고 그 기본 구현이 id를 버린다는 점이 발견됐다(`f3caea80`). 이제 단일 행 진입점과 같이 id를 거부한다.

### 3.2 새 스냅샷 포맷 대신 기존 인코더 재사용

`state_snapshot`은 텐서를 직접 인코딩하지 않는다. 패밀리의 `LayerStateSnapshotSpec`이 `Windowed { standard, rotating }`(패밀리 고유 이름을 쓰는 기존 `kv_snapshot`과 `kv_ring_checkpoint`, Gemma 3) 또는 `Hybrid`(추출된 `hybrid_snapshot`, Qwen 3.5)를 고른다. generic 경로가 쓴 스냅샷은 모델 훅이 쓰던 것과 바이트 단위로 호환되므로, 실행 중인 서버의 저장된 스냅샷과 영속화된 상태가 업그레이드 후에도 살아남는다. 테스트가 두 경로를 텐서 단위로 비교하고 이름을 고정한다. 대가는 generic 경로가 패밀리별 이름 체계를 물려받는다는 점이며, 이를 쓰는 저장 스냅샷이 남아 있는 한 없앨 수 없다.

### 3.3 Qwen 3.5 디코더를 다시 쓰지 않고 상태를 빌려 줌

Qwen 3.5의 디코더 레이어, batched decode, speculative verify, rollback은 `Qwen3NextCache` 위에 작성되어 있고, 이 타입은 아직 상태를 소유하는 Qwen3-Next와 Qwen4-Exp가 공유한다. 이 경로들을 `LayerState` 위로 다시 쓰면 마이그레이션하지 않는 두 패밀리를 건드리게 된다. 대신 `LentCaches`가 forward 한 번 동안 각 레이어를 풀의 벡터에서 typed view로 옮기고 drop 시 되돌린다. 텐서 복사는 없고, forward 사이에는 상태가 풀에만 존재한다. 어떤 레이어든 옮기기 전에 모든 레이어를 먼저 검사하므로 거부되면 상태는 그대로다. 리뷰에서 첫 구현이 drop 시 linear 레이어마다 `RecurrentState`를 다시 만들어, linear 레이어, 행, 디코드 스텝마다 힙 할당이 한 번씩 일어난다는 점이 발견됐다. `00940c54`는 recurrent 슬롯 테이블을 제자리에 두고 텐서만 옮긴다. Qwen3-Next와 Qwen4-Exp가 마이그레이션되면 typed view가 저장소가 되고 빌려 주는 단계는 사라진다.

### 3.4 모든 복원을 설치 전에 검증

복원은 새 벡터를 채우고, 모든 레이어가 검증을 통과했을 때만 설치한다. 검증 항목은 패밀리 태그, 레이어 수, 테이블 대비 레이어별 저장소 종류, 각 attention 레이어의 key/value 텐서(`check_restored_layer`가 `check_kv_tensors` 호출)다. 초기 구현은 전체 복원과 절단 복원에서만 텐서를 검사하고 체크포인트 복원에서는 검사하지 않았으며, hybrid 분기는 key, value, offset을 검사 없이 대입하는 `hybrid_snapshot::restore_attention`에 의존했다(`9a274d0e`). 두 번째 빈틈은 Qwen 3.5 attention 디코더가 스냅샷에 attention 행이 없어도 offset을 스냅샷 토큰 길이로 설정하는데, 빈 쌍을 무조건 허용했다는 점이다(`4a879e9c`). 현재 검사는 다음과 같다.

```rust
(None, None) => {
    return match full_offset {
        Some(offset) if offset > 0 => {
            Err(format!("no keys or values behind an offset of {offset}"))
        }
        _ => Ok(()),
    };
}
```

실패하면 이름 있는 에러가 나고 시퀀스는 그대로이며, 스케줄러는 이전과 같이 `LayoutConstraints`를 기록하고 cold prefill을 한다. 윈도우 레이어와 KV-shared 레이어는 빈 텐서를 offset 0으로 복원하므로 영향이 없다.

### 3.5 전체가 아니라 두 개의 레퍼런스 패밀리만 마이그레이션

Gemma 3(full + sliding)와 Qwen 3.5 텍스트(full + recurrent)는 이질적 상태의 두 형태를 대표한다. Gemma 4와 AFMoE는 #3255 훅을 유지하는데, 이 훅은 generic 경로와 같은 `kv_ring_checkpoint` 함수를 호출하므로 링 체크포인트 로직은 이미 구현 하나에 호출자 둘이다. ADR 0010의 감사는 모든 패밀리에서 model-owned가 된 실제 이유가 하나(레이어별 이질적 벡터와 동질적 슬라이스의 불일치)뿐이고, 상태가 모델 안에 있어야 할 이유는 없다는 결론을 냈다. 나머지는 여섯 개의 후속 작업으로 묶였다(Gemma 4/AFMoE, 기타 full + sliding 패밀리, Llama 4 chunked, GatedDeltaNet 및 linear hybrid, SSM 및 convolution hybrid, 레이어당 저장소가 둘 이상인 패밀리). `exaone_moe`와 `baichuan_m1`은 model-owned가 아니어서 표에서 빠졌다.

### 3.6 root-to-leaf 경로별 체크포인트 상한

삽입된 체크포인트를 지나는 어떤 경로에 그보다 긴 체크포인트가 이미 `N`개 있으면 체크포인트를 버린다. 포크는 브랜치마다 최대 `N`개를 유지하고, 결과는 삽입 순서에 의존하지 않으며, 모든 브랜치 전체의 상한은 기존 체크포인트 바이트 예산이 맡는다. 상한만으로는 더 이상 대화당 전체 체크포인트 수를 제한하지 않고, 바이트 예산이 이를 제한한다.

## 4. 구현 상세

```
[이전]
Scheduler ─ CachePool 엔트리 (길이 미러만)
    │
    └─ 모델 훅 ─> Wrapper { HashMap<SequenceId, Vec<Cache>> }   (패밀리마다 하나)
                    패밀리별 snapshot / restore / checkpoint / trim / rewind

[이후]
Scheduler ─ CachePool 엔트리 { layer_state: Vec<LayerState> }   (단일 소유자)
    │            │
    │            └─ Engine: forward_layer_state(request, rows)
    │                       LayerState 위에서 패드 트림 / 디코드 되감기 / visible_len
    └─ state_snapshot::* ─> kv_snapshot / kv_ring_checkpoint / hybrid_snapshot 인코더
                             (마이그레이션되지 않은 패밀리만 모델 훅)
```

Pooled MRoPE 상태: `Pooled::prepare_sequence_state`가 처음에는 모델의 release 훅을 호출했고, Qwen 3.5에서는 이 훅이 시퀀스의 MRoPE 위치를 지운다. MRoPE와 텍스트를 섞은 디코드 parity 테스트는 prepare 전에 위치를 설치하므로, 멀티모달 행이 조용히 텍스트 행이 됐다. 이제 prepare는 풀 엔트리만 버린다(`f3caea80`).

MTP slice 가드: 풀 엔트리에 layer state가 없는 Qwen 3.5 시퀀스의 MTP slice 스텝은 타깃 어댑터에서 panic하는 대신 slice 0과 같이 클라이언트에 보이는 에러 하나로 해당 요청을 실패시킨다(`f3caea80`, 테스트는 `3ddbeb0d`).

## 5. 학습 포인트

### 5.1 소유 벡터 위의 빌려 주고 돌려받기

`LentCaches::take`는 `&mut [LayerState]`를 잡고, attention 레이어는 `std::mem::replace(layer, LayerState::Empty)`로 옮기고 recurrent 레이어는 슬롯 테이블에서 텐서 두 개를 꺼내 `Vec<Qwen3NextCache>`를 만든다. `Drop`이 모두 되돌린다. 이 패턴은 디바이스 텐서를 복사하지 않고, 장기 소유자를 둘로 만들지 않으면서 기존 typed API에 임시 소유권을 준다. 하나라도 옮기기 전에 모든 레이어를 검증해야 하며, 그렇지 않으면 중간에 거부될 때 반쯤 빈 벡터가 남는다.

### 5.2 attention 트레이트를 구현하는 저장소 enum

`LayerState`는 내부 캐시로 match하여 `KvAttention`을 구현하므로, `impl KvAttention`을 받는 attention 코드가 풀 상태 위에서 그대로 동작한다. `Sliding`을 링 버퍼로 매핑하는 곳은 생성자 하나(`LayerState::new`)뿐이다. 링을 윈도우 제한 paged 블록으로 바꾸는 작업(#3256)은 이 생성자와 `Rotating` 분기만 바꾸면 되고 모델 파일은 건드리지 않는다.

## 6. 관련 PR/이슈

- 이슈 #3254: 이 PR의 이슈. 에픽 #3259: 슬라이딩 윈도우와 hybrid 상태 작업.
- #3283 / #3255: `--ctx-checkpoints`와 링 체크포인트. 이 PR은 Gemma 3의 링 체크포인트를 generic 경로로 옮기고, 그 리뷰에서 나온 브랜치 비구분 상한 문제를 고친다.
- #3047(상태 체크포인트), #1145(절단), #1755(패드 트림), #2159(디코드 되감기), #1346(paged decode override): 마이그레이션된 패밀리에서 이제 한 번만 구현되는 기능.
- ADR 0007, 0008, 0009: 이 ADR이 기반으로 하는 제약.

## 7. 후속 조치

### 완료 필요
- [ ] ADR 0010에 나열된 여섯 개의 마이그레이션 이슈를 #3259 아래에 등록하고 구현.
- [ ] `src/models/qwen3_5_tests.rs`의 ignored 테스트 `rollback_speculative_cache_panics_on_ragged_batched_accepts` 처리 결정. 이 브랜치에서 실패한다. 기대하는 panic 메시지가 `main`의 `qwen3_5.rs`에도 없지만 `main`에서 실행하지 않았으므로 원래부터 깨져 있었는지는 확인되지 않았다.

### 검증되지 않은 항목
- Gemma 4와 AFMoE의 디코드 벤치마크 없음. `LayerKv` 이름 변경 외에는 경로가 바뀌지 않아 실행하지 않았다.
- MTP slice 가드 테스트(`scheduler_qwen35_slice_guard_tests`)는 통과하지만, 가드를 제거했을 때 실패하는지는 확인하지 않았다.
- `make verify-test-cuda`는 기록된 검증에 포함되지 않았다.

### 모니터링 필요
- 업그레이드 후 Gemma 3와 Qwen 3.5의 `/v1/cache/stats`에서 `LayoutConstraints` 거부 수. 증가하면 저장된 스냅샷이 새 복원 검증에서 실패한다는 뜻이다.
- `layer_state_nbytes()`는 이전과 같이 `SequenceCacheSet::nbytes()`에 포함되지 않는다. `nbytes()`에 의존하는 메모리 계산은 여전히 이 상태를 보지 못한다.

## 부록

### A. 검증 결과

최종 브랜치에 대한 오케스트레이터 검증:

- core와 lib(테스트 포함)에 대해 `cargo clippy -D warnings`.
- 변경 영역 lib 테스트 2,134개와 `mlxcel-core` 테스트 919개를 직렬로 실행, 통합 계약 테스트.
- Engine parity: `qwen3.5-0.8b`는 동일. `gemma-3-1b`와 `gemma-4-e2b`는 각각 dense 대 prompt-cache-miss 불일치가 하나씩 있으며, 둘 다 `main` `bd931f76`에서 동일하게 재현되므로 이 PR 이전부터 있던 문제다.
- 실제 체크포인트에서 `main`과 브랜치 비교(프롬프트 캐시 켜짐, `--ctx-checkpoints 4`; 3턴 대화, 편집, 재실행, 동시 신규 요청 4개), `gemma-3-1b-it-4bit`와 `qwen3.5-0.8b-4bit`: 모든 단계에서 응답이 같고 `/v1/cache/stats` 값도 벽시계 필드 하나를 빼면 같다. Qwen 3.5에서 동시 요청 단계 후 스냅샷 하나 차이는 두 바이너리 모두에서 반복되었고 요청 도착 순서에 따라 달라진다.

### B. 성능 벤치마크

`docs/benchmark_results/scheduler-owned-state-decode-gb10-2026-11-23.md`: GB10, CUDA 릴리스 빌드, 암 `main-a`, `branch`, `main-b`(null 암, `main-a`와 같은 바이너리), 순서 순환, 프롬프트 캐시 꺼짐, `P` {256, 8192}, `B` {1, 4}. 모든 셀에서 응답 해시가 암 간에 동일.

Run 1에서 4는 부하가 있는 공유 호스트에서 실행됐고, 거기서 Gemma 3 `B = 4`는 -2.2%에서 -4.1%로 나왔다. Run 5는 브랜치를 `4a879e9c`에서 다시 빌드해 조용한 호스트에서 해당 셀을 반복했다.

| 모델 | P | B | main-a | main-b | branch | null | branch 대 main |
|---|---|---|---|---|---|---|---|
| gemma-3-1b-it-4bit | 256 | 4 | 366.21 | 363.91 | 388.52 | 0.6% | +6.4% |
| gemma-3-1b-it-4bit | 8192 | 4 | 350.60 | 344.62 | 357.85 | 1.7% | +3.1% |

이전의 음수 측정치는 호스트 부하 때문이었다. 암당 샘플 12개 기준으로 `B = 1` 셀은 null 범위 안에 있다(Gemma 3 256에서 -0.1%, Qwen 3.5 +3.9%와 -1.4%). Qwen 3.5 `B = 4`는 조용한 실행에서 -0.1%와 -0.3%였다. 회귀를 보이는 셀은 없다. +6.4%를 속도 향상으로 읽어서는 안 된다. 브랜치는 스텝당 같은 디바이스 작업을 하고, 향상을 설명하는 코드 변경은 없다.
