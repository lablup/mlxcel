# PR #2934: ROCm GQA 디코드 score matmul 튜닝

**작성일**: 2026-11-10
**상태**: 완료 (PR은 draft로 열려 있고 #2921을 닫음)

## 요약

단일 시퀀스 XLA 그래프의 디코드 attention은 dispatch 하나가 지배하고 있었다. GQA score contraction이 1509.75 us짜리 attention 블록에서 1187.17 us를 차지했고, 16개 dispatch가 각각 74.20 us였다. 같은 MAC 수를 가진 context matmul은 6.42 us에 실행됐다. dispatch별 `rocprofv3` 표가 elementwise folding과 dispatch 수 축소를 배제했다. IREE가 그래프 재작성을 dispatch 형성 이전에 canonicalize하기 때문이다. 출시된 수정은 ROCm 튜닝 스펙에 추가한 좁은 두 번째 match다. batch 8, group 4, `head_dim` 64, 키 128개 이상에 128스레드 `[32, 4, 1]` workgroup과 64폭 전체 reduction을 적용한다. 스텝당 attention은 1509.75에서 337.64 us로, score dispatch는 호출당 74.20에서 1.47 us로, n=128, capacity 256에서 디코드 스텝은 8.80 ms에서 중앙값 7.33 ms(최선 7.07 ms)로 줄었다. greedy 출력은 이전 native 바이너리와 f16 경로에 두 프롬프트에서 바이트 단위로 동일하고, `xla_batch_bench`는 B=4와 B=8에서 REFERENCE-EXACT이며, 커밋된 golden은 갱신하지 않았다. 6개 파일, +328/-14.

## 1. 문제 정의

#2890, #2892, #2907 이후 투영 블록은 스텝당 약 21 ms에서 3.7 ms로 줄었지만 attention은 약 1.31 ms에 머물러 가장 큰 남은 커널 블록이 됐다. 이슈는 추측이 아니라 dispatch별 표를 먼저 만들고 표가 선택한 수정을 하라고 요구했다.

n=1과 n=128(127 디코드 스텝, 차분) 측정에서 attention은 스텝당 1509.75 us다. 그중 GQA score contraction 하나가 16개 dispatch로 1187.17 us, 호출당 74.20 us다. 같은 수의 multiply-accumulate를 하는 context matmul은 호출당 6.42 us다. 따라서 격차는 수학, 마스크, 커널 수가 아니라 score matmul이 dispatch로 lowering되는 방식에 있다. 조사 전후 attention dispatch는 레이어당 10개로 같았다.

## 2. 시도했다가 배제한 것

- attention scale을 레이어 루프 이전에 q로 접어 넣고 K를 head-major로 emit해 contraction 축을 연속으로 만드는 변경. 둘 다 구현하고 측정했지만 IREE가 dispatch 형성 이전에 canonicalize해 컴파일된 그래프에서는 no-op이었다. 되돌렸고 이 PR에 없다.
- 레이어당 dispatch 수 축소나 KV 레이아웃 변경. 같은 표에서 배제됐다. elementwise와 reduction dispatch는 score contraction에 비해 작다.

## 3. 출시된 변경

`src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`에 기존 long-K vecmat 규칙 다음으로 `match_gqa_score`를 추가했다.

- 조건: batch 8, group 4, `head_dim` 64, 키 128개 이상.
- Lowering: 74 us dispatch를 만들던 일반 lowering 대신 128스레드 workgroup `[32, 4, 1]`과 64폭 전체 reduction.

지원 변경은 `src/lib/mlxcel-xla/src/tuning.rs`의 튜닝 테스트, 단일 디코드 score contraction이 튜닝 스펙 형태를 유지하는지 확인하는 표적 emitter 테스트(`single_decode_score_contraction_keeps_the_tuning_spec_shape`), `src/lib/mlxcel-xla/src/iree.rs`의 주석 갱신이다. `AttnLayout::Ragged`와 `AttnLayout::Prefill`의 emit 텍스트는 바이트 단위로 동일하고 golden은 갱신하지 않았으며 모델 수학은 바뀌지 않았다.

## 4. 측정

모든 실행은 `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, `MLXCEL_XLA_CONTEXT_CAPACITY=256`, `scripts/rocm_gpu_guard.sh --hold` 아래에서 수행했다.

| 지표 | 이전 | 이후 |
|------|------|------|
| 스텝당 attention | 1509.75 us | 337.64 us |
| GQA score dispatch | 호출당 74.20 us | 호출당 1.47 us |
| 전체 그래프 커널 시간 | 5291 us | 4099 us |
| 디코드 스텝, n=128 | 8.80 ms (113.6 tok/s) | 중앙값 7.33 ms, 최선 7.07 ms (136.4 / 141.5 tok/s) |

A/B는 한 바이너리에서 세 라운드 교차 실행이다. greedy 출력은 변경 전 native 바이너리와 f16 경로에 두 프롬프트에서 바이트 단위로 동일하다. `xla_batch_bench`는 B=4와 B=8에서 REFERENCE-EXACT PASS를 출력한다.

이슈 목표인 7.1 ms 미만 스텝은 최선 샘플(7.07 ms)에서는 만족하지만 중앙값(7.33 ms)에서는 만족하지 않는다. attention은 더 이상 병목이 아니며 남은 비용은 토큰당 호스트 측 작업이다.

## 5. 기술적 선택과 그 이유

### 5.1 그래프가 아니라 튜닝 스펙 변경

프로파일은 그래프가 이미 canonical하고 비용이 dispatch 형성에 있다는 것을 보여줬다. emitter를 고치면 컴파일러가 버리는 코드만 늘어난다. 튜닝 match는 이 경로에서 형성되는 dispatch를 바꾸는 유일한 레버이며 그래프, 수학, golden을 그대로 둔다.

### 5.2 일반 match 대신 좁은 match

end-to-end로 추적한 설정인 batch 8, group 4, `head_dim` 64, 키 128개 이상으로 제한했다. batch 조건은 장식이 아니라 핵심이다. IREE의 StableHLO 입력 lowering은 ragged와 uniform-B `dot_general`을 같은 indexing map의 같은 rank-3 contraction으로 접으면서 batch를 `bsz * n_kv`로 만들기 때문에, 처음 버전의 match는 한 행용 설정을 ragged score matmul에도 적용했다. 측정한 `n_kv = 8`로 batch를 고정하면 단일 시퀀스 그래프만 match되고 여러 행 그래프는 모두 컴파일러 기본값으로 돌아간다. matcher로 확인한 결과 single decode 16개 match, prefill 0개, ragged B=4 0개, ragged B=8 0개다. 나중에 넓히는 것은 새 trace를 근거로 조건 한 줄을 바꾸는 일이다.

## 6. 검증과 한계

- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings`: 깨끗함.
- `cargo test -p mlxcel-xla --lib` 표적 실행: `validation::tests::emit_all_reproduces_registered_goldens`, `tuning::tests` 2개, 새 `single_decode_score_contraction_keeps_the_tuning_spec_shape` 테스트가 모두 통과한다.
- 구현 리뷰에서 첫 버전의 match가 이슈 범위 밖인 ragged와 uniform-B score contraction에도 걸린다는 점을 찾았다. 위의 batch 조건이 이를 고쳤고, lowered golden에 matcher를 다시 실행해 single decode 16개 match, prefill/ragged B=4/ragged B=8 0개를 확인했다. `pr-security-checker`와 `pr-finalizer` 단계는 코디네이터의 중단 지시에 따라 실행하지 않았으며, 위의 제한된 gate가 실제로 실행된 검증이다. 변경 표면은 컴파일러 튜닝 스펙과 테스트이며 런타임 입력 처리나 자격 증명 표면은 없다.
- capacity 256만 end-to-end로 추적했다. KV read convert(스텝당 103 us)와 context matmul(스텝당 103 us)이 이제 attention의 가장 큰 항목이며 후속 후보다.

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 6 |
| 추가 라인 | +328 |
| 삭제 라인 | -14 |
| 추가 테스트 | 튜닝 assertion 2개, emitter 테스트 1개 |

| 카테고리 | 요약 |
|----------|------|
| 성능 | 좁은 GQA score 튜닝으로 스텝당 attention 1509.75에서 337.64 us |
| 정확성 | 토큰 동일, 배치 reference-exact, golden 불변 |
| 문서 | `iree.rs` 주석 갱신 |

## 8. 후속 조치

1. KV read convert와 context matmul(현재 각 약 103 us/step)을 추적하고 줄인다.
2. 새 형태(다른 head 차원이나 키 범위)는 trace를 근거로만 튜닝 match를 넓힌다.
3. capacity 256만 측정했으므로 다른 capacity에서 n=128 결과를 확인한다.
