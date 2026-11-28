# PR #2960: GQA decode context contraction을 위한 좁은 tuning matcher

**작성일**: 2026-11-11
**상태**: 완료 (draft PR, #2956 close)

## 요약

#2921의 decode attention 항목은 capacity 256에서 각각 약 100 us를 쓰는 두 개의 dispatch로 나뉘었다. 그중 context matmul은 lowering 문제였다. `group`이 4라서 ROCm이 wave 단위 tiling을 충분히 고르고, IREE 컴파일러는 reduction을 한 tile로, batch 차원을 1로 둔다. MAC당 비용은 튜닝된 score dispatch의 네 배였다. ROCm tuning spec에 세 번째 `foreach_match` 분기를 추가해 네 개 query row 전체와 64개 출력 열 전체를 workgroup 단위로 tile하고 key 축을 64 원소 chunk로 나누면, context dispatch가 capacity 256에서 `6.343 us`에서 `1.806 us`로 줄어든다.

KV read convert는 negative result다. `elementwise_transpose_256x8x64_f32xf16`는 호출당 768 KB를 3.2 us에 옮기므로 약 240 GB/s이고, `xla_validate_kv_pair`가 두 KV 캐시를 f32로 고정한 상태에서는 대역폭 한계다. lowering 변경으로 없앨 수 없고 f16 캐시가 필요하지만 이는 ABI와 수치 결정이라 이슈가 제외했다. 따라서 구현하지 않고 기록만 한다.

capacity 256에서 합산 step 절감은 `72.6 us`로 206 us 항목의 약 35퍼센트(이슈 목표 30퍼센트)지만, 실제 wall time은 그 지점에서 평평하다(5회 교차 라운드에서 `140.5` 대 `140.0 tok/s`). 개선은 dispatch가 지배하는 큰 capacity에서만 보인다(`123.8`에서 `130.0 tok/s` at 512, `82.6`에서 `87.4 tok/s` at 1024). 정확한 서술은 dispatch 시간은 고정적으로 줄었고 end-to-end 이득은 측정된 capacity에 달려 있다는 것이다.

## 1. 문제 정의

이슈 #2956은 마지막 decode-attention 프로파일의 두 항목을 요청한다.

1. KV read convert(`elementwise_transpose_256x8x64_f32xf16`), step당 32회, 회당 3.2 us, step당 102.9 us.
2. context matmul(`main_dispatch_11_batch_matmul_8x4x64x256_f16xf16xf32`), step당 16회, 회당 6.343 us, step당 101.5 us.

목표는 합산 205 us의 30퍼센트, 약 62 us였다. 프로파일은 `gfx1151`, `llama-3.2-1b-instruct-4bit`, `MLXCEL_XLA_CONTEXT_CAPACITY=256`에서 `n=128`에서 `n=1`을 뺀 값으로 기록해, 시작 비용이 step별 귀속을 희석하지 않게 했다.

## 2. 원인

### 2.1 context matmul의 lowering

contraction은 `[8, 4, K] x [8, K, 64] -> [8, 4, 64]`이다. batch 8(KV head), `M = 4`(query group), `N = 64`(head width), `K`는 key다. capacity 256에서 출력 원소당 256 MAC뿐인데, 컴파일러 기본값은 출력 원소당 workgroup 하나를 배정하고 네 query row 사이의 재사용을 하지 않는다. grid `256x8`, workgroup `32x8`(256 thread), VGPR 104다.

#2921 전의 score matmul과 같은 출력 원소당 workgroup 배치인데, 원소당 연산이 64 MAC으로 score의 2048보다 작다. score matcher가 발화하지 않는 이유는 역할이 뒤바뀌었기 때문이다. score는 `head_dim`을 줄이고 `keys`를 쓰며, context는 `keys`를 줄이고 `head_dim`을 쓴다.

### 2.2 KV read convert

호출마다 512 KB f32 블록을 읽고 256 KB f16 블록을 쓴다. 3.2 us에 768 KB면 약 240 GB/s로, 이 host의 실제 달성 가능한 비율의 상당 부분이며 명백히 잘못 lowering된 접근 패턴은 아니다. 이를 바꾸려면 KV 캐시 dtype을 바꿔야 하고 이슈가 이를 제외했다.

## 3. 변경

`src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`에 `@match_gqa_score` 뒤에 세 번째 named sequence `@match_gqa_context`를 추가하고 `@__kernel_config`에 연결한다.

```mlir
transform.iree.match.contraction %arg0,
  lhs_type = f16, rhs_type = f16, output_type = f32,
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
transform.iree.match.dims_equal %batch_dims, [8]
transform.iree.match.dims_equal %m_dims, [4]
transform.iree.match.dims_equal %n_dims, [64]
transform.iree.match.dim_bounds %rhs[1], umin = 128, none
#iree_gpu.lowering_config<{reduction = [0, 0, 0, 64], thread = [1, 1, 2, 0],
                           workgroup = [1, 4, 64, 1]}>
pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 4, 1] subgroup_size = 32
```

네 조건이 다른 곳에서 발화하지 않게 한다.

- `batch = 8`은 단일 시퀀스 그래프의 `n_kv`다. ragged와 uniform-B 그래프는 `bsz * n_kv`를 같은 rank-3 shape의 batch로 접어 넣으므로, 이 조건이 없으면 한 row용으로 측정된 tile을 조용히 가져간다. #2921이 score contraction에서 세운 것과 같은 필수 조건이며 header 주석에 명시했다.
- `M = 4`는 query group이다. prefill의 M 축은 prompt 길이를 실으므로 prefill은 매치될 수 없다.
- `N = 64`는 head width다. 다른 폭은 컴파일러 기본값을 유지한다.
- `K >= 128`은 아주 짧은 KV에서 tiling보다 reduction predication이 더 중요해지는 것을 막는다.

`src/lib/mlxcel-xla/src/tuning.rs`는 세 번째 matcher를 문서화하고 `tuning_spec_matches_the_gqa_decode_context_shape`를 추가한다. 이 테스트는 batch, M, N 조건과 reduction bound, workgroup 크기를 고정해 의도치 않은 확장을 막는다. 기존 score 테스트는 entrypoint에 context 줄이 여전히 있는지도 확인한다.

## 4. 측정

프로토콜: `llama-3.2-1b-instruct-4bit`, `gfx1151`, GPU guard 독점, step별 귀속은 `n = 128`에서 `n = 1`을 뺀 `rocprofv3` trace, wall time은 capacity 256에서 5회, 512와 1024에서 3회 교차 라운드.

### 4.1 회당 dispatch 시간

| Capacity | 기본 us/call | 기본 us/step (16) | 신규 us/call | 신규 us/step | 절감 |
|----------|--------------|-------------------|--------------|--------------|------|
| 256 | 6.343 | 101.5 | 1.806 | 28.9 | 72.6 |
| 512 | 11.967 | 191.5 | 2.813 | 45.0 | 146.4 |
| 1024 | 24.890 | 398.2 | 5.923 | 94.8 | 303.5 |

512와 1024의 기본값은 같은 프로토콜로 그 capacity에서 측정했고, 신규 값은 해당 shape의 runtime dump와 trace에서 얻었다.

### 4.2 End to end

| Capacity | 기본 tok/s | 신규 tok/s | 라운드 |
|----------|-------------|-------------|--------|
| 256 | 140.5 | 140.0 | 5 |
| 512 | 123.8 | 130.0 | 3 |
| 1024 | 82.6 | 87.4 | 3 |

capacity 256에서 step당 72.6 us는 약 7 ms step의 1퍼센트라 중앙값이 평평하고, 1024에서 5퍼센트가 보인다. 이 shape에서 처리량 주장은 아니며, 프로파일이 이 항목에 귀속한 dispatch 시간을 제거한 것이다.

## 5. 평가

### 5.1 matcher가 발화하는 그래프

컴파일된 configuration dump와 trace 호출 수 기준.

| 그래프 | bigk | score | context | step당 호출 |
|--------|------|-------|---------|-------------|
| 단일 시퀀스 decode | 1 | 1 | 1 | 각 16 |
| Prefill | 0 | 0 | 0 | - |
| Ragged B=4 | 0 | 0 | 0 | - |
| Ragged B=8 | 0 | 0 | 0 | - |
| f16 weight decode 그래프 | 1 | 1 | 1 | - |

f16 weight 행은 이 변경이 기존 matcher를 꺼뜨리지 않았다는 대조군이다. f16 projection에서는 bigk vecmat이 살아 있고 여전히 매치한다.

### 5.2 검증

- `cargo fmt --all -- --check` clean.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings` clean.
- `cargo test -p mlxcel-xla --lib tuning::` 3 passed.
- `single_decode_score_contraction_keeps_the_tuning_spec_shape` passed.
- `validation::tests::emit_all_reproduces_registered_goldens` passed (emitter와 asset 변경 없음).
- 두 prompt에서 greedy 출력이 기본 spec과 동일.
- `xla_batch_bench`에 `MLXCEL_XLA_PRECISION=f16`: B=4(N=8, maxcap 24)와 B=8(N=16, maxcap 48)에서 `REFERENCE-EXACT PASS`.

## 6. 남은 항목

KV read convert는 step당 102.9 us에 머문다. 두 캐시가 f32이고 transpose가 호출당 768 KB를 옮기기 때문이다. f16 KV 캐시는 convert를 없애지만 수치 경로와 `xla_validate_kv_pair` 계약을 바꾸므로, 정밀도 근거를 갖춘 별도 이슈에 속한다.

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 2 |
| 추가 줄 | +108 |
| 삭제 줄 | -18 |
| 커밋 | 1 |

관련: #2921(이 변경이 모방한 score matcher), #2960(이 PR), #2956(이 이슈).
