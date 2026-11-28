# 기술 보고서: PR #2845 - perf: absorbed MLA attention for multi-token steps below crossover

**작성일**: 2026-11-03
**상태**: 리뷰 중 (열린 PR, #2408 종료)
**언어**: Rust, Markdown
**위험도**: 중간

---

## 요약

MLA attention은 `l == 1`일 때만 matrix-absorbed 경로를 선택했기 때문에, 모든 speculative verify block, MTP verify step, chunked-prefill continuation이 쿼리를 2개에서 16개만 담고 있어도 캐시된 latent 전체에 대해 per-head K/V를 materialize했다. 이 PR은 그 gate를 공유 cost-model helper인 `mlxcel_core::mla::use_absorbed_mla`로 교체한다. upstream bound `L < r*d / (r*d/T + 2*r - d)`를 구현하며, 호출당 한 번 계산한 boolean을 일곱 MLA 모델 지점에 연결한다. cold prefill은 여전히 materialize하고(`T == L`이면 모든 배포 geometry에서 bound가 false가 된다), 단일 쿼리는 항상 absorbed이므로 기존 prefill 동작은 보존된다. 긴 캐시 뒤의 짧은 verify block은 이제 absorbed로 실행된다. 1024 토큰 뒤에서 width 4와 16으로 두 4-bit 체크포인트를 teacher-forced trace한 결과, 되돌린 `l == 1` arm과 decided 위치 불일치가 0이었고, chunked-prefill continuation의 traced-forward 시간은 15.8%에서 53.0% 감소했다. MTP speculative decode는 명확한 개선을 보이지 않았다.

---

## 1. 문제 정의

### 1.1 배경

DeepSeek 계열 MLA는 K와 V를 `kv_lora_rank` 폭의 단일 latent `c`와 공유 rope stream `k_pe`로 압축한다. Absorbed 방식은 K up-projection을 쿼리에, V up-projection을 attention 뒤에 접어 넣어 모델이 latent에 직접 attend하고 훨씬 작은 캐시를 유지하게 한다. mlxcel에는 이미 두 경로가 있었다. absorbed decode 경로와, live latent window를 `kv_b_proj`로 up-project한 뒤 dense SDPA를 실행하는 materialize 경로다.

경로 선택은 단일 조건, 즉 `let output = if l == 1 { ... } else { ... }`이었다. absorbed 경로는 decode 전용으로 존재했고 multi-token step은 prefill로 간주됐다. 이 가정은 이미 부정확했다. speculative decoding은 step마다 여러 draft token을 verify하고, MTP는 block을 verify하며, chunked prefill은 긴 cached context 뒤에 소수의 토큰을 append할 수 있다. 세 경우 모두 쿼리 수는 작고 캐시 길이 `T`는 크며, 바로 이 영역이 absorption이 더 저렴한 곳이다.

### 1.2 기존 문제점

- **문제 1**: 모든 multi-token step이 쿼리 수와 무관하게 `O(T)` 비용으로 latent window 전체에 걸쳐 K와 V를 materialize했다.
- **문제 2**: `l == 1` gate가 "decode"와 "쿼리 하나"를 혼동했다. 1024 cached token 뒤의 width 4 또는 16 verify block이 16 토큰짜리 cold prefill과 같은 경로를 탔다.
- **문제 3**: absorbed 경로가 multi-token width에서 실제로 사용됐다는 증거가 없었다. 이슈 #899에서 이미 한 번도 활성화되지 않은 fused decode 경로를 출시했고, 벤치마크가 fallback끼리 비교되어 깔끔해 보이는 null 결과를 냈다.
- **문제 4**: 각 MLA family가 경로 선택을 로컬에서 재구현했으므로 정책을 바꾸려면 일곱 파일을 동시에 수정해야 했다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|--------|------------|
| multi-token absorbed step이 causal mask를 빠뜨려 미래 위치에 attend | 높음 | 중간 (`deepseek_v2`가 mask를 생성하고, `deepseek_v32`가 indexer top-k를 mask에 접으며, 나머지 family는 branch 전에 causality를 접어 완화) |
| absorbed와 materialized 경로가 multi-token width에서 수치적으로 발산 | 중간 | 중간 (측정: decided 불일치 0이나 GLM에서 2 미만 gap의 top-1 뒤집힘 존재) |
| cost model이 cold prefill에서 absorbed를 골라 수치적으로 동일한 baseline을 잃음 | 중간 | 낮음 (`T == L`이면 bound가 `2*r < d`로 줄고 배포 config에서 false) |
| MTP speculative decode 성능 회귀 | 중간 | 중간 (측정: 평균은 노이즈 범위로 불확정) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

해당 없음. 입력 처리, 인증, 직렬화, 네트워크 경계를 변경하지 않는다.

- **호환성 파괴**: 없음. `use_absorbed_mla`는 추가 API이며, 기존 `MLXCEL_MLA_ABSORBED` gate가 absorbed 기구를 빌드할지 여부를 계속 결정한다.
- **신규 의존성**: 없음.

### 2.2 성능 관점

cost model은 per-head FLOP 수에서 유도된다. `r = kv_lora_rank`, `d = qk_nope_head_dim + v_head_dim`, `L`개 쿼리, 이번 step의 update 이후 cached latent 위치 `T`개일 때:

- Absorbed: `L*r*d + 2*L*T*r`
- Materialize: `T*r*d + L*T*d`

Absorbed는 `L < r*d / (r*d/T + 2*r - d)`일 때 이긴다. `T`가 커지면 bound는 `r*d / (2*r - d)`에 접근한다. DeepSeek-V3 geometry(512/128/128)에서는 170 쿼리, GLM-4.7-Flash의 `glm4_moe_lite` geometry(512/192/256)에서는 398 쿼리다. cold cache(`T == L`)에서는 bound가 `2*r < d`로 줄어 모든 배포 구성에서 false이므로 새 prefill은 계속 materialize한다. 단일 쿼리는 나눗셈 전에 true를 반환하고, non-positive denominator는 의미 없는 bound를 만들지 않고 거절한다.

**성능 영향 (chunked-prefill continuation, 1024 cached token 뒤 4 및 16 토큰, traced forward만 측정):**

| 모델 | Width | Before | After | 변화 |
|------|--------|-------|-------------|-------------|
| DeepSeek-V2-Lite | 4 | 17.57 ms | 14.80 ms | -15.8% |
| DeepSeek-V2-Lite | 16 | 31.52 ms | 25.34 ms | -19.6% |
| GLM-4.7-Flash | 4 | 46.45 ms | 21.83 ms | -53.0% |
| GLM-4.7-Flash | 16 | 60.31 ms | 47.44 ms | -21.3% |

이는 `logit_trace` 내부의 traced-forward wall time이며 end-to-end 서빙 처리량이 아니다.

### 2.3 호환성/의존성 관점

일곱 호출 지점 모두 자체 `ModelArgs` geometry로 helper를 계산한다. `kimi_k3`는 cost model에 필요한 `v_head_dim`을 구조체 필드로 저장하는 변경을 얻는다. 기존 projection shape에는 구조체 멤버로 노출되지 않았다. 제거되는 공개 API는 없다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 새 테스트 6개. 다섯 개는 helper 경계 동작(단일 쿼리, `T=100000`에서 DeepSeek-V3 crossover, `T == L`, verify-block width, non-positive denominator)을 고정한다. 하나는 하나의 warm cache에서 full-model logits를 두 branch로 실행한다.
- **코드 복잡도**: 각 모델 지점은 이제 `absorbed` boolean 하나를 계산하고 경로를 기록하며, 인라인 `l == 1` branch를 대체한다. `deepseek_v32`는 branch 본문을 재구성해 single-token sparse gather와 multi-token absorbed attention을 분리한다.
- **기술 부채**: 감소. 경로 정책이 일곱 호출 지점을 명시한 `Used by:` 목록을 가진 하나의 helper에 있다.

---

## 3. 기술적 선택과 그 이유

### 3.1 모델별 threshold 대신 공유 cost-model helper

**컨텍스트:**
일곱 모델 파일이 동일한 `l == 1` gate를 갖고 있었고, 올바른 crossover는 head geometry에 의존한다. upstream 공식을 각 파일에 따로 port하면 drift가 생긴다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|--------|------|------|
| `l == 1`을 유지하고 모델별 예외 추가 | 최소 diff | 일곱 개로 갈라지는 정책, 원리적 경계 없음 |
| upstream 부등식을 각 모델에 복사 | cross-crate 의존 없음 | 중복 공식, 한 곳만 고치기 쉬움 |
| 모델별 경험적 threshold | 측정 하드웨어에 fit 가능 | geometry나 하드웨어에 이식 불가 |
| **선택: `mla/mod.rs`의 하나의 helper** | 단일 진실 공급원, 독립 테스트 가능, upstream과 일치 | 모든 family가 core MLA 모듈에 의존 |

**선택 이유:**
이 부등식은 하드웨어 휴리스틱이 아니라 닫힌 형식의 FLOP 비교다. upstream이 이미 Blaizzy/mlx-vlm#2150(mlx-lm 대응은 ml-explore/mlx-lm#1817)에서 이를 도입했으므로 upstream 형태와 맞추면 port가 정렬된다. helper의 doc comment는 공식, 두 배포 geometry의 점근 bound, cold-cache 축소, 그리고 공유 함수 규칙이 요구하는 `Used by:` 목록을 담는다.

**트레이드오프:**
helper는 각 지점에서 `v_head_dim`을 필요로 한다. 대부분 family에는 이미 있었고 `kimi_k3`에는 없어서 이제 저장한다. helper는 geometry 전용이다. FLOP는 모델링하지만 캐시 트래픽, 커널 런치 오버헤드, quantization 차이는 반영하지 않으므로 crossover는 근사값이다.

### 3.2 `T`는 update 이후 cached 길이이며 호출당 한 번 계산

**컨텍스트:**
bound는 step의 쿼리 수를 attention이 실제로 읽을 캐시, 즉 이번 step이 append하는 행을 포함한 길이와 비교한다. 길이를 잘못된 시점에 읽으면 crossover가 이동한다.

**결정:**
각 지점은 `absorbed`를 한 번 계산한다. `cache.update_and_fetch` 이후에는 `t = cache.seq_len()`을, `MlaLatentCache::attend`가 내부에서 update를 수행하는 `deepseek_v2`에서는 `live_before + l`을 사용한다. boolean 아래의 어떤 코드도 경로 선택을 `l`로 gate하지 않는다. helper doc은 `t`가 monotonic RoPE offset이 아니라 live window 길이임을 명시한다.

**트레이드오프:**
`deepseek_v2`는 append를 수행하는 attend 호출 전에 캐시를 측정하므로 유일하게 모양이 다르다. 이 비대칭은 helper와 호출 지점에 문서화되어 있다.

### 3.3 multi-token absorbed step에는 명시적 causal mask가 필요

**컨텍스트:**
materialize 경로는 `causal_attention`에서 causality를 얻는다. absorbed 경로는 `pe_scores`를 직접 전달하므로, multi-token absorbed step은 causal 제한 없이 전체 latent를 보게 될 수 있다.

**결정:**
- 자체 `pe_scores` mask를 만드는 family(`deepseek_v3`, `glm4_moe_lite`, `kimi_linear`, `kimi_k3`, `longcat_flash_ngram`)는 branch 전에 이미 causality를 접는다.
- `deepseek_v2::forward_absorbed`는 `l > 1`이고 호출자가 mask를 주지 않았을 때 `create_causal_mask(l, live_before)`를 생성한다. 호출자가 준 padded mask가 우선한다.
- `deepseek_v32`는 `l > 1`일 때 `apply_sparse_prefill_mask`를 branch 앞으로 옮겨, indexer의 top-k 선택이 multi-token absorbed step의 additive mask 일부가 되게 한다. single-token decode는 더 저렴한 `gather_topk_kv` fast path를 유지한다.

**트레이드오프:**
`deepseek_v32` 재구성이 diff에서 가장 큰 부분이다(+88/-42). 서로 다른 mask를 가진 하나의 호출을 공유하는 대신 single-token과 multi-token absorbed branch에 latent attention 호출을 중복시킨다.

### 3.4 선택된 경로를 관측 가능하게 만듦

**컨텍스트:**
이슈 #899의 교훈은 경로가 조용히 한 번도 활성화되지 않을 수 있고, 그러면 벤치마크가 두 arm 모두 fallback을 측정한다는 것이다.

**결정:**
모든 지점이 `mla::stats::MlaDecodePath`를 기록한다(`l == 1`이면 `AbsorbedComposed`, multi-token absorbed면 `AbsorbedPrefill`, 모델이 거절하면 `Decompressed`). `examples/logit_trace`는 `MLXCEL_TRACE_MLA_STATS=1`에서 이 카운터를 출력하고, 각 chunk의 traced-forward wall time을 context pass와 분리해 보고한다. 이 PR의 trace 증거는 그 카운터로 traced width가 실제로 `absorbed_prefill`을 탔음을 보인다(DeepSeek-V2-Lite 216회, GLM-4.7-Flash 376회). 되돌린 arm은 0을 기록했다.

**트레이드오프:**
카운터는 프로세스 전역 atomic이다. 한 arm에 카운트를 귀속하려면 `take()`(카운터를 비움)를 호출해야 하며 trace harness가 그렇게 한다. `logit_trace`에 추가된 `eval(&logits)`는 lazy graph를 일찍 강제해 타이밍에서 host-side top-k를 제외한다. 이는 graph materialize 시점을 바꾸지만 계산 결과는 바꾸지 않는다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before]
model forward ──> l == 1 ? absorbed : materialize   (7개 파일에서 독립적으로 결정)

[After]
model forward ──> use_absorbed_mla(l, t, r, nope, v)
                      │
                      ├── true  ──> absorbed (single-query 또는 multi-token)
                      └── false ──> per-head K/V materialize, dense SDPA
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-core/src/mla/mod.rs`**

```rust
pub fn use_absorbed_mla(
    l: usize,
    t: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
) -> bool {
    if l == 1 {
        return true;
    }
    let r = kv_lora_rank as f64;
    let d = qk_nope_head_dim as f64 + v_head_dim as f64;
    let denom = r * d / t as f64 + 2.0 * r - d;
    if denom <= 0.0 {
        return false;
    }
    (l as f64) < r * d / denom
}
```

**변경 이유:** 공유 정책이다. `f64`로 평가해 crossover에서 정수 절단을 피하고, `l == 1`에서 true를 반환해 부등식에 의존하지 않고 decode 경로를 보존하며, non-positive denominator는 음수 bound 대신 거절한다.

**파일: `src/models/deepseek_v2.rs`**

```rust
let live_before = view.seq_len();
let absorbed = mla::use_absorbed_mla(
    l as usize,
    live_before as usize + l as usize,
    self.kv_lora_rank,
    self.qk_nope_head_dim,
    self.v_head_dim,
);

let output = if absorbed {
    let mask = if l == 1 {
        None
    } else if let Some(m) = mask {
        Some(m)
    } else {
        causal_storage = mlxcel_core::utils::create_causal_mask(l, live_before);
        Some(causal_storage.as_ref().unwrap())
    };
    view.attend(q_nope, &q_pe, ckv, k_pe, proj, self.scale, mask)
} else { /* per-head K/V materialize */ }
```

**변경 이유:** `attend`가 step의 행을 직접 append하므로 `T`는 `live_before + l`로 계산해야 한다. causal mask는 호출자 mask가 없는 multi-token absorbed 경우에만 생성한다.

**파일: `src/models/deepseek_v32.rs`**

```rust
// multi-token step은 indexer의 top-k 선택을 additive mask에 접는다
let pe_scores = if l > 1 {
    match &topk_indices {
        Some(indices) => apply_sparse_prefill_mask(
            &pe_scores, indices, mlxcel_core::array_shape(&kv_latent)[2],
        ),
        None => pe_scores,
    }
} else {
    pe_scores
};
```

**변경 이유:** multi-token absorbed step은 전체 latent에 대해 latent attention을 실행하므로 top-k 제한이 gathered KV subset이 아니라 mask에 있어야 한다. single-token decode는 `gather_topk_kv`를 유지한다.

**파일: `src/models/glm4_moe_lite.rs` 및 `src/models/glm4_moe_lite_tests.rs`**

`forward_with_gate(x, cache, mask, forced_absorbed: Option<bool>)`는 테스트용 진입점이며 production `forward`는 `None`으로 호출한다. production 모델이 스스로 absorbed를 선택하는 width에서 하나의 warm cache를 두 branch로 재생할 수 있게 한다.

**변경 이유:** pinnability가 없으면 테스트 대상 width에서 materialize 쪽을 실행할 수 없다.

**파일: `examples/logit_trace.rs`**

```rust
let forward_started = std::time::Instant::now();
let logits = ...;
eval(&logits);
let forward_ms = forward_started.elapsed().as_secs_f64() * 1e3;
...
if std::env::var("MLXCEL_TRACE_MLA_STATS").as_deref() == Ok("1") {
    eprintln!("# mla_paths\t{}", mlxcel_core::mla::stats::take().summary());
}
```

**변경 이유:** trace harness를 multi-token continuation을 시간 측정하고 traced width가 어느 branch를 탔는지 증명하는 측정 도구로 바꾼다.

### 4.3 데이터 모델 변경

없음. 두 경로가 같은 latent 캐시에 쓰므로 step이 재패킹 없이 두 경로를 오갈 수 있다.

---

## 5. 학습 포인트

### 5.1 구조적 휴리스틱보다 닫힌 형식 경로 선택

**개념:**
두 알고리즘이 같은 수학을 서로 다른 작업 분포로 구현할 때, "쿼리 수가 1" 같은 proxy 대신 FLOP 수를 기호적으로 비교해 더 저렴한 쪽을 선택할 수 있다.

**이 PR에서의 적용:**
absorption은 up-projection을 쿼리에 접고, materialize는 캐시된 window에 대한 up-projection 비용을 낸다. `L*r*d + 2*L*T*r`과 `T*r*d + L*T*d`를 비교하면 다섯 shape 값에 대한 하나의 부등식이 나온다. 같은 부등식이 decode, verify block, prefill을 모두 포괄하며 양 끝점이 자연스럽게 도출된다(`T == L`은 항상 materialize, 큰 `T`는 더 많은 쿼리를 허용).

**일반적 사용 사례:**
- 행렬 shape에 따라 transposed/비transposed contraction 선택.
- sequence 길이와 batch로 chunked/monolithic 커널 선택.
- 쿼리 대 캐시 비율로 fused/decomposed 연산자 결정.

### 5.2 부정 증거에는 control이 필요

**개념:**
"불일치 0"을 보고하는 parity 비교는 불일치가 존재할 때 이를 탐지할 수 있어야만 의미가 있다.

**이 PR에서의 적용:**
trace 비교는 MLA rope score를 2배로 키운 control 바이너리에 대해서도 실행되어 decided 불일치 41개 중 6개(14.6%)를 기록했다. 새 gate는 0이었다. 이로써 comparator의 민감도를 가정하지 않고 입증한다.

**일반적 사용 사례:**
- 두 커널 사이의 logit-trace diff.
- quantized 경로의 수치 동등성 테스트.
- 두 arm이 우연히 같은 코드를 실행할 수 있는 A/B 벤치마크 harness.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|---------|-------------|-----------|
| Matrix-absorbed MLA | `kv_b_proj`를 쿼리와 출력 contraction에 접어 attention이 latent에서 실행되게 함 | 이 PR이 multi-token step으로 확장하는 경로 |
| `kv_lora_rank` (`r`) | K와 V가 공유하는 latent 폭 | cost model의 중심 항 |
| Crossover bound | `L < r*d / (r*d/T + 2*r - d)` | 호출당 결정 규칙 |
| `absorbed_prefill` | cost model 아래의 multi-token absorbed 경로 | 카운터가 traced width가 이를 탔음을 증명 |
| MTP | Multi-token prediction; target이 draft block을 verify | 변경 대상 shape 중 하나이며 decode 결과는 불확정 |
| Teacher-forced trace | 고정 토큰 시퀀스를 실행하고 위치별 logits를 비교 | parity 증거 방법 |

### 관련 PR/이슈

- Issue #2408: `perf: Use absorbed MLA attention for multi-token steps below crossover` — 이 PR이 종료하는 이슈.
- PR #907: compressed-latent 캐시 위의 matrix-absorbed MLA decode를 도입했고 `stats` 카운터를 포함한다.
- Issue #899: 한 번도 활성화되지 않은 fused decode 경로. 경로 카운터의 동기.
- Issue #898: Stage 2 split-KV가 사용하는 `paged_attention_merge_states`.
- Blaizzy/mlx-vlm#2150 및 ml-explore/mlx-lm#1817: 이 port가 따르는 upstream cost-model threshold.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 13 |
| 추가 라인 | +512 |
| 삭제 라인 | -93 |
| 추가 테스트 | 6 |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|-------|---------|
| 성능 | 8 | 공유 cost model과 일곱 MLA 호출 지점 |
| 관측성 | 2 | `mla::stats` 경로 기록과 `logit_trace` 카운터/타이밍 |
| 테스트 | 2 | helper 경계 테스트와 warm-cache parity 테스트 |
| 문서 | 2 | `docs/mla-absorbed-decode.md`와 `docs/benchmarks.md` |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `b2131d94` | perf | use absorbed MLA attention for multi-token steps below crossover |

---

## 8. 후속 조치

### 완료 필요

- [ ] per-model parity 테스트를 `glm4_moe_lite` 외 나머지 여섯 MLA family로 확장한다. 특히 branch 재구성이 가장 큰 `deepseek_v32`가 중요하다.
- [ ] MTP speculative decode를 더 많은 run 또는 더 엄격한 harness로 재측정한다. 보고된 샘플은 불확정이었다.

### 모니터링 필요

- `MLXCEL_MLA_ABSORBED=1`에서 서빙 중 `mla::stats`를 관찰한다. verify 형태 step에서 `decompressed`가 증가하면 geometry 기반 bound가 배포 shape와 어긋난다는 뜻이다.
- speculative decoding에서 logit gap 2 미만의 top-1 뒤집힘이 관측된 DeepSeek-V3.2와 GLM-4.7-Flash의 수치 drift 보고를 주시한다.

### 향후 개선 사항

- 현재 bound가 무시하는 캐시 트래픽과 커널 오버헤드도 모델링한다.
- mask 변형이 안정되면 `deepseek_v32`의 중복 latent-attention 호출을 통합한다.
- verify 비중이 큰 워크로드의 end-to-end 서빙 벤치마크를 추가한다.

---

## 부록

### A. 테스트 결과

| 테스트 | 결과 |
|------|--------|
| `cargo test --profile test-fast -p mlxcel-core mla::` | 통과: `T=100000`에서 `L=169` absorbed, `L=171` materialized, `T == L`이면 absorbed 아님, `L == 1`이면 항상 absorbed |
| `models::glm4_moe_lite::tests::four_token_step_on_a_warm_cache_matches_across_both_attention_paths` | 통과: 64-token warm cache 위 4-token step의 full-model logits가 두 branch에서 `2e-2` threshold 아래로 일치 |
| `cargo fmt --check` | 통과 |
| `cargo clippy --lib --tests --example logit_trace --features metal,accelerate -- -D warnings` | 통과 |

**Teacher-forced logit trace** (1024 토큰 뒤 width 4와 16, 새 gate 대 되돌린 `l == 1` gate, `--decided 2`):

| 모델 | Width | 위치 수 | Decided | Decided 불일치 | Top-1 불일치 |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat-4bit-mlx | 4 | 32 | 21 | 0 | 0 |
| DeepSeek-V2-Lite-Chat-4bit-mlx | 16 | 128 | 66 | 0 | 0 |
| GLM-4.7-Flash-4bit | 4 | 32 | 14 | 0 | 1 (gap 0.5) |
| GLM-4.7-Flash-4bit | 16 | 128 | 41 | 0 | 10 (gap 2 미만) |

**Branch 카운터:** traced width가 `absorbed_prefill`을 탔다(DeepSeek-V2-Lite 216회, GLM-4.7-Flash 376회). 되돌린 arm은 0을 기록했다.

**Control:** 동일한 GLM trace를 MLA rope score를 2배로 키운 임시 바이너리에 대해 실행하면 decided 불일치 6/41(14.6%)을 기록하므로 comparator가 실제로 다른 arm을 탐지한다.

### B. 성능 벤치마크

**Chunked-prefill continuation** (1024 cached token 뒤 4 및 16 토큰, `logit_trace` 내부 측정, arm당 세 번 실행, 첫 chunk 제외, cell당 21 샘플):

| 모델 | Width | Before | After | 변화 |
|---|---|---|---|---|
| DeepSeek-V2-Lite | 4 | 17.57 ms | 14.80 ms | -15.8% |
| DeepSeek-V2-Lite | 16 | 31.52 ms | 25.34 ms | -19.6% |
| GLM-4.7-Flash | 4 | 46.45 ms | 21.83 ms | -53.0% |
| GLM-4.7-Flash | 16 | 60.31 ms | 47.44 ms | -21.3% |

**GLM-4.7-Flash MTP speculative decode** (4-bit target와 `mlxcel split-mtp --q-bits 4`로 `zai-org/GLM-4.7-Flash`에서 분리한 drafter, block 2, 1621-token prompt, `-n 64 --profile`, arm당 세 번 실행):

| Arm | Run 1 | Run 2 | Run 3 | 중앙값 |
|---|---|---|---|---|
| Before | 93.14 tok/s | 93.81 tok/s | 92.92 tok/s | 93.14 tok/s |
| After | 80.99 tok/s | 98.42 tok/s | 98.55 tok/s | 98.42 tok/s |

첫 after-run은 outlier다. 중앙값 차이는 +5.7%이나 평균은 노이즈 범위이므로 block-2 MTP decode는 명확한 개선이 아니다. verify window가 짧고 MoE weight 트래픽이 지배한다.

### C. 검증되지 않은 항목

PR 본문은 전체 workspace 실행을 나열하지 않는다. 보고된 것은 `cargo test --profile test-fast -p mlxcel-core mla::`, 단일 `glm4_moe_lite` parity 테스트, `cargo fmt --check`, 그리고 `--lib --tests --example logit_trace`로 범위를 한정한 clippy 호출이다. 이슈의 verification 블록에는 `cargo test --workspace --profile test-fast --features metal,accelerate`와 `cargo clippy --workspace --all-targets`도 있었으나 PR 본문은 이를 보고하지 않는다.

per-model parity 테스트는 `glm4_moe_lite`에만 있다. 나머지 여섯 지점은 helper 단위 테스트와 두 체크포인트, 두 width의 teacher-forced trace로 커버되며 family별 both-branch logits 비교는 없다. 특히 `deepseek_v32`의 mask 재구성에는 직접적인 branch-parity 테스트가 없다.

MTP speculative-decode 벤치마크는 명시적으로 불확정이다. 첫 after-run이 outlier였고 평균은 노이즈 범위다. end-to-end 서빙 처리량이나 메모리 측정은 보고되지 않았다. 타이밍 수치는 `logit_trace` 내부의 traced-forward wall time이다.

### D. 참고 자료

- `docs/mla-absorbed-decode.md` — absorbed MLA 설계와 개정된 multi-token 섹션.
- `docs/benchmarks.md` — `MLXCEL_TRACE_MLA_STATS`와 chunked-prefill trace 사용법.
- Issue #2408 및 upstream Blaizzy/mlx-vlm#2150, ml-explore/mlx-lm#1817.
