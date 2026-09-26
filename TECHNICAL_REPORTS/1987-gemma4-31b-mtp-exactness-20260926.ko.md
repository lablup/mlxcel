# 기술 보고서: PR #1987, Gemma 4 31B MTP 정확성 복구

**작성일**: 2026-09-26
**상태**: 구현, 리뷰, 로컬 검증 완료. 머지 대기 중.
**언어**: Rust, Python, Markdown
**위험도**: 높음

## 요약

이 변경은 Apple Silicon에서 Gemma 4 31B MTP의 temperature 0 정확성을 복구한다. 기존 block forward와 single-token decode chain은 full attention 폭, sliding cache의 reduction 순서, 긴 프롬프트의 prefill chunk 구성이 달라 QAT와 non-QAT 31B 체크포인트 모두 startup gate에서 거절됐다. 수정된 경로는 각 query row마다 decode attention을 재현하고 classic decode의 physical ring 순서를 보존하며 scheduler와 같은 prefill chunk를 사용한다. 대체 산술은 측정된 31B geometry에만 적용된다.

M5 Max 측정에서 수정된 QAT와 non-QAT teacher-forced trace는 256개 위치 모두에서 top-1이 일치했다. 대상별 3개 프롬프트의 256-token free-running 출력과 513-token 긴 프롬프트 출력도 corrected classic decode와 byte 단위로 일치했다. GB10 호스트에 접근할 수 없어 CUDA는 검증하지 못했다.

## 1. 문제 정의

Gemma 4 31B MTP는 여러 candidate token을 한 번의 forward로 검증하지만 classic decode는 한 번에 한 token씩 진행한다. `qmv_wide`를 꺼도 두 shape가 서로 다른 reduction을 선택했다. QAT는 full-attention layer 41에서, non-QAT는 layer 5에서 처음 달라졌다. 따라서 원인은 QAT 체크포인트의 8-bit MLP에 한정되지 않는다. 긴 context probe에서는 두 번째 차이도 발견됐다. speculative buffer는 key를 chronological 순서로 보관하지만 classic decode는 rotating ring의 physical 순서로 reduction한다.

긴 server prompt에서는 세 번째 차이가 나타났다. 측정한 server의 classic 경로는 512-token chunk로 prefill하지만 MTP adapter는 6,449-token prompt 전체를 한 번에 forward했다. 이 prefix state 차이 때문에 attention 수정 뒤에도 기본 classic serving과 출력이 달랐다.

수정 전 exactness gate는 안전하게 classic decode로 fallback했으므로 기본 설정에서 의도한 MTP 경로를 사용할 수 없었다. `MLXCEL_MTP_ALLOW_INEXACT=1`로 강제하면 계약을 우회하며 near-tie 위치에서 greedy 출력이 달라질 수 있었다.

## 2. 기술적 검토 사항

### 2.1 정확성 및 호환성

산술 수정은 query head 32개, dimension 256인 sliding KV head 16개, dimension 512인 global KV head 4개인 Gemma 4에만 적용된다. B=1 linear verification만 수정 경로를 사용한다. Batched MTP는 classic decode로 fallback하고 nonlinear tree round는 지원되는 linear dispatch 경로를 사용한다. Unified 12B는 512-wide full head를 공유한다는 이유만으로 이 수정을 적용받지 않는다.

공용 `RotatingKVCache` metadata 경로는 Gemma 3, AFMoE, Gemma 4, Muse Glimmer가 사용한다. 새 reference-ring anchor는 rollback, compaction, scheduler slice 재구성, snapshot restore, detach/adopt를 거쳐 유지된다. Unbuffered snapshot의 호환성은 유지된다. 새 anchor가 없는 buffered snapshot은 이전 reduction 순서를 안전하게 복원할 수 없어 거절한다.

Gemma 3 4B와 Qwen 2.5 7B의 classic before/after smoke에서 출력이 동일했다. Qwen 3.8 27B MTP 검사도 수정 전후 출력과 acceptance count가 같았다. 92 round에서 proposal 274개 중 164개를 accept했다. 로컬 Qwen 3.6 target과 drafter의 hidden size가 호환되지 않아 Qwen 3.6 MTP는 검증하지 못했다.

### 2.2 성능

수정 경로는 정확한 M=1-compatible reduction을 위해 일부 병렬 attention 작업을 줄인다. Free-running 측정은 arm과 prompt마다 1회였고 cooldown 없이 고정 순서로 실행했으며 thermal pressure는 `Fair`였다. 따라서 출력 일치는 증명하지만 안정적인 throughput 변화는 증명하지 않는다. 6개 QAT/non-QAT prompt에서 corrected MTP는 corrected classic의 0.85x에서 1.38x였다. MTP 사용 여부는 계속 adaptive profitability policy가 결정한다.

강화된 31B startup probe는 QAT 1회 측정에서 약 5.51초가 걸렸고 기존 short probe는 약 2.26초였다. Failure localization과 긴 buffered draw는 request forward가 아닌 startup에서 실행된다. 다른 Gemma geometry는 기존 short probe를 유지한다.

### 2.3 보안

네트워크 surface, 인증 경로, unsafe block은 추가되지 않았다. Snapshot restore는 cache state를 변경하기 전에 ring origin, logical bound, K/V rank, shape, dtype을 검증한다. 완료된 31B MTP request의 buffered rotating cache는 ordinary suffix prefill이 같은 layout을 재현할 수 없으므로 automatic prompt cache에 기증하지 않는다.

## 3. 기술적 선택과 그 이유

### 3.1 수치적 근사 허용 대신 decode attention 재현

구현은 각 verify query를 M=1에서 보이는 key prefix로 자른다. Full-attention row는 maskless single-query dispatch를 사용하고 sliding row는 visible window를 classic decode의 physical ring 순서로 재배치한다. 기존 block reduction을 유지하면 병렬성은 더 높지만 byte identity 계약을 만족하지 못한다.

### 3.2 Reference ring 위치를 cache metadata로 보존

현재 speculative storage만으로 classic ring cursor를 계산하면 rollback, compaction, snapshot restore, detach/adopt 이후 정확하지 않았다. Cache는 speculative buffering 시점의 logical offset과 physical cursor를 기록하고 이후 cursor를 이 anchor로부터 계산한다.

### 3.3 Adapter 경계에서 prefill geometry 일치

측정된 31B geometry에서 MTP adapter는 scheduler의 configured prefill chunk size를 사용하고 마지막 chunk에서만 assistant seed를 capture한다. Offline generation은 classic generation과 같은 `MLXCEL_PREFILL_CHUNK` 정책을 사용한다. 다른 Gemma variant는 기존 prefill 경로를 유지한다.

### 3.4 authoritative probe 실패 뒤에만 localization 수행

일반 startup probe가 최종 verdict를 유지한다. 실패할 때만 diagnostic rerun이 attention, MLP, layer, norm, head 출력을 capture하고 첫 divergence의 layer, kind, sub-operation을 decline reason에 추가한다. Request forward는 capture scope에 들어가지 않는다.

## 4. 구현 상세

```text
Classic decode:       prefill chunks -> M=1 query -> physical rotating ring
기존 MTP verification: whole prefill  -> M=K query -> chronological buffered keys
수정된 31B MTP:       prefill chunks -> K x M=1-compatible rows -> reference ring order
```

구현에는 query-row attention helper, failure-only probe capture, cache anchor 직렬화와 검증, scheduler prefill-size 전달, B=1/tree capability guard, 회귀 테스트, Gemma-aware teacher-forced trace mode, 재현 가능한 측정 harness와 machine-readable evidence가 포함된다.

## 5. 학습 포인트

Attention mask가 visible key set을 보존해도 산술은 달라질 수 있다. 값이 모두 0인 mask도 다른 kernel을 선택할 수 있으며 chronological key 순서와 physical-ring key 순서는 서로 다른 floating-point reduction을 만든다. 정확한 speculative decoding에는 shape, visible width, mask 존재 여부, physical order, prefix 구성까지 일치해야 한다.

Model family 이름만으로 수치 수정을 적용하기에는 범위가 넓다. Gemma 4 Unified 12B는 512-wide full head를 공유하지만 query/KV geometry와 forward behavior가 다르다. Diagnostic long draw에서 별도의 12B sliding-attention mismatch가 발견됐으며 issue #1986에서 추적한다. 이 PR은 12B long-context exactness를 주장하지 않는다.

## 6. 변경 요약

| 항목 | 값 |
|---|---:|
| 구현 커밋 변경 파일 | 28 |
| 구현 커밋 추가 라인 | 2,409 |
| 구현 커밋 삭제 라인 | 78 |
| 주요 커밋 | `fb5dc631` |

| 분류 | 요약 |
|---|---|
| 정확성 | 31B B=1 linear MTP의 attention과 prefill 동작을 exact하게 복구 |
| 진단 | 첫 divergence localization과 actionable decline reason 추가 |
| Cache 안전성 | cache lifecycle 전체에서 reference-ring metadata 보존 및 검증 |
| 테스트 | attention, probe, cache, adapter, scheduler, prefill 회귀 테스트 추가 |
| 문서 | QAT/non-QAT 측정, evidence JSON, 환경 변수 지침, model scope 설명 추가 |

## 7. 후속 조치

- Issue #1986에서 Unified 12B long-context mismatch를 조사한다.
- GB10/CUDA 호스트 접근이 복구되면 동일한 backend-neutral 검사를 실행한다.
- 성능을 주장하기 전에 arm 순서 randomization, cooldown, 반복 샘플로 corrected throughput을 재측정한다.

## 부록

### A. 측정 증거

- 수정된 QAT와 non-QAT teacher-forced trace: 256개 위치에서 top-1 disagreement 0, reference-choice logit delta 0.
- QAT/non-QAT free-running triple 6개: baseline classic, corrected classic, corrected MTP의 256-token 출력이 byte-identical.
- 수정된 QAT 긴 실행: input 6,449 tokens, generated 513 tokens, accepted/proposed 280/698, verify당 emitted tokens 2.1974, corrected classic과 byte-identical.
- Qwen 3.8 MTP before/after: 256-token 출력과 92 round의 acceptance 164/274가 동일.
- Repository `make verify` gate가 통과했고, ignored buffered-cache donation 회귀 테스트도 새로 빌드한 root test binary에서 직접 실행해 1/1 통과했다.
- 최종 Unified 12B startup은 inexact override 없이 automatic narrow retry 뒤 기존 short gate를 통과했다. Health 도달 14.09초는 정확성 확인이며 성능 결과가 아니다.

### B. 한계

CUDA는 측정하지 못했다. Throughput 샘플은 변동하는 thermal 조건에서 수행한 diagnostic single run이다. Issue의 원래 긴 prompt를 확보하지 못해 재현 가능한 6,449-token workload를 사용했으며 이를 원래 workload로 제시하지 않는다.

### C. 참고 자료

- Issue #1983: Gemma 4 31B MTP exactness
- Issue #1279: 31B exactness policy 및 benchmark qualification
- Issue #1986: Unified 12B long-context exactness 후속
- `docs/benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.md`
- `docs/benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.json`
