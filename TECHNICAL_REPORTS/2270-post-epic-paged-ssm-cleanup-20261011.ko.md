# 기술 보고서: PR #2270 - 에픽 이후 paged 및 SSM 정리

**날짜**: 2026-10-11
**상태**: CUDA(GB10)에서 구현 및 검증 완료
**언어**: C++, Rust, Python, Bash, Markdown
**위험 수준**: 중간

## 요약

PR #2270은 이슈 #2230의 에픽 이후 작업 세 가지를 완료합니다. 사용하지 않는 dense-pointer 및 rotating paged 호환 표면을 제거하고, pooled decode 테스트에서 그 역할을 독립적인 float32 attention oracle로 교체했습니다. 또한 완전한 paired/null serving sweep으로 Gemma 3와 Llama 4의 model-owned storage 플래그를 결정하고, 수치가 처음 어긋나는 단계에서 CUDA bf16 SSM parity를 수정했습니다.

SSM 결함의 원인은 고정된 MLX type-promotion 표가 `float32 + bfloat16` 결과를 bf16으로 선택하는 데 있었습니다. `compute_dt_compiled`는 `dt`를 float32로 변환했지만 `dt_bias`는 bf16으로 남겨 두었으므로, 어떤 backend kernel이 값을 받기 전에 compiled softplus 결과가 반올림되었습니다. 두 피연산자를 모두 float32로 변환하여 허용 오차를 바꾸지 않고 Metal, CUDA, HIP에서 문서화된 recurrence를 복원했습니다. 실제 Granite checkpoint의 deterministic 비교에서는 fused CUDA kernel과 graph path가 생성한 64개 token ID가 모두 같았습니다.

Storage sweep은 fresh server 60개 arm과 성공한 client cell 180개를 실행했습니다. 두 family 모두 여섯 prompt/concurrency cell 중 어느 것도 repeated-paged null spread를 어느 방향으로도 벗어나지 못했으므로 두 `supports_paged_decode_backend()` 플래그는 계속 `true`입니다.

## 1. 문제

### 1.1 오래된 호환 표면

#2210과 #2217 이후 production model은 dense-pointer 또는 rotating paged decode 호환 경로를 더 이상 호출하지 않았습니다. C++ entry point, CXX bridge declaration, Rust wrapper와 validator, cache metadata, 직접 FFI test는 서로를 유지하기 위해서만 남아 있었습니다. 네 개의 활성 pooled-kernel test가 기존 dense fallback을 oracle로 사용하고 있었으므로, 독립 reference를 먼저 마련해야 이 코드를 제거할 수 있었습니다.

### 1.2 미결정 model-owned storage 기본값

Gemma 3와 Llama 4는 자체 KV row를 소유하고 dense storage에서 attention을 계산합니다. 그러나 paged 지원을 알리는 플래그 때문에 multi-slot server는 paged storage를 할당하고 각 decode step을 여기에 mirror합니다. 기존 Gemma 3 수치는 실행 간 분산과 겹쳤고 Llama 4 측정은 없었습니다. 따라서 family별 플래그는 paired dense-versus-paged delta와 repeated-paged null arm으로 결정해야 했습니다.

### 1.3 CUDA bf16 SSM parity 실패

Granite bf16 SSM case는 float32 graph reference 대비 normalized state RMS 약 `1.8e-3`에서 `2.5e-3`으로 실패했지만, f32 case와 bf16 activation/f32 `A_log`를 쓰는 Nemotron-H case는 통과했습니다. 같은 실패가 에픽 이전에도 존재했습니다. Acceptance criteria는 production code를 바꾸기 전에 `dt'`, decay, state를 측정하고 근거 없이 tolerance를 넓히지 않도록 요구했습니다.

## 2. 구현

### 2.1 호환 경로 제거와 독립 oracle

PR은 dense 및 rotating 호환 C++ 구현과 declaration, bridge entry, Rust fallback wrapper와 validator, `PagedDecodeMetadata`, `RotatingPagedDecodeMetadata`, 그리고 해당 API만 검증하던 test를 삭제합니다. Pooled empty-batch case는 현재 pooled 동작을 대상으로 유지합니다.

네 pooled decode test는 이제 test 전용 `dense_reference_decode_attention` helper를 사용합니다. 이 helper는 query, key, value를 float32로 변환하고 grouped-query attention을 위해 KV head를 반복한 뒤 `softmax(q k^T * scale) v`를 계산하고 reference row를 연결합니다. 각 test는 원래의 absolute RMS `<5e-3` assertion을 유지하고 normalized RMS `<5e-3`를 추가합니다. Pooled output에 임시로 `1.01`을 곱하면 네 test가 모두 normalized RMS 약 `0.010`으로 실패했으며, mutation은 commit 전에 제거했습니다.

### 2.2 Model-owned storage 측정

Benchmark는 arm마다 fresh `mlxcel serve --parallel 8` process를 시작합니다. 두 모델, 512 및 4096 token prompt, 다섯 round, 회전 순서의 `paged-a`, `dense`, `paged-b` arm을 측정합니다. 각 arm은 128개 token 생성에 대해 concurrency 1, 4, 8을 실행합니다. 바깥 command는 `gpu-lock`을 유지하고, 안쪽 fail-closed gate는 지속적인 quiet-host 구간, wait 전후의 빈 port, CI/build/foreign-model activity 없음, 충분한 memory, 깨끗한 NVIDIA driver counter를 요구합니다.

Analyzer는 completion marker, clean checkout 및 artifact provenance, 일치하는 configured/built MLX commit, 측정 당시 script hash, 순서가 정확한 60-arm schedule, 승인된 quiet-gate record 61개, backend marker가 있는 nonempty server log 60개, 정확히 180개의 성공한 client key가 없으면 decision을 계산하지 않습니다. `paged-a`가 paired baseline이고 `paged-b / paged-a - 1`이 null spread입니다. Dense는 median paired delta가 `max(0, null_high)`보다 클 때만 cell에서 이기고, paged는 `min(0, null_low)`보다 작을 때만 이깁니다.

### 2.3 공유 SSM 정밀도 수정

첫 번째 큰 편차는 compiled `dt'`에서 나타났으며 output dtype은 bf16이었습니다. 고정된 MLX promotion 표가 원인을 설명했습니다. Graph는 float32 `dt`와 bf16 `dt_bias`를 더했고, 결과가 표의 bf16 항목을 따랐습니다. PR은 compiled add, softplus, clip 전에 `dt_bias`를 `dt`와 함께 명시적으로 float32로 변환합니다.

`compute_dt_compiled`는 backend dispatch 전에 실행됩니다. 따라서 backend별 code를 복제하지 않고 Metal, CUDA, HIP 모두에 동일한 float32 `dt'` semantics를 제공합니다. CUDA recurrence kernel, float32 reference, tolerance는 변경하지 않았습니다.

## 3. 측정 결과

### 3.1 SSM 중간 단계 편차

Granite seed 2069에서 안정적인 float32 graph 대비 normalized RMS/max 편차는 다음과 같습니다.

| 단계 | 수정 전 | 수정 후 |
|---|---:|---:|
| Compiled `dt'` | `1.957140831e-3 / 5.875451502e-3` | `2.890811210e-8 / 2.002812756e-7` |
| Compiled decay | `3.008783834e-3 / 1.075881018e-2` | `6.809246822e-10 / 4.717584582e-9` |
| Kernel state | `2.532402801e-3 / 4.127457397e-2` | `2.838613609e-8 / 1.701009653e-6` |

수정 후 downstream 편차가 약 다섯 자릿수 줄었고, 이전에 실패하던 두 CUDA test 모두 기존 budget 안으로 들어왔습니다.

### 3.2 Storage sweep

아래 속도는 aggregate generated token/s입니다. Paired 열은 다섯 round의 median과 전체 paired range이고 null 열은 repeated-paged range입니다.

| 모델 | Prompt | C | paged-a | dense | paged-b | Dense paired delta | Null spread | 결정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gemma 3 | 512 | 1 | 154.4 | 154.0 | 154.0 | +0.52% (-2.76..+3.63) | -2.37..+3.84% | unresolved |
| Gemma 3 | 512 | 4 | 361.0 | 363.6 | 360.6 | +1.29% (-2.11..+2.49) | -1.35..+3.52% | unresolved |
| Gemma 3 | 512 | 8 | 194.9 | 197.2 | 194.4 | +1.16% (+0.25..+6.37) | -0.88..+3.80% | unresolved |
| Gemma 3 | 4096 | 1 | 114.6 | 114.4 | 113.0 | -0.18% (-1.82..+0.26) | -2.33..+1.31% | unresolved |
| Gemma 3 | 4096 | 4 | 219.4 | 216.6 | 218.6 | -0.60% (-2.74..+0.54) | -3.45..+0.50% | unresolved |
| Gemma 3 | 4096 | 8 | 184.4 | 185.8 | 184.0 | +0.49% (-0.53..+1.98) | -3.13..+1.91% | unresolved |
| Llama 4 | 512 | 1 | 11.7 | 11.8 | 11.8 | +0.85% (+0.00..+83.08) | -0.85..+81.54% | unresolved |
| Llama 4 | 512 | 4 | 31.0 | 30.8 | 30.7 | +0.65% (-3.53..+2.61) | -0.97..+3.53% | unresolved |
| Llama 4 | 512 | 8 | 17.8 | 17.9 | 17.7 | +0.00% (-1.69..+13.29) | -1.13..+10.76% | unresolved |
| Llama 4 | 4096 | 1 | 8.0 | 8.0 | 8.0 | +1.25% (+0.00..+1.27) | +0.00..+2.53% | unresolved |
| Llama 4 | 4096 | 4 | 22.9 | 23.1 | 23.0 | +0.44% (-0.43..+2.25) | -2.99..+4.05% | unresolved |
| Llama 4 | 4096 | 8 | 14.7 | 14.7 | 14.5 | +0.68% (-0.68..+3.55) | -2.03..+2.84% | unresolved |

어느 cell도 clear-win 조건을 충족하지 않습니다. Multi-slot server의 `auto`와 명시적인 `paged` request는 두 model family 모두에서 paged storage를 계속 선택하며, 여기에는 pool allocation과 per-step mirror가 포함됩니다. Attention path는 계속 dense row를 사용합니다.

### 3.3 관찰된 첫 arm 이상값

Llama 4의 첫 512-token round는 `paged-a`로 시작했습니다. Concurrency 1에서 이 arm은 aggregate 6.5 tok/s와 mean TTFT 12690.4 ms를 기록했습니다. 이어진 dense와 repeated-paged arm은 aggregate 11.9 및 11.8 tok/s, TTFT 4072.6 및 4128.2 ms였습니다. Round 2부터 5까지는 모든 arm에서 약 11.7에서 11.9 aggregate tok/s였습니다.

따라서 첫 round는 +83.08% dense paired delta와 +81.54% null delta를 만듭니다. 모든 record를 결과에 그대로 유지했습니다. Host gate는 arm을 승인했지만 evidence는 원인을 입증하지 않습니다. Cold 또는 warm-up 효과는 하나의 가설일 뿐입니다. 결과를 본 뒤 run을 필터링하거나 다시 실행하지 않았습니다. 사전 정의한 rule에서 paired median +0.85%는 null upper bound +81.54%를 넘지 못하므로 이 cell은 unresolved입니다.

## 4. 검증

| 영역 | 결과 |
|---|---|
| `src/`의 제거 대상 호환 표현식 | 일치 항목 없음 |
| 네 focused pooled CUDA test | Passed |
| 임시 pooled-output `1.01` mutation | 네 test 모두 실패, mutation 제거 |
| Focused CUDA SSM parity | 이전 두 실패를 포함해 5 passed |
| 실제 Granite checkpoint | Fused kernel과 graph가 greedy token ID 64/64개 동일하게 생성 |
| Granite token sequence SHA-256 | `9df6f4fc2abbfcb121fa5dca3ad1e140e8db5b67d28118a905b93022f8905829` |
| CUDA release build | 19분 43초에 성공 |
| GPU lock 아래 전체 `make verify-test-cuda` | Exit 0, 실패 0 |
| Workspace/all-target CUDA clippy와 formatting | Passed |
| Dtype-key, port-dispatch, llama-compat contract | Passed |
| Storage analyzer | 60 arm과 성공한 180 cell 검증 |
| Benchmark protocol fixture | Warning을 error로 처리하고 9 passed |
| Raw storage 및 SSM manifest | Passed |
| 독립 구현, 보안, evidence review | 남은 finding 없음 |

Granite 비교는 prompt `Explain the Mamba state update.`, greedy token 64개, temperature 0, deterministic SDPA를 사용했습니다. Graph arm은 `MLXCEL_SSM_CUDA_KERNEL=0`을 설정했습니다.

## 5. Provenance와 한계

Storage run 당시의 `environment.txt`는 clean benchmark checkout `35da2dbe`를 기록합니다. 측정한 release binary는 production commit `2cb92404`에서 build했습니다. Binary에는 commit이 내장되어 있지 않으므로, 이 연결은 보존된 build-session context, artifact hash와 mtime, ancestry, 정확한 intervening path set으로 사후 재구성했습니다. `2cb92404`는 `35da2dbe`의 ancestor이고, 그 사이 변경은 네 benchmark protocol file로 제한됩니다. Rust, C++, model, build-definition source에는 차이가 없습니다. 원래 environment와 측정 당시 script byte는 그대로 유지하고, 별도의 derived provenance record가 증명과 한계를 기록합니다.

Committed release log는 release build가 19분 43초에 완료되었음을 증명하지만 HEAD는 출력하지 않습니다. Raw package는 모든 client/server output, 승인된 host-gate record 61개, ordered schedule, analyzer result, completion marker, script hash, MLX pin, derived provenance, SSM stage output, Granite comparison, checksum을 보존합니다.

## 6. 호환성, 보안, 성능

제거한 API에는 production caller가 없었습니다. SSM 수정은 bf16 bias가 의도하지 않은 반올림을 만들던 numeric preprocessing만 바꾸며 token당 host synchronization이나 새 dispatch를 추가하지 않습니다. 새로운 runtime dependency나 public option도 없습니다.

두 model-owned flag가 계속 enabled이므로 serving 동작은 유지됩니다. Benchmark는 관찰된 반복성 범위 안에서 명확한 throughput 승자를 보여 주지 않습니다. 독립 security review는 production-code 문제를 찾지 못했습니다. Benchmark review는 처음에 fail-open host/provenance 처리를 발견했으며, 최종 gate와 analyzer는 계산 전에 nonquiet, incomplete, reordered, dirty, backend-substituted evidence를 거부합니다.

## 7. 관련 기록

- 이슈 #2230: 에픽 이후 정리 요구 사항
- PR #2270: 구현과 evidence
- `docs/benchmark_results/model-owned-storage-gb10-2026-10-11.md`: 날짜가 기록된 storage 결정
- `docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/`: 완전한 storage protocol 및 raw output
- `docs/benchmark_results/data/ssm-bf16-parity-gb10-2026-10-11/`: 원래 stage 편차 및 Granite parity artifact
