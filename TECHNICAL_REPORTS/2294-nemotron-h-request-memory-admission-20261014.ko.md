# PR #2294: 검증된 메모리 예산으로 Nemotron-H 요청 승인

**작성일**: 2026-10-14
**상태**: 구현 및 검증 완료, 남은 CI job 대기
**위험도**: HTTP 스트리밍과 GPU 실행 사이에서 동작하며 의도적으로 fail-closed하는 보호 장치이므로 높음

## 요약

PR #2294는 Metal에서 단일 슬롯으로 동작하는 text-only Nemotron-H serving에 보수적인 요청 메모리 admission을 추가한다. checked allocation envelope가 런타임 한도와 `MLXCEL_REQUEST_MEMORY_BUDGET` 중 낮은 값을 넘으면 GPU prefill이나 SSE header 전송 전에 typed HTTP 503을 반환한다. context overflow는 계속 HTTP 400이 우선하며, 지원하지 않는 모델과 실행 모드는 기존 동작을 유지한다.

## 1. 문제 정의

PR #2290은 Nemotron-H SSD prefill scratch를 제한했지만 유효한 context 안의 요청도 resident weights와 cache, output KV, recurrent state, activation, logits, attention fallback, bounded SSD workspace 때문에 Metal 메모리를 초과할 수 있었다. 이런 할당 실패는 예측 가능한 단일 요청을 거절하는 대신 serving process를 종료할 수 있다. Startup memory estimate로는 실제 호출 지점의 요청을 판단할 수 없다. prompt, 실제 prefill plan, 요청 output, retained allocation, 설정된 runtime ceiling은 prefill 직전에만 함께 알 수 있기 때문이다.

이 보호 장치는 universal OOM guarantee가 아니다. 다른 process, driver allocation, fragmentation, 예상하지 못한 kernel workspace, admission 이후의 allocation은 여전히 메모리를 소비할 수 있다.

## 2. 기술적 검토 사항

### 2.1 Admission 경계와 오류 계약

최종 검사는 앞선 width-one 작업이 끝난 뒤 scheduler thread에서 request GPU work 전에 실행된다. Provider는 HTTP route가 streaming header를 확정하기 전에 one-shot admission event를 기다린다. 거절 시 `error.type`과 `error.code`가 `request_memory_budget_exceeded`이고 `estimated_bytes`와 `budget_bytes`를 포함하는 HTTP 503을 반환한다. 기존 context 검증은 우선순위를 유지하며 HTTP 400을 반환한다. Cancellation과 worker disconnect 경로는 handshake를 완료하거나 취소하므로, 중단된 요청이나 지원하지 않는 요청이 waiter를 영구 대기시키지 않는다.

리뷰에서 두 route-level gap을 닫았다. Guarded request는 final single-request check를 우회하던 batched prefill의 sequential fallback에 진입할 수 없으며, unsupported path는 자신이 발생시키지 않는 admission event를 기다리지 않고 즉시 acknowledge한다. Legacy `--no-batch`는 순차 실행을 유지하면서 같은 size-one scheduler guard를 사용한다.

### 2.2 Checked estimate와 budget

Pure estimator는 normalized loaded `NemotronHConfig`, loaded embedding output dtype, 실제 cold `PrefillPlan` 조각, architecture-aware KV geometry, 현재 MLX memory snapshot을 사용한다. 모든 곱셈과 합산은 checked arithmetic을 사용하며 dimension이나 budget을 알 수 없거나 dtype이 잘못되거나 overflow가 발생하면 fail closed한다. Envelope는 현재 active allocation, growth와 snapshot ownership을 포함한 cold attention KV, recurrent 및 convolution state, layer별 activation, logits, MoE intermediate, 보수적인 float32 attention score, model internal chunk 크기의 float32 SSD buffer를 계산한다.

Default budget은 device recommendation, mlxcel이 실제 적용한 wired cap, allocator cap의 양의 유한값 중 최솟값이다. `MLXCEL_REQUEST_MEMORY_BUDGET`은 기존 binary size grammar를 사용하지만 이 ceiling을 낮출 수만 있다. 명시적으로 설정한 empty, zero, `none`, negative, nonfinite, overflow 값은 startup을 중단한다.

### 2.3 Residency와 cache 정책

Metal Nemotron-H weight는 warmup이 꺼져 있어도 shared construction boundary에서 평가된다. 따라서 lazy graph 때문에 첫 요청이 실제보다 작은 resident baseline을 관측하지 않는다. Constructor는 fused checkpoint adapter와 VLM 또는 VoiceChat에서 포함해 사용하는 경우에도 적용되지만, admission은 eligible text serving에서만 활성화된다. Arbitrary-rank float32 workspace가 모델링되지 않은 runtime unfused LoRA는 startup에서 거절하고 staged adapter state를 비운다.

Guarded request는 cold prefill하며 admission 전에 prompt snapshot을 restore하거나 donate하지 않고 background prompt-cache warmup도 건너뛴다. Retained snapshot은 active memory에 계속 포함된다. 이 보수적 정책은 optimistic cache reuse나 fused kernel을 사용하면 실행 가능한 요청도 거절할 수 있지만, 최종 측정 전에 request work를 실행하지 않는다.

### 2.4 범위와 호환성

정책은 effective width one인 local, non-speculative, text-only Metal Nemotron-H로 제한된다. CUDA, ROCm, CPU, 다른 family, multimodal input, tensor 또는 pipeline parallel, disaggregated worker handoff, speculative decoding, multi-slot reservation은 기존 정책을 유지한다. 새 dependency나 data format 변경은 없으며 model numeric도 바꾸지 않는다.

## 3. 기술적 선택과 그 이유

| 결정 | 선택한 방식 | 선택하지 않은 방식 | 트레이드오프 |
|------|-------------|--------------------|-------------|
| 검사 위치 | Production prefill 호출 지점의 최종 scheduler admission | Startup estimate 또는 HTTP-only heuristic | 실제 request geometry와 resident state를 쓰지만 HTTP-worker handshake가 필요함 |
| Cache credit | Cold KV와 retained active allocation을 계산 | Prompt snapshot hit가 요청 크기를 줄인다고 가정 | 예측 가능한 안전성을 얻는 대신 false rejection과 cache-hit latency 손실 가능 |
| Override | 알려진 runtime limit보다 낮추기만 가능한 cap | Environment variable로 budget을 높이거나 새로 생성 | 더 엄격한 정책은 시험할 수 있지만 device limit을 우회할 수 없음 |
| Unknown value | Fail closed | Geometry, budget, arithmetic을 알 수 없을 때 admit | 안전한 실패지만 잘못된 설정이나 unsupported geometry는 service를 거절할 수 있음 |
| Runtime LoRA | Unfused adapter를 construction에서 거절 | Arbitrary-rank workspace를 추측 | Workspace 모델이 생기기 전까지 명확한 startup failure 제공 |

## 4. 검증

- Request-memory suite: 26개 통과. Budget boundary, output growth, checked overflow, invalid override, cache accounting, context precedence, cancellation, zero-forward rejection, unsupported path, batched-prefill bypass, legacy serving, pre-SSE error를 포함한다.
- Nemotron-H suite: 23개 통과, 기존 3개 ignored. Runtime-LoRA rejection과 staging cleanup을 포함한다.
- Focused provider 및 native-route binary에서 각각 30개와 34개 test가 통과했다. 이 test들은 26개 request-memory test와 중복되므로 unique case로 합산하지 않는다. 5개 generation dialect와 provider admission lifecycle을 포함한다.
- `cargo check --tests --features metal,accelerate`는 51.50초에 통과했고, `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`는 5분 5초에 통과했다. 알려진 C++ unused-capture warning은 Rust Clippy gate를 실패시키지 않았다.
- `cargo build --release --features metal,accelerate`는 측정 source에서 9분 42초에 통과했고 in-place-setter 수정 뒤에도 8분 59초에 통과했다.
- Reviewed source의 real-checkpoint lifecycle은 normal fresh server 3개, `--no-warmup` server 1개, legacy `--no-batch` server 1개에서 통과했다. Post-setter normal 및 legacy lifecycle smoke도 같은 rejection/recovery 동작을 반복했다. Qwen 2.5 7B second-family smoke에서는 setter 수정 전후 모두 streaming과 non-streaming request가 계속 admit되었다.
- GitHub two-host logical pipeline job은 unoptimized worker에 추가된 by-value `BatchScheduler` builder frame 하나 때문에 stack overflow를 보고했다. Commit `9d2a1584`는 admission 동작을 바꾸지 않고 이를 in-place setter로 교체했으며 targeted test 26개가 통과했다. 이전에 실패한 logical job도 수정된 commit에서 통과했다. 전체 workflow와 수정된 release build에는 아직 대기 중인 job이 있어 fully green으로 보고하지 않았다.

## 5. 벤치마크 근거

추적되는 benchmark record는 `docs/benchmark_results/nemotron-h-request-memory-m1ultra-2026-10-13.md`이고 machine-readable artifact는 같은 이름의 `data/` directory에 있다. M1 Ultra 128 GB의 초기 calibration에서 32 GiB budget 아래 6,001-token 요청이 admit되었고 estimate는 27.34 GiB, 측정된 process peak는 17.8 GiB였다. Harness는 이 실행을 rejection evidence로 잘못 보고하지 않았다. 낮은 cap을 사용한 중간 실행은 calibration으로 보존하지만 final routing 및 cache review fix 전 결과이다.

최종 reviewed binary(`1c4673addeb123b8d3ef2758f69dc0c54811d1d061911dbcb86aa486cc6065c5`)로 normal fresh-server 3회 반복을 실행했다. Long streaming 및 non-streaming request 12건은 모두 SSE 전에 typed 503을 반환했고, 바로 뒤의 short request 12건과 medium request 3건은 모두 200을 반환했다. Beyond-context request 3건은 400 우선순위를 유지했고 health는 계속 200이었다. Medium estimate는 세 번 모두 22.328 GiB이고 process physical peak는 17.6 GiB였다. Long estimate는 27.337–27.351 GiB, short estimate는 17.813–17.865 GiB였으며 long-rejection process peak는 16.9 GiB였다. 1분 host load가 6.44, 6.90, 11.01이었으므로 이는 latency 주장이 아니라 lifecycle 및 safety 근거이다.

No-warmup server 1개에서 long 503 4건, 이어지는 short 200 4건, medium 200, context 400, 정상 health를 반복했다. Legacy `--no-batch` server 1개에서 long 503 4건, 이어지는 short 200 4건, medium 200을 반복했다. Unchunked largest forward 때문에 legacy estimate는 의도적으로 보수적인 45.506–45.634 GiB였다. Qwen 2.5 7B server는 같은 budget에서 streaming 및 non-streaming request 모두 200을 반환하고 health를 유지해 shared route handshake가 second family에 Nemotron-H rejection policy를 적용하지 않음을 확인했다.

Post-setter release artifact(`e7a5cc9997a8c0eeceba6a69769353eed35395e906b270d2c62a5f437e2a7062`)는 normal lifecycle 1회와 legacy lifecycle 1회를 반복했다. 각각 typed long-request 503 4건을 SSE 전에 반환했고, short-request 200 4건이 모두 완료되었으며 medium 200과 health 200을 유지했다. Normal context check는 계속 400이었다. Estimate는 측정 artifact의 범위와 일치했다. Qwen control도 streaming 및 non-streaming request에서 다시 200을 반환하고 health 200을 유지했다. 이 post-setter artifact와 benchmark record는 `62c1cf03`에 commit되었다.

## 6. 변경 요약

| 카테고리 | 요약 |
|----------|------|
| 안전성 | Checked fail-closed request envelope와 lower-only runtime budget 추가 |
| HTTP | 지원하는 generation route에서 SSE header 전에 typed 503 반환 |
| Scheduler | 최종 width-one prefill 경계에서 guarded admission을 직렬화하고 batched-prefill bypass 차단 |
| Model loading | Metal Nemotron-H weight를 materialize하고 모델링하지 않은 runtime LoRA를 안전하게 거절 |
| Cache 동작 | Guarded request의 pre-admission snapshot restore/donation 및 background warmup 비활성화 |
| 테스트와 문서 | Estimator, scheduler, provider, route, model regression test와 operator 문서 및 benchmark harness 추가 |

## 7. 후속 조치와 모니터링

- Merge 전에 수정된 release build와 남은 CI job을 확인한다. 근거 없이 fully green workflow로 보고하지 않는다.
- False-rejection rate, estimate 대비 observed peak 비율, retained prompt-cache 기여도, post-admission allocation failure를 모니터링한다.
- 향후 정책은 다른 family, runtime LoRA rank, multi-slot reservation, distributed path, optimistic cache reuse를 모델링할 수 있지만 이 PR은 이를 보장하지 않는다.

## 참고 자료

- Issue #2291: Metal prefill 전 over-budget Nemotron-H request 거절.
- PR #2290 및 issue #2281: Nemotron-H SSD prefill scratch 제한과 long-prompt 측정 baseline.
- `docs/request-memory-admission.md`: 운영자 계약, 제한 사항, error shape.
