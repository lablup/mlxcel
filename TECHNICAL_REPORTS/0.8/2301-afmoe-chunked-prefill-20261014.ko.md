# PR #2301: AFMoE no-ID chunked prefill 계약 수정

**날짜**: 2026-10-14
**상태**: 구현 및 독립 리뷰 완료, CUDA·focused GPU·contract 검증 통과
**위험도**: 낮음 (model capability 정책 및 loaded-model 회귀 검증)

## 요약

AFMoE는 sequence ID가 없는 multi-token forward가 model-owned cache set을 교체하는데도 `supports_chunked_prefill() == true` 기본값을 상속했습니다. 따라서 legacy caller가 하나의 프롬프트를 여러 no-ID forward로 나누면 앞선 sliding-attention 및 full-attention KV state를 모두 잃을 수 있었습니다.

이제 모델은 token-ID chunked prefill을 지원하지 않는다고 선언합니다. Loaded synthetic regression은 production loader와 실제 mixed-cache lifecycle을 6개 계층에서 실행합니다.

## 상태 동작

AFMoE는 heterogeneous cache vector를 소유합니다. 축소한 Trinity fixture는 실제 hybrid 설계를 따라 rotating KV cache를 사용하는 sliding-attention 계층 4개와 standard KV cache를 사용하는 full-attention 계층 2개로 구성됩니다. 처음 2개 계층은 dense MLP를 사용하고 나머지 계층은 MoE 경로를 사용합니다.

No-ID 경로는 multi-token forward를 시작할 때마다 전체 cache vector를 의도적으로 교체합니다. 이 동작은 서로 독립적인 프롬프트의 상태를 격리합니다. 안정적인 sequence ID는 per-sequence model-owned state를 선택하고 모든 계층의 cache를 호출 사이에 유지합니다.

DirectEngine은 이미 prefill 조각 전체에 같은 sequence ID를 전달하므로, 이 이슈는 현재 CLI가 AFMoE 프롬프트를 잘랐다는 증거가 아닙니다. 하지만 model capability는 cache-less 및 legacy no-ID caller에도 안전해야 합니다.

## 변경

- `supports_chunked_prefill()`이 `false`를 반환하도록 override합니다.
- 결정적으로 채운 0이 아닌 synthetic tensor와 `AfmoeModel::from_weights`를 사용해 6-layer AFMoE 모델을 구성합니다.
- No-ID로 길이 2와 3을 차례로 실행한 뒤 6개 cache offset이 모두 3인지 검사해 두 번째 프롬프트가 이전 상태를 교체했음을 증명합니다.
- 하나의 안정적인 sequence ID로 같은 길이를 실행한 뒤 6개 cache offset이 모두 5인지 검사해 rotating 및 standard KV state가 모두 이어짐을 증명합니다.

테스트는 대표 계층 하나가 아니라 모든 cache를 확인하므로 일부 계층만 초기화되거나 sliding 및 full-attention cache 처리가 달라지면 실패합니다.

## 검증

소스는 독립 구현 리뷰를 통과했습니다. Root 및 surgery CUDA `test-fast` build graph는 14분 05초에 통과했습니다. GPU lock 아래 전체 `models::afmoe::tests::` 모듈은 loaded-model lifecycle regression과 기존 snapshot restore coverage를 포함해 29개 테스트 모두 0.85초에 통과했습니다. Contract gate는 앞선 모든 검사와 Python tooling 123개, WebUI schema·DTO·fixture 검사 49개를 통과했습니다. Formatting 및 patch whitespace 검사도 통과했습니다. Fixture는 결정적이고 0이 아닌 synthetic weight를 적재하며, 외부 AFMoE 체크포인트 실행을 수행하거나 주장하지 않습니다.

## 의존성 갱신

위 검증 결과는 브랜치가 PR #2298의 main commit `4b0c4519`를 병합하기 전에 기록했습니다. Merge commit `4cc9b87d`는 새 MLX pin과 CUDA/Metal overlay base를 가져오지만 검토한 model file을 바꾸지 않았습니다. `afmoe.rs`의 SHA-256은 `7f00690db48937f9e5a2810f9c26307e50dd1d5b14ea1810cd216f75d1d3466c`, `afmoe_tests.rs`는 `844cbce7258e750cface1aa24b7f64a925b2c8c79524159a411c53de25d71a12`로 유지됩니다. 최종 pin 통합 후 새 focused runtime 검증 결과는 아래에 기록합니다.

## 메모리 트레이드오프

`PrefillCaps`는 이 model-wide capability를 모든 token-ID prefill planner에 적용합니다. 따라서 DirectEngine과 cache-less caller를 포함해 capability를 따르는 모든 호출자가 AFMoE에서 하나의 whole-prompt graph를 사용합니다. 이 동작은 안전하지 않은 no-ID state 손실을 막지만 chunk-bounded transient memory를 포기하므로 긴 프롬프트의 peak memory 사용량이 늘 수 있습니다.

## 범위

이 변경은 독립 프롬프트 초기화, stable-ID server state, cache mode, quantization loading, padded-prefill 정책, snapshot 동작을 유지합니다. Chunking capability만 바꾸고 lifecycle coverage를 추가합니다.

## 최종 통합 검증

MLX `243bdf09`, mlxcelverse, 앞서 머지한 모델 수정을 포함한 main `a2698835`를 소스 충돌 없이 통합했다. 다시 빌드한 CUDA `test-fast` binary에서 `models::afmoe` 테스트 29개가 모두 통과했다. 가중치를 적재한 sliding/full attention 혼합 캐시 lifecycle과 snapshot 검증을 포함한다. ID 없는 prompt는 6개 캐시를 모두 초기화하고, 같은 ID의 청크는 모든 캐시를 연속해서 전진시킨다. Formatting과 diff 검사도 통과했다. 합성 가중치 적재 검증이며 외부 체크포인트 실행을 주장하지 않는다.
