# PR #2628: Gemma 3n 네이티브 배치 디코드

**작성일**: 2026-10-27
**상태**: 완료; 기존 비전 테스트 실패 기록
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

PR #2628(이슈 #2561)은 Gemma 3n 서버 디코드의 trait 기본 행별 실행을 활성 단일 토큰 행을 위한 네이티브 배치 경로로 바꾼다. Embedding, AltUp, LAuReL, token 기반 per-layer input, projection, normalization, MLP, tied LM head는 전체 `[B, 1, hidden]` tensor에서 실행하고, attention은 각 행의 concrete 또는 KV-shared cache 상태와 sliding-window 동작을 보존한다.

Text model은 이제 `BatchedDecodeKind::Native`를 보고하고 Gemma 3n VLM은 text-only batched decode를 이 경로에 위임한다. 체크포인트가 필요 없는 E2B/E4B 형태 regression과 decode classification coverage가 포함되었다. 최종 release, Clippy, 포맷, checkpoint parity, trace, server path 검증을 완료했다. C=4 총 처리량은 E2B에서 72.2%, E4B에서 62.5% 증가했다. 전체 workspace suite에는 기존 비전 테스트 실패 10개가 남았고 새 실패는 발견되지 않았다.

## 1. 문제 정의

Gemma 3n은 batching 지원을 표시했지만 `LanguageModel::forward_batched`를 override하지 않았다. 따라서 서버는 여러 활성 행을 받아도 완전한 single-sequence `forward` graph를 행마다 호출해, weight read와 projection 연산을 batch 전체에서 분담하지 못하고 반복했다.

이 모델은 단순히 입력을 쌓아 처리할 수 없다. AltUp plane, LAuReL residual branch, token 기반 per-layer input, 교대로 배치된 sliding/full attention, 앞선 owner cache의 K/V를 갱신 없이 읽는 logical layer를 함께 사용한다. E2B와 E4B는 KV-shared layer 수도 다르다. 네이티브 경로는 기존 single-row graph를 정확히 보존하면서 행별 offset과 cache ownership을 유지해야 한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Routing contract | `Gemma3nModel`이 native decode를 보고하고 `route_batched_decode`를 사용한다. 공유 B=0 empty result와 B=1 direct single-sequence dispatch를 보존한다. |
| Batched transformer 연산 | Embedding과 scaling, AltUp predict/correct, LAuReL, per-layer input gather/projection, norm, attention projection, MLP, final norm, softcap, LM head를 전체 batch graph에 유지한다. |
| Concrete attention | `Gemma3nAttention`이 `SplitAttention`을 구현하고 공유 `forward_split_attention` helper를 호출한다. Sliding layer는 기존 행별 windowed attention 동작을 유지한다. |
| KV-shared attention | 각 logical shared layer가 모든 행을 concrete owner cache에 연결하고, owner offset에서 query를 회전하며, append 없이 live K/V 범위를 읽은 뒤 행 결과를 한 번 합친다. |
| Cache 진단 | Shape, owner-cache 수, cache 누락, offset 불일치 오류에 관련 행과 layer를 표시한다. |
| VLM 통합 | `Gemma3nVLModel`이 batched text decode와 decode-kind classification을 text model에 위임한다. Multimodal prefill은 기존 embedding 경로를 유지한다. |
| 테스트와 문서 | E2B/E4B 형태의 서로 다른 길이 regression이 sliding boundary와 shared-layer layout을 다룬다. Coverage list와 `docs/CONTINUOUS_BATCHING.md`가 Gemma 3n text/VLM을 올바르게 분류한다. |

## 3. 기술적 선택과 그 이유

### 3.1 Weight 비중이 큰 graph는 배치하고 cache-sensitive attention은 행별로 유지

새 cache 표현이나 paged backend를 도입하지 않고 공유 split-attention 경계를 재사용했다. Q/K/V와 output projection은 batch에서 계산하지만 cache append, live-range 선택, sliding attention은 각 sequence의 독립 history를 보존한다. 따라서 single-sequence inference의 cache contract를 바꾸지 않고 반복 transformer 연산을 제거한다.

Attention 내부에 행별 연산이 남는 것이 트레이드오프다. 더 큰 storage 재설계는 이슈 #2561의 범위 밖이며, 추가 batching은 측정한 serving 결과와 향후 profiling 결과를 근거로 결정해야 한다.

### 3.2 KV sharing을 읽기 전용 ownership mapping으로 처리

뒤쪽 Gemma 3n layer는 자체 cache를 소유하지 않는다. Batched path는 모든 행에서 `layer_idx_to_cache_idx`를 사용하고, owner의 live K/V tensor를 읽으며, cache를 갱신하지 않은 채 owner의 logical offset을 RoPE에 적용한다. 이를 하나의 helper에 모아 shared layer마다 행 loop를 다시 구현하지 않게 했고, 특히 동일 token을 owner cache에 여러 번 append하는 오류를 방지한다.

### 3.3 B=1을 근사 parity가 아니라 routing으로 보존

`route_batched_decode`는 한 행 입력을 기존 `forward` method에 직접 전달한다. 따라서 single-row의 정확한 동작은 서로 다른 shape의 kernel 사이 수치 tolerance에 의존하지 않는다. 새 구현은 행이 둘 이상일 때만 실행되고, 빈 batch는 공유 empty-logit contract를 따른다.

### 3.4 VLM 경계를 좁게 유지

Input embedding이 없는 decode 행은 정확히 text-model 호출이므로 VLM wrapper는 `forward_batched`를 text model에 위임한다. Image processing과 multimodal per-layer input 생성은 prefill에서 바꾸지 않아 네이티브 decode 범위를 불필요하게 확장하지 않았다.

## 4. 검증 상태

### 4.1 최종 보고 전에 완료된 항목

- Gemma 3n 집중 테스트 통과: 3개.
- Batched-decode coverage 테스트 통과: 12개.
- `cargo check --lib` 통과.
- `cargo fmt --all -- --check` 통과.
- 최종 release build가 12분 37초에 통과.
- 전체 workspace suite는 13,826 passed, 10 failed, 441 ignored로 완료되었다. 실패 10개는 기존 수치 oracle 문제다. 이름, panic 위치, 실제·기대 값이 #2546의 clean baseline 조사와 정확히 같고 관련 파일도 동일하다. 전체 suite가 모두 성공한 것은 아니다.
- 구현 및 보안 리뷰에서 필수 수정 사항 없음.
- 최종화 중 `git diff --check origin/main...HEAD` 통과.

Native regression은 prompt 길이가 서로 다른 세 행을 사용해 4-token sliding boundary의 아래, 경계, 이후를 실행한다. E2B 형태 설정의 KV-shared layer 1개와 E4B 형태 설정의 2개를 각각 검사하고, full/sliding attention, token 기반 per-layer input, multi-row sequential parity, 정확한 B=1 dispatch, B=0 output도 다룬다. Static coverage는 text model을 native로, VLM을 delegating으로 고정한다.

### 4.2 최종 체크포인트와 서버 근거

- `metal,accelerate`와 `-D warnings`를 사용한 workspace/all-targets Clippy 통과.
- E2B와 E4B greedy 출력 및 전체 B=1 trace 파일이 main `1cb18ccc`와 byte-identical이다. Candidate source는 `f01241ba`이며 두 빌드는 MLX `7ca40033`을 사용한다.
- B=4 대 B=1 trace: E2B top-1 불일치 2/256, decided 0/147; E4B top-1 불일치 4/256, decided 0/150. 모든 불일치에서 reference gap은 최대 0.125다. 배치 graph가 bit-identical한 것은 아니며 reference token의 최대 logit 차이는 6.25와 10.9375다.
- 두 checkpoint startup log가 `decode path: native batched forward`를 보고한다.
- E2B C=1/2/4 총 tok/s: main 87.6/124.3/128.6, native 88.3/146.0/221.5. E4B: main 68.5/97.9/100.2, native 68.4/110.8/162.8. 단일 측정에서 C=1 성능은 사실상 유지됐다(+0.8%/-0.15%).

조건, perplexity 차이, 실패 분류와 측정 한계는 [벤치마크 기록](../docs/benchmark_results/gemma3n-native-batched-decode-2026-10-27.md)에 정리했다. CUDA/ROCm 성능은 측정하지 않았고 선택적 server metrics도 사용할 수 없었다.

## 5. 학습 포인트

- Batching capability flag만으로 하나의 batched model graph가 실행된다고 볼 수 없다. Decode-kind coverage는 입력 수용 여부가 아니라 실제 실행 경로를 분류해야 한다.
- 독립 sequence cache를 합치지 않고도 transformer 연산 대부분을 네이티브로 배치할 수 있다. Attention 주변의 split boundary는 복잡한 cache 의미를 가진 모델에 실용적인 중간 아키텍처다.
- Logical KV-sharing layer에는 owner tensor뿐 아니라 owner position도 필요하다. K/V는 재사용하면서 별도로 계산한 logical-layer offset을 쓰면 query가 잘못된 위치에서 회전한다.
- 정확한 B=1 보존은 control flow로 강제할 때 가장 강하다. 서로 다른 kernel shape의 tolerance 비교를 graph identity의 증거로 사용하는 문제를 피한다.

## 6. 후속 조치와 제한 사항

- 더 넓은 checkpoint/backend 검증은 후속 과제이며, 현재 근거는 Metal의 E2B/E4B를 다룬다.
- Attention 내부는 계속 행별로 실행하며 paged backend를 추가하지 않는다. 둘 다 이 PR의 범위 밖이다.
- VLM 변경은 text-only decode 위임만 다루며 audio 또는 vision tower batching은 다루지 않는다.
- Backend별 reduction 순서 때문에 multi-row logit은 bit-identical이 아니라 수치적으로 가까울 수 있다. 관찰된 근소한 점수 차이 구간의 불일치와 logit 차이는 위에 기록했다.

## 참고 자료

- 이슈 #2561: Gemma 3n batched decode forward
- PR #2628: 활성 행 전체에서 Gemma 3n decode batching
- 이슈 #2546 / PR #2611: Gemma 4 네이티브 배치 디코드 참조 설계
- `docs/CONTINUOUS_BATCHING.md`: 서버 decode 분류와 동작
