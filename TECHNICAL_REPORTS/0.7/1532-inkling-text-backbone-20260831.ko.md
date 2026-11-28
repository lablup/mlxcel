# 기술 보고서: PR #1532 - Inkling 텍스트 백본

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 완료. 합성 수치 검증과 CI는 통과했고 실제 체크포인트 검증은 보류
**언어**: Rust, Markdown
**위험 수준**: 높음

---

## 요약

PR #1532는 에픽 #1313의 선행 조건인 Inkling 텍스트 아키텍처를 추가한다. 하이브리드 슬라이딩/전역 디코더, RoPE 없는 학습형 밴드 상대위치 어텐션, 레이어당 네 개의 short-convolution 상태, dense 및 sparse MoE, 세 가지 체크포인트 가중치 형식, 모델 소유 캐시 스냅샷, 서버의 효율적인 last-logits 투영, Inkling reasoning 마커를 구현했다. 이미지·오디오·비디오·MTP·융합 커널·텐서 병렬·패딩 배치는 의존 이슈의 범위로 남겼다.

핵심 성과는 새 `ModelType`을 등록한 것에 그치지 않는다. Inkling은 상태를 가진 convolution과 앞쪽이 잘리는 sliding KV window를 결합하고, correction bias로 routed expert를 선택한 뒤 routed/shared expert를 한 분포에서 함께 정규화한다. 둘 다 잘못 구현해도 유한하고 모양이 맞는 출력이 나올 수 있다. 그래서 이 PR은 모양 검사만이 아니라 소형 결정론적 수치 기준, 캐시 롤백 테스트, 잘못된 가중치 거부 테스트, 독립 리뷰로 정확성을 확인했다.

## 1. 문제 정의

Inkling 체크포인트는 `inkling_mm_model` / `InklingForConditionalGeneration`으로 식별되지만, mlxcel에는 계열을 위한 감지·설정·로더·디코더·캐시 계약·출력 마커 지원이 없었다. 기존 모델의 작은 변형으로 표현하기도 어렵다. RoPE를 쓰지 않고 학습된 거리 밴드를 additive attention mask로 사용하며, K·V와 두 잔차 분기 모두에 depthwise short convolution을 적용하고, routed expert와 항상 켜진 shared expert를 하나의 logsigmoid-softmax 분포로 합친다.

체크포인트 호환성은 별도의 위험 영역이다. 원본 bf16/f32 가중치는 `model.llm.*` 이름과 interleaved gate/up 평면을 사용하고, native ModelOpt NVFP4 릴리스는 packed expert 텐서와 중첩된 `hf_quant_config.json` 메타데이터를 사용하며, MLX 커뮤니티 변환본은 affine 4-bit expert triplet을 사용한다. 불완전한 sidecar나 설정 입력 차원과 맞지 않는 packed width를 조용히 허용하면 실패를 복구하기 어려운 MLX/CXX 호출까지 도달하므로 모델을 만들기 전에 거부해야 한다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| 아키텍처 | embedding RMSNorm, 하이브리드 sliding/global NoPE attention, banded relative logits, log-position temperature, f32 short-convolution 상태, dense SwiGLU, routed/shared MoE 추가 |
| 상태 | bounded sliding-KV 복원, visible-window snapshot, 레이어당 convolution slot 네 개, tail rollback 의미론 추가 |
| 가중치 | 원본 bf16/f32, native NVFP4, affine MLX 4-bit sanitize와 shape·packing·dtype·sidecar 검증 추가 |
| 로딩 | 감지·메타데이터·디렉터리 로더·owned-weight 로더·`LoadedModel` dispatch에 `inkling_mm_model`과 `inkling` 등록 |
| 서버 | prompt prefill이 201,024-token vocabulary 전체를 투영하기 전에 마지막 hidden row를 자르는 sequence-aware last-logits hook 추가 |
| 출력 | `<|content_thinking|>` / `<|end_message|>`를 tokenizer·streaming·non-streaming·thinking-budget 경로에 연결 |
| 문서 | `docs/supported-models.md`에 Inkling 텍스트 전용 지원 범위 추가 |

## 3. 기술적 선택과 이유

### 3.1 융합 커널이 아니라 upstream 연산 의미론을 따르기

공개 mlx-vlm Inkling 파일인 [language.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/language.py), [inkling.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/inkling.py), [config.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/inkling/config.py)와 구현을 대조했다. relative bias, routing, shared expert, cache 순서, 가중치 매핑 의미론은 따르되 이슈가 요구한 graph path를 유지했다. upstream의 mask·short convolution·router·q4 down-combine·QKVR Metal 융합 커널은 성능 후속 작업으로 남겼다.

### 3.2 Convolution 상태와 sliding-KV 상태를 분리하기

각 레이어는 KV cache 하나와 f32 convolution 상태 네 개를 가진다. 오래된 attention key를 앞에서 잘라도 convolution 상태는 지우면 안 된다. 각 slot은 이미 가장 최근 `kernel_size - 1`개 activation을 담고 있기 때문이다. 반대로 padding이나 speculative position을 뒤에서 되돌릴 때는 KV 길이만으로 convolution 상태를 되감을 수 없으므로 KV tail을 자르고 미래 상태가 남지 않도록 convolution 상태를 초기화한다.

Snapshot은 예약된 backing capacity가 아니라 보이는 KV window만 직렬화한다. 복원할 때 absolute offset을 유지하고 내부 live-window 시작점도 다시 만든다. 리뷰 전 구현은 10-token cache를 256-token live window처럼 직렬화해 복원을 거부하거나 상대 거리를 어긋나게 만들 수 있었다.

### 3.3 Correction bias는 선택에만 사용하고 가중치에는 섞지 않기

Routed expert 선택은 `sigmoid(logit) + correction_bias`를 사용하지만, 기여 가중치는 선택된 routed expert의 원본 logit과 shared-expert 원본 logit으로 계산한다. 공통 분포는 logsigmoid 뒤 softmax를 적용하고 route scale과 학습된 global scale을 곱한다. Native NVFP4는 SwiGLU 앞뒤에 expert별 gate/output sidecar를 각각 적용한다. Token 수와 top-k가 서로 다른 CPU 수치 기준을 사용해, 두 차원이 우연히 같을 때 broadcast 축 오류가 가려지는 문제를 막았다.

### 3.4 체크포인트 메타데이터를 신뢰하지 않는 입력으로 취급하기

Sanitizer와 load validation은 MLX 연산 전에 불완전한 triplet, 반대 방향 mixed dtype, 잘못된 packed width, scale/bias 선행 shape 불일치, 미지의 NVFP4 sidecar, 정수가 아닌 group size, 잘못된 schedule, overflow가 나는 dimension 변환을 거부한다. Native NVFP4 감지는 필드가 최상위에 있다고 가정하지 않고 실제 중첩 구조인 `quantization.quant_algo` / `quantization.group_size`를 읽는다.

### 3.5 프롬프트 전체 vocabulary 투영을 피하기

Inkling의 padded vocabulary는 201,024행이다. 서버 prefill에서 모든 prompt position을 투영하면 샘플링에는 한 행만 필요한데도 긴 요청에서 수십 GiB의 임시 logits가 생길 수 있다. Core generation interface에 하위 호환 sequence-aware last-logits hook을 추가하고 full·chunked·prompt-cache server prefill에 연결했다. 기존 모델은 기본 동작을 유지하고 Inkling만 LM head 전에 hidden state를 자른다.

## 4. 리뷰와 보강

독립 정확성 리뷰와 finalizer 리뷰가 머지 전에 다음 결함을 발견하고 수정했다.

- Visible-window snapshot 직렬화가 cache reserved-capacity 표현과 맞지 않았다.
- Cache rollback hook이 최신 tail을 제거해야 하는데 `trim_front`로 오래된 prefix를 제거했다.
- Sequence-ID server prefill이 last-logits 최적화를 우회해 vocabulary 전체를 투영했다.
- Native NVFP4 sidecar 승격이 잘못된 JSON 중첩 위치를 읽었다.
- Quantized plane의 packed input width와 scale/bias 선행 축 검증이 불완전했다.
- Floating/byte expert plane 혼합이 잘못된 runtime path로 들어갈 수 있었다.
- Token 수와 top-k가 다를 때 expert-scale broadcast 축 순서가 틀렸다.
- Non-streaming parser가 일반 Inkling 답변의 `<|end_message|>`를 reasoning close로 오인해 답변을 삭제할 수 있었다.
- Schedule 오타, sidecar I/O/type 오류, 악의적인 정수 곱셈을 더 엄격히 거부해야 했다.

머지 시점에 해결되지 않은 CRITICAL/HIGH 정확성·보안·성능 문제는 없었다.

## 5. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo fmt --all -- --check` | 통과 |
| `cargo check --lib --features metal,accelerate` | 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 로컬 및 CI 통과 |
| Inkling 집중 단위 테스트 | 26/26 |
| Expert-scale CPU 기준 | 1/1, N=3 및 K=2 |
| Registry 완전성 | 2/2 |
| GitHub CI | format, clippy, deny, OpenXLA feature compile, manifest, cross-repo reference, CLA 모두 통과 |

집중 테스트는 config 우선순위, width alias, banded bias, log-temperature, short-convolution 연속성, sliding/global causal prefill, token-by-token decode parity, router correction-bias 분리, shared-expert CPU 등가성, native NVFP4 및 affine sanitizer 성공 경로, 잘못된 입력 거부, cache snapshot/restore, EOS, 감지 alias, reasoning-marker 전 경로를 포함한다.

## 6. 검증 한계와 후속 작업

로컬에 Inkling 체크포인트가 없었다. 공개 affine MLX 체크포인트는 약 153.5 GB, native NVFP4 릴리스는 약 170.7 GB로 121 GiB 메모리 호스트에서 실용적으로 실행할 수 없다. 따라서 `Inkling-0.6B-A0.6B` 토큰 단위 비교, Inkling-Small 유창성, 실제 처리량, CUDA 검증은 적절한 하드웨어에서 수행해야 한다.

에픽의 의존 작업은 HMLP vision, dMel audio, temporal-pair video, native MTP drafter를 추가한다. 실제 체크포인트 패리티를 확립한 뒤 mask·short-convolution·router·q4 down-combine·QKVR 융합 커널을 성능 후속 작업으로 추가할 수 있다.

## 참고

- 에픽 #1313과 이슈 #1318
- PR #1532, squash commit `5690012fee9ec9053a2cde5984c4b5ada0eb27ec`
- 3.1절에 연결한 공개 mlx-vlm Inkling 구현
- `docs/supported-models.md`
