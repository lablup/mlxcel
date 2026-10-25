# 기술 보고서: PR #2596 — VLM 계열 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2563, #2566, #2567, #2587

## 요약

Sequence-aware VLM text tower 6개가 native grouped decode를 실행한다. 대상은 GLM-4V/GLM-OCR, GLM-4V-MoE, Hunyuan-VL, ERNIE-4.5-MoE-VL, PaddleOCR-VL, Cohere Compass다. 실제 vision wrapper 7개도 text-only decode를 이 tower entry point에 위임한다. 행을 plain과 multimodal-RoPE state 기준으로 묶고, weight 사용량이 큰 projection은 호환되는 group별로 한 번 실행하며, 요청별 attention cache는 격리하고, logits는 scheduler의 원래 행 순서로 scatter한다.

검증된 Qwen VL의 position/group/cache/scatter 로직을 공통 helper로 추출하면서 각 tower의 서로 다른 rotary, normalization, expert, attention-window 규칙을 보존했다. Prefill, 누락되거나 불일치하는 sequence ID, 저장된 명시적 position, 안전하게 sequence 격리되지 않은 Compass의 model-global DeepStack state는 정확한 per-row fallback을 유지한다. Metal checkpoint parity와 처리량 검증은 #2587로 명시적으로 이관했다.

## 문제 정의

이 VLM tower들은 batching에 필요한 sequence state를 가지고 있었지만 serving decode에서는 여전히 per-row forward를 상속했다. 단순히 행을 concatenate하면 올바르지 않다. 각 행의 MRoPE delta가 다를 수 있고, plain text와 multimodal row가 서로 다른 rotary path를 요구하며, cache history 길이도 다르고, scheduler는 grouping 순서가 아니라 원래 요청 순서의 logits를 기대하기 때문이다.

계열별 차이도 크다. Hunyuan은 RoPE 뒤에 learned Q/K normalization을 적용하고 position axis 4개를 사용한다. GLM은 설정된 prefix만 회전하며 even/odd 또는 half-split rotary pairing을 보존해야 한다. ERNIE는 interleaved 3D MRoPE와 분리된 text·multimodal expert bank를 사용한다. Compass는 full과 sliding-window attention, layer type별 optional RoPE, 아직 sequence 격리되지 않은 model-global visual/DeepStack state를 함께 사용한다.

## 기술적 선택과 그 이유

### 호환되는 decode row만 grouping

Native entry point는 single-token decode width와 행 수에 맞는 완전한 sequence-ID 목록을 요구한다. 각 행의 monotonic cache offset과 저장된 rope delta를 읽고 `[axes, B, 1]` position을 만든 뒤 plain group과 MRoPE group으로 나눈다. 각 group은 오름차순 요청 순서를 유지하고 해당 cache row만 빌려 하나의 model graph를 실행한 다음, logits를 원래 scheduler 순서의 slot에 scatter한다. Family별 진단은 cache layer 누락이나 생성되지 않은 행을 명확히 거부한다.

명시적으로 저장된 position tensor가 있는 행은 scalar-delta fast path로 표현할 수 없다. Prefill width, 누락되거나 불일치하는 ID, stored-position case는 position을 근사하지 않고 기존 per-row dispatcher를 사용한다. 이 fallback은 batching 실패가 아니라 correctness boundary다. 호환되는 행만 native grouping이라고 표시한다.

Grouping, row 선택, cache 선택, scatter, concatenation helper를 공통 Qwen MRoPE state module로 옮겼다. 기존 Qwen 2 VL, Qwen 3 VL, Qwen 3 VL MoE caller도 같은 함수를 재사용하며, regression test로 plain/MRoPE grouping, cache update, wrapper 동작에 drift가 없음을 검증했다.

### Rotary와 model 의미는 각 tower가 소유

공통 메커니즘은 row, position, cache를 제공하고 수학적 transform은 각 tower가 계속 소유한다. Hunyuan은 4축 rotary transform 뒤의 Q/K normalization을 유지한다. GLM-4V와 MoE variant는 설정된 partial dimension만 회전하고 각자의 pair layout을 보존한다. ERNIE는 f32에서 adjacent-pair interleaved 3D MRoPE를 적용하고, decode width에서는 항상 text expert bank를 선택하며, image-placeholder expert 선택은 prefill에만 남긴다. 기존 stacked-expert quantization geometry guard도 유지한다.

Compass의 초기 구현은 sliding layer가 항상 RoPE를 사용하고 full layer는 사용하지 않는다고 가정했다. 리뷰에서 이것이 유효한 configuration invariant가 아님을 확인했다. 최종 구현은 첫 sliding layer type과 첫 full layer type에서 optional cosine/sine table을 각각 독립적으로 만들고, 각 layer에 맞는 optional table을 재사용한다. Windowing은 계속 attention layer 자체가 제어하므로 sliding-NoPE, full-MRoPE, 표준 variant가 모두 single-row 의미를 보존한다.

Compass DeepStack input은 여전히 model-global state로 저장된다. 이 state가 있으면 전체 call이 정확한 per-row path를 사용한다. Global state를 sequence isolation으로 잘못 표현하지 않기 위한 경계이며, request-owned state로 전환하는 작업은 #2567에서 계속 추적한다.

## 검증

Tower/configuration 테스트 9개는 tower 6개와 Compass sliding-NoPE, full-layer-MRoPE, global-DeepStack fallback variant를 다룬다. 실제 wrapper fixture 7개는 각 concrete wrapper를 실행한다. 모든 synthetic tower fixture는 유의미한 weight와 서로 다른 길이의 prompt history 3개 이상을 사용하고, plain row, sequence별 서로 다른 delta, mixed group/scatter order, stored-position fallback, exact B=1 dispatch, B=0을 검증한다.

ERNIE expert-bank guard의 focused test 1개가 통과했다. Qwen regression은 공통 helper 테스트 2개, Qwen 3 VL MoE tower 테스트 1개, wrapper 5개 통합 테스트 1개가 통과했다. 최종 classification ledger 9개와 logit-trace example 7개가 통과했고, library test suite는 `--no-run`으로 성공적으로 compile됐다. Format과 diff 검사도 통과했다.

Workspace crate version, 320개 `MLXCEL_*` 이름의 문서 coverage, environment-documentation 테스트 15개, Llama compatibility manifest와 모든 validator case에 대한 repository source gate가 통과했다. Warnings-denied 전체 workspace·all-target Clippy는 2분 5초에 통과했다. Compass rotary-table 수정을 반영한 최종 architecture, correctness, security, performance 리뷰에는 지적 사항이 없었다.

실제 checkpoint, Metal trace, greedy-generation 비교, server concurrency benchmark는 실행하지 않았다. 해당 hardware acceptance는 #2587에 남아 있으므로 이 보고서는 실측 speedup이나 production parity를 주장하지 않는다.

## 학습 포인트

Sequence-aware batching은 tensor shape 문제인 동시에 partition-and-scatter 문제다. 행은 weight를 공유하면서 서로 다른 position state와 cache ownership을 유지할 수 있으므로 grouping은 명시적이고 안정적이며 역변환 가능해야 한다. Stored position을 scalar delta로 손실 없이 줄일 수 없을 때 fallback하는 편이 근사 표현으로 native path를 넓히는 것보다 안전하다.

Attention type과 rotary type은 서로 독립적인 configuration axis다. Compass는 “sliding이면 RoPE”, “full이면 NoPE”라는 shortcut이 왜 위험한지 보여준다. Window size는 어떤 key를 볼 수 있는지 결정하고, optional rotary table은 Q/K를 어떻게 encode하는지 결정한다. 리뷰 가능한 helper는 이 선택을 독립적으로 드러내야 한다.

## 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 26 |
| 추가 라인 | 2,957 |
| 삭제 라인 | 303 |
| Native text tower | 6 |
| Delegating wrapper | 7 |
| 커밋 | `9aae72bc` |

이 PR은 text tower 6개에 grouped native decode를 추가하고 wrapper 7개를 위임하며, Qwen에서 유래한 MRoPE grouping/cache/scatter helper를 공통화하고 runtime·coverage regression과 native family 문서를 확장한다. Metal 인수 검증은 #2587, Compass DeepStack state의 sequence 격리는 #2567에서 계속 추적한다.
