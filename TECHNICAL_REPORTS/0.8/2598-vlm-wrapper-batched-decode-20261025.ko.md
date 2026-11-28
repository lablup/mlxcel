# 기술 보고서: PR #2598 — VLM 래퍼 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2564, #2566, #2567, #2587

## 요약

Molmo, Molmo2, MolmoPoint, Moondream2, Moondream3, Jina VLM, Mllama, dots.ocr, MiniCPM-o의 9개 VLM 래퍼가 네이티브 텍스트 디코드 경로를 사용합니다. 자체 텍스트 타워 6개는 공통 split attention을 사용하고, Mllama는 이미지 상태가 없을 때 기존 self-attention 블록을 배치 처리합니다. 위임만 필요한 두 래퍼는 이미 네이티브인 텍스트 타워를 재사용합니다. 가중치 연산을 배치 단위로 실행하면서 attention은 요청별 캐시 이력을 유지합니다.

B=1은 기존 단일 시퀀스 그래프를 사용하고 B=0은 공통 빈 결과를 반환합니다. Mllama에 모델 전역 이미지 상태가 있으면 기존 행별 동작을 유지하며, 이것이 요청 간 격리를 제공하지는 않습니다. 상태 격리는 #2567, Metal 및 실제 체크포인트 검증은 #2587에 남아 있습니다.

## 문제 정의

기존 래퍼들은 행별 디코드를 상속하거나 행별 처리 타워에 위임했습니다. 따라서 스케줄러 동시성을 높이면 embedding, projection, normalization, MLP, 출력 head 연산이 행마다 반복됐습니다. 타워마다 normalization과 rotary 규칙이 다르고, MolmoPoint는 정규화 전후의 hidden state를 함께 사용하며, Moondream3는 토큰별 expert 라우팅을 수행하므로 공통 transformer 어댑터만으로는 정확성을 보장할 수 없었습니다.

## 기술적 선택과 그 이유

`SplitAttention`, `forward_split_attention`, 캐시 진단 도구와 `route_batched_decode`를 재사용합니다. 타워별 기존 normalization 순서, rotary 파라미터, attention scale을 보존하고 요청별 attention을 배치 가중치 연산 사이의 경계로 둡니다. 명시적으로 보존하는 Mllama 이미지 상태 fallback 외에는 새로운 행별 모델 루프를 추가하지 않았습니다.

MolmoPoint는 `forward_with_pre_ln` 의미와 추가 vocabulary 경로를 포함한 `ExtendedLmHead`를 유지합니다. Moondream3는 temperature와 expert 계산을 보존하면서 배치 형태의 라우팅 토큰을 처리합니다. 실제 expert 선택 결과를 assertion으로 확인해 동일 expert와 서로 다른 expert fixture가 의도대로 구성됐는지 검증한 뒤 parity를 비교합니다.

Mllama의 이미지 없는 cross-attention 층은 항등 연산입니다. 배치 경로는 원래 층 인덱스의 self-attention 캐시만 진행시키고 cross-attention dummy slot은 유지합니다. 이미지 상태가 있으면 기존 단일 행 forward를 행마다 호출합니다. 테스트에서 0이 아닌 gate와 이미지 feature를 사용하고 이미지 상태가 실제 logits를 바꾸는지 확인하므로 비활성 fixture가 fallback 검증을 허위로 통과시키지 않습니다.

DotsOCR와 MiniCPM-o는 기존 Qwen2/Llama3 및 Qwen3 네이티브 진입점에 위임합니다. 실제 래퍼 객체를 호출하는 테스트로 위임, 디코드 marker, B=1 라우팅을 검증하며 타워 테스트만으로 래퍼 동작을 추정하지 않습니다.

## 검증

최종 통합 CPU 라이브러리 배치 디코드 테스트는 129개 통과, 기존 Gemma4 ignore 1개, 실패 0개입니다. coverage 분류 및 source guard 테스트 11개가 통과했습니다. 집중 fixture는 자체 타워 6개, MolmoPoint 정규화 전후와 확장 head, Moondream3 동일·서로 다른 expert, Mllama 외 실제 래퍼 8개, Mllama 타워·래퍼 상태 테스트 3개를 포함합니다. 공통 fixture는 길이가 다른 prompt 3개, 정확한 B=1 및 빈 B=0을 검증합니다.

Python 도구 테스트 139개가 통과했습니다. workspace 버전, 환경변수 문서, Llama 호환성, 라이선스 헤더, 커널 dtype key, 커널 port dispatch, ROCm overlay 기록, binary asset, Windows 경로 검사도 통과했습니다. warnings를 오류로 처리한 workspace 전체·모든 target Clippy, formatting 및 diff 검사도 통과했습니다.

최종 정확성·보안·성능 리뷰는 clean입니다. 리뷰에서 `LanguageModel`을 구현하지 않는 Mllama 텍스트 타워 대신 래퍼를 분류하도록 coverage 항목을 수정했고, Mllama 파일에 직접 split-attention 호출을 요구하던 검사는 재사용하는 Llama 블록을 확인하도록 고쳤습니다.

Metal 체크포인트 trace, greedy 생성 baseline 비교, serving ladder 및 측정된 속도 향상은 주장하지 않습니다. 해당 검증은 #2587, 모델 전역 이미지 상태 격리는 #2567에 남아 있습니다.

## 학습 포인트

coverage 분류는 실제 trait 구현 주체를 따라야 합니다. `LanguageModel`이 없는 타워는 래퍼를 통해 검증하고, source guard는 중복 구현을 강요하지 않고 기존 코드 재사용을 인식해야 합니다. 활성 이미지 gate와 명시적인 expert 선택 assertion은 실질적인 parity 검증을 가능하게 합니다.

## 변경 요약

현재 구현 가능한 마지막 래퍼·타워 그룹에 네이티브 디코드를 추가하고, 실제 연산을 수행하는 합성 runtime fixture를 등록했으며, coverage 목록과 네이티브·조건부 fallback 문서를 갱신했습니다. 기존 단일 행 진입점과 이미지 prefill 인터페이스는 보존했습니다. 구현 커밋: `788f2689`.
