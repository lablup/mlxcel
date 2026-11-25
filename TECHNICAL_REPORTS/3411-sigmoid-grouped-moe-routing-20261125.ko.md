# PR #3411: Sigmoid 그룹 MoE 라우팅

**작성일**: 2026-11-25
**상태**: #3345 검증 완료. 실제 30B 수용 검증은 #3350 범위.

## 요약

Nemotron-H E 블록이 XLA 백엔드에서 sigmoid 그룹 라우팅과 relu2 라우팅·공유 expert를 실행한다. 기본 연산 수치 검사에서 드러나지 않았던 체크포인트 검증과 전체 GPU 그래프 컴파일 문제를 통합 검사로 찾아 수정했다.

## 문제 정의

기존 softmax/SwiGLU 라우터는 Nemotron-H MoE를 표현하지 못했다. E 스케줄 허용만으로는 recurrent 체크포인트 검증기가 dense up/down 가중치를 요구하는 문제가 남았고, 전체 CUDA 그래프에 융합된 tuple-reduction argmax는 컴파일에 실패했다.

## 기술적 선택과 그 이유

선택에는 sigmoid 점수와 보정 bias를 사용하고, 결합에는 보정 전 점수와 epsilon을 포함한 선택적 정규화 및 routed scaling을 사용한다. 그룹 점수는 상위 두 점수의 합이며 동점은 낮은 인덱스가 우선한다. 단일 결과 reduction으로 first-NaN·낮은 인덱스 의미를 유지하면서 전체 그래프의 tuple reduction 실패를 피했다. packed routed/shared 텐서는 올바른 scale·bias shape를 요구하며 선택 불가능한 그룹 용량, 폭 0 shared expert, 미지원 latent projection은 거부한다.

## 변경 요약

설정, 텐서 스키마, expert 로딩, 그룹 라우팅, E 포함 통합 fixture를 함께 확장했다. 기존 softmax MoE golden은 byte 단위로 동일하다. tokenizer fixture는 EOS 31을 예약하고 독립 출력 head를 사용하며 native HF greedy 기준과 출력 ID 12개가 모두 일치한다.

## 검증과 한계

실행 코드 `8466cabbfb596a15dfe7a2c912ed3d32670b54ae`에서 library 543개 통과(21개 ignored), local-task oracle 7개, CUDA hybrid 5/5, CUDA CLI/HF 12/12를 확인했다. `bf8188d20177cde215ca1b54c6444b86396bd8bd`는 테스트 스위치 allowlist만 추가하며 guard와 Python 관련 테스트 15개가 통과했다.

외부 ROCm 검증은 bf8188d2에서 CPU/HIP 기본 연산 7개, 각 장치 hybrid 5/5, CLI/HF 12/12, B4/N8, 취소·슬롯 재사용, HTTP 동시 요청 4개의 텍스트 비교 통과를 보고했다. 실제 Nemotron-H-4B oracle/CLI 40/40 및 batch와 Mamba2 합성·실제 모델 회귀도 통과했다. HTTP 증거는 표시 텍스트 일치이며 HTTP token ID 검증은 아니다. 근거: https://github.com/lablup/mlxcel/pull/3411#issuecomment-7147301101.

배치 정책에 따라 hosted CI는 취소했으며 CI 전체 통과로 기록하지 않는다. 최적화 test-fast 검증은 정확성 증거이지 처리량 증거가 아니다. MoE 전체 f32 업로드와 모든 expert의 dense 계산은 #3350에서 확인할 제약으로 남으며 30B 용량이나 active-expert 성능을 주장하지 않는다.

## 학습 포인트

기본 연산 oracle 통과만으로 체크포인트 로딩, 컴파일러 융합, 서빙 수용 기준을 충족하지 않는다. 독립 전체 모델 기준과 정확한 소스 해시를 수치 검사와 함께 유지해야 한다.
