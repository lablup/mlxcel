# PR #3414: Falcon-H1 병렬 믹서와 일반 CLI 출력 보존

**날짜**: 2026-11-25

**상태**: 통합 CPU/CUDA 검증 통과. 최종 head의 ROCm 재검증은 외부 환경에서 대기 중

## 요약

Falcon-H1의 원본 Hugging Face 가중치를 OpenXLA/IREE에서 실행한다. 각 계층은 동일하게 정규화된 입력으로 attention과 Mamba2를 계산하고 두 출력을 잔차에 더한 다음, 별도로 정규화한 MLP를 실행한다. Prefill, 단일 decode, ragged decode가 이 순서를 공유하며 계층마다 KV 상태와 recurrent 상태를 함께 유지한다.

검증 과정에서 공통 CLI 표시 버그도 확인했다. 일반 completion의 끝 공백이 채팅용 필터에서 제거되었다. 전체 CLI 출력은 이제 일반 completion의 바이트를 보존하고 reasoning, 도구 호출, Harmony, Muse 및 기본 채팅의 기존 정리 정책은 유지한다.

## 문제 정의

기존 `parallel_block`은 attention과 MLP의 병렬 실행을 뜻하므로 Falcon-H1 구조와 다르다. 원본 HF 가중치에는 muP 배율을 명시적으로 적용해야 하며, 미리 배율을 반영한 MLX 변환 가중치를 가정하면 출력이 달라진다. Attention과 SSM의 서로 다른 head 폭, 명시적인 중간 차원, 선택적 SSM 정규화, 독립적인 입력·출력 bias도 그대로 반영해야 한다.

이전 서버 검증 도구는 표시된 CLI stdout 대신 CLI 토큰 ID를 다시 디코딩했다. 외부 ROCm 검증에서 Germany 응답 끝의 공백 두 개와 줄바꿈을 HTTP는 보존하지만 CLI는 제거한다는 사실을 발견했다. 따라서 이전 결과는 원본 디코딩 텍스트와 토큰 수의 일치이며 실제 표시 텍스트의 엄격한 일치는 아니었다.

## 변경 내용과 기술적 선택

로더는 정식 키 `mamba_d_head`를 사용하고 원본 체크포인트 shape를 검증한다. Embedding, attention, key, SSM 채널, MLP 및 최종 출력 배율을 HF와 같은 위치에 적용한다. 선택적 grouped SSM 정규화는 두 gating 순서와 정규화 생략을 지원한다. 지원하지 않는 attention 스케줄과 RoPE 변형은 명시적으로 거부한다. 기존 recurrent f32 기본값과 prefix-cache 제한도 유지한다.

PR #3411 및 #3413과의 통합은 MoE 라우팅과 Mamba1 상태 구조를 보존한다. 정규화된 입력을 받는 공통 믹서 함수가 `SsmMixerW`로 분기한다. Falcon-H1은 `SsmKind::Mamba2`를 명시하며 전용 정규화, bias, 배율 규칙을 해당 variant에만 적용한다. 로더와 그래프 분기는 순수 Mamba1/Mamba2 경로를 모두 유지한다. 등록된 golden과 전체 모델 계열의 가중치 순서 일치 검증으로 통합을 확인한다.

`StreamFilter::flush`의 기본 채팅 계약은 그대로다. CLI가 명시적으로 선택하는 flush 모드는 구조 구분자를 소비하지 않았고 프롬프트가 reasoning을 미리 열지 않은 경우에만 끝 공백을 보존한다. 스트리밍 raw CLI는 원래부터 텍스트를 그대로 전달한다. 테스트는 문제의 공백 suffix, Unicode와 공백만 있는 텍스트, 청크 위치 계산, 열린 reasoning, 도구 호출, Harmony/Muse 채널 및 EOF 구분자 처리를 다룬다.

## 검증과 증거 범위

통합 실행 소스는 `8c7b4412d37b91807f8d688ba2f3febbc7a7e41f`이며 기반은 `2e46fc09e47208898e708d7090b196ec47b22357`이다. XLA crate 전체에서 560개 테스트가 통과하고 22개가 ignored였다. 등록된 golden의 바이트 일치와 모든 계열의 로더·emitter 가중치 일치도 포함된다. 동일한 통합 소스에서 표시 테스트 7개와 StreamFilter 테스트 146개도 모두 통과했다. CPU와 CUDA 모두 세 정규화 모드의 합성·ragged·슬롯 재사용 검증 및 실제 체크포인트의 독립 HF 40토큰 oracle 일치를 통과했다. CUDA CLI도 같은 40토큰 oracle과 일치했다. 실제 표시 stdout은 일반 HTTP 네 응답과 SSE 네 응답 모두와 일치했고 Germany의 끝 공백 두 개와 줄바꿈도 보존했다. 토큰 수도 일치했다. 두 요청 그룹 모두 동시 실행이 겹쳤고 직접 시작한 서버를 종료·회수했으며 GPU lock을 해제했다. Hybrid 회귀 5개와 Mamba2 단일 요청 회귀 6개가 통과했다. 포맷, Python 문법 및 `-D warnings`를 사용한 XLA 범위 clippy 검사도 통과했다. 최종 CI allowlist는 테스트 전용 Falcon 제어 변수 두 개를 분류한다. 환경 변수 검사(발견 398개, 문서화 265개, allowlist 133개)와 Python 회귀 테스트 15개가 모두 통과했다. 검증한 런타임 소스 이후의 변경은 이 CI 메타데이터와 검증 문서뿐이다. 커밋된 `integrated-validation.json` 및 `integrated-cuda-displayed-*` 파일에 결과, 바이너리 hash, Germany의 원본 stdout과 HTTP/SSE 응답을 기록했다.

이전 CPU/CUDA 소스 `205d0a3dd2d92eca2dbcba3dffcefba54e825eca`는 세 합성 정규화 모드, 네 ragged 궤적과 슬롯 재사용, 실제 Tiny90M의 HF 40토큰 oracle 및 batch 4 / requests 8의 170토큰 완전 일치를 통과했다. 해당 HTTP 증거는 토큰 재디코딩을 사용했으므로 위에서 그 범위를 바로잡았다.

외부 ROCm gfx1151 검증은 `852129eb1b4bf45d957f4c63b082d9d0276e4951`에서 수치 비교, 실제 40토큰 CLI, batching·슬롯 재사용, recurrent 회귀 및 release 아키텍처 검증을 통과했다. 표시 CLI/SSE 텍스트의 엄격한 일치는 네 프롬프트 중 세 개였다. 네 SSE 문자열 모두 CLI 토큰의 원본 디코딩과 일치했다. [검증 기록](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147795531)과 [공백 원인 분석](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147960509)에 근거한다.

커밋된 `falcon_h1_cli_server_check.py`는 실제 CLI stdout을 각각 네 개의 동시 40토큰 일반 HTTP completion 및 SSE completion과 비교한다. 생성 공백, 토큰 수, 스트림 종료를 검증하고 바이너리 hash와 원본 응답을 기록하며 자신이 시작한 서버를 종료·회수한다. HTTP는 출력 토큰 ID를 제공하지 않으므로 이 증거는 정확한 텍스트·토큰 수 일치다.

## 학습 내용과 후속 작업

출력 일치는 사용자가 실제로 보는 바이트를 비교해야 한다. 토큰 ID에서 텍스트를 복원하면 표시 단계 변형을 놓칠 수 있다. 채팅 정리와 raw completion 보존에는 명시적인 호출자 정책이 필요하다.

최종 통합 head에서 공통 출력 동작과 병합된 recurrent 경로에 대한 외부 ROCm 재검증을 요청한다. 커밋된 안내에 따라 `MLXCEL_XLA_DEVICE=hip`, `MLXCEL_XLA_ROCM_TARGET=gfx1151`, `MLXCEL_XLA_PRECISION=f32`를 명시하고 `scripts/rocm_gpu_guard.sh --hold --`로 실행한다. 이전 HIP 증거, 새 CPU 테스트, CUDA 재검증 및 취소된 hosted CI는 서로 다른 관측이며 새로운 HIP 실행을 대신하지 않는다.
