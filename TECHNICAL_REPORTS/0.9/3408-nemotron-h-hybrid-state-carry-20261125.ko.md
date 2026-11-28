# PR #3408: Nemotron-H 하이브리드 상태 전달

**작성일**: 2026-11-25

**상태**: CUDA 및 ROCm 수용 검증 통과

## 요약

이 PR은 Nemotron-H 하이브리드 스케줄을 OpenXLA/IREE에서 직접 실행할 수 있게 한다. NoPE attention 레이어는 압축 KV 행을 사용하고, Mamba2 레이어는 recurrent state를 전달하며, relu2 전용 레이어는 두 mixer를 모두 생략한다. 설정 파싱, 가중치 순서, 체크포인트 검증, prefill, 단일 시퀀스 decode, continuous batching이 동일한 스케줄을 사용한다.

## 문제 정의

기존 recurrent engine은 모든 레이어가 attention 전용이거나 Mamba2 전용이라고 가정했다. Nemotron-H는 Mamba2(`M`), attention(`*`), feed-forward 전용(`-`) 블록을 섞으며, attention projection 폭과 SSM inner 폭이 hidden 폭과 독립적이다. 이를 dense KV 모델로 처리하면 불필요한 상태를 할당하고 가중치 순서가 어긋난다. 순수 recurrent 모델로 처리하면 attention 상태가 사라지고 batch slot의 admit과 재사용을 원자적으로 처리할 수 없다.

공식 `nvidia/Nemotron-H-4B-Base-8K` 체크포인트는 SSD chunk size를 256으로 지정한다. 이 값을 유지해야 reference와 같은 reduction partition 및 fp32 token trajectory를 비교할 수 있다.

## 변경 요약

Emitter는 하이브리드 스케줄을 파싱하고 attention KV와 recurrent state에 별도의 압축 인덱스를 만든다. NoPE attention은 hidden 폭 3072와 projection 폭 4096이 다른 체크포인트의 128차원 head를 그대로 사용한다. Mamba2는 inner 폭 7168, grouped gated RMSNorm, f32 recurrence를 사용한다. Feed-forward 전용 블록은 체크포인트의 nongated `relu2` 연산을 생성한다. MoE `E` 블록은 #3345에서 sigmoid router를 제공할 때까지 명시적으로 거부한다.

Runtime은 KV와 SSM 행을 함께 전달하고 필요한 모든 전송이 끝난 뒤에만 slot을 valid로 표시한다. Admit 실패, 잘못된 행, 부분 저장은 재사용 전에 validity를 제거한다. Recurrent 계열의 prompt cache와 APC는 계속 비활성화된다. 체크포인트 검증과 weight-name mapping은 graph emission과 같은 스케줄을 따른다.

등록된 StableHLO golden은 hidden 폭과 attention 폭이 다르고 KV head 2개, SSM group 2개를 사용하는 합성 `M*-M-` 모델을 다룬다. IREE 통합 테스트는 단일 reference 일치, staggered admission, slot 재사용, batch 점유율, allocation 재사용, decode 실패 시 무효화, KV 일부 저장 뒤 실패, KV 전체와 첫 SSM tensor 저장 뒤 실패를 검증한다.

## 기술적 선택과 그 이유

### KV와 recurrent state에 별도의 압축 인덱스 사용

체크포인트 이름과 residual 순서는 원래 레이어 위치를 기준으로 유지하고, attention 및 SSM tensor는 필요한 레이어에만 저장한다. 따라서 recurrent 또는 feed-forward 전용 블록에 KV를 할당하거나 attention 블록에 recurrent state를 할당하지 않는다.

### Recurrent state를 KV와 원자적으로 commit

Hybrid slot은 쓰기 전에 무효화하고 모든 KV 및 SSM 전송이 성공한 뒤에만 공개한다. 테스트 훅은 K 저장 뒤와 KV 전체 및 첫 SSM tensor 저장 뒤에 각각 실패시킨다. 두 torn-store 경계의 복구를 코드 검토에만 의존하지 않고 실제로 실행한다.

### 체크포인트의 수치 파라미터 보존

Parser는 표준 `chunk_size`를 읽고 기존 `mamba_chunk_size` fallback을 유지하며 잘못된 값을 거부한다. 실제 체크포인트의 값 256을 보존한다. 합성 fixture는 128을 명시해 이 수정이 기존 golden을 바꾸지 않게 한다.

## 검증

커밋 `93ddebad382478540fdfc987fbca655744be941d`에서 등록된 모든 byte-exact golden, Nemotron-H 구조 테스트 5개, recurrent 구조 회귀 27개가 통과했다. IREE local-task와 NVIDIA GB10(driver 580.178.04) CUDA에서 합성 실행 테스트 5개가 모두 통과했으며, CUDA GPU 작업은 `gpu-lock`으로 실행했다. XLA library 범위 clippy는 `-D warnings`로 통과했고 변경된 Rust 파일의 formatting도 통과했다.

독립적인 Transformers 5.12.1 CPU fp32 실행은 체크포인트 revision `faba3b731ad7ea5781b9518ae75fb610a94affcf`의 tensor 311개를 missing 또는 unexpected-weight 경고 없이 로드하고 provenance hash가 포함된 40-token oracle을 생성했다. 이는 reference artifact를 확립한 결과이며 XLA와의 일치를 의미하지 않는다.

같은 커밋에서 실행한 실제 4B CUDA 검사는 원본 체크포인트, context 256, f32 실행, NVIDIA GB10, driver 580.178.04 조건에서 커밋된 HF fp32 oracle 40개 token과 모두 일치했다. `gpu-lock` 아래에서 정상 종료했으며 전체 로그는 `/tmp/epic564-plan/3344-cuda-real-oracle.log`에 있다. 실행 파일은 `test-fast` build profile을 사용했으므로 이 결과는 token correctness만 입증하며 release build의 성능을 주장하지 않는다.

실제 체크포인트 CUDA batch gate도 B=4에서 요청 8개를 처리하고 context 256 및 fp32 조건에서 생성한 170개 token을 요청별 단일 시퀀스 reference와 모두 일치시켰다. Test-fast의 참고용 처리량은 batch 4.974 tok/s, sequential 6.02 tok/s(0.8x)였으며 release benchmark 또는 성능 개선 근거가 아니다. 원시 oracle 및 batch 출력은 `spike/openxla/nemotron_h_reference/`에 포함한다.

Test-fast CLI는 HF oracle token ID 40개와 모두 정확히 일치했다. HTTP 요청 4개는 16.958초 동안 겹쳐 실행됐고, 각 요청의 completion token 24개에 대한 text와 count가 별도로 캡처한 CLI reference와 일치했다. Server API는 token ID를 노출하지 않으므로 HTTP 검증은 rendered text 및 count 일치를 입증한다. 성공한 server는 `--ctx-size 2048`, B=4, 명시적 XLA capacity 256, prompt cache/APC 비활성화 조건을 사용했다. 앞선 `--ctx-size 1024` 시도는 slot당 256 token이 최소값 512보다 작아 정상적으로 거부됐다.

CUDA에서 `xla_ssm_single_seq` 테스트 6개, `xla_ssm_state_carry` 테스트 5개, 별도의 `xla_hybrid_nemotron_h` 테스트 5개가 통과했다. Nemotron-H prompt-cache 제외 unit test도 통과했다. 검증 binary는 `93ddebad` 소스에서 빌드했고 CLI/server 증거는 문서만 변경된 `d7e8a040` checkout에서 캡처했다. 원시 증거와 binary hash는 `spike/openxla/nemotron_h_reference/`에 포함한다.

ROCm gfx1151도 실행 소스 `93ddebad382478540fdfc987fbca655744be941d`에서 수용 검증을 통과했다. HF oracle을 독립 재생성했고 IREE-HIP 및 CLI 40/40 token, B4/N8 170-token, 동시 streaming HTTP 4개 출력 문자열 일치를 확인했다. HIP과 local-task에서 hybrid 테스트 5개와 SSM probe 11개가 각각 통과했으며 Mamba2 단일 시퀀스·상태 전달 및 실제 체크포인트 회귀도 통과했다. ROCm HTTP token ID와 usage count는 수집하지 않았으므로 HTTP 증거는 문자열 일치에 한정한다. 모든 빌드와 장치 작업은 ROCm guard를 사용했고 runtime은 release 빌드였으며 성능 향상을 주장하지 않는다. 명령과 로그 경로는 [ROCm 수용 기록](https://github.com/lablup/mlxcel/pull/3408#issuecomment-7138393314)에 있다. 이번 실행의 local-validation 정책에 따라 원격 CI는 취소했으며, 취소를 CI 통과 증거로 간주하지 않는다.

## 학습 포인트

하이브리드 모델 지원에는 설정, tensor 이름, 체크포인트 shape, graph emission, runtime state를 관통하는 하나의 authoritative schedule이 필요하다. 각 영역을 따로 검증하면 compile은 되지만 다른 레이어의 가중치를 읽거나 재사용한 slot의 상태 일부만 노출하는 graph가 만들어질 수 있다.

내부 단일 시퀀스 reference는 batch consistency를 증명하지만 architecture correctness를 증명하지 못한다. 따라서 token acceptance는 별도로 생성한 HF fp32 oracle, 고정된 체크포인트 revision, 정확한 token ID를 사용한다.

## 후속 조치

- [x] 정확한 실행 소스 SHA를 기록하고 ROCm gfx1151의 실제 모델, batching, server, Mamba2 증거를 남겼다([수용 기록](https://github.com/lablup/mlxcel/pull/3408#issuecomment-7138393314)).
- [ ] Nemotron-H MoE 스케줄을 활성화하기 전에 #3345에서 `E` 블록을 구현하고 검증한다.

## 참고 자료

- 이슈 #3344
- 에픽 #564
- `spike/openxla/NEMOTRON_H_VALIDATION.md`
- `spike/openxla/nemotron_h_reference/`
