# PR #3461: Raon-1 mini decode, 로드, prompt 재사용 성능 (이슈 #3458)

**작성일**: 2026-11-27

**상태**: PR 열림(#3458을 닫음). M3 Ultra에서 `main` `477564e2e` 기준으로 측정. 24개 파일, +2025 / -39. CUDA와 ROCm은 실행하지 않았다.

**위험도**: 중간(near tie에서 plain greedy 텍스트가 바뀔 수 있음, 새 모델 소유 cache snapshot, 장치 쪽 NVFP4 transcode)

## 요약

Raon-1 mini 포트의 후속 여섯 가지다. 셋은 유지, 하나는 검증, 둘은 측정 후 넣지 않았다. 반복된 2132토큰 chat 턴의 TTFT가 1.00 s에서 0.04 s로, 단일 스트림 decode가 33.5에서 54.2 tok/s로, ModelOpt NVFP4 로드가 63~70 s에서 2.2 s로 줄었다. batched DFlash2와 sampled verify 블록당 host read 1회는 측정 결과 손해라 트리에 없다.

## 1. 문제 정의

- **prompt 재사용 없음.** Raon의 state는 RDA conv window, f32 recurrent state, MLA latent다. 서버에 이에 대한 모델 소유 snapshot이 없어 동일한 반복 요청도 cold prefill했다.
- **느린 decode step.** 약 8 ms weight floor 대비 33.5 tok/s. 이슈는 작은 dispatch가 많은 탓으로 가정했다.
- **느린 ModelOpt 로드.** 약 70 s, 대부분 host 쪽 작업.
- **NVFP4에서 DFlash2 미확인.** #3453과 #3455가 따로 머지됐다.
- **batched DFlash2와 sampled-verify host sync**는 #3455에서 남은 질문이었다.

## 2. 기술적 선택과 그 이유

### 2.1 decode 병목은 dispatch 수가 아니라 커널 하나

profiling 결과 step은 MLX의 transposed gemv 커널로 돈 1행 mHC selector product 하나에 묶여 있었다. row-major 형태 35 us 대비 255 us이고, 토큰당 50곳이다. 1행 product를 row-major 형태로 바꾸는 것만으로 51.5 tok/s가 되고, compiled mHC site graph가 나머지를 더하며 eager와 비트 단위로 같다. 그 product의 합산 순서가 바뀌어 plain greedy 텍스트가 near tie에서 이전 빌드와 다를 수 있다(essay 프롬프트에서 NVFP4 export 두 개 모두에서 관찰, 원본 체크포인트에서는 아님). Transformers trace는 prefill과 decode 경로 모두 결정 위치 불일치 0 / 578을 유지한다. `MLXCEL_RAON_DECODE_FUSION=mhc`는 34.5 tok/s로 이전 바이트를 유지한다.

이슈가 제안한 RDA projection 융합은 빼냈다. +0.6~+0.8%에 실제 폭에서 비트 일치가 아니었다(prefill trace 1024행 중 1008행 변화).

### 2.2 snapshot은 추측 대신 거부

Raon은 정확한 prefix snapshot(RDA conv window, f32 state와 offset, MLA latent pair)을 기부하고 복원한다. 다른 계열, 레이어가 빠진 snapshot, rank·batch·offset이 맞지 않는 RDA state는 복원을 거부하고 cold prefill한다. reasoning effort는 이미 cache 키에 있고(#3445), 다른 effort가 miss임을 scheduler 테스트가 확인한다. DFlash2 burst는 snapshot을 저장하지 않는다.

이후 hit가 cold 실행과 같도록 cold 서버 chat은 이제 history 경계에서 prefill을 나눈다. miss는 1.00 s 대신 1.14 s이고, cold 서버 텍스트가 near tie에서 `mlxcel generate`와 다를 수 있다.

### 2.3 장치 쪽 NVFP4 transcode

63~70 s 로드는 host weight copy 50 s, scale decode 10 s, re-encode 3 s였다. canonical triplet은 이제 장치에서 재해석되고, 이미 scale을 decode한 loader는 host 경로를 유지한다. plane은 host transcode와 바이트 단위로 같지만, scale 바이트 `0xFF`(NaN)를 더 이상 `0x7F`로 바꾸지 않는다. resident 메모리(16.39 GB)와 decode 속도는 그대로다.

### 2.4 측정 후 뺀 두 항목

| 항목 | 측정 | 결정 |
|---|---|---|
| batched DFlash2 | exact verify block이 2행에서 1행보다 42~59% 비싸고, batched decode step은 9%. 4행에서 상한이 classic batched decode보다 13~22% 낮다 | `supports_batched`는 `false`, B > 1 코드 없음 |
| sampled verify 블록당 host read 1회 | verify round당 49.5 대 48.7 ms(첫 불일치 뒤 행을 쓸데없이 샘플링) | `sampled_verify.rs` 변경 없음 |

#3459 이후 서버 기본 `presence_penalty` 1.5가 기본 요청을 batched speculative window에서 빼므로, batched DFlash2에는 더 싼 exact verify와 함께 행별 token history도 필요하다.

### 2.5 DFlash2 기본 width 5에서 3으로

decode step이 빨라지자 width 5는 prose에서 plain decode의 0.97x로 떨어졌다. width 3은 프롬프트·체크포인트 9쌍 평균 1.18x, width 5는 1.11x다.

## 3. 검증

| 확인 | 결과 |
|---|---|
| Transformers trace, 4 x 256 corpus | 결정 위치 불일치 0 / 578, prefill과 decode 경로 |
| greedy drafted 대 plain, CLI, 체크포인트 3개 | 54 / 54 동일 |
| 서버, cold / cache hit / drafted | 36개 텍스트, 차이 0 |
| NVFP4 export 두 개에서 DFlash2 | gate가 width 3, 5, 7 허용, drafted 36 / 36 plain과 동일 |
| snapshot 왕복과 거부, scheduler hit와 effort miss, compiled 대 eager mHC, 장치 transcode 경로 | 단위 테스트 |

`d43915faf`와 병합한 브랜치에서 로컬 `make verify` 전체를 통과했고, main과 마지막으로 병합한 뒤 대상 suite, fmt, clippy를 통과했다. 벤치마크 문서의 서버 측정은 #3459 이전이라 presence penalty 0에서 돌았다.

## 4. 후속 조치

- batched DFlash2(더 싼 exact 다중 행 verify와 batched window의 행별 token history 필요): 별도 이슈.
- 동일한 두 번째 요청은 여전히 miss이고 세 번째부터 hit한다. scheduler의 idle warm-up이 경계 entry를 대체하기 때문이다. 공유 scheduler 정책이라 별도 이슈.
- CUDA와 ROCm은 실행하지 않았다.
