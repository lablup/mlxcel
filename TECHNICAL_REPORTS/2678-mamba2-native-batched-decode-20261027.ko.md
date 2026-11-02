# 기술 보고서: PR #2678 — Mamba2 네이티브 배치 디코드

**작성일**: 2026-10-27
**상태**: 검증 완료; merge 대기
**위험도**: 중간

## 요약

Mamba2, Nemotron-H, Granite Hybrid, Falcon-H1, PLaMo2에 독립적인 recurrent 상태와 서로 다른 attention 캐시 길이를 지원하는 네이티브 디코드를 추가했다. 완료된 다섯 개 서빙 측정에서 동시성 4의 총 처리량은 기준 대비 2.00–3.43배였다. Admission을 수정한 Nemotron은 전체 서빙 측정을 통과했다. Granite Vision은 자유 생성 출력 동일성을 유지하도록 wrapper의 행별 fallback을 적용했으며, 최종 세 요청 동시 실행 결과는 모두 단독 실행과 일치하고 텍스트 tower는 네이티브 배치를 유지한다.

## 문제 정의

각 계열은 이미 시퀀스별 상태를 보유했지만 서빙 디코드 행 수를 1로 제한했다. 배치 활성화에는 상태 공유 없이 recurrence를 결합하고, 단일 행 실행을 보존하며, 실제 attention 메모리를 계산하고, provider의 admission 절차를 scheduler와 일치시키는 작업이 필요하다.

## 기술적 선택과 그 이유

공유 `recurrent_batch` 모듈은 axis 0에서 행을 결합하고 분리 결과를 복사하여 한 시퀀스가 전체 배치 버퍼를 붙잡지 않도록 한다. 초기화 상태가 섞인 경우 0 상태를 만들어 넣지 않고 ID를 전달하는 행별 실행으로 전환한다. 임베딩, projection, recurrence, feed-forward 연산을 함께 배치 처리하고 attention은 시퀀스별 offset을 유지한다. 잘못된 배치 ID는 상태 변경 전에 거부한다. Paged shadow 저장소는 attention 길이를 반영하고 recurrent 슬롯은 0으로 유지한다.

Mamba2의 네이티브 SSD carry에서 잘못 추가되던 broadcast 축을 제거하고, scalar carry의 선택적 singleton 토큰 축은 결합할 때만 정규화한다. B=1의 기존 그래프는 보존하며 native→B1→native 회귀 테스트로 rank 전환을 검증한다. Nemotron의 fused projection 조건은 bias를 가진 affine 양자화 입출력 projection을 요구하여 지원되는 실제 체크포인트 경로를 유지하면서 float 및 비-affine projection에는 기존 fallback을 적용한다. Nemotron Nano Omni는 네이티브 디코드를 텍스트 tower에 위임한다. Granite Vision은 ID를 유지하는 wrapper의 행별 디코드를 사용하며, 텍스트 tower는 네이티브 배치를 유지하고 두 wrapper 모두 모델 소유 저장소를 동기화한다.

실제 서빙에서 Nemotron admission 불일치가 드러났다. 여러 행을 처리하는 scheduler는 일반 paged admission을 사용해 single-slot request-memory handshake를 보내지 않지만 HTTP provider는 이 이벤트를 기다렸다. `ce00333c`는 이 대기를 설정상 single-slot인 scheduler로 제한하고 Done-only provider 회귀 테스트를 추가한다. 수정된 서버는 전체 서빙 재측정을 통과했다. 앞선 HTTP-500 실행은 성능 결과에서 제외한다.

## 검증 및 측정

집중 GPU 테스트 16개, 라이브러리/테스트 검사와 rebase 후 coverage 13개, wrapper 테스트 2개, 포맷 및 대상 strict clippy를 통과했다. 테스트는 6단계 상태 격리, 정확한 B=1, 빈 배치와 잘못된 입력, 새 상태와 혼합 상태, native/scalar 전환, Granite 위치 모드, paged 메모리 계산, VLM 위임을 다룬다. 통합 수정 후 main 라이브러리 테스트 10,603개와 integration 테스트가 통과했다. Seeded core device-lock 회귀 테스트는 14/14가 통과했다. 수정 후 core 재실행은 2,199개 통과(4개 ignored), core doctest는 2개 통과(6개 ignored), 나머지 crate는 618개 통과(19개 ignored)였다. Main doctest는 1개 통과(4개 ignored)했으며 명령 종료 코드는 0이었다. 이는 분리 실행한 검증 범위이며 하나의 연속된 workspace 명령이 성공 종료한 것으로 표현하지 않는다. 전체 workspace all-targets clippy는 `098b569a`에서, 수정된 core-test clippy와 포맷은 `12182cad` 이후 통과했다. 후속 source guard와 오래된 Nemotron VLM assertion 수정은 테스트만 변경하며 runtime 동작은 바꾸지 않는다.

환경은 Apple M3 Ultra, 512 GB, macOS 26.6.2와 release `metal,accelerate`다. 네 텍스트 계열은 기준 `a29a2692`와 후보 `00e33351bf5bb1ac63f9d0b6ce22f3845caf272a`를 사용한다. 이후 main PR #2679가 Nemotron의 Metal sorted prefill을 변경해 Nemotron만 공통 변경을 포함하는 기준 `b4cdf023`와 후보 `1249c337`로 재측정했다. 나머지 네 구현은 이 merge로 바뀌지 않았다. 기준 trace에는 ID가 있는 prefill 후에도 시퀀스 ID를 유지하는 명시적 scalar-reference 측정 수정을 적용했다. 다섯 체크포인트 모두 greedy 텍스트가 바이트 단위로 일치하며, 기록된 B=1 trace 256개 위치도 logit 차이 0으로 일치한다.

| 체크포인트 | 기준 총 처리량 1/2/4 tok/s | 후보 총 처리량 1/2/4 tok/s | B=4 top-1 불일치 | Decided 불일치 | Trace PPL 변화 |
|---|---|---|---:|---:|---:|
| granite-4.0-h-tiny-4bit | 97.3 / 109.1 / 129.9 | 120.0 / 173.3 / 260.2 | 10/256 | 0/105 | +0.755% |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 94.0 / 71.0 / 91.8 | 88.6 / 211.4 / 313.5 | 6/256 | 0/113 | +0.361% |
| mamba2-1.3b-4bit | 85.0 / 92.3 / 100.0 | 98.2 / 157.9 / 283.4 | 1/256 | 0/109 | +0.053% |
| Falcon-H1-Tiny-90M-Instruct-4bit | 180.8 / 194.4 / 165.5 | 208.0 / 324.7 / 567.2 | 4/256 | 0/92 | -0.448% |
| plamo-2-1b | 88.3 / 95.2 / 103.8 | 99.3 / 129.2 / 273.5 | 0/256 | 0/97 | 0.000% |

Nemotron의 warmed B=1 decode 다섯 표본은 기준에서 141.1, 141.3, 139.8, 139.8, 140.3 tok/s, 후보에서 140.0, 139.4, 138.5, 137.5, 137.2 tok/s였다. 중앙값은 140.3→138.5 tok/s(-1.3%)이며 관측 범위는 겹친다. 다섯 표본만으로 통계적 동등성을 주장하지 않는다. 반복 측정은 09:41:27 UTC에 종료됐고 집중 GPU 테스트는 이후 09:42:45 UTC에 시작했다.

동시성은 클라이언트 동시성이며 총 처리량에는 서빙 오버헤드가 포함된다. Decided는 기준 상위 두 logit 차이가 2 이상인 위치다. Decided 불일치가 0이어도 전체 logit이나 자유 생성 텍스트가 동일하다는 의미는 아니다. 서빙은 `--parallel 4`, `--prompt-tokens 512`가 요청하는 명목상 약 512토큰의 텍스트 prompt, 생성 상한 128토큰을 사용한다. 실제 tokenized prompt 길이는 tokenizer마다 다르며 Nemotron 로그는 448이었다. Trace는 B=1/B=4에서 `MODEL CORPUS 64 4 8 512`를 사용한다. 여기서 512는 최대 prefill/history 예산이며 모든 위치의 고정 context 길이가 아니다. Chunk는 corpus 위치 0에서 시작하고 사용 가능한 history가 증가한다. 기준에는 `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`을 적용하고 네이티브 후보에는 적용하지 않는다.

## VLM 결과와 최종 검증

Nemotron Nano Omni에서는 텍스트 요청 두 개와 이미지 요청 한 개를 동시 실행한 결과가 모두 단독 실행과 일치했다. Granite Vision의 이전 네이티브 wrapper 자유 생성 진단은 이미지 요청만 일치해 `[false, false, true]`였고 기준 구현은 세 요청 모두 일치했다. 제약된 Granite prompt에서는 세 요청 모두 일치했지만, 원래 자유 생성 기준의 실패를 대체하지 않는다. 해당 네이티브 wrapper의 이미지 조건부 강제 trace는 top-1 불일치 4/64, decided 불일치 0/11, 최대 불일치 gap 0.125, perplexity 122.3245→122.9666(+0.525%)였다.

최종 Granite Vision wrapper는 `PerRowFallback`을 사용하고 텍스트 tower는 네이티브 배치를 유지한다. 원래 자유 생성 workload를 다시 실행한 결과는 `[true, true, true]`였으며 서버는 clamp 없이 행별 fallback을 보고했다. 최종 Nemotron Nano Omni도 원래 자유 생성 요청 세 개가 모두 일치했고 clamp 없는 네이티브 배치를 보고했다. 따라서 원래 요구를 완화하지 않고 이 workload의 VLM 출력 동일성을 검증했다. 앞선 Granite 네이티브 진단은 fallback 선택의 근거로 보존하며, 최종 fallback 경로의 측정값이 아니다. main `cfa14e7b`의 vision loading 변경 후 `098b569a9aac3eac151d46f721a5b88cb1964478`에서 두 VLM workload를 다시 실행해 각각 3/3 정확한 일치와 clamp 없는 fallback/native 경로를 확인했다. 새 원본에는 명령과 소스 정보, 관측한 서버 바이너리 및 내장 Metal 라이브러리 경로의 파일 해시를 기록했고 이전 진단은 원래 revision을 유지한다. 앞서 설명한 분리 실행으로 검증을 완료했다. HEAD `12182cad`는 통합 runtime 측정 revision `098b569a` 이후 테스트 변경만 추가한다.

## 변경 요약

공유 recurrent helper, 모델 tower 다섯 개, VLM wrapper 두 개, 상태 격리 harness, coverage 목록, provider admission 조건과 continuous-batching 문서를 변경했다. 새로운 SSM kernel이나 의존성은 추가하지 않는다. 관련 이슈는 #2571, #2574, #2555이며 후속 계열은 공유 helper를 재사용한다. 간결한 측정 근거는 `docs/benchmark_results/data/native-batched-decode-m3ultra-2026-10-27/mamba2-families.json`에 기록한다.

## 통합 검증 수정

전체 테스트의 병렬 실행에서 BitLinear 커널 초기화 경쟁이 드러나 Metal·CUDA·ROCm holder에 `std::call_once`를 적용했다. Metal에서 BitNet 테스트 5개가 assertion을 완화하지 않고 병렬로 통과했다. CUDA·ROCm 런타임은 이 호스트에서 실행하지 않았다. Paged decode 카운터 assertion은 테스트 전용 스레드 로컬 관찰값을 사용하며 공개 운영 통계는 프로세스 전체 집계를 유지한다.
