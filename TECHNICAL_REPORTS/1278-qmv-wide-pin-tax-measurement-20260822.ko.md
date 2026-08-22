# 기술 보고서: PR #1278 - qmv_wide narrow pin이 나머지 프로세스에 물리는 비용 측정

## 요약

이슈 #1261은 MTP exactness gate가 temperature-0 byte-identity를 되사려고 `qmv_wide`를 프로세스 전체에서 꺼버릴 때, 그 프로세스의 나머지 작업이 무엇을 지불하는지 물었다. 이슈는 의도적으로 측정을 선행 조건으로 걸었고("숫자가 나오기 전에는 아래 어떤 것도 만들 가치가 없다"), 명시적인 종료 조건을 함께 달았다. 프로덕션 shape에서 비용이 작으면 숫자를 기록하고 거기서 멈추라는 것.

이 PR은 그 종료 조건을 택한다. Rust 소스는 한 줄도 바뀌지 않는다. 벤치마크 하네스 두 개를 추가하고, 이슈가 요구한 Apple GPU generation 15 호스트에서 측정을 기록하며, #1199가 gate의 순서를 바꾸면서 낡아버린 `MLXCEL_MTP_ALLOW_INEXACT` 레시피를 문서 세 곳에서 바로잡는다.

측정된 답은 batched decode 비용이 최대 1%라는 것인데, 이유가 수치적이라기보다 구조적이다. 이슈가 가정한 `M = B` projection을 두 MTP 계열 모두 dispatch하지 않는다. 실제로 존재하는 비용은 요청당 prompt cache에 붙는 suffix prefill 한 번이고, Gemma 타깃에서 forward당 +15.4 ms, Qwen 타깃에서 +12.6 ms다.

## 1. 문제

PR #1199는 exactness gate에 retry를 붙였다. multi-token verify block이 `qmv_wide` 아래에서 single-token decode chain과 갈라지면 gate가 `qmv_wide`를 끄고 다시 probe하며, 그렇게 해서 byte-identity가 돌아오면 프로세스가 끝날 때까지 꺼둔다. 스위치를 되돌리지 않는 것은 의도적이다. 다시 켜면 gate가 방금 승인한 바로 그 block이 깨지기 때문이다.

문제는 이 스위치가 프로세스 전역이라는 점이다. `dispatch_qmv`에서 모든 quantized matmul의 dispatch 경로에 놓여 있어서, 프로세스를 공유하는 서버는 byte-identity를 요구한 적 없는 작업에도 이 비용을 문다. #1199는 이 점을 명시하고 scoping 작업을 후속으로 넘겼다. #1261이 그 후속이고, 첫 단계는 수술을 정당화할 만큼 부수 비용이 큰지 확인하는 일이었다.

verify 쪽 비용은 이미 가격표가 붙어 있었다(Qwen verify forward 17~20%, Gemma 4 verify forward 약 23%). 옆에 있던 작업이 무엇을 물었는지는 한 번도 측정된 적이 없었다.

## 2. 무엇을 측정했나

Mac Studio M3 Ultra(`applegpu_g15d`, generation 15, macOS 26.6.1)에서 `scripts/with_indexers_paused.sh` 아래, Time Machine을 끈 상태로 두 arm을 돌렸다.

**arm 1, drafter 없는 batched-decode B-sweep.** `MLXCEL_QMV_WIDE=1` 고정과 `MLXCEL_QMV_WIDE=0` 고정을 B = 1, 2, 4, 8에서 비교했고, 타깃은 `models/gemma-4-31b-it-4bit`, 확인은 `models/qwen3.8-27b-4bit`로 했다. 균형 잡힌 ABBA 블록 두 개로 8 boot, boot마다 warm-up 한 pass를 버리고 측정 pass를 두 번 돌려 cell당 arm당 8 샘플을 얻었다(Qwen 확인은 4). 결과는 Gemma 0.0~0.9%, Qwen 0.0~0.2%였고 모든 spread가 1% 이하로, 하네스의 4% 신뢰 한계 안에 넉넉히 들어온다.

**arm 2, 혼합 워크로드.** Qwen MTP 페어링에서 tick-slice speculative slot을 붙잡은 MTP 스트림 하나에 classic 스트림 넷을 붙였고, arm A는 gate의 retry가 프로세스를 narrow로 고정하는 기본 env, arm B는 `MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1`이다. 읽는 값은 classic 스트림의 decode rate뿐이다. classic 스트림은 0.1~2.6%를 잃었는데, 그 손실은 classic 스트림 자신의 kernel이 아니라 narrow verify forward가 공유 worker를 더 오래 점유하는 데서 온다.

## 3. 기술적 판단

### 3.1 Step 2를 만들지 않고 이슈의 종료 조건을 택한다

이슈의 Step 2는 exact kernel을 verify forward로 좁히는 방법을 세 가지 제시했다. 명시적 동기화로 구간을 감싸는 방식부터 dispatch 쪽 stream predicate까지. 아무것도 만들지 않았고, B-sweep 차이가 작을 때는 그쪽이 이슈가 승인한 결말이다.

이 결말을 방어하는 것은 숫자가 작다는 사실이 아니라 그 뒤의 메커니즘이다. `M = 1`은 항상 `qmv`로 가고, pin은 `M >= 2`에서만 `qmv`와 `qmv_wide` 사이를 고른다. 두 MTP 계열 모두 decode batch를 forward당 한 row씩 처리하므로, 고정된 프로세스의 batched decode는 pin이 끄는 kernel에 애초에 닿지 않는다. Gemma 3와 Llama 4는 decode row를 하나의 `M = B` forward로 쌓지만 MTP 계열이 아니어서, 그들이 도는 프로세스는 gate에 고정될 일이 없다.

기록은 이 답이 뒤집히는 경계도 함께 지목한다. MTP 계열이 진짜 joint batched decode를 갖는 순간이다. Gemma 4나 Qwen 3.5가 Gemma 3처럼 decode row를 쌓기 시작하면 batched decode가 qmv window로 들어가고 pin이 거기에 세금을 물리기 시작한다.

### 3.2 gate에 맡기지 않고 두 arm을 모두 명시적으로 고정한다

`qmv_wide_pinned_by_operator()`는 `std::env::var("MLXCEL_QMV_WIDE").is_ok()`를 읽는다. 값이 `0`이든 무엇이든 변수를 설정한 것 자체가 operator pin이고, gate의 retry는 양방향 모두 건너뛴다. 두 arm에 모두 걸어야 B-sweep이 깨끗한 A/B가 되는데, 그래야 gate가 run 도중에 arm을 뒤집을 수 없기 때문이다. drafter를 안 띄우면 돌릴 probe 자체가 없고, `mlxcel_core::set_qmv_wide`의 호출자는 gate 하나뿐이라 다른 무엇도 이 flag를 움직이지 못한다.

### 3.3 오염된 cell은 평균에 섞지 않고 그대로 보고한다

long-context B = 4 cell은 narrow arm 쪽 throughput이 오히려 높게 나왔고, 16 샘플 전체에서 안정적으로 그랬다. kernel 효과가 아니다. temperature 0에서도 두 kernel은 정당하게 다른 텍스트를 만들고, 이 프롬프트에서는 그 분기가 생성 길이 자체를 바꿨다(wide 130 토큰, narrow 182 토큰). 두 컬럼이 서로 다른 워크로드를 비교하고 있는 셈이다. 그래서 이 cell은 text-divergence로 오염됐다고 적고 비용 집계에서 뺐다. TTFT 컬럼은 prefill이 생성보다 앞서므로 영향을 받지 않고, chunked-prefill 대조군으로 그대로 쓴다.

ladder cell들은 애초에 이 문제를 피하도록 구성됐다. 스트림별 `usage` 카운트로 확인한 결과 두 arm 모두 생성이 59 토큰에서 끝났고, 중간 바이트는 갈라져도 rate 비교는 길이가 맞춰져 있다.

### 3.4 낡은 `MLXCEL_MTP_ALLOW_INEXACT` 레시피를 바로잡는다

`mtp_exactness_gate`는 `allow_inexact()`를 보기 전에 `retry_without_qmv_wide`를 먼저 돌린다(`let decision = exact || allow_inexact();`). narrow retry가 통과하는 호스트, 즉 지금까지 측정된 모든 generation 15+ 호스트에서는 override 단독이 무력하다. retry가 프로세스를 먼저 narrow로 고정해버려서 flag가 결정에 관여할 자리가 없다. #1199가 머지되기 전에 쓰인 문서들은 override만으로 fast kernel에 닿는다고 적어두었는데, 이제는 그렇지 않다.

이 정정은 주장이 아니라 서로 독립적인 세 갈래로 검증됐다. 로그 줄(override만 건 실행은 retry의 INFO 줄을 남기고 큰 경고는 끝내 뜨지 않는다), 바이트(기본 실행과 override 단독 실행의 생성 텍스트가 byte-identical이고, wide로 고정한 실행은 여섯 단어째에서 갈라진다), 그리고 throughput(117 대 139 tok/s로, `docs/benchmarks.md`가 이미 싣고 있는 byte-identical 수치와 fast kernel 수치를 재현한다). 동작하는 레시피는 `MLXCEL_QMV_WIDE=1`과 `MLXCEL_MTP_ALLOW_INEXACT=1`을 함께 거는 것이다.

### 3.5 concurrency 하네스가 `reasoning_content` delta를 센다

Qwen 3.8은 reasoning 채널을 `reasoning_content` delta로 흘린다. `bench_serving_concurrency.py`는 `content`만 세고 있었고, 예산 전부를 생각에 쓰는 요청은 TTFT도 decode rate도 아예 보고되지 않았다. 두 채널 모두 디코드된 토큰이므로 이제 둘 다 센다. reasoning이 없는 모델에는 `reasoning_content` 자체가 오지 않으니 기존에 기록된 측정값의 의미는 달라지지 않는다.

## 4. 리뷰에서 나온 지적과 수정

구현 리뷰는 결론을 떠받치는 코드 주장 두 개를 소스에 직접 대조했고, 그중 하나가 측정이 지나가지 않은 경로를 인용하고 있었다.

**Gemma 4 메커니즘 인용(수정됨).** 기록은 Gemma 4가 "`forward_batched`를 override하지 않으므로 trait default를 상속한다"고 적었다. `src/models/gemma4.rs`에 대해서는 맞는 말이지만, `models/gemma-4-31b-it-4bit`는 `embed_vision.*` 가중치를 들고 있어서 `gemma4_has_vision_weights`가 이를 `LoadedModel::Gemma4VLM`으로 보낸다. 스케줄러의 `execute_batched_decode`는 batch의 sequence id를 실어 `forward_batched_with_context_and_ids`를 호출하고, `Gemma4VLModel`은 바로 그 진입점을 override해서(`src/vision/gemma4_vl.rs:687`) `forward_batched_with_seq_ids_dispatch`로 넘긴다. 측정한 구성에서 trait default에는 닿지 않는다.

결론은 살아남고 오히려 더 단단한 근거 위에 선다. 그 dispatch helper 자체가 `forward_with_sequence_id`를 도는 명시적 per-row 루프이기 때문이다. 다만 이 기록의 가치는 나중에 읽는 사람이 다시 검증할 수 있다는 데 있고, 인용을 따라간 독자는 override를 발견하고 주장이 틀렸다고 판단했을 것이다. 기록은 이제 두 경로를 모두 서술하고 측정이 탄 쪽을 지목한다.

Qwen 쪽 주장은 손댈 데가 없었다. `src/models/qwen3_5.rs:3388`은 `shape[1] <= 1`이면, 즉 모든 decode step에서 per-row 루프로 분기하고, `Qwen35VLModel` 래퍼는 같은 함수로 그대로 위임하므로 인용이 두 variant 모두에 대해 정확하다.

**수치의 이름(수정됨).** `docs/benchmarks.md`에는 23%가 둘 있다. Gemma 4 계열의 verify forward 비용과 M5 Max code row의 end-to-end decode 비용인데, 문서 스스로 둘을 섞지 말라고 경고한다. 기록은 어느 쪽인지 밝히지 않은 채 "Gemma 4에서 ~23%"라고 적었고, Qwen 페어링만 측정한 절에서 "17~23%"라는 합성 범위를 인용했다. 이제 둘 다 수치의 정체와 계열을 명시한다.

**피팅 오차(수정됨).** suffix forward 피팅이 "모든 row에서 0.1 ms 이내"로 일치한다고 적었으나 B = 8 row는 0.2 ms 벗어난다(4.5 곱하기 15.4는 69.3, 측정값은 69.5).

검증해서 정확한 것으로 확인된 항목은 다음과 같다. C++ flag 파싱과 `qmv_wide_pinned_by_operator`에 대조한 `MLXCEL_QMV_WIDE` 문서 행, 인용된 네 가지 gate 레시피 로그 줄과 실제 `tracing` 호출, `set_qmv_wide`의 호출자가 하나뿐이라는 주장, `applegpu_g15d` 파트에서 31B 타깃의 모든 projection에 대해 qmv batch limit이 12라는 계산, `mtp_capable_target`의 계열 목록, 그리고 Gemma 3와 Llama 4의 joint decode 반례.

## 5. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `scripts/bench_qmv_wide_pin.sh` | 신규. 두 arm을 모두 구동하는 ABBA boot 드라이버. `sweep`은 drafter 없이 wide 고정과 narrow 고정 boot를 번갈아 띄우고, `mixed`는 gate 레시피를 번갈아 쓰면서 boot마다 exactness-gate 로그 줄을 grep해 arm 정체를 증거로 남긴다. |
| `scripts/bench_qmv_pin_mixed.py` | 신규. 혼합 워크로드 클라이언트. 보고 대상은 classic decode rate이고, MTP 스트림이 window의 95% 이상을 디코드하며 지나가지 않은 window는 무효 처리한다. |
| `scripts/bench_serving_concurrency.py` | `content`와 나란히 `reasoning_content` delta를 센다. |
| `docs/benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md` | 신규. 제외된 오염 cell과 gate 레시피 검증을 포함한 측정 기록. |
| `docs/benchmarks.md` | 기록을 링크하고, fast row 재현 레시피와 declining probe 문장을 #1199 이후 retry 순서에 맞게 고친다. |
| `docs/environment-variables.md` | `MLXCEL_MTP_ALLOW_INEXACT` 행을 정정하고 빠져 있던 `MLXCEL_QMV_WIDE` 행을 추가한다. |
| `docs/benchmark_results/mtp-drafter-step-profile-m5max-2026-08-17.md` | 재현 레시피가 #1199 retry보다 앞선다는 날짜 표기 주석. |
| `.gitignore` | 드라이버가 쓰는 원시 `bench-results/` 실행 디렉터리를 무시한다. |

Rust 소스가 바뀌지 않았으므로 되돌아갈 런타임 동작 변화도 없다. 검증은 두 Python 하네스에 대한 `python3 -m py_compile`, 드라이버에 대한 `bash -n`, 그리고 측정 실행 자체였다.

## 6. 후속

작업 도중 발견해 해결하지 않고 기록만 한 것이 하나 있다. 현재 `main`에서 31B + bf16 assistant 페어링은 이 호스트에서 두 kernel 모두에 대해 non-identical로 probe되고, 그래서 기본 env의 gate가 #1217이 켜놓은 batch-capable burst를 거절한다. #1217의 1.95x~2.65x 행은 `9e2c6675`에서 측정됐는데 이는 #1258의 Gemma probe보다 앞선다. 현재 `main`에서 그 행들을 다시 돌리는 사람은 이 상황을 만나게 되고, 그 기본값을 어떻게 할지는 별도 이슈의 몫이다.

scoping 작업 자체는 MTP 계열이 진짜 joint batched decode를 갖기 전까지 만들지 않는다. 그 시점이 오면 이 기록의 B-sweep을 다시 돌려 유의미한 숫자를 기대해야 하고, 이슈의 Step 2 후보들을 저울질할 가치는 그때 생긴다.
