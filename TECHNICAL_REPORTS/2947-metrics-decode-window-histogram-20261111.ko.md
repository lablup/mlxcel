# PR #2947: /metrics decode 히스토그램에서 decode window 안의 토큰만 계산

**작성일**: 2026-11-11
**상태**: 완료 (#2938을 닫음. `make verify-test`와 workspace clippy 통과)

## 요약

#2931은 프로젝트의 모든 decode tok/s 수치에 하나의 규칙을 정했다. 측정된 decode window 안에서 생성된 토큰 수를 window 길이로 나누고, 계산은 `mlxcel_core::generate::decode_tok_per_sec`가 한다. 서버의 `/metrics` 히스토그램 `mlxcel_request_decode_tok_s`는 이 규칙에서 빠져 있었다. 이 히스토그램은 completion 토큰 N개 전부를 `generation_only_ms`로 나눴는데, 이 window는 prefill에서 샘플링된 첫 토큰 이후에 열린다. 이슈 #2938이 이 차이를 추적했다. 이 PR은 `GenerationResult`의 `generation_only_ms` 옆에 `decode_window_tokens: Option<usize>`를 추가하고, 두 필드의 의미를 문서로 정의하고, recorder가 `decode_tok_per_sec(decode_window_tokens, generation_only_ms)`를 기록하게 한다. batch scheduler 경로는 N-1을 보고한다. Florence-2와 Nemotron-Parse는 encoder pass와 decode loop를 분리할 수 없으므로 `None`을 보고하고, wall time 기준 속도를 기록하는 대신 건너뛴다. llama-server 카운터는 upstream 입력을 그대로 유지한다. 마지막 커밋은 `mlxcel run`이 출력하는 속도 줄에도 같은 규칙을 적용한다. 17개 파일, +306/-70 (이 보고서 제외).

## 1. 문제 정의

### 1.1 같은 양에 대한 두 가지 규칙

#2931 이후 `docs/benchmarks.md`의 "How decode tok/s is counted"는 decode tok/s를 window 토큰 수를 window 시간으로 나눈 값으로 정의한다. 첫 토큰이 prefill에서 나오면 N-1, 모든 토큰이 window 안에서 생성되면 N이다. 벤치 도구, `generate`, speculative engine, native `/completion` 응답의 `timings.predicted_per_second`가 모두 이 규칙을 따른다. `src/server/batch/observability.rs`의 `BatchObservability::record_request_completion`은 따르지 않았다. 이 함수는 `completion_tokens * 1000 / generation_only_ms`를 직접 계산했다. batch scheduler에서 `generation_only_ms`는 `elapsed_ms - prompt_eval_ms`이고 첫 토큰 시점에서 시작하므로, 첫 토큰이 자신을 포함하지 않는 window에 계산되었다. 그 결과 히스토그램 기반 대시보드는 같은 요청의 응답 timings보다 N/(N-1)만큼 높게 읽혔다. N = 128에서 128/127 (0.79%), N = 2에서는 두 배다.

### 1.2 producer마다 다른 `generation_only_ms`의 의미

이슈는 이 필드에 정해진 의미가 없다는 점도 지적했다. `src/server/model_worker.rs`의 공용 builder는 decode window를 보고한다. `src/server/florence2_worker.rs`와 `src/server/nemotron_parse_worker.rs`는 `prompt_eval_ms: 0`과 함께 encoder pass를 포함한 전체 wall time을 보고한다. 소비자는 둘을 구분할 방법이 없었고, 이 필드로 계산한 속도는 decode window와 wall time을 아무 표시 없이 섞었다.

### 1.3 토큰 하나짜리 경우

completion이 토큰 하나뿐이면 그 토큰은 prefill에서 나오므로 window에는 토큰이 없다. 이전 코드는 `generation_only_ms`가 0으로 반올림될 때만 기록을 건너뛰었다. 0이 아닌 꼬리 시간이 있으면(예: prefill 토큰 다음의 EOS step) 그 꼬리 시간에 대해 토큰 1개를 기록했고, 그 속도는 아무것도 측정하지 않았다.

## 2. 변경 요약

### 2.1 `GenerationResult`의 계약 (`src/server/model_provider.rs`)

`decode_window_tokens: Option<usize>`가 `generation_only_ms` 옆에 추가되었고, 두 필드 모두 둘의 관계를 정의하는 doc comment를 갖는다. `Some(n)`이면 `generation_only_ms`는 첫 토큰부터 종료까지의 구간이고 `n`은 그 안에서 생성된 토큰 수다. prefill에서 첫 토큰을 얻는 producer는 `completion_tokens - 1`, 모든 토큰을 window 안에서 생성하는 producer는 `completion_tokens`를 보고한다. `None`이면 producer에 분리된 decode window가 없고 `generation_only_ms`는 전체 wall time이다. doc은 llama-server의 `predicted_ms` 필드와 카운터가 어느 경우든 `generation_only_ms`를 읽는다는 점도 적는다.

### 2.2 Producer

- **공용 builder** (`src/server/model_worker.rs`의 `build_generation_result_with_cache`): `Some(completion_tokens.saturating_sub(1))`을 보고한다. `ab113b39`과 `f50016cc`에서 작성된 주석에 따르면, 이 builder를 쓰는 모든 producer는 `StreamingDecodeState`를 통해 첫 토큰 시점을 기록한다. 대상은 batch scheduler(히스토그램에 도달하는 유일한 producer), XLA worker, diffusion worker다. diffusion은 블록 하나를 한 번에 확정하므로 첫 블록의 일부가 window가 열리기 전에 계산된다. 따라서 diffusion의 N-1은 정확한 수가 아니라 상한이다. 이 보고서는 producer에 대한 이 설명을 주석과 PR 본문에서 가져왔고, producer별 기록 방식을 직접 추적하지는 않았다.
- **count 전용 결과** (`src/server/model_provider_prompt_count.rs`): `Some(0)`. EOS at prefill과 같은 빈 window다.
- **Florence-2와 Nemotron-Parse**: `Done` 결과 생성 코드가 요청 handler에서 순수 함수 `florence2_generation_result`, `nemotron_parse_generation_result`로 옮겨졌다. 둘 다 `generation_only_ms: elapsed_ms`와 `decode_window_tokens: None`을 보고하고, 이유를 주석으로 남긴다. 다른 필드는 바뀌지 않았다. 함수로 분리한 이유는 모델 없이 결과를 테스트하기 위해서다.
- **테스트 지원** (`src/server/model_provider_test_support.rs`): mock provider가 새 필드를 채운다(`Some(0)`과 `Some(tokens_emitted - 1)`).

### 2.3 Recorder (`src/server/batch/observability.rs`)

`record_request_completion`에 `decode_window_tokens: Option<usize>` 인자가 추가되었고, `src/server/batch/scheduler/decode_tick.rs`의 유일한 호출 지점에서 그대로 전달된다. decode 속도는 값이 `Some(n)`이고 `n > 0`이며 `generation_only_ms > 0`일 때만 기록되고, 로컬 나눗셈 대신 `decode_tok_per_sec`로 계산된다. TTFT와 llama-server 카운터는 `None` 결과를 포함해 이전과 같이 기록된다. doc comment는 llama 카운터가 b10621의 방식대로 모든 completion 토큰을 `generation_only_ms`에 대해 누적한다고 적는다.

### 2.4 `mlxcel run` (`src/commands/run.rs`, 커밋 `a04b39db`)

결과가 `Some(window_tokens)`를 가지면 요약 줄은 이제 `[Generated N tokens; decoded W in S s = R tok/s]`이고, R은 `decode_tok_per_sec(W, generation_only_ms)`이므로 출력된 수치가 맞아떨어진다. `None`이면 기존 `[Generated N tokens in S s = R tok/s]` 줄과 `completion_tokens / seconds`를 유지한다. diffusion, Florence-2, Nemotron-Parse의 one-shot은 `generate` 흐름에 남으므로(`one_shot_stays_on_generate`) 이 줄에 도달하지 않는다. 테스트가 아닌 소스를 grep한 결과, decode 속도를 얻으려고 `completion_tokens`를 `generation_only_ms`로 나누는 소비자는 더 없었다. native timings는 #2931부터 이미 N-1을 쓰고, engine probe는 밀리초 값만 전달한다.

### 2.5 문서

- `docs/benchmarks.md`의 "How decode tok/s is counted"에 히스토그램의 입력, N-1 계산, 건너뛰는 경우, 바뀌지 않은 llama 카운터를 설명하는 문단이 추가되었다.
- `mlxcel_request_decode_tok_s`의 `/metrics` HELP 문구가 decode window 안에서 생성된 토큰만 센다고 명시한다.
- `CHANGELOG.md`의 Unreleased에 히스토그램 변경과 `mlxcel run` 요약 줄 항목이 있다.
- `llamacpp:tokens_predicted_*` 카운터와 `timings.predicted_*`가 바뀌지 않으므로 `docs/llama-server-compat.md`는 수정하지 않았다.

## 3. 검증

추가되거나 바뀐 테스트:

| 테스트 | 고정하는 동작 |
|--------|---------------|
| `observability::decode_rate_counts_only_window_tokens_for_a_prefill_seeded_request` | 토큰 128개 중 127개가 1270 ms window 안에 있으면 정확히 100 tok/s를 기록하고, `decode_tok_per_sec(127, 1270.0)`과 같음 |
| `observability::decode_rate_skips_a_request_without_a_decode_window` | `None` 결과는 decode 속도를 기록하지 않지만 TTFT와 llama 카운터(토큰 30개, 900000 us)에는 반영됨 |
| `observability::record_request_completion_skips_zero_token_requests` | 15 ms 꼬리를 가진 토큰 하나짜리 경우 추가: 0 tok/s 기록 없음 |
| `observability::record_request_completion_populates_histograms`, `histogram_counts_plus_inf_observations` | 기존 기대값을 window 토큰 수로 다시 씀 |
| `model_worker_tests::build_generation_result_computes_finish_reason_and_generation_split` | window 토큰 N-1, 빈 생성은 `Some(0)` |
| `florence2_worker_tests`, `nemotron_parse_worker_tests` (`result_reports_wall_time_without_a_decode_window`) | wall time, `None`, recorder에 넣은 뒤 decode 히스토그램이 비어 있음 |
| `metrics_llama_tests` | 새 인자를 넣어도 llama 카운터 값은 그대로 |

- **게이트 결과.** `make verify-test`(직렬, CI와 같은 방식)가 15405개 테스트 모두 통과했고, `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`도 경고가 없다.
- **실제 체크포인트.** `mlx-community/Llama-3.2-1B-Instruct-4bit`에서 `mlxcel run -p ... -n 64`가 `[Generated 64 tokens; decoded 63 in 0.26s = 240.46 tok/s]`를 출력했다. decode window 안의 63개 토큰을 window 시간으로 나눈 값이다. `/metrics` scrape는 기록하지 않았고, 히스토그램 경로는 단위 테스트가 다룬다.
- **실제 체크포인트 측정 기록은 없다.** PR 본문은 절차를 제시한다(`mlxcel-server --metrics`에서 `n_predict: 128`로 native `/completion`을 한 번 보내고, `mlxcel_request_decode_tok_s_sum`의 변화량을 응답의 `timings.predicted_per_second`와 비교). 하지만 실행한 결과는 PR에 없다.
- `a04b39db`의 `mlxcel run` 변경은 `src/commands/run.rs`만 수정하고 테스트를 추가하지 않는다.

## 4. 기술적 선택과 그 이유

### 4.1 recorder에서 계산하지 않고 결과에 window 토큰 수를 담음

**배경:** recorder의 호출자는 batch scheduler 하나뿐이므로 recorder가 직접 `completion_tokens - 1`을 계산할 수도 있었다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `GenerationResult`에 `decode_window_tokens: Option<usize>` 추가 (선택) | window가 어디서 열리는지 아는 producer가 수를 보고함. `None`으로 "window 없음"을 표현할 수 있음. struct literal로 만드는 구조체에 필드를 추가하면 모든 생성 지점이 컴파일 시점에 값을 정해야 함 | 테스트 지원 코드를 포함해 모든 `GenerationResult` literal을 고쳐야 했음 |
| `record_request_completion` 안에서 1을 뺌 | 한 줄 변경 | scheduler의 prefill 기반 구조를 공용 recorder에 박아 넣고, window에 N개가 모두 있는 producer에는 틀림 |
| "prefill seeded" boolean 플래그 | 카운트보다 작음 | window 토큰 수가 N도 N-1도 아닌 producer를 표현할 수 없고, "window 없음"을 위한 세 번째 상태가 여전히 필요함 |

두 번째와 세 번째 행은 이 보고서의 분석이고, PR 본문은 이를 다루지 않는다.

### 4.2 encoder-decoder worker는 근사하지 않고 건너뜀

Florence-2와 Nemotron-Parse는 모델이 소유한 하나의 호출 안에서 encode와 decode를 실행하므로, worker는 decode가 언제 시작했는지 볼 수 없다. wall time 기준 속도를 기록하면 decode 히스토그램에 encoder 시간이 섞인다. 호출을 나누려면 내부 timestamp를 노출하도록 모델 코드를 바꿔야 하고, 이는 metrics 수정의 범위를 벗어난다. `None`은 이들의 속도를 제외하면서 이유를 문서로 남긴다. 현재 이 worker들은 `Done`을 직접 보내고 `record_request_completion`에 도달하지 않으므로, 이 건너뛰기는 동작 변경이 아니라 향후 라우팅 변경에 대비한 guard다. PR은 이 worker들을 recorder에 연결하지 않는다고 명시한다. 연결하면 `prompt_eval_ms: 0`도 TTFT 히스토그램과 llama 카운터에 들어가기 시작하기 때문이다.

### 4.3 llama-server 카운터는 upstream 입력을 유지

`llamacpp:tokens_predicted_total`과 `llamacpp:tokens_predicted_seconds_total`은 b10621의 `n_decoded`, `t_token_generation` 누적 방식을 따르고, 배포 스크립트는 이 값을 실제 `llama-server`와 비교한다. 이를 바꾸면 내부 일관성을 얻는 대신 upstream 호환성을 잃고, 호환성 manifest에도 항목이 필요해진다. PR은 이 카운터를 그대로 두었으므로 `/metrics`는 이제 분자가 다른 두 속도를 노출한다. 히스토그램(window 토큰)과 카운터 비율(전체 토큰)이다. 이 차이는 `docs/benchmarks.md`에 문서화되어 있다.

### 4.4 나눗셈은 helper 하나로

속도를 로컬 식이 아니라 `decode_tok_per_sec`로 계산하면, 앞으로 규칙이 바뀔 때(예: 0 처리 방식) 히스토그램에도 함께 적용된다. recorder 자체의 `window_tokens > 0 && generation_only_ms > 0` 조건은 여전히 필요하다. helper는 그런 경우 0.0을 반환하고, 0 tok/s 기록은 히스토그램을 왜곡하기 때문이다.

## 5. 한계와 남은 일

- **토큰 하나짜리 `run` turn은 0.00 tok/s를 출력한다.** decode window에 토큰이 없으니 규칙대로지만, 이전에는 window가 0이 아니면 1 / S를 출력했다.
- **히스토그램 변화의 크기.** 각 기록값은 이전 값의 (N-1)/N이 된다. N = 128에서 0.78% 낮아지고(이전 값이 128/127, 즉 0.79% 높았음), N = 2에서는 50% 낮아진다.
- **diffusion의 수는 상한이다.** diffusion은 블록 단위로 확정하므로 N-1은 상한이다. 지금은 히스토그램에도 `mlxcel run` 출력 줄에도 도달하지 않으므로, 어느 한쪽이 이 값을 쓰기 시작할 때만 문제가 된다.
- **builder 주석의 producer 설명은 테스트되지 않았다.** XLA나 diffusion worker가 주석대로 첫 토큰 시점을 기록하는지 확인하는 테스트는 없다.
- **실제 체크포인트 `/metrics` scrape 없음**, 3절 참고.

## 6. 학습 포인트

- 구간이 정의되지 않은 duration 필드에는 각자 다른 것을 측정하는 producer가 모이기 쉽다. 필드 doc에 계약을 쓰고 duration을 그것으로 나눌 카운트와 짝지은 것이 Florence-2와 Nemotron-Parse의 불일치를 드러냈다.
- "측정할 수 없음"은 `0`이나 wall time 같은 sentinel보다 문서화된 `None`을 가진 `Option`으로 표현하는 편이 낫다. 이를 처리하지 않는 소비자는 그럴듯한 숫자를 내는 대신 컴파일에 실패한다.
- channel로 보내는 handler 안의 struct literal을 순수 함수로 옮기면, 모델을 로드하지 않고도 worker 출력을 테스트할 수 있다.
- 계산 규칙을 고칠 때는 이슈가 지목한 소비자만이 아니라 입력을 쓰는 모든 소비자를 찾아야 한다. 여기서는 같은 N/window 나눗셈이 `mlxcel run`에도 있었고, 후속 커밋에서야 발견되었다.
