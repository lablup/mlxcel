# PR #2931: decode tok/s에서 디코드 윈도 안의 토큰만 세기

**작성일**: 2026-11-10
**상태**: 완료 (2026-11-10 머지, #2929를 닫음)

## 요약

mlxcel의 생성 경로 대부분은 첫 토큰을 prefill 중에, 즉 디코드 타이머가 시작되기 전에 샘플링하고 평가하면서도 생성된 N개 토큰 전체를 `decode_ms`로 나눴다. 측정 윈도에는 forward 스텝이 N-1개뿐이므로 보고된 디코드 속도는 N/(N-1)만큼 높았다. 128 토큰에서 0.79%다. speculative 엔진은 이미 N-1을 쓰고 있어서 speculative 행과 일반 디코드 행이 서로 다른 규칙으로 측정되고 있었다. 이 PR은 헬퍼 하나 `mlxcel_core::generate::decode_tok_per_sec(window_tokens, decode_ms)`를 추가해 모든 경로의 속도 계산을 이 함수로 모으고, 첫 토큰이 prefill에서 나오는 곳에서는 N-1을 넘긴다. `scripts/bench_mlxlm.py`는 mlx-lm과 mlx-vlm의 `generation_tps`를 같은 규칙으로 변환해 "vs mlx-lm" 열이 계속 같은 기준끼리 비교되게 한다. `decode_ms`와 `generated_tokens`는 바뀌지 않고 속도만 바뀐다. Llama-3.2-1B-Instruct-4bit(M1 Ultra, 512 토큰 프롬프트, 128 토큰)에서 `bench_decode.sh`는 392.16 tok/s = 127 / 323.84 ms를 보고하며 이전 규칙으로는 395.26이다. `bench_mlxlm.py`는 mlx-lm 값으로 389.61을 보고하고 mlx-lm 자체 값은 392.69다. 15개 파일, +244/-54.

## 1. 문제 정의

디코드 속도는 개수를 시간으로 나눈 값이고, 분자와 분모는 같은 구간을 기술해야 한다. direct 엔진에서 `complete_prefill`은 이미 샘플링과 평가가 끝난 첫 토큰을 반환하고, `decode_start`는 그 뒤에 기록된다. 따라서 `decode_ms`가 재는 윈도에는 2번부터 N번 토큰까지의 forward 스텝만 들어 있다. N을 이 윈도로 나누면 윈도가 만들지 않은 토큰 하나를 윈도의 몫으로 계산하게 된다.

오차는 작지만(N/(N-1)) 체계적이었고, 경로마다 달랐다.

- speculative 엔진은 prefill 토큰을 제외한 `decode_count`로 이미 나누고 있었다.
- engine direct completion(`mlxcel-bench-decode`, `bench_decode.sh`, 일반 `generate`), MTP 단일 및 배치, `generate`의 MTP와 DFlash 재계산, `mlxcel-bench-engine` 서버 프로브는 N으로 나눴다.
- 서버의 `predicted_per_second`는 llama-server와 마찬가지로 이미 N-1 규칙을 따랐다.

그래서 speculative 대 일반 디코드 비교에는 128 토큰에서 일반 디코드 쪽으로 0.79%의 편향이 있었고, N이 작아질수록 커진다(32 토큰에서 약 3.2%). speculative 디코딩의 속도 향상은 한 자릿수 퍼센트로 측정되는 경우가 많아서 이 정도 편향이면 결론이 바뀔 수 있다.

## 2. 변경 요약

### 2.1 규칙을 담은 헬퍼 하나

`src/lib/mlxcel-core/src/generate.rs`에 다음이 추가됐다.

```rust
pub fn decode_tok_per_sec(window_tokens: usize, decode_ms: f64) -> f64
```

`window_tokens / (decode_ms / 1000)`을 반환하고, 개수나 윈도가 0이면 0.0을 반환한다. doc 주석에 규칙, N/(N-1) 오차, llama-server 및 mlx-lm과의 관계가 적혀 있다. `GenerationStats::decode_tok_per_sec` 필드 doc도 이 함수를 가리킨다. 이 PR 이전에는 경로마다 `if decode_ms > 0.0 { n / (decode_ms / 1000.0) } else { 0.0 }` 식을 각자 복사해 갖고 있었고, 그래서 분자가 서로 달라졌다.

### 2.2 경로별 분자

| 경로 | PR 이후 분자 | 이유 |
|------|--------------|------|
| `engine/direct.rs` (bench-decode, `bench_decode.sh`, 일반 `generate`) | `gen_count - 1` | 첫 토큰은 `decode_start` 이전의 `complete_prefill`에서 나온다 |
| `speculative/mtp/generator.rs::build_stats` | `gen_count - 1` | 첫 bonus는 `prefill_and_seed`에서 나온다 |
| `speculative/mtp/round_loop_batched.rs` | 행별 `len - 1`의 합 | 행마다 prefill bonus가 하나씩 있다 |
| `commands/generate.rs`(offline MTP)와 `generate_dflash.rs` | `tokens.len() - 1` | 같은 prefill bonus를 커맨드 계층에서 다시 계산 |
| `server/engine_probe/server_engine.rs` | `tokens.len() - 1` | `decode_ms`는 첫 토큰이 도착한 시점에 시작된다 |
| `bin/bench_engine/output.rs` | `generated_tokens - 1` | 두 측정 경로 모두 첫 토큰을 윈도 이전에 만든다. 사람이 읽는 출력 줄은 이제 윈도 토큰 수를 출력한다 |
| `engine/speculative.rs` | 변경 없음(`decode_count`) | 이미 올바름. 헬퍼를 거치게만 바꿈 |

모든 토큰을 윈도 안에서 만드는 경로는 N을 유지한다. pipeline-parallel `generate`와 diffusion 생성기(`diffusion_gemma`, `llada2_moe`)가 그렇다. DFlash round loop와 서버의 `predicted_per_second`는 이미 올바르므로 건드리지 않았다.

모든 뺄셈은 `saturating_sub(1)`을 쓴다. prefill 토큰만 만든 실행은 개수 0이 되고 헬퍼는 음수나 정의되지 않은 값 대신 0.0을 반환한다.

### 2.3 mlx-lm 기준값 변환

mlx-lm과 mlx-vlm은 `stream_generate`에서 첫 토큰 이후 시계를 다시 시작하지만 여전히 N개 토큰 전부를 센다. 따라서 이들의 `generation_tps`에는 mlxcel이 방금 제거한 것과 같은 N/(N-1) 부풀림이 있다. 변환 없이 mlxcel만 고쳤다면 모든 "vs mlx-lm" 비율이 mlxcel에 불리하게 0.79% 움직였을 것이다.

`scripts/bench_mlxlm.py`에 `decode_window(gen_tokens, generation_tps)`가 추가됐다. 윈도를 `N / generation_tps`로 복원하고 `(window_ms, (N - 1) / window)`를 반환한다. 입력이 없거나 숫자가 아니거나 양수가 아니면 `(None, None)`을 반환하며, 이것이 CSV 작성기의 `except Exception: pass`를 대체한다. 토큰 하나짜리 실행은 윈도는 있지만 속도는 0.0이다.

### 2.4 문서

`docs/benchmarks.md`에 "How decode tok/s is counted" 절이 추가됐다. 경로별 표, mlx-lm과의 차이, 이전 행을 다시 계산하는 식 `(generated_tokens - 1) / decode_ms`가 들어 있다. `benchmarks/`와 `docs/benchmark_results/` 아래 기록된 CSV는 기록된 그대로 둔다. CHANGELOG 항목은 0.79% 이동을 명시한다.

## 3. 측정

Llama-3.2-1B-Instruct-4bit, M1 Ultra, 512 토큰 프롬프트, 128 토큰 생성.

| 항목 | 값 |
|------|----|
| `decode_ms`(PR로 바뀌지 않음) | 323.84 ms |
| mlxcel decode tok/s, 새 규칙(127 / 323.84 ms) | 392.16 |
| mlxcel decode tok/s, 이전 규칙(128 / 323.84 ms) | 395.26 |
| mlx-lm `generation_tps` 원본 | 392.69 |
| `bench_mlxlm.py`가 변환한 mlx-lm 값 | 389.61 |

산술 확인: 127 / 0.32384 s = 392.17, 128 / 0.32384 s = 395.26, 392.69 × 127/128 = 389.62. 마지막 자리 차이는 `decode_ms`와 보고된 속도의 반올림에서 온다.

mlxcel 대 mlx-lm 비율은 두 규칙에서 같다(395.26 / 392.69와 392.16 / 389.61 모두 약 1.0065). 변환이 지키려는 성질이 바로 이것이다. 절대값은 N/(N-1)만큼 내려가고 런타임 간 비교는 움직이지 않는다.

## 4. 기술적 선택과 그 이유

### 4.1 윈도를 넓혀 N을 유지하지 않고 N-1을 쓴다

다른 일관된 수정은 디코드 타이머를 첫 토큰 이전에 시작하고 N을 유지하는 것이다. 그러면 첫 토큰의 샘플링과 평가(일부 경로에서는 prefill의 끝부분까지)가 `decode_ms`에 섞이고, 기존 CSV, 대시보드, 서버가 이미 보고하는 필드의 의미가 바뀌며, llama-server의 `predicted_per_second`와도 맞지 않게 된다. 분자만 바꾸면 `decode_ms`와 `generated_tokens`가 과거 기록과 그대로 호환되므로 이전 행도 이미 가진 필드만으로 다시 계산할 수 있다.

### 4.2 mlx-lm 규칙을 따르지 않고 기준값을 변환한다

mlx-lm의 "N개를 N-1 스텝으로 나누는" 규칙을 채택하면 참조 하나와는 맞지만 llama-server, 그리고 mlxcel 자신의 speculative 엔진과는 어긋난다. PR은 측정된 구간을 그대로 기술하는 규칙을 택하고, 외부 숫자는 그것이 mlxcel CSV로 들어오는 한 지점에서 맞춘다.

### 4.3 지점별 수정 대신 공유 헬퍼

버그는 일곱 지점이 각자 나눗셈을 복사해 갖고 있었기 때문에 생겼다. 규칙을 doc 주석이 달린 함수 하나에 두면 "어떤 개수를 넣는가"가 모든 호출 지점에서 명시적인 인자가 된다. 그 결정은 호출 지점에 있어야 한다. 헬퍼는 특정 경로가 첫 토큰을 윈도 안에서 만들었는지 알 수 없지만, 이제 모든 호출자가 그것을 밝혀야 한다.

### 4.4 기록된 CSV는 건드리지 않는다

과거 행을 다시 쓰면 기록으로 취급되는 파일에 재계산 값과 측정 값이 섞인다. 대신 문서가 재계산 식과 기준 날짜를 제공한다(2026-11-10 이전 행은 speculative 엔진을 제외한 모든 경로가 N을 썼다).

## 5. 검증과 한계

- 3절의 실제 체크포인트 실행.
- 새 단위 테스트: 헬퍼용 `rate_counts_only_window_tokens`와 `rate_is_zero_without_a_window`, MTP 단일용 `session_stats_rate_excludes_the_prefill_bonus`, MTP 배치용 `batched_stats_rate_excludes_each_rows_prefill_bonus`(bonus만 있는 배치가 0.0을 보고해야 하는 경우 포함), 단일 토큰 경우를 추가하고 이름을 `decode_rate_excludes_the_first_token_from_the_window`로 바꾼 서버 프로브 테스트, 기대 속도를 유지하도록 100에서 101 토큰으로 바꾼 bench-engine fixture, CSV 문자열 입력과 누락 값을 포함한 Python 변환용 `tests/test_bench_mlxlm_decode_window.py`.
- `cargo clippy --workspace --all-targets -D warnings`와 `cargo fmt --check` 통과. `cargo test --workspace --profile test-fast --no-fail-fast`는 15373개 테스트가 통과했고 변경 범위 밖에서 8개가 실패했다. 병렬 실행에서만 실패하고 변경을 stash해도 실패하는 root-lib 모델 테스트 4개(dots1, qwen4_exp snapshot)와, 단독 실행이나 `--test-threads=1`에서는 통과하는 `mlxcel-server` 인자 테스트 4개다.

유지보수자가 알아야 할 남은 항목:

- 서버 `/metrics`의 요청별 디코드 히스토그램은 범위 밖이다. `generation_only_ms`의 출처가 워커마다 다르므로(florence2와 nemotron_parse는 전체 경과 시간을 쓴다) 별도 작업이 필요하고, 그때까지 CLI 속도와 어긋날 수 있다.
- `decode_window`는 mlx-lm과 mlx-vlm이 "첫 토큰 이후 시계를 다시 시작하고 N개 전부를 센다"는 동작을 유지한다고 가정한다. upstream이 `stream_generate`를 바꾸면 변환은 조용히 틀려진다. upstream 동작을 고정하는 테스트는 없다.
- 과거 CSV 행과 새 행이 이제 서로 다른 규칙으로 나란히 있다. 2026-11-10을 사이에 두고 비교하려면 이전 행을 다시 계산해야 한다.
- 실제 체크포인트로 검증한 것은 일반 direct 경로뿐이다. MTP, DFlash, 서버 프로브 변경은 stats 빌더 단위 테스트로만 확인했고 실제 모델 실행은 없다.

## 6. 변경 요약 표

| 항목 | 값 |
|------|----|
| 변경 파일 | 15 |
| 추가 줄 | +244 |
| 삭제 줄 | -54 |
| 추가 또는 변경된 테스트 | 헬퍼 테스트 2, MTP stats 테스트 2, 서버 프로브 테스트 1, bench-engine fixture 2, Python 테스트 4 |

| 분류 | 요약 |
|------|------|
| 정확성 | prefill로 시작하는 경로에서 디코드 속도가 N-1을 셈. 모든 경로가 헬퍼 하나를 거침 |
| 벤치마크 도구 | mlx-lm과 mlx-vlm 기준값을 같은 규칙으로 변환 |
| 문서 | `docs/benchmarks.md`의 계산 규칙과 재계산 식, CHANGELOG 항목 |

## 7. 학습 포인트

- 타이머가 어떤 사건 이후에 시작되면 개수도 그 이후부터 세야 한다. LLM 벤치마크에서는 첫 토큰이 흔히 틀리는 지점이다. prefill이 자연스럽게 그 토큰을 만들면서 끝나기 때문이다.
- 작고 일정한 편향이라도 비교 대상 사이에서 다르면 고칠 가치가 있다. 여기서는 일반 경로와 speculative 경로가 서로 달랐고, 그 차이가 절대값 0.79%보다 더 중요했다.
- 규칙을 고칠 때는 그 규칙과 비교되는 외부 숫자를 모두 확인해야 한다. 비율의 한쪽만 고치면 비율이 움직인다.
