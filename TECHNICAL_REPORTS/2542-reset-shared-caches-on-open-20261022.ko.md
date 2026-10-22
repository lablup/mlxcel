# PR #2542: CLI가 시퀀스를 열 때 모델 내부 공유 캐시를 초기화

**날짜**: 2026-10-22
**상태**: 완료 (PR 열림; #2494 종료)

## 요약

이슈 #2494는 #2225가 CLI를 `CxxGenerator`에서 `DirectEngine`으로 옮긴 뒤 exaone-4.0-1.2b-4bit의 `mlxcel generate` 디코드 처리량이 떨어졌다고 보고했다(이슈의 198 vs 158 tok/s, release 빌드). PR #2542는 이 손실이 엔진 스텝이 느려진 것이 아니라 남아 있던 상태 때문에 생긴 호출당 고정 비용임을 보인다. ExaOne 4는 배칭을 지원하지 않고 KV 캐시를 래퍼 안에 보관한다. 이 모델의 model-owned 풀 항목은 `make_caches()`를 호출하지 않으므로 `DirectEngine`은 캐시를 한 번도 초기화하지 않았고, `mlxcel generate`의 측정 실행은 1토큰 워밍업이 남긴 KV 위에 프롬프트를 다시 prefill했다. 출력이 틀렸고, 오프셋이 어긋난 prefill이 측정 구간 안에서 cuBLAS 커널을 로드했다. 이제 `DirectEngine::open_sequence`는 `supports_batching()`이 false이면 `make_caches()`를 호출한다. `BatchScheduler`가 prefill 시점에 이미 적용하던 규칙이다. GB10(test-fast)에서 exaone4 `-n 384`는 main의 179.28 tok/s에서 202.04로 올라 d379d2f8 기준의 202.73과 같은 수준이 되었고, greedy 출력도 기준과 다시 일치한다. 4개 파일, +239/-0.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/lib/mlxcel-core/src/engine/direct.rs` | `open_sequence`가 `engine.open` 성공 후, 모델이 배칭을 지원하지 않으면 `apply_pool_kv_cache_mode` 전에 `self.model().make_caches()`를 호출; doc comment에 해당 계열과 따르는 스케줄러 규칙을 명시 |
| `src/lib/mlxcel-core/src/engine/mod.rs` | 새 테스트 모듈 등록 |
| `src/lib/mlxcel-core/src/engine/shared_cache_tests.rs` (신규) | `make_caches`만 초기화하는 `Cell`에 오프셋을 둔 stub 모델로 테스트 2개 |
| `docs/benchmark_results/exaone4-engine-decode-gb10-2026-10-22.md` (신규) | 측정 기록: 메커니즘, 절편/기울기 fit, null arm을 포함한 교차 측정, 대조군 |

대상 계열은 배칭을 지원하지 않고 캐시 한 벌을 `RefCell`에 보관하는 ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum, Laguna다. 서버 경로는 바뀌지 않았다.

## 2. 기술적 선택과 그 이유

### 2.1 손실은 정확성 버그에서 나온 고정 비용이었다

`-n 2, 32, 128, 384`(각 3회, 같은 세션)에 대해 `time = fixed + n * per_token`을 fit하면 두 arm은 기울기가 아니라 절편에서 갈린다.

| arm | 고정 | 토큰당 | n=2 / 32 / 128 / 384 중앙값 시간 |
|---|---|---|---|
| ref (d379d2f8) | 70.5 ms | 4.770 ms | 85 / 226 / 668 / 1905 ms |
| main (8116e266) | 302.0 ms | 4.851 ms | 318 / 457 / 898 / 2175 ms |

`mlxcel generate`는 1토큰 워밍업 후 같은 클라이언트로 측정 실행을 하므로, 측정 실행의 prefill은 워밍업이 남긴 26개 위치에서 시작했다. 결과는 두 가지였다. 모델이 프롬프트를 두 번 보았으므로 `--show-reasoning -n 48`에서 main의 출력은 첫 토큰부터 ref와 달랐다. 그리고 비어 있지 않은 캐시에 대한 attention이 다른 커널 경로를 탔다. 모든 `cuLibraryLoadData`에 건 gdb backtrace를 보면 워밍업 prefill은 cuDNN SDPA, RoPE, RMSNorm을 로드했고, 측정 prefill은 cuBLAS GEMM 커널(`CublasGemm::run_batched`, `execute_matmul` 로드 10회)과 `Softmax`를 로드했다. nsys로 보면 이 라이브러리 로드 중 9개가 각각 9~71 ms, 합계 약 290 ms(nsys 하에서)로 측정 구간 안에 있었다. 수정 후 prefill을 포함한 호출당 고정 비용 전체는 57 ms다.

즉 처리량 회귀는 상태 누수의 증상이었다. 엔진의 디코드 스텝을 튜닝해서는 고칠 수 없었다.

### 2.2 클라이언트가 open 시점에 초기화하고, `supports_batching()`으로 분기

초기화는 `DirectEngine::open_sequence`에 있다. `generate`(임베딩 prefill 포함), `loglikelihoods`, `generate_with_drafter`, `warm_up_verify_widths`가 모두 지나는 단일 진입점이다. 서버가 이미 적용하던 규칙을 따른다. `BatchScheduler`는 배칭을 지원하지 않는 모든 모델에 대해 prefill 시점(`execute_full_prefill`, `start_chunked_prefill`)에 `make_caches()`를 호출하고, 폐기된 CLI 루프는 워밍업과 측정 실행 사이에 `reset_with_model`로 초기화했다. 이 위치를 택하면 더 좁은 세 가지 방안이 배제된다.

- 래퍼마다 고치는 방안(ExaOne 4 등에 `prepare_sequence_state` 추가)은 여섯 계열을 건드리고, 다음에 추가될 비배칭 래퍼는 여전히 노출된다.
- `CachePool::allocate_with_layout`에서 model-owned 항목에 `make_caches()`를 호출하는 방안은 스케줄러도 쓰는 풀의 계약을 바꾼다. 스케줄러는 이미 prefill에서 초기화하므로 서버에서는 두 번 초기화된다.
- `mlxcel generate` 명령에서 워밍업과 측정 사이에 초기화하는 방안은 그 호출자만 고친다. 같은 클라이언트를 쓰는 `loglikelihoods`, `MlxInferenceSession`, `bench_decode`, `speculative_bench`, `engine_parity`, 엔진 probe는 그대로 남는다.

초기화는 `engine.open`이 성공한 뒤에만 실행되므로 open이 실패하면 아무것도 초기화하지 않는다. open당 한 번이며 스텝마다 실행되지 않는다. 배칭 계열은 풀 항목이 이미 `make_caches()`를 호출했으므로 건너뛴다.

### 2.3 모델이 시퀀스별 상태를 소유하는 계열은 영향받지 않는다

일부 비배칭 계열(`ModelOwnedSequenceState`: Mamba, Jamba, NemotronH, LFM2, Kimi, RWKV7 등)은 `prepare_sequence_state`에서 시퀀스별 상태를 만든다. 이들에게 `make_caches()`는 엔진 경로가 쓰지 않는 fallback 슬롯만 교체하고, id별 상태와 KV 모드는 그대로 둔다. 비용은 open당 가벼운 객체 할당 하나다. 배칭을 지원하지 않고 시퀀스 키 상태가 이미 올바르던 granite-4.0-h-tiny는 이제 open마다 초기화를 받으며, 대조군으로 측정했다(3절).

`make_caches`는 `Result`가 아니라 `Vec<KVCache>`를 반환하고 캐시 생성은 지연된다(`RotatingKVCache::new`를 포함해 `keys: None`). 따라서 `let _ =`가 버리는 것은 더미 캐시뿐이고 오류를 담을 수 있는 값은 없다.

### 2.4 동작 변경: 대상 계열의 CLI 출력이 바뀐다

`mlxcel generate`, `loglikelihoods`, `MlxInferenceSession`은 이제 이 계열들을 새 캐시에서 디코드하므로 출력이 올바른 쪽으로 바뀐다. exaone4 greedy 출력은 d379d2f8 `CxxGenerator` 빌드와 일치한다. gpt-oss-20b는 더 이상 다른 답변("The user wants...")으로 시작하지 않고 ref처럼 ("We need to write a detailed essay...") 시작한다. laguna-xs-2.1-nvfp4의 워밍업 후 greedy `-n 32` 출력은 워밍업 없는 `--profile` 출력과 같다. `--prompt-lookup`은 배칭을 지원하지 않는 모델을 모두 거부하므로 영향이 없다.

### 2.5 리뷰 중 변경

- `210968ef`: 구현 리뷰에서 Mellum과 Laguna도 `prepare_sequence_state` 없이 `make_caches()`로만 초기화되는 `RefCell` 캐시를 가진다는 점이 확인되었다. doc comment, 테스트 헤더, PR 본문, 측정 기록이 이제 여섯 계열을 모두 명시한다. 같은 커밋에서 `--prompt-lookup` 출력이 바뀐다는 PR 본문의 서술도 바로잡았다.
- `01c5a298`: 측정 기록의 prefill 비용 문단을 측정한 값만으로 다시 썼다(nsys 하의 합계와 fit에서 얻은 수정 후 고정 비용).

## 3. 검증

GB10(sm_121), 커널 `7.0.0-1019-nvidia`, 드라이버 `580.178.04`, `MLX_CUDA_ARCHITECTURES=121`, test-fast 프로파일. Arm: ref d379d2f8(마지막 `CxxGenerator` 빌드), main 8116e266, fix, 그리고 null arm으로 fix 바이너리를 한 번 더. 라운드마다 arm 순서를 회전했고, arm마다 미리 데운 별도 `MLX_PTX_CACHE_DIR`을 썼으며, `gpu-lock` 하에서 실행했다.

| 항목 | 결과 |
|---|---|
| exaone-4.0-1.2b-4bit, `-n 384`, 7라운드 | ref 202.73, main 179.28, fix 202.04, null 201.94 tok/s (중앙값) |
| exaone4 paired | main vs ref -12.27% (-12.91..-10.26); fix vs ref -0.72% (-1.14..+0.82); null 하한(null vs fix) +0.11% (-1.48..+1.46) |
| 수정 후 exaone4 fit (새 ref fit과 같은 블록) | fix 고정 56.8 ms, 토큰당 4.970 ms; ref 고정 69.7 ms, 토큰당 4.888 ms |
| llama-3.2-1b-4bit 대조군(배칭, 경로 불변), `-n 384`, 5라운드 | ref 251.33, main 254.95, fix 254.37, null 254.45; main vs fix +0.07% (-0.10..+0.60) |
| granite-4.0-h-tiny-4bit 대조군(비배칭, 이제 open마다 초기화), `-n 256`, 7라운드 | main 99.07, fix 97.38, null 99.37; main vs fix +2.14% (-2.43..+3.69)는 null vs fix +1.55% (-0.63..+4.01) 범위 안; `-n 48` 출력 main과 fix 동일 |
| Greedy `--show-reasoning -n 48` 출력 | exaone4 fix == ref; granite-4.0-h와 ministral-3b fix == main; gpt-oss-20b fix는 ref처럼 시작 |
| `engine::shared_cache_tests` (신규 2개) | 초기화를 제거하면 둘 다 실패, 있으면 통과 |
| `mlxcel-core` lib 테스트, CUDA, `--test-threads=1` | `engine::` 70, `exaone` 3, `granitemoehybrid` 17, `lfm2` 72, `gpt_oss` 11, `ministral3` 10, `session` 13 |
| `cargo clippy --profile test-fast --features cuda -p mlxcel-core --lib --tests -- -D warnings`, `cargo fmt --check` | 경고 없음 |

`warmup_does_not_leak_into_the_next_completion`은 stub에서 `mlxcel generate`의 형태를 재현한다. 1토큰 워밍업 후 같은 클라이언트로 6토큰을 실행하며, 이 실행의 prefill은 오프셋 0에서 시작해 새 클라이언트와 같은 토큰열을 디코드해야 한다. stub의 최상위 logit은 내부 오프셋에 의존하므로 오래된 오프셋에서 시작하면 토큰열이 달라진다. `every_open_resets_the_shared_caches`는 `run`, 두 번째 `run`, `loglikelihoods`에 걸쳐 open당 한 번 초기화되는지, scoring이 오프셋 0에서 시작하는지, 토큰 예산 0이면 아무것도 열지 않고 초기화도 하지 않는지 확인한다.

## 4. 미검증 및 미해결 항목

- **토큰당 기울기:** 수정 후 소규모 fit에서 토큰당 기울기는 d379d2f8 기준보다 1.7% 높다(4.970 vs 4.888 ms). 교차 측정한 `-n 384` 비교(fix vs ref -0.72%, null 하한 안)로는 어느 쪽으로도 차이를 판별하지 못했다. 추가 조사는 하지 않았다.
- **release 프로파일:** 모든 수치는 test-fast다. 이슈의 198 vs 158 tok/s는 release 빌드였고, release 프로파일 수치는 측정하지 않았다.
- **실제 체크포인트 없음:** Step 3.5, Mellum, `Ministral3` 텍스트 래퍼는 실제 체크포인트로 실행하지 않았다. 로컬에 없고, ministral-3b-4bit는 `Ministral3` 텍스트 래퍼를 쓰지 않는 `mistral3` VLM으로 로드된다. 이들은 공통 `supports_batching()` 분기와 stub 테스트로만 커버된다.
- **측정하지 않음:** gpt-oss 처리량(출력만 확인); 비배칭 래퍼에서의 VLM `--image` prefill. 이 경로도 `open_sequence`를 지나며, 서버는 임베딩 prefill 전에 같은 초기화를 적용한다.
- **테스트 공백(LOW):** `shared_cache_tests.rs`는 배칭 모델이 open당 정확히 한 번 `make_caches`를 받는지 확인하지 않는다. 분기가 `if` 하나이므로 회귀 방지용일 뿐이다.
- **설계상 한계, 이 PR과 무관:** 같은 비배칭 모델 위의 `DirectEngine` 클라이언트 두 개는 여전히 `RefCell` 캐시 한 벌을 공유한다.
- **기존 문제, 별도 추적:** `NemotronNAS`는 `supports_batching() == false`를 보고하지만 `forward`가 호출자의 `caches` 슬라이스를 인덱싱한다. model-owned 풀 항목은 이 슬라이스를 비워 두므로 엔진 경로에서는 첫 attention 레이어에서 panic할 것으로 보인다. `LongCat-Flash-Ngram`은 forward마다 새 캐시를 만든다(`make_caches`에 초기화 부수효과가 없음). 둘 다 이 PR에서 바뀌지 않는다.

## 5. 학습 포인트

- 캐시를 내부에 보관하는 모델은 서버뿐 아니라 그 모델을 구동하는 모든 클라이언트에서 시퀀스 경계마다 초기화가 필요하다. 생성 루프가 `DirectEngine`으로 옮겨질 때 `reset_with_model`이 하던 초기화가 조용히 사라졌는데, model-owned 풀 항목이 완전해 보였기 때문이다.
- 측정 실행과 상태를 공유하는 워밍업은 정확성 버그를 처리량 회귀로 보이게 만들 수 있다. 디코드 A/B를 엔진 탓으로 돌리기 전에 생성 출력을 기준 빌드와 비교한다.
- 토큰당 작업을 프로파일링하기 전에 여러 `-n` 값으로 절편과 기울기를 fit한다. 여기서는 절편(302.0 vs 70.5 ms, 기울기는 거의 같음)이 호출당 한 번 드는 비용을 가리켰고, `cuLibraryLoadData`에 건 gdb breakpoint가 어느 호출이 어떤 커널을 로드하는지 보여 주었다.
- null arm(같은 바이너리를 한 번 더 실행)이 paired delta로 주장할 수 있는 범위를 정한다. granite의 main vs fix +2.14%는 추가 초기화의 비용처럼 보였지만, null arm이 동일 코드에서 +1.55%를 보였다.
