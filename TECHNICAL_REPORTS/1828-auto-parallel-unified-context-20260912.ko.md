# 기술 리포트: PR #1828 - fix: share auto parallel context budget

**날짜**: 2026-09-12
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클(구현 리뷰, 보안·성능 리뷰)
**상태**: 완료(이슈가 지정한 정확한 QAT 체크포인트가 없어 같은 계열의 로컬 체크포인트를 사용함)
**언어**: Rust, JSON, Markdown
**위험도**: 중상(스케줄러 admission 및 decode 제한 변경; 추론 산술은 바뀌지 않음)

---

## 요약

PR #1828은 llama-server b10621의 자동 parallel-context 동작을 완성한다. `--parallel`을 기본값 `-1`로 두면 이제 표시상 슬롯 네 개와 unified context budget이 함께 활성화된다. 각 슬롯은 설정한 전체 컨텍스트를 사용할 수 있지만 모든 live sequence는 하나의 논리 토큰 예산을 공유한다. 명시적인 `--parallel` 또는 `--max-batch-size`는 `--kv-unified`도 명시하지 않는 한 분할 창을 유지하며, `--no-kv-unified`는 항상 분할 모드를 선택한다.

mlxcel은 계속 sequence별로 KV 상태를 할당한다. 따라서 unified 모드는 물리 cache layout을 바꾸지 않고 스케줄러 계측으로 b10621의 외부 계약을 구현한다. 스케줄러는 admission, prefill 직전, 첫 sampled token, 매 decode 성장 전에 prompt와 생성 토큰의 합을 검사한다. 더 성장할 공간이 없으면 재사용 가능한 prompt cache를 먼저 비우고, 그래도 부족하면 모든 live sequence를 context-exhausted metadata와 함께 length 종료한다.

---

## 문제와 해결된 동작

변경 전 auto `--parallel`은 슬롯 네 개를 만들면서 `--ctx-size`도 네 개의 고정 share로 나눴다. 따라서 `--ctx-size 20480`만 지정한 배포는 5,120토큰 슬롯 네 개를 노출했다. 반면 b10621은 각 슬롯의 개별 창이 20,480토큰인 슬롯 네 개를 노출하고 live 사용량의 합만 제한한다. 이 차이는 load 후에야 non-batching임을 알 수 있는 모델에서 더 크게 드러났다. Worker가 decode width를 하나로 줄여도 유일하게 실행 가능한 요청에는 설정한 컨텍스트의 1/4만 남았다.

해결된 정책은 다음과 같다.

- Auto `--parallel -1`: 슬롯 네 개, unified mode 활성화, 모든 슬롯에 전체 컨텍스트 보고.
- 명시적 `--parallel N` 또는 `--max-batch-size N`: 기본적으로 분할 창.
- 명시적 `--kv-unified` / `-kvu` / `LLAMA_ARG_KV_UNIFIED=true`: 슬롯별 전체 창과 하나의 공유 live-token budget.
- 명시적 `--no-kv-unified` / `-no-kvu`: 분할 창.
- Load 후 non-batching clamp: 하나의 유효 decode row가 설정한 전체 컨텍스트를 되찾고, runtime metadata를 다시 게시해 load 전의 오래된 share가 남지 않음.

b10621 호환성 manifest는 이제 `--parallel`, `--ctx-size`, `--kv-unified`를 supported로 분류하고 load 후 non-batching 경로의 좁은 차이를 기록한다.

---

## 구현과 기술적 선택

CLI는 해석된 값 4로부터 추측하지 않고 원래 parallel 값이 auto였는지를 보존한다. Startup은 이 source bit를 명시적 unified/split flag 및 명시적 max-batch override와 결합하고, 전체 컨텍스트와 유효 슬롯별 컨텍스트를 모두 `ServerConfig`에 저장한다.

`BatchScheduler`는 선택적인 공유 예산을 받는다. Live 사용량은 완료되지 않은 모든 decode-active sequence와 대기 중인 chunked-prefill sequence의 prompt 및 생성 토큰을 saturating 합산한 값이다. Checked addition으로 산술 overflow를 거부한다. 요청이 queue에서 기다리는 동안 기존 sequence가 성장할 수 있으므로 admission과 prefill에서 예산을 다시 확인한다. Decode tick 실행 전에는 active row마다 논리 토큰 하나를 예약하며, unified mode에서는 한 번의 검사를 여러 토큰 성장이 우회하지 못하도록 speculative burst를 비활성화한다.

Prompt-cache entry는 live sequence 소유량에 포함하지 않지만, 공유 예산 거부나 최종 고갈 전에 b10621의 선회수 정책에 맞춰 비운다. Decode tick이 그래도 들어가지 않으면 모든 live sequence를 `FinishReason::Length` 및 `context_exhausted`로 표시해 문서화한 결정적 fail-all 결과를 만든다.

일부 non-batching family는 모델 load 후에야 판별할 수 있다. Worker는 이제 release/acquire atomic을 통해 수정한 runtime context 및 KV 제한을 `BatchMetrics`에 게시한다. `/props`, `/slots`, `/v1/models`는 유효 runtime 값을 읽고 두 번째 geometry log가 load 후 결과를 기록한다. 따라서 자주 조회하는 metadata는 lock-free로 유지하면서 control-plane 보고와 실제 worker 제한이 어긋나지 않는다.

---

## 리뷰, 보안 및 성능

구현 리뷰에서는 호환성 shard pointer를 고치고 issue #1815를 runtime-and-context shard에 배정했으며, 공유 예산 계측 테스트와 admission, prefill, decode, runtime publication 경계를 모두 확인하는 정적 wiring 테스트를 보강했다. 이후 실제 모델 smoke에서 non-batching clamp 뒤 명시적 split metadata가 오래된 값으로 남는 문제가 드러났고, 커밋 `b039be14`가 이를 수정하고 회귀 테스트를 추가했다.

보안·성능 리뷰 후 남은 CRITICAL 또는 HIGH 문제는 없다. 토큰 합계는 checked 또는 saturating 연산을 사용하고, budget 결정은 모델 작업이 live 상태를 늘리기 전에 수행한다. 사용자 입력에 따른 새 무제한 allocation, filesystem 표면, credential 경로, lock ordering은 추가되지 않았다. 공유 계측 비용은 크기가 제한된 active batch에 대해 선형이다. Metadata 게시에는 atomic을 쓰며 prompt-cache eviction은 정상 용량 검사가 실패한 뒤에만 수행한다.

Tensor 연산, kernel 선택, quantization 규칙, 수치 산술은 바뀌지 않았으므로 이 control-plane 및 scheduler-budget 변경에는 teacher-forced logit trace가 해당하지 않는다.

---

## 검증 및 제한 사항

- 최종 구현 head에서 `cargo test --workspace --profile test-fast --features metal,accelerate`가 모든 workspace test binary와 doctest를 포함해 실패 없이 통과했다.
- Unified 해석과 창 크기, 공유 live-token 계측과 overflow 거부, non-batching clamp, metadata geometry, load 후 publication, scheduler wiring 집중 테스트가 통과했다.
- `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`, `cargo fmt --check`, `git diff --check`가 통과했다.
- `scripts/ci/check_llama_compat_manifest.py`가 help entry 249개, spelling 323개, 환경 변수 136개, route 53개, native field 74개, deferred issue 0개로 통과했다.
- 리뷰를 마친 구현 head의 GitHub CI에서 crate version, kernel dtype key, license, compatibility, cross-repository reference, dependency policy, formatting, clippy, OpenXLA compile 검사가 통과했다.
- 이슈의 정확한 체크포인트 `models/gemma-4-12B-it-qat-4bit`는 없었다. 같은 계열의 로컬 체크포인트 `models/gemma-4-12b-it-4bit`를 대체재임을 명시하고 사용했다.
  - Auto mode에서 `/props`, `/slots`, `/v1/models`가 슬롯 네 개, `kv_unified: true`, `n_ctx: 20480`을 보고했다.
  - 16,002토큰 completion 요청은 성공했고, 21,002토큰 요청은 가용 컨텍스트 20,480의 `exceed_context_size_error`로 거부됐다.
  - `--parallel 4 --kv-unified`도 같은 unified geometry를 보고했다.
  - Unified 없이 `--parallel 4`로 실행하면 모델의 non-batching clamp가 발생하고 요청 창을 전체 20,480토큰으로 복구했으며, 모든 metadata 표면이 수정된 load 후 값을 보고했다.

이 대체 체크포인트는 같은 family와 정확한 load 후 non-batching 경로를 검증하지만, 사용할 수 없었던 QAT 체크포인트의 weight 자체에 대한 증거는 아니다. Batching 가능한 구성의 명시적 5,120토큰 split geometry는 결정적인 startup 및 route 테스트가 계속 검증한다.
