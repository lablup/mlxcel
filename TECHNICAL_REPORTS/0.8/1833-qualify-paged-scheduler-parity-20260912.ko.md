# 기술 리포트: PR #1833 - test: qualify paged scheduler parity

**작성일**: 2026-09-12
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클(구현 리뷰, 보안·성능 리뷰, 최종화)
**상태**: 완료(보고된 LLM-jp VLM 차이는 현재 CUDA 빌드에서 재현되지 않았으며 검증한 계약은 의도적으로 더 좁음)
**언어**: Rust, Markdown
**위험도**: 낮음(문서와 ignored 실모델 테스트만 변경하며 요청 경로 코드는 바꾸지 않음)

---

## 요약

PR #1833은 paged decode가 dense decode와 바이트 단위로 같다는 두 곳의 무조건적인 주장을 저장소가 실제로 증명할 수 있는 범위로 좁힌다. 통합 테스트는 이제 greedy token마다 전체 selector-logit 행을 기록하고, greedy token의 정확한 일치와 relative RMS `5e-5` 이하를 함께 요구한다. CUDA qwen3 arm은 TF32를 끄고 실행한다. 서빙 문서는 처리량 기본값과 token-exact oracle 설정을 구분하고, 자동 저장소 선택까지 바꾸는 `--max-batch-size 1`보다 `--decode-storage-backend dense`를 먼저 써서 backend만 이분하도록 안내한다.

원래 보고에 사용된 `llm-jp/llm-jp-4-vl-9b-beta`는 NVIDIA GB10에서 다시 실행했다. CLI, 기본 서버, 기본 batch width에 dense를 강제한 서버, `--max-batch-size 1` 서버가 모두 같은 15토큰 연속 출력을 만들었다. 이는 보편적인 VLM parity나 fused paged-v2 parity를 증명하지 않는다. 297토큰 프롬프트는 fused dispatch 기준보다 짧아 gather-then-SDPA를 실행한다.

---

## 문제 정의

기존 문서는 paged decode가 dense backend와 바이트 단위로 같다고 서술했다. 근거로 든 `tests/paged_scheduler_parity.rs`는 qwen3와 llama3의 Fp16 dense-natural cache만 덮는 ignored 실모델 테스트였다. CUDA 전용 arm과 VLM arm이 없으므로 VLM front end, model-owned cache, Turbo나 quantized KV 저장소, 모든 CUDA reduction geometry까지 포괄하는 보장을 뒷받침할 수 없었다.

이슈 #1773은 LLM-jp VLM 체크포인트에서 `mlxcel generate`와 continuous-batching 서버가 일본어 토큰 하나만큼 다르다고 보고했다. 하지만 `--max-batch-size 1` 결과는 batch width만 비교한 것이 아니다. 자동 저장소 선택을 쓰면 이 옵션은 decode storage도 paged에서 dense로 바꾼다. 기본 admission width에서 명시적으로 dense를 실행하지 않으면 회복 원인을 잘못된 차원에 귀속할 수 있다.

기존 token-only parity 테스트는 수치 차이가 argmax를 바꿀 때까지 drift를 볼 수 없었다. 근접 동률에서는 전체 오차가 작아도 출력이 바뀔 수 있는 반면, 특정 일본어 연속 출력을 고정하면 cache backend 계약이 특정 모델, processor, prompt, vocabulary에 결합된다. 따라서 테스트는 문장이 아니라 backend 속성을 직접 비교해야 했다.

### 기존 계약을 유지할 위험

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 운영자가 처리량 기본값을 보편적인 token-exactness 약속으로 해석 | 중간 | 중간 |
| `--max-batch-size 1` 결과를 저장소 backend가 아니라 batch width 효과로 오진 | 높음 | 중간 |
| CUDA 수치 drift가 greedy 결정을 바꿀 때까지 보이지 않음 | 중간 | 중간 |
| 재현되는 이분 없이 VLM 증상이 관련 없는 kernel, reduction, MLX pin 변경으로 확대 | 높음 | 중간 |

---

## 변경 요약

- 문서의 paged/dense parity 범위를 scheduler 모양의 qwen3·llama3 Fp16 dense-natural-cache B=1/B=2 실행과 명시적 CUDA qwen3 수치 계약으로 한정했다. VLM front end, model-owned cache, Turbo/quantized KV mode, 모든 CUDA reduction geometry는 주장 범위 밖에 남긴다.
- continuous-batching 안내서는 서버 기본값이 CLI와의 보편적인 token-exact 동일성이 아니라 서빙 처리량 계약이라고 명시한다. `--no-batch`는 legacy worker를 고르고, `--max-batch-size 1`은 scheduler를 유지하지만 자동 저장소가 dense를 고르게 하며, `--decode-storage-backend dense`는 기본 admission width를 유지한 채 저장소만 격리한다.
- `DecodeStepTrace`는 출력 토큰마다 그 토큰을 선택한 logit 행을 연결한다. 첫 행은 prefill 마지막 행이고 이후 행은 앞선 decode step에서 온다. 리뷰에서는 post-token next logits만 비교하던 초기의 한 단계 어긋남을 고쳤다.
- 전체 vocabulary logits를 host f32로 변환한다. 테스트는 빈 행과 NaN·무한대 값을 거부하고 `RMS(actual - reference) / RMS(reference)`를 계산해 `5e-5` 이하를 요구하며, greedy-token 동일성은 별도로 요구한다.
- CUDA-gated qwen3 테스트는 기존 테스트와 같은 scheduler 모양 paged layout을 만들고 dense allocation과 비교한다. 증거 명령은 `MLX_ENABLE_TF32=0`을 쓰고, `MLXCEL_REQUIRE_MODELS=1`은 체크포인트 누락을 soft-skip이 아니라 실패로 만든다.

---

## 기술적 선택과 그 이유

### 저장소 backend 이분을 batch width에서 분리

첫 비교로 기본 admission width와 `--decode-storage-backend dense`의 조합을 선택했다. 조사하는 decode storage 선택만 바뀌기 때문이다. `--max-batch-size 1`도 scheduler 모양의 단일 요청 진단에는 유용하지만, 그 폭에서는 `auto`가 dense로 결정되므로 순수한 batch-width arm으로 취급하지 않는다.

Scheduler 동작은 바꾸지 않았다. 기존 선택 방식만 문서화해 이후 조사가 교란된 실험 위에 인과 주장을 세우지 않도록 했다.

### 문장이 아니라 수치와 선택 속성을 테스트

계약은 두 층으로 구성된다. 정확한 token 일치는 사용자가 보는 greedy divergence를 잡고, 전체 selector-logit 행의 relative RMS는 argmax가 유지되는 동안에도 backend drift를 잡는다. 이는 #1773의 일본어 단어를 단언하는 방식보다 재사용 가능하고, 모델 processing·tokenizer·prompt 문구를 cache storage 테스트 밖에 둔다.

Trace는 forward 호출이 아니라 결정에 맞춰 정렬한다. 토큰 `t[n]`이 logits `L[n]`에서 출력되면 `(t[n], L[n])`을 저장한다. 따라서 첫 비교는 한 단계 늦게 시작하지 않고 prefill의 마지막 logits를 포함한다.

### 바이트 동일성 대신 제한된 상대 오차 선택

바이트 동일성은 가용한 증거보다 강하고 보편적 CUDA 주장으로 부적절했다. TF32를 끈 상태에서 relative-RMS `5e-5` 한계는 측정 가능한 수치 계약을 정의하면서 greedy 결정은 정확히 같게 유지한다. 빈 입력과 non-finite 입력을 명시적으로 실패시켜 NaN이 비교를 우회하지 못하게 한다.

### Gather와 fused paged 증거를 구분

297토큰 LLM-jp 요청은 4,096토큰 fused paged-v2 dispatch 기준보다 짧다. 따라서 기본 서버 실행은 gather-then-SDPA paged 경로만 검증한다. 리포트와 문서는 이 성공적인 이분을 fused-kernel 증거로 표현하지 않고 경계를 유지한다.

---

## 검증

### LLM-jp VLM CUDA 이분

spark-101에서 실행했다. 환경은 NVIDIA GB10, compute capability 12.1, driver 580.173.02, Linux aarch64다. 체크포인트는 `/home/inureyes/models/mlx/llm-jp-4-vl-9b-beta`, 단색 빨강 이미지 SHA-256은 `5eafcdbe57b88e9c12ef8ac4cc3eee45f9c3433b7f929d867e5dd4d7832a8aed`였으며 모든 수치 실행에 `MLX_ENABLE_TF32=0`을 설정했다.

| Arm | Admission/storage 선택 | 결과 |
|-----|------------------------|------|
| `mlxcel generate` | CLI cache 경로 | 같은 15토큰 연속 출력 |
| `--decode-storage-backend dense` 서버 | 기본 max batch, dense 강제 | 같은 15토큰 연속 출력 |
| 기본 서버 | 기본 max batch, `decode_storage=auto` | 같은 15토큰 연속 출력 |
| `--max-batch-size 1` 서버 | scheduler 유지, `auto`가 dense 선택 | 같은 15토큰 연속 출력 |

보고된 `キャンバス` 차이는 현재 빌드에서 재현되지 않았다. 오래된 설치 바이너리는 Apple이나 CUDA 세대 차이의 증거로 해석하지 않았고, `llmjpvl`을 읽지 못해서 제외했다.

### Paged/dense 계약

최종 HEAD `2dc42d9db33072a45fe26c36849eefacfbcdccee`에서 다음 GB10 명령은 체크포인트 존재를 강제한 채 통과했다.

```text
MLXCEL_REQUIRE_MODELS=1 MLX_ENABLE_TF32=0 cargo test --test paged_scheduler_parity --release --features cuda cuda_paged_scheduler_qwen3_matches_dense_within_logit_contract -- --ignored --nocapture
```

Dense와 paged의 greedy trace는 같았다.

```text
[1079, 264, 5458, 304, 279, 220, 16, 15, 339, 11972, 13, 358, 614, 311, 3270, 264]
```

결과는 1 통과, 0 실패였다. 더 넓은 ignored integration target은 리뷰 커밋 `363d6648`에서 5/5 통과했다. qwen3는 실제 실행했고 llama3는 호스트에 체크포인트가 없어 soft-skip했다. CUDA clippy, 로컬 format 검사, `git diff --check`, Metal/Accelerate integration target 컴파일, focused clippy, 필수 CI 검사도 모두 통과했다.

---

## 학습 포인트

- **진단 옵션은 이름보다 많은 것을 바꿀 수 있다.** `--max-batch-size 1`은 admission width와 자동 KV storage 선택을 함께 바꾼다. 유효한 이분은 여러 control을 조합하기 전에 한 차원을 명시적으로 바꾼다.
- **수치 trace는 결정에 정렬해야 한다.** 토큰 출력 뒤에 반환된 logits를 그 토큰을 선택한 logits로 표시하면 첫 결정을 누락하고 모든 행을 한 칸 민다. Selector logits를 기록하면 불변 조건이 명확해진다.
- **soft-skip 통과는 하드웨어 증거가 아니다.** 실모델 증거 명령은 모델 존재를 강제해야 한다. 탐색이나 CI 실행에서는 soft-skip을 유지할 수 있지만 체크포인트 기반 통과로 보고해서는 안 된다.
- **Backend와 하드웨어 세대는 별도 가설이다.** 실제 CUDA 호스트를 사용했고 호환되지 않는 옛 바이너리는 버렸다. Apple 세대나 CUDA 세대 원인을 추정하지 않았으며 재현 없이 MLX pin이나 kernel 변경으로 범위를 넓히지 않았다.

---

## 변경 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 3 |
| 추가 줄 | 219 |
| 삭제 줄 | 37 |
| Runtime 요청 경로 변경 | 0 |
| 새 CUDA 계약 테스트 | 1 |

| 범주 | 요약 |
|------|------|
| 테스트 | Selector-logit trace, relative-RMS 검증, finite-value guard, 체크포인트 강제 증거 mode, CUDA qwen3 arm 추가 |
| 문서 | 보편적 바이트 동일성 문구를 측정 범위로 교체하고 교란되지 않은 진단 control 문서화 |
| Runtime | Production scheduler, cache, kernel, reduction, model 코드는 바꾸지 않음 |

### 관련 커밋

| 해시 | 요약 |
|------|------|
| `51f277f` | 범위를 한정한 parity 계약과 CUDA arm 도입 |
| `032353c` / `9e5f192` | 임시 branch-scoped GB10 진단 workflow를 추가했다가 완전히 되돌림 |
| `2967db0` | 리뷰가 정렬 문제를 드러내기 전 trace 용어를 명확화 |
| `363d664` | 모든 logit 행을 그 행이 선택한 토큰에 정렬 |
| `2dc42d9` | 증거 실행의 모델 존재 강제와 수치 검증 강화 |

---

## 후속 조치

- 현재 비재현 결과로는 kernel, reduction 순서, scheduler, MLX pin 수정을 정당화할 수 없다.
- LLM-jp 차이가 현재 pin의 빌드에서 다시 나타나면 네 arm backend 이분을 유지하고 처음 갈리는 결정의 selector logits를 수집한 뒤, 범위를 넓히기 전에 별도 correctness 이슈로 기록한다.
- Fused paged-v2, VLM 전용 cache, model-owned cache, Turbo/quantized KV storage는 각각 증거가 생기기 전까지 문서 주장 범위를 넓히지 않는다.

---

## 참고 자료

- 이슈 #1773: server and CLI diverge by one token on LLM-jp-4-vl-9B
- `tests/paged_scheduler_parity.rs`
- `docs/CONTINUOUS_BATCHING.md`
- `docs/turbo-kv-cache.md`
