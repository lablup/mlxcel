# PR #3455: Raon-1 mini DFlash2 speculative drafting (이슈 #3436)

**작성일**: 2026-11-26

**상태**: PR 열림(#3436을 닫음). M3 Ultra에서 검증. 34개 파일, +3794 / -33.

**위험도**: 중간(새 drafter와 target capture 추가. drafter를 넘기면 출시된 모델의 verify forward가 바뀌며, exactness gate가 이를 막는다)

## 요약

KRAFTON은 target 레이어 다섯 개의 mHC residual stream 네 개로 토큰 블록을 초안하는 DFlash2 speculator(`KRAFTON/Raon-1-mini-dflash2`)를 배포한다. 이 PR은 drafter를 `mlxcel-core`에 추가하고, `raon1_mini`를 `DFlashTargetModel`로 만들고, 둘을 기존 `--draft-model` / `--model-draft` 플래그로 `mlxcel generate`와 `mlxcel-server`에 연결한다. Greedy 초안 출력은 모든 실행에서 plain decode와 같고, 기본 width 5는 plain decode보다 1.30x에서 1.76x 빠르다.

## 1. 문제 정의

### 1.1 배경

drafter는 5레이어 sliding-window transformer(`block_size 8`)이고, 짧은 grouped conv와 top-16 candidate selector가 붙어 있다. 입력은 `[n, 51200]` feature다. target 레이어 1, 6, 11, 16, 21 뒤의 폭 2560 mHC stream 4개다. 자체 embedding과 head가 없어 target의 것을 빌린다.

### 1.2 기존 문제점

- **feature tap이 없었다.** `raon1_mini`(#3225)는 logit만 내놓아 forward 중간의 stream별 residual을 읽을 수 없었다.
- **verify가 decode와 달랐다.** Raon의 다중 토큰 forward는 1토큰 decode와 반올림이 다르다(RDA chunked recurrence, batched matmul). 여러 위치를 한꺼번에 채점하는 verify forward는 plain decode와 다른 greedy 토큰을 고를 수 있고, 그러면 exactness gate(`mtp_exactness_gate`)가 drafting을 거부한다.
- **샘플링.** mlxcel의 DFlash는 greedy 전용이었다. Raon의 `generation_config.json`은 샘플링(`temperature 1.0`, `top_p 0.95`, `top_k 20`)이라 greedy 전용 drafter는 기본 요청에서 거의 돌지 않는다.

## 2. 기술적 선택과 그 이유

### 2.1 decode와 바이트 단위로 같은 verify forward

| 선택지 | 장점 | 단점 |
|---|---|---|
| 모든 초안 위치를 batched verify | 가장 빠르다 | decode와 반올림이 달라 exactness gate에 막힌다 |
| **위치별 RDA와 MLA, row-invariant dense product** | 1토큰 decode와 같은 바이트라 greedy 출력이 바뀔 수 없다 | 완전 batched보다 verify가 느리다 |

verify forward 안에서 RDA conv, recurrence, MLA attention은 한 위치씩 진행한다. dense product(RDA projection, mHC selector, router, SiTU MLP, head)는 core `row_invariant`를 거친다. 한 행을 혼자 계산하든 블록으로 계산하든 같은 바이트가 나온다. `MLXCEL_RAON_ROW_INVARIANT=1`은 모든 forward를 row-invariant 커널로 돌려 초안이 더 빠르지만(code 프롬프트에서 약 2.1x) plain greedy 텍스트가 바뀌므로 기본은 꺼져 있다.

### 2.2 speculative 규칙으로서의 sampled verify

초안이 결정적이므로 샘플링 요청에 확률비 수락 검사가 필요 없다. round loop는 plain decode가 쓸 sampler와 토큰 history로 각 verify 행을 샘플링하고, 샘플이 초안과 같은 동안 수락한다. 그러면 내보내는 토큰 하나하나가 target 분포에서 뽑은 plain 샘플이다. 구현은 `drafter/dflash/sampled_verify.rs`이고 drafter별 opt-in(`Drafter::supports_sampled_verify`)이다. 호출 사이에 상태를 갖는 sampler(Mirostat, adaptive-p)는 거부한다. 행별 sampler 호출이 plain decode와 다르게 상태를 갱신하기 때문이다. 서버는 classic decode로 돌아가고 CLI는 조합을 거부한다(리뷰 지적, `3523d76a4`에서 수정).

### 2.3 prefill에서 drafter window만 capture

첫 버전은 프롬프트 모든 행의 `[L, 51200]` feature를 capture했다. 긴 프롬프트에서는 2048위치만 읽는 drafter를 위해 수 GB가 된다. `Drafter::context_window`, `Drafter::skip_context`, `Raon1MiniModel::limit_prefill_capture`가 prefill tap을 마지막 window로 제한한다. 다른 `DFlashTargetModel` 구현에는 no-op hook이 붙는다(보안 리뷰 MEDIUM, `ad9d715ee`). MLX slice는 원본 버퍼를 공유하므로 전체 길이 tap tensor는 첫 초안까지 살아 있고, peak 메모리는 0 근처가 아니라 약 절반으로 준다.

## 3. 구현 상세

- `src/lib/mlxcel-core/src/drafter/raon_dflash2/`: 이름 붙은 오류로 config 검증(`target_layer_ids`나 stream 수 불일치, dimension 곱의 i32 overflow), 벤더 NVFP4 키 형식을 거부하는 sanitize, 블록 경계를 넘지 않는 grouped block conv, grouped KV head 위의 non-causal attention, selector chain. feature 폭 불일치는 panic이 아니라 `DraftFailed`이고, `add_context` panic은 오류로 바뀌었다.
- `src/models/raon1_mini_speculative.rs`: 레이어 다음 stream 순서의 tap capture, decode와 같은 verify forward, 기록된 RDA state를 복원하고(`raon_rda.rs`의 `RdaStep`, `forward_recording`) MLA cache를 잘라내는 rollback. 요청의 reasoning-effort alpha(#3435)는 burst cache에 실린다. 서버에서 high와 low effort는 각각 plain 출력과 같고 서로는 다르다.
- 공유 Raon 모듈의 작은 수정: `raon_rda.rs`, `raon_mhc.rs`, `raon1_mini_moe.rs`, `raon1_mini.rs`, `raon1_mini_lm.rs`.
- width 기본값은 5이고, 학습된 8을 넘는 width는 거부한다.
- 문서: `docs/supported-models.md`, `docs/environment-variables.md`, `CHANGELOG.md`, 벤치마크 문서.

## 4. 검증

| 확인 | 결과 |
|---|---|
| Greedy, offline, 프롬프트 3개(short, code, 3137토큰) x width 3/5/7 x 2회 | 21 / 21 plain decode와 동일 |
| Greedy, 서버, effort high와 low | 각각 plain 서버 출력과 동일 |
| prefill window 변경 후 width 5, short와 3284토큰 프롬프트 | 여전히 동일 |
| 샘플링 분포 | Markov round-loop 테스트가 width 2와 4에서 joint distribution 일치, presence penalty가 수락된 초안을 본다 |
| 부분 수락 후 rollback | 새 decode와 동일 |

plain decode 대비 속도(GPU lock 안 ABAB, `docs/benchmark_results/raon-dflash2-m3ultra-2026-11-26.md`):

| 프롬프트 | Width 3 | Width 5 | Width 7 |
|---|---|---|---|
| Short | 1.39x | 1.30x | 1.10x |
| Code | 1.60x | 1.76x | 1.62x |
| Long | 1.76x | 1.69x | 1.43x |

수락률은 0.26에서 0.74다. 샘플링 width 5는 1.51x(1회 측정). babf65660에서 로컬 `make verify` 전체 게이트를 통과했고, main과 병합한 뒤 저장소 contract target도 통과했다.

## 5. 후속 조치

- 동시 배치는 classic decode로 돈다. 이 target에는 batched DFlash가 없다.
- drafter를 #3453의 NVFP4 export에서는 확인하지 않았다. 시작 시 exactness gate가 결정한다.
- sampled verify는 verify 행마다 host sync를 한 번 한다.
- 이슈가 지목한 기존 distribution-preservation 테스트는 sampled 경로에 돌리지 않았고, 새 Markov 테스트가 대신 다룬다.
- CUDA와 ROCm은 실행하지 않았다.
