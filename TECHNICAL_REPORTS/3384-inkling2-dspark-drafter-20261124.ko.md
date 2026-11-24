# PR #3384: Inkling 2 내장 DSpark drafter

**날짜**: 2026-11-24
**상태**: Draft PR 열림 (#3374 종료). 좁은 테스트 묶음 통과, M3 Ultra (512 GB)에서 `models/inkling-2-mini`로 mlx-vlm 466069f5와 비교 검증. Inkling 2 Small은 보류.

## 요약

Inkling 2 체크포인트 두 개 모두 `mtp.safetensors`에 DSpark drafter를 함께 싣고 있다. 이 PR은 그것을 mlx-vlm에서 포팅해 기존 DFlash 라운드 루프에서 오프라인과 서버 양쪽으로 실행한다. `--draft-model`에는 대상 디렉터리 자체를 넘긴다. 이 호스트에서는 block-versus-chain 정확성 프로브가 모든 폭에서 거부하므로, 기본 동작은 CLI가 이름 붙은 오류로 거절하고 서버가 일반 디코드로 처리하는 것이다. `MLXCEL_MTP_ALLOW_INEXACT=1`을 주면 Mini의 256토큰 greedy 디코드가 한 프롬프트에서 35.5에서 38.6 tok/s로, 다른 프롬프트에서 35.9에서 55.0 tok/s로 오른다. drafter 제안은 두 구현의 컨텍스트가 같은 모든 라운드에서 mlx-vlm과 토큰 단위로 일치한다.

## 1. 문제

#3367은 내장 drafter를 범위에서 뺐다. 그래서 mlx-vlm은 같은 체크포인트로 드래프트할 수 있는데 mlxcel은 Inkling 2를 한 토큰씩 디코드했다. 이 drafter는 일반 DSpark가 아니다. 48개 레이어 방문 중 네 곳에 들어가는 부분 블록을 읽고, 드래프트 레이어마다 융합된 대상 컨텍스트를 캐시하며(sliding 두 개, global 하나), 출력 투영이 대상 LM head에 접혀 들어가는 `rnn_hidden` Markov head를 쓴다. LFM2 DSpark 코드는 이 head를 다루지 않는다.

| 위험 | 영향 | 대응 |
|---|---|---|
| drafter 수치가 upstream과 어긋남 | 수락률 하락, 눈에 띄는 오류 없음 | 호스트 f64 기준 테스트, 실제 체크포인트에서 mlx-vlm과 라운드별 비교 |
| 검증되지 않은 드래프트 토큰이 출력됨 | 잘못된 출력 | 출력된 모든 토큰이 그 라운드 verify argmax인지 확인하는 테스트 |
| 여러 행 verify가 디코드와 바이트 단위로 같지 않음 | greedy 텍스트가 일반 디코드와 달라짐 | 정확성 프로브로 drafter를 막고, 우회는 명시적으로만 허용 |
| 엉뚱한 drafter와 짝지어짐 | forward 안에서 MLX shape abort | 별도 drafter family, bind 시 model type, 크기, 레이어 수, loop 비교 |

## 2. 설계

### 2.1 Drafter (`mlxcel-core/src/drafter/inkling2_dspark/`)

| 부분 | Upstream | 비고 |
|---|---|---|
| `config.rs` | `config.py` | `mtp_config`를 최상위 또는 `text_config`에서 읽고, `loop_config`로 방문 수를 계산하며, 잘못된 필드를 이름으로 거부하고, verify 폭 2..=`block_size`를 받는다 |
| `model.rs` | `DraftAttention`, `DraftMLP`, `DraftLayer`, `Model.mix` | local 레이어는 rope 적용된 최신 1024개 컨텍스트 키만 유지한다. 더 긴 컨텍스트는 앞쪽 행을 건너뛰되 offset은 그대로 진행한다 |
| `markov.rs` | `MarkovHead` | `hidden_out`을 LM head에 접는 연산을 호스트에서 4096행 타일 단위 f64로 다중 스레드 실행한 뒤, upstream의 numpy 경로처럼 f32와 BF16으로 반올림한다 |
| `drafter.rs` | `Model.bind`, `Model.draft_block` | 항상 학습된 블록 전체를 돌린다. 드래프트 logit은 대상의 BF16 head에서 나오고, softcap 30, unpadded 어휘에서 argmax |

대상은 새 `LanguageModel::inkling2_dspark_binding` 훅으로 임베딩, `embed_norm`, LM head를 빌려준다. dense head는 대상 자체 곱셈을 위해 f32로 보관되므로, 바인딩은 이를 체크포인트 dtype으로 되돌린다. 이 변환은 정확하다.

### 2.2 대상 (`src/models/inkling2_speculative.rs`)

verify forward는 모델의 hidden 경로를 재사용하면서, 지정된 각 방문 직전과 id `visit_count`에 대한 최종 norm 직후의 상태를 캡처한다. `language.py`와 같은 의미다. `DecodeLookaheadAppendScope` 안에서 실행되어 grouped expert 커널과 블록 하나 분량의 convolution 여유가 켜지며, 이는 upstream의 `speculative_verify=True`에 해당한다. 모든 verify 행에 모델의 억제 id(이미지 placeholder)를 `-inf`로 적용한다. 일반 디코드가 구성하는 bias와 같으므로 CLI는 정확히 그 bias만 허용할 수 있다. 롤백은 모든 방문 캐시를 `block_size - (accepted + 1)`만큼 잘라낸다.

### 2.3 연결

- CLI: `generate_dflash.rs`가 Laguna와 Inkling 2를 공통 꼬리 함수 하나로 처리한다. 이미지 프롬프트는 병합된 임베딩으로 prefill한다.
- 서버: `DFlashTargetModel for Inkling2VlModel`(프롬프트 모든 행이 drafter를 채움, B = 1 전용, 별도 family `Inkling2DSpark`)와 dispatch, exactness, label, drive 분기.
- `/v1/internal/mtp-policy`: DFlash dispatch는 그동안 pairing 필드가 빈 `no_mtp_dispatch`를 발행했다. 이제 pairing을 채운 `unavailable`(reason `dflash_dispatch`) 또는 프로브 문장이 담긴 `exactness_declined`를 발행한다. `docs/mtp-policy-api.md`에 따라 reason label 추가는 스키마 버전 변경 없이 가능하다.

## 3. 검증

모든 GPU 실행은 M3 Ultra에서 공유 lock 아래 직렬로 돌렸다.

### 3.1 정확성 판정

프로브는 폭 8(verify 위치 0이 체인과 800232 logit 바이트 중 464983개가 다름)과 폭 2(559076개)에서 거부했고, `qmv_wide`를 꺼도 같았다. mlx-vlm은 이런 게이트가 없고 같은 현상을 보인다. 두 프롬프트에서 drafter를 켠 텍스트가 자기 일반 디코드 텍스트와 213자, 214자 뒤에 갈라진다.

### 3.2 처리량 (ABAB, 256토큰, `MLXCEL_MTP_ALLOW_INEXACT=1`)

| 프롬프트 | 일반 tok/s | drafter tok/s | 수락률 | 라운드당 평균 수락 | mlx-vlm 일반 / drafter |
|---|---|---|---|---|---|
| Sky is blue | 35.54 | 38.55 | 0.170 | 1.18 | 33.9 / 42.2 (1.27 수락) |
| Fibonacci 함수 | 35.93 | 54.95 | 0.307 | 2.12 | 33.1 / 65.9 (2.71 수락) |

블록 동안 다른 worktree가 컴파일 중이라 load average가 2.2에서 8.4로 올랐다. 절대값에는 몇 퍼센트의 잡음이 있다.

### 3.3 mlx-vlm과의 일치

라운드 기록을 켜면 프롬프트 2의 라운드 0부터 2까지 mlx-vlm과 같은 7개 토큰 id를 제안하고, 같은 bonus 토큰에서 시작하는 라운드 4와 5도 같다. 두 실행은 라운드 2에서 갈라진다. 둘 다 제안 7개를 모두 수락하지만 bonus 행이 여기서는 토큰 149899를, mlx-vlm에서는 쉼표를 고른다. 8행 verify에서의 근소한 동률이다. 따라서 3.2의 수락률 차이는 drafter가 아니라 서로 다른 텍스트 경로에서 온다. mlx-vlm drafter 실행과의 텍스트 일치 길이는 213자와 71자다.

### 3.4 기타 경로

- drafter를 켠 이미지 프롬프트는 일반 디코드와 같은 48토큰을 출력한다.
- `models/inkling-2-mini-4bit`는 sidecar가 없고 이전처럼 일반 디코드로 실행된다(43.05 tok/s).
- 서버: 채팅 요청 하나가 DSpark burst로 실행되었고(34라운드, 수락률 0.272), mtp-policy가 pairing과 함께 `dflash_dispatch`를 보고했다. 우회 없이는 `exactness_declined`를 보고하고 일반 디코드로 처리했다.
- 단위 테스트: f64 호스트 기준과 비교하는 drafter 테스트 13개(mix, Markov step, fold, 윈도를 넘고 블록을 좁히는 세 라운드의 `draft_block`, bind 거부, config 거부), 작은 loop 모델에서의 대상 테스트 5개(캡처, 억제, 롤백, 프로브, 매 라운드 출력 토큰이 verify argmax와 같음), 기존 `inkling`, `drafter::`, coverage, detection, dispatch, mtp_policy 묶음.

## 4. 한계와 후속

- Inkling 2 Small은 검증하지 않았다. 다운로드에 `.done` 표시가 없고 #3373이 열려 있다. 코드 경로에는 Mini 전용 상수가 없다.
- 이 호스트에서는 inexact 우회 없이 drafter가 동작하지 않는다. 정확하게 만들려면 verify 행이 디코드 단계와 같은 커널을 타야 하며, 이는 #3369가 기록한 문제와 같다.
- f64 fold는 첫 bind에서 약 8.8초 걸린다. 서버는 warm-up 요청이 흡수하고, CLI는 로드 시간에 부담한다.
- 서버는 이미지 요청, B > 1 윈도, 채택된 prompt-cache prefix를 일반 디코드로 처리한다.
