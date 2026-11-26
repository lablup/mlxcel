# PR #3397: 실제 체크포인트로 검증한 Inkling 2 Small

**날짜**: 2026-11-24
**상태**: Draft PR 열림 (#3373 종료). 좁은 범위 테스트 통과. M3 Ultra (512 GB)에서 `models/inkling-2-small`을 mlx-vlm 466069f5와 비교 검증

## 요약

Inkling 2 Small은 512 GB 호스트에서 로드와 생성이 되고, 출력이 mlx-vlm과 일치한다. greedy 128 토큰이 동일하고, teacher-forced trace에서 decided 위치 불일치가 없다 (텍스트 0/409, 이미지 0/88). 첫 실행에서 Small에서만 드러나는 문제가 하나 나왔다. 첫 토큰이 원본을 쥔 채로 expert 가중치 대부분의 contiguous 복사본을 만들어서, 프로세스가 459 GiB까지 올라가고 호스트 여유 메모리가 10%까지 떨어졌다. 이제 로더는 mlx-vlm README가 Small에 권하는 대로 가중치를 레이어 하나씩 평가한다. 피크는 367 GiB로, 첫 prefill은 8.2 s에서 0.34 s로 줄었고 출력은 그대로다. decode는 19.5 tok/s (mlx-vlm 20.5)이고, 서버 동시 요청 4개의 합계는 36.8 tok/s이다. 번들 drafter는 코드 프롬프트에서 71% 빨라지고 일반 문장에서는 이득이 없다.

## 1. 문제

#3367은 Small (`inkling2`, 48 레이어, hidden 4608, 384 expert, loop 없음)을 tiny 모델 테스트로만 다뤘다. 타깃 가중치가 호스트의 wired 한도 480 GiB 중 363 GiB를 차지한다. 그래서 mlxcel과 mlx-vlm 레퍼런스는 한 번에 하나만 돌릴 수 있고, 로드 중 생기는 일시적 복사본도 적은 여유분에서 빠진다. upstream은 MLX 캐시를 끄고 레이어마다 `mx.eval`을 한 번씩 호출해 Small을 로드한다. mlxcel에는 이런 제한이 없었다.

| 위험 | 영향 | 대응 |
|---|---|---|
| 로드가 메모리에 안 들어가거나 가중치보다 훨씬 높게 피크 | Small 사용 불가 또는 스왑 | 측정 후 레이어 단위로 제한 |
| Small 고유 수치 (loop 없음, 384 expert) | 오류 없이 틀린 텍스트 | mlx-vlm과 greedy 및 trace 비교 |
| Small에서 loop 전용 커널 사용 | upstream과 어긋남 | 로드 시 게이트 로그, 디스패치 집계 |

## 2. 설계

### 2.1 메모리가 쓰인 곳

eager materialization 정책은 로드 시 체크포인트 텐서 1348개 (389.40 GB)를 평가한다. 그 뒤 sanitize가 interleave된 expert `w13`을 `contiguous`로 gate와 up으로 나누는데, 이것이 expert 바이트의 약 2/3이다. 이 복사본은 첫 토큰까지 lazy로 남아 있었다. 첫 토큰은 48 레이어의 복사본을 한 그래프에서 전부 만들었다. 원본은 각 복사가 끝날 때까지 참조가 남았고, 해제된 원본 버퍼는 MLX 버퍼 캐시에 남았다. 새 복사본은 크기가 달라 그 버퍼를 재사용하지 못한다. 프로세스 footprint는 458.9 GiB에 이르렀고, 22 토큰 첫 prefill에 8.2 s가 걸렸다.

### 2.2 레이어 단위 materialization (`src/loading/vlm_inkling2.rs`)

`materialize_by_layer`는 eager 정책에서만, sanitize와 모델 구성 사이에 실행된다. 캐시 한도를 0으로 두고 guard가 원래 값을 되돌린다. sanitize된 텐서를 디코더 레이어 순서대로 한 그룹씩, 마지막에 나머지를 한 그룹으로 평가하고 detach한다. 각 레이어의 복사본을 만들고 그 레이어의 `w13` 원본을 해제한 다음에야 다음 레이어를 읽는다. 정책상 modality tower는 요청이 필요로 할 때까지 lazy로 두므로 vision tower 텐서는 제외한다. 값은 첫 토큰이 계산했을 값과 같아서 출력은 바뀌지 않는다. 로드 `[TTFT]` 줄은 이제 정책이 materialize한 텐서 868개와 참조 없는 텐서 480개를 보고한다. 480개는 이미 소비된 `w13` 원본이다. `layer_groups`는 순수 함수다. 테스트는 순서, `layers.1`과 `layers.10` prefix 구분, config보다 큰 인덱스, tower 제외를 확인한다.

### 2.3 커널 게이트 증거 (`src/models/inkling2/kernels.rs`)

커널 디스패치는 스레드별로 집계하고, 커널별 첫 디스패치는 debug 레벨로 로그를 남긴다. `Inkling2Model::from_weights`는 게이트 판단을 로그로 남긴다.

```
Inkling 2 kernels: short convolution on, attention head preparation off, grouped expert matvec off
Inkling 2 short convolution kernel dispatched (first use)
```

집계가 스레드별이라 테스트는 자기 디스패치만 읽는다. 테스트는 게이트가 거절한 경우를 세지 않는지 확인한다.

### 2.4 레퍼런스 스크립트

`mlx_vlm_image_logit_trace.py --bounded-load`는 upstream README와 같은 방식으로 로드한다. `mlx_lm_logit_trace.py --mlx-vlm`은 이미 파라미터를 나눠 평가하고 있었고, 이제 그 루프 동안 캐시 한도도 0으로 둔다.

## 3. 검증

모든 실행은 GPU 락 아래에서 한 번에 런타임 하나씩 순서대로 했다.

| 항목 | 결과 |
|---|---|
| 수정 전 / 후 로드 | 54.7 s / 59.3 s; MLX 피크 362.66 / 367.04 GiB; 프로세스 footprint 458.9 / 368.5 GiB; 최저 여유 메모리 10% / 27% |
| `mlxcel inspect` | 가중치, KV, 활성값 364.82 GiB, 측정 피크보다 0.6% 낮음; 일반 1.2x 여유분 포함 437.78 GiB |
| Greedy, `Explain why the sky is blue.`, 128 토큰 | 수정 전후 모두 mlx-vlm과 텍스트 동일 |
| 텍스트 trace, 4 x 256 위치 | top-1 30/1024, decided 0/409, 불일치 최대 gap 0.320, perplexity -0.68% |
| 이미지 trace (1280 x 720) | prompt id 600개 동일, 4/126, decided 0/88, 최대 gap 0.260 |
| 서버 | 텍스트, `image_url`, `get_weather` (`{"city":"Warsaw"}`), 스트리밍과 비스트리밍 결과 동일 |
| Decode, ABAB 커널 off / on | 18.91 / 19.45 tok/s; mlx-vlm 20.5 |
| Prefill, 2274 토큰 | 361.0 / 362.5 tok/s |
| 서버 합계, C1 / C4 | 10.8~17.2 / 36.5~37.0 tok/s |
| Mini greedy | 레이어 단위 로드와 제한 없는 lazy 경로 사이 동일 |

DSpark: Small의 5 레이어 drafter는 코드 변경 없이 바인드된다. exactness probe는 Mini처럼 거절한다 (800232 logit 바이트 중 503855). `MLXCEL_MTP_ALLOW_INEXACT=1`로 256 토큰 decode를 돌리면 일반 문장은 19.3에서 19.3 tok/s (acceptance 0.20), 코드는 19.3에서 33.0 tok/s (0.46)다. 8행 verify가 약 107 ms로 decode 한 스텝 52 ms의 두 배라서, 라운드당 약 2.3 토큰보다 많이 내보내야 이득이 난다.

테스트 selector: `--lib inkling` (172 통과), `--lib coverage_tests` (18). clippy, fmt, contract 스크립트 통과.

## 4. 한계와 후속 작업

- Small에서 drafter는 일반 문장에서 손익분기다. verify forward를 싸게 만들면 달라진다. grouped expert 커널은 upstream처럼 looped 모델에만 켜진다.
- Small에서 mlx-vlm drafter는 돌리지 않았다. 라운드별 일치는 #3384에서 Mini로 확인했다.
- 서버 C1의 같은 128 토큰 wall time이 7.4~11.9 s로 흔들렸다. 측정값 그대로 기록했다.
- Small decode는 mlx-vlm보다 약 5% 느리다.
