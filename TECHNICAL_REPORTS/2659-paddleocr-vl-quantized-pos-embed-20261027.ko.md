# PR #2659: PaddleOCR-VL vision 위치 테이블 역양자화

**날짜**: 2026-10-27
**상태**: Metal(M1 Ultra)에서 PaddleOCR-VL 4-bit, bf16 검증 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2659(이슈 #2657)는 `mlx-community/PaddleOCR-VL-4bit`가 동작하도록 고칩니다. 이 export는 vision tower의 학습된 위치 테이블까지 양자화해 저장하는데, `PositionEmbedding`은 이 테이블을 양자화되지 않은 일반 테이블로 읽었습니다. 그 결과 보간한 위치 임베딩이 1152개 특징값이 아니라 패킹된 144개 워드 폭이 되었고, patch 임베딩에 더하는 단계에서 CLI와 서버 모두 `[broadcast_shapes] Shapes (2664,1152) and (2664,144)`로 중단되었습니다. 이제 이 테이블은 `UnifiedEmbedding`으로 다룹니다. #2459에서 Qwen3-VL에 적용한 수정과 같은 방식입니다. 수정 후 4-bit 체크포인트는 bf16과 같은 OCR 텍스트를 내고, bf16 출력은 base와 바이트 단위로 같습니다.

## 1. 문제 정의

4-bit export에서 `visual.embeddings.position_embedding.weight`는 U32 `[729, 144]`이고, `.scales`와 `.biases`는 `[729, 18]`입니다(group size 64, 4 bit, 1152 / 8 = 144). `src/vision/encoders/paddleocr_vl.rs`의 `PositionEmbedding::from_weights`는 `.weight`를 그대로 복사하고 임베딩 폭을 `shape[1]`에서 가져왔습니다. 그러다 보니 `interpolate`가 패킹된 행을 gather했고, `[N, 1152]` patch 임베딩에 residual로 더하는 단계에서 실패했습니다.

같은 결함이 다른 텐서에도 있는지 확인하려고 개발 에이전트가 vision tower와 projector의 safetensors 헤더를 읽었습니다. `patch_embedding`, `fc2`, layer norm들, projector `pre_norm`은 export에서 BF16입니다. `qkv`, `out_proj`, `fc1`, projector `linear_1`/`linear_2`는 양자화되어 있지만 이미 `UnifiedLinear`로 올바르게 로드됩니다. 양자화되어 있는데 일반 테이블로 읽히던 텐서는 위치 테이블 하나뿐이었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `PositionEmbedding` | `UnifiedEmbedding`을 보관하고, vision 양자화 설정 `group_size`/`bits`를 인자로 받습니다. `interpolate`는 `UnifiedEmbedding::forward`로 행을 가져오는데, 이 함수는 양자화된 테이블이면 역양자화하고 일반 테이블이면 단순 `take`로 동작합니다. `embed` 필드와 아무 효과가 없던 reshape는 제거했습니다. |
| 로드 시점 검증 | 테이블이 2차원이고 행 수가 양의 제곱수인지 확인합니다. 보안 점검 후 추가했습니다(3장). |
| 테스트 | `paddleocr_vl_pos_embed_tests.rs`에 테스트 3개를 넣었습니다. 양자화 테이블의 보간 결과가 역양자화한 일반 테이블 결과와 같은지 확인합니다(base에서는 `[60, 128]` 대신 `[60, 16]`이 나와 실패). bf16 테이블 조회가 기존 행 gather와 비트 단위로 같은지도 확인하고, 15행과 17행 테이블은 로드 시 거부되는지 확인합니다. |

## 3. 기술적 선택과 그 이유

- **`config.hidden_size`를 읽지 않고 저장된 폭 자체를 없앴습니다.** 이슈는 폭을 config에서 가져오라고 제안했습니다. 하지만 조회가 `UnifiedEmbedding`을 거치면 폭을 읽는 곳이 없고, 기존의 `[h_in * w_in, embed]` reshape도 2차원 테이블에는 아무 효과가 없었습니다. 필드를 지우면 두 가지 레이아웃 중 한쪽에서는 틀리는 값을 들고 있을 일이 사라집니다.
- **로드 시점에 그리드 모양을 검증합니다.** 제거한 reshape는 행 수가 완전제곱수가 아닌 테이블을 걸러내던 유일한 장치였습니다. `interpolate`는 `sqrt(num_positions)`로 인덱스를 계산하고 범위를 검사하지 않고 gather하므로, 잘못된 체크포인트가 마지막 행 너머를 읽을 수 있습니다. 보안 점검에서 MEDIUM으로 지적되었고, 로드 시점 검사로 명확한 오류를 다시 냅니다.
- **조회할 때마다 역양자화합니다.** `UnifiedEmbedding`은 forward마다 가져온 행을 역양자화합니다. 테이블이 729 x 1152이고 gather가 이미지당 4번 실행되므로, 로드 시 일반 테이블 사본을 캐시하는 데 드는 메모리만큼의 가치는 없다고 판단했습니다. 비용은 측정하지 않았습니다.

## 4. 검증

M1 Ultra, greedy, 입력 이미지는 `tests/fixtures/got_ocr_page.png`입니다. base 바이너리는 cbe7919f에서 빌드했습니다.

| 확인 항목 | base | 브랜치 |
|---|---|---|
| 4-bit CLI `generate --image` | 중단(exit 134), broadcast 오류 | "GOT OCR two point zero renders this page" |
| 4-bit `mlxcel-server` `/v1/chat/completions`, base64 이미지 | 실행 안 함 | CLI와 같은 텍스트 |
| bf16 이미지 프롬프트 | 기준 | 바이트 단위 동일 |
| bf16 텍스트 프롬프트 | 기준 | 바이트 단위 동일 |
| 4-bit와 bf16 OCR 텍스트 비교 | 해당 없음 | 같은 문자열 |

오케스트레이터 게이트는 main cbe7919f 기준 브랜치에서 돌렸습니다. fmt, 크로스 레포 참조 검사, `-D warnings` workspace clippy, `vision::`, `models::paddleocr`, `vlm_family_batched_decode_tests` 테스트 모듈을 실행했습니다.

## 5. 검증하지 못한 것

- 4-bit 체크포인트를 base 서버로는 돌리지 않았습니다. CLI 크래시가 확인됐고 인코더 경로가 같으므로 서버 결과도 예상할 수 있지만, 이번 실행에서 직접 관찰하지는 않았습니다.
- bf16 텍스트 프롬프트는 토큰 1개("France")만 생성했습니다. 텍스트 경로는 바뀐 코드를 거치지 않습니다.
- Linux/CUDA에서는 실행하지 않았습니다.
- 보안 점검의 LOW 3건이 남아 있고 PR 본문에 적혀 있습니다.
  - 폭이 `hidden_size`와 다른 테이블은 로드 시점이 아니라 여전히 residual add에서 실패합니다. 이 PR 이전부터 있던 동작입니다.
  - 조회 비용은 측정하지 않았습니다.
  - 일반 테이블에 엉뚱한 `.scales` 텐서가 함께 있으면 양자화 경로를 탑니다. Qwen3-VL도 같은 동작입니다.

## 6. 학습 포인트

- MLX 양자화 export는 linear만이 아니라 vision tower 안의 임베딩 테이블도 양자화할 수 있습니다. `.weight`를 그대로 복사하는 로더는 옆에 `.scales`가 있는지 확인해야 합니다. 공용 해법은 `UnifiedEmbedding`이고, 이를 필요로 한 VLM은 #2459의 Qwen3-VL에 이어 이번이 두 번째입니다. 다른 vision 인코더에서 임베딩 테이블의 `.weight`를 직접 복사하는 곳을 한 번 훑어보면, 사용자의 체크포인트가 먼저 터뜨리기 전에 남은 사례를 찾을 수 있습니다.
- 아무 효과가 없어 보이는 코드를 지우면 암묵적인 보호 장치까지 함께 사라질 수 있습니다. 이번에는 그 reshape가 유일한 모양 검사 역할을 겸하고 있었습니다.
