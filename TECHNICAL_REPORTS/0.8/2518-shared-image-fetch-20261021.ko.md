# PR #2518: media route 공용 이미지 fetch/decode helper

**Date**: 2026-10-21
**Status**: Completed (PR open)

## 요약

PR #2518(#1692 해결)은 이미지 URL을 읽고 decode하는 private 코드 사본 세 개를 helper 하나로 바꾼다. helper는 `src/server/media.rs`의 `fetch_labeled_image`다. 사본은 embeddings, rerank, images route에 있었다. 이슈에는 앞의 두 개만 적혀 있었고, images route의 사본은 그 뒤 #2497과 함께 들어왔다. 동작은 바뀌지 않는다. 모든 route가 이전과 같은 status, error type, 메시지를 반환한다. 이 PR 전에는 이 메시지들을 확인하는 테스트가 없었다. 그래서 helper 테스트 4개를 추가했다. 리뷰에서 테스트 강도에 관한 MEDIUM 지적이 하나 나와 고쳤고, 보안 점검에서는 지적 사항이 없었다.

## 1. 변경

| 영역 | 변경 |
|---|---|
| `src/server/media.rs` | `fetch_labeled_image(url, label, limits) -> Result<DynamicImage, ErrorResponse>`: `try_read_image_url_with_limits`, `decode_request_images_with_limits`, `pop()` 순서로 실행 |
| `routes/embeddings.rs`, `routes/rerank.rs`, `routes/images.rs` | private 코드를 helper 호출로 교체. label은 각 호출부가 계속 만든다(`input[{index}]`, rerank label, `image[{index}]`). |
| `src/server/media_tests.rs` | 테스트 4개: 성공 경로, 지원하지 않는 scheme, 읽기 실패, decode 실패 |

helper가 내는 메시지는 네 가지이고, 모두 400 `invalid_request_error`다.

| 경우 | 메시지 |
|---|---|
| 지원하지 않는 scheme | `{label}: unsupported image URL scheme` |
| 읽기 오류 | `{label}: failed to read image: {err:#}` |
| decode 오류 | `{label}: {err:#}` |
| decode 결과 없음 | `{label}: image decoded to nothing` |

## 2. 기술적 선택

- **세 번째 사본도 포함했다.** 이슈는 공용 helper 하나를 요구했다. `images.rs` 사본을 남기면 이슈가 없애려던 중복이 그대로 남는다.
- **route 고유 검사는 route에 둔다.** `images.rs`는 fetch 전에 자체 검사를 계속 실행한다. 빈 목록 검사, `validate_image_count`, 항목별 빈 값 검사, raw path 거부가 그것이다. 이 검사들은 fetch가 아니라 그 route의 계약에 속한다.
- **label은 호출부가 만든다.** 사본 사이의 차이는 label뿐이었다. label을 인자로 넘기면 모든 메시지가 바이트 단위로 같게 유지된다.
- **테스트 강도(리뷰 지적, MEDIUM).** 처음에는 decode 실패 테스트가 label 접두사만 확인했다. 이렇게 하면 "decode 결과 없음" 경우에도 테스트가 통과한다. 이제 전체 메시지 `documents[0]: Failed to decode any images`를 확인한다. 읽기 실패 테스트는 여전히 접두사만 확인한다. 메시지 뒷부분이 하위 reader의 오류에서 오기 때문이다.

## 3. 검증

| 항목 | 결과 |
|---|---|
| `server::media` | 82개 통과 |
| `server::routes::embeddings`, `rerank`, `images` | 각각 22, 20, 17개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. 보안 점검은 각 route가 같은 `ImageInputLimits` 값을 그대로 넘긴다는 점을 확인했다. 로컬 파일을 읽거나 네트워크에 접근하는 새 경로는 생기지 않았다.

## 4. 남은 항목

아래 두 문제는 리뷰에서 발견했다. 이 PR 전부터 있던 문제이고, 이 PR은 고치지 않는다.
- **깨진 PNG fixture.** `media_tests.rs`의 `tiny_png_bytes()` fixture는 IDAT CRC가 잘못되어 decode되지 않는다. `prompt_count_route_tests.rs`의 `PNG_BASE64`도 같은 데이터다. 현재 이를 decode하는 테스트는 없으므로 약해진 검증은 없다. 다만 `/// A 1x1 PNG` 주석은 실제보다 많이 주장한다. `rerank_tests.rs`의 `TINY_PNG`는 정상 PNG라서 대신 쓸 수 있다.
- **실행 순서에 따라 결과가 바뀌는 audio 테스트.** `audio_acquisition_preserves_order_and_rejects_any_invalid_item`은 약 11번에 1번 실패한다. 공용 테스트 media root가 설치되어 있으면 `read_audio_input`이 raw base64를 파일 경로로 처리한다. 그래서 결과가 테스트 실행 순서에 따라 달라진다.
