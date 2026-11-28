# PR #2993: EXIF 방향 태그에 맞춰 이미지를 바로 세우기

**날짜**: 2026-11-14
**상태**: 완료 (보고서 작성 시점 PR open, #2992 종료)

## 요약

mlxcel은 이미지에 저장된 픽셀을 그대로 모델에 넘기고 EXIF `Orientation` 태그를 무시했습니다. 휴대폰으로 세로로 찍은 사진은 옆으로 눕힌 픽셀과 태그 6 또는 8로 저장되므로, 모델은 돌아간 이미지를 받았고 위치를 틀리게 설명하거나 회전된 글자를 읽지 못했습니다. mlx-vlm과 transformers는 `load_image`에서 `PIL.ImageOps.exif_transpose`를 호출하며, 체크포인트도 그 경로로 측정되었습니다. PR #2993은 디코드 직후 태그를 적용합니다. 적용 지점은 모든 서버 모델 계열과 `mlxcel generate`가 공유하는 디코드 함수, 디코드 전에 헤더로 프롬프트 크기를 정하는 Kimi K3 경로, 그리고 이미지를 직접 여는 명령들입니다. 태그가 없는 이미지는 이전과 같은 픽셀로 디코드됩니다. 리뷰에서 CRITICAL/HIGH 문제는 없었고, 하드닝 단계에서 8개 태그 전체에 대한 WebP 테스트와 Metal 실모델 검증을 추가했습니다. 제출된 PR은 16개 파일, +550/-21이며 대부분 테스트이고, 하드닝에서 47줄 WebP 테스트가 더해졌습니다.

## 1. 문제 정의

`image` crate(0.25.10)는 저장된 픽셀을 디코드하고 방향 처리는 호출자에게 맡깁니다(`ImageDecoder::orientation`, `DynamicImage::apply_orientation`). `src/server/model_worker.rs`의 `decode_request_image_with_limits`는 `reader.decode()` 후 바로 반환했습니다. 이 함수 하나가 chat `image_url`, `/v1/embeddings`와 `/v1/rerank`의 이미지 항목, `/v1/images/edits` 참조 이미지(모두 `media::fetch_labeled_image` 경유), diffusion, Florence-2, Nemotron-parse 워커, OpenXLA 전처리기, 그리고 `decode_image_payloads_with_limits`를 거치는 `mlxcel generate --image`를 담당합니다. CLI의 `embed`, `rerank`, `generate-image`, `generate`의 diffusion 경로, YOLO11과 RT-DETRv2 검출기는 `image::open`이나 `image::load_from_memory`를 직접 호출해 같은 문제가 있었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `multimodal/image_orientation.rs` (신규) | 헤더만 파싱해 태그를 읽는 `exif_orientation(bytes)`, `turn_upright`, `load_upright_from_memory`, `open_upright`, 헤더 전용 호출자를 위한 `upright_dimensions` |
| `server/model_worker.rs` | 공유 디코드가 `reader.decode()` 성공 후 태그를 적용 |
| `server/chat_request.rs` | `kimi_k3_image_prompts`에서 분리한 `planned_image_size`가 저장 크기로 제한을 검사하고 바로 선 크기를 반환 |
| `commands/{embed,rerank,generate_image,generate_vlm}.rs`, `vision/detection/{yolo11,rt_detr_v2}` | `image::open` / `load_from_memory`를 upright 헬퍼로 교체 |
| 테스트 | 메모리에서 만든 이미지(PNG `eXIf`, JPEG `APP1`, WebP `EXIF`), EXIF 정의로 계산한 저장 픽셀, 태그 1~8 전체 |
| 문서 | `docs/llama-server-compat.md`, `docs/embeddings.md` |

나머지 검출기(RF-DETR, PP-DocLayoutV3), OCR 명령, Florence-2는 이미 바이트를 읽어 공유 함수로 디코드하므로 코드 변경 없이 수정이 적용됩니다.

## 3. 기술적 선택과 그 이유

### 3.1 태그는 두 번째 헤더 파싱에서 읽음

단일 패스 방식(`into_decoder()`, `decoder.orientation()`, `DynamicImage::from_decoder`)은 채택하지 않았습니다. `ImageReader::decode`는 디코드 전에 출력 버퍼를 `max_alloc`에 대해 예약(`limits.reserve(decoder.total_bytes())`)하지만 `into_decoder`는 이 예약을 건너뜁니다. `reader.decode()`를 유지하면 decompression bomb 및 할당 검사가 그대로 남습니다. 방향 파싱은 디코드가 성공한 뒤에만 실행되므로, 그 헤더와 메타데이터 파싱은 이미 요청의 제한을 한 번 통과한 상태입니다.

비용은 이미지당 헤더 파싱 한 번입니다. PNG와 WebP는 이미지 데이터 직전까지 청크를 읽습니다. JPEG는 image 0.25.10의 `JpegDecoder::new`가 헤더를 읽기 전에 입력 전체를 `Vec`으로 복사하므로 JPEG마다 복사가 한 번 더 일어납니다. 입력 크기는 `--max-image-payload-size`로 제한되고, 그 크기의 memcpy는 디코드와 비전 인코더 비용에 비해 작습니다.

### 3.2 읽을 수 없는 태그 때문에 이미지를 거부하지 않음

태그가 없거나, 1~8 범위 밖이거나, EXIF가 없는 포맷이거나, 파서가 메타데이터를 거부하면 모두 "저장된 그대로 사용"으로 처리합니다. 요청의 유효성은 디코드가 결정하고 태그는 표시 힌트이므로, 디코드 가능한 이미지를 태그 때문에 거부하면 지금 잘못된 EXIF를 보내는 클라이언트에게 회귀가 됩니다.

### 3.3 제한은 저장된 픽셀 기준

너비, 높이, 할당 제한은 디코더가 저장 프레임에서 검사하고, 그 뒤 1/4 회전이 너비와 높이를 바꿉니다. Kimi K3도 같은 규칙을 따릅니다. `planned_image_size`는 저장된 헤더 크기로 제한을 검사하고 바로 선 크기를 반환하므로, 계획된 `<|media_pad|>` 개수가 회전된 이미지로 만든 그리드와 일치합니다. 1/4 회전은 같은 크기의 버퍼를 잠시 하나 더 할당하며, 이 복사본은 `max_alloc`에 포함되지 않으므로 이미지당 일시적 최대 메모리는 디코드 크기의 약 2배입니다.

### 3.4 사용자가 체감하는 동작 변화

태그가 있는 이미지는 이제 모든 모델에 바로 선 상태로 전달되며, mlx-vlm, transformers, 이미지 뷰어와 같습니다. 픽셀은 이미 바로 서 있는데 오래된 태그가 남아 있는 이미지는 그 도구들과 마찬가지로 회전됩니다. 검출 박스와 Florence-2, Nemotron-parse의 박스 출력은 이제 저장 프레임이 아니라 사용자가 보는 바로 선 프레임 기준입니다. 저장소 fixture를 읽는 벤치마크 바이너리와 진단 코드, EXIF가 없는 비디오 프레임은 바뀌지 않습니다.

## 4. 검증

- 단위 테스트: 8개 태그 전체를 PNG `eXIf`(픽셀 단위 일치), JPEG `APP1`(블록 허용 오차), WebP `EXIF`(픽셀 단위 일치, 하드닝에서 추가)로 검사합니다. 디코드만 한 결과가 저장 픽셀임을 함께 단언하므로 우연히 통과할 수 없습니다. 정의되지 않은 태그(0, 9, 600)와 이미지가 아닌 입력은 그대로 둡니다.
- 공유 디코드 테스트와 Kimi K3 크기 테스트는 수정을 제거하면 실패합니다(PR에 기록된 mutation 실행).
- macOS M3 Ultra(Metal), main 병합 후 트리: 대상 lib 테스트(120개 통과), `cargo clippy --lib --tests -D warnings`, release 빌드 통과.
- Metal 실모델, `gemma-3n-e2b-it-4bit`, greedy, shapes fixture: 바로 선 JPEG와 태그 3, 6, 8 JPEG 모두 "아래에 삼각형, 위에 사각형과 원"으로 답하고, 같은 태그 6 픽셀을 태그 없이 저장한 이미지는 "아래에 사각형"으로 답합니다. 서버(`POST /v1/chat/completions`, base64 `image_url`)에서도 태그 6과 태그 없는 쌍이 같은 결과로 갈립니다.
- 기여자의 GB10 실행은 Qwen3-VL-4B와 gemma-3-4b의 JPEG, PNG, WebP를 다루고, 태그 없는 이미지에서 35개 체크포인트가 알려진 디코드 비결정성 외에 변화가 없음을 보였습니다.

## 5. 학습 포인트

- `image` crate는 `open`, `load_from_memory`, `ImageReader::decode` 어디에서도 EXIF 방향을 스스로 적용하지 않습니다. 새 입력 경로는 `image::open` 대신 `decode_request_images_with_limits`(요청 바이트) 또는 `image_orientation::open_upright` / `load_upright_from_memory`(CLI 경로)를 거쳐야 합니다.
- 디코드 전에 헤더로 프롬프트 크기를 정하는 코드는 `upright_dimensions`를 써야 하며, 그렇지 않으면 태그 5~8에서 디코드된 그리드와 어긋납니다.

## 6. 미검증 항목

`embed`, `rerank`, `generate-image` 참조, YOLO11과 RT-DETRv2 검출기는 태그가 있는 입력으로 실제 체크포인트에서 실행하지 않았습니다. 이들은 테스트된 헬퍼로 한 줄씩 교체한 것입니다. 실제 휴대폰 사진은 쓰지 않았고 모든 이미지는 합성입니다. HEIC와 AVIF는 활성화된 `image` 기능이 아니므로 범위 밖입니다.
