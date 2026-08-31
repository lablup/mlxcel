# 기술 보고서: PR #1546 - Inkling 인접 프레임 비디오 지원

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 리뷰 중. 보고된 HIGH 항목을 모두 수정하고 결정론적 검증 완료
**언어**: Rust, Markdown
**위험 수준**: 높음

---

## 요약

PR #1546은 PR #1535의 HMLP 이미지 shell에 Inkling 네이티브 비디오 입력을 추가한다. CLI `--video`와 OpenAI 호환 서버 `video_url`은 이제 하나의 제한된 host 준비 경로를 공유한다. 이 경로는 clip별로 인접 프레임 pair를 선택하고, clip마다 독립된 시간축을 유지하며, 선택한 source frame만 디코딩하고, 두 프레임을 별도 이미지 entity로 만드는 대신 HMLP temporal axis를 사용한다.

최종 설계는 독립 리뷰에서 보고된 HIGH 세 건을 해결한다. 구조화된 timestamp와 image part는 전역 BOS 뒤가 아니라 검증된 현재 사용자 turn 경계 안에 삽입된다. 여러 비디오는 clip 경계를 유지한 채 하나의 요청 전체 16-pair 예산을 공유하며 clip 사이 pair를 만들지 않는다. Probe로 계산한 frame 및 pixel 예산은 첫 디코딩 전에 검사하고, 전처리는 전체 샘플 시퀀스를 깊은 복사하지 않고 compact selection을 참조한다. Timestamp는 실제 선택한 source index와 source FPS를 사용해 일반 768-frame 샘플링 상한 이후의 drift도 제거한다.

이 브랜치는 main commit `092d3dd0`의 Inkling MTP 구현도 일반 merge로 통합했다. 결합된 wrapper는 image/video 요청의 HMLP prepared-embedding prefill을 유지하면서 MTP가 사용하는 같은 공개 text backbone과 상태 lifecycle을 제공한다.

## 1. 문제 정의

Inkling visual input shape는 `[N, 2, 40, 40, 3]`이다. Still image는 같은 pixel을 두 temporal slot에 복제한다. Video는 선택한 첫 프레임을 slot 0에, 바로 인접한 프레임을 slot 1에 넣으면서 tile마다 prompt image entity와 HMLP feature row를 하나만 유지해야 한다.

올바른 host 경로는 대화 구조를 보존하고 신뢰하지 않는 media 작업량도 제한해야 한다. Pairing 전에 clip을 평탄화하면 한 비디오의 마지막 프레임과 다음 비디오의 첫 프레임이 결합될 수 있고, 뒤쪽 clip은 잘못된 timestamp 원점을 사용한다. 16-pair 제한을 적용하기 전에 모든 sampled frame을 디코딩하면 메모리는 허용된 작업량이 아니라 입력 크기에 비례한다. Render된 image placeholder를 BOS 다음으로 옮기면 system message나 history가 있을 때 media가 현재 사용자 turn 밖에 놓일 수 있다.

이러한 오류는 tensor shape가 정상인 상태에서도 시간 순서를 바꾸고, 잘못된 temporal row를 수정하고, 관련 없는 prompt 위치에 feature를 scatter하거나, 명목상의 pair 제한이 적용되기 전에 메모리를 소진할 수 있다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| Clip-local planning | 비디오별로 인접 pair index를 독립 계산하고 media 경계를 넘는 pair를 만들지 않음 |
| 요청 전체 할당 | 허용한 각 clip에 레퍼런스 최소값인 두 pair를 먼저 주고 남은 capacity를 16-pair 총량 안에서 round-robin으로 배분. 최대 8 clip 허용 |
| Timestamp 정확도 | 768-frame 샘플링 상한 이후를 포함해 sampled anchor를 실제 source-frame index로 다시 매핑하고 probed source FPS로 나눔 |
| Resource admission | 모든 clip을 probe한 뒤 첫 ffmpeg decode 전에 고유 선택 frame 최대 32개와 RGBA 최악 기준 512 MiB를 검사 |
| 선택적 decode | 기존 fd-safe single-pass extractor로 고유 source index만 디코딩하고 compact clip별 pair index를 저장 |
| Prompt 안전성 | 완전한 Inkling text/image content part를 마지막 사용자 text part 바로 앞에 삽입. System message, history, companion still part, generation tail 위치 유지 |
| Plain CLI mode | 구조화된 turn이 없는 명시적 `--no-chat-template`에서는 media를 BOS 뒤에 prepend하는 동작을 별도로 유지 |
| Temporal splice | Video tile에 해당하는 suffix의 slot 1만 교체. Companion still row는 복제 temporal plane 유지 |
| Borrowed preprocessing | Compact decoded frame 참조를 처리해 first/second frame 목록 생성 시 image buffer deep clone 방지 |
| MTP 통합 | Main에서 merge된 공개 `InklingVlModel.text` target 및 prepared-embedding sequence/last-logit entry point 유지 |
| Capability 및 문서 | Inkling 서버 video admission 유지, clip·pair·frame·byte·prompt·checkpoint 제한 문서화 |

## 3. 구조와 데이터 흐름

1. CLI path는 `VideoSource`가 되고 서버 part는 이미 admission을 통과한 fd 기반 `ResolvedVideo.source` handle을 유지한다.
2. 모든 clip을 기존 `VideoLimits`로 probe하고 요청 FPS를 검사한 뒤 일반 sampled-frame 수를 계산한다.
3. 요청 전체 allocator가 clip마다 두 pair를 예약하고 남은 16-pair 예산을 clip 경계를 넘지 않게 배분한다.
4. Clip별 sampled anchor를 uniform source index에 매핑한다. 첫 decode 전에 compact 고유 source frame 집합, 실제 anchor timestamp, decoded-frame 총량, 최악 decoded byte 수를 알 수 있다.
5. 고유 index만 디코딩한다. Pair index는 compact clip별 vector를 가리키므로 짧은 clip의 반복 anchor도 image buffer를 복제하지 않는다.
6. 구조화된 prompt는 완전한 Inkling content part를 현재 사용자의 마지막 text part 앞에 받는다. 각 clip은 독립된 시간 순서 안내문과 해당 clip source time에서 시작하는 timestamp를 갖는다.
7. Companion still과 pair의 첫 프레임을 함께 tile한다. Pair의 두 번째 프레임은 borrowed reference로 tile하며 pair별 tile 수가 같아야 한다.
8. 마지막 video-tile row의 slot 1만 교체한다. 공통 HMLP tower가 visual feature를 만들고 placeholder run은 실제 tile 수로 확장되며 `merge_llava`가 normalized text embedding에 feature를 scatter한다.
9. 결합된 `InklingVlModel`의 prepared-embedding entry point는 해당 embedding을 공개 text backbone으로 직접 전달한다. 따라서 정규화를 중복하지 않으면서 MTP 호환 sequence-state lifecycle을 유지한다.

## 4. 기술적 선택과 이유

### 4.1 공개 인접 pair 공식을 clip 내부에서 유지하기

Sampled-frame 수가 홀수인 clip은 마지막 프레임을 개념적으로 반복한다. Pair 수는 `max(2, min(clip_budget, padded_len / 2))`다. Anchor `i`는 `min(round(i * (padded_len - 2) / max(pair_count - 1, 1)), padded_len - 2)`이며 두 번째 위치는 `anchor + 1`이다. 개념적으로 반복한 마지막 프레임에서만 실제 frame 범위로 clamp한다. 따라서 한 프레임 clip도 서로 같은 `[0, 0]` pair 두 개를 만들며 별도 short-clip graph를 발명하지 않고 레퍼런스 최소값을 따른다.

16-pair 제한은 요청 전체에 적용하지만 공식은 clip마다 계산한다. 각 clip에 두 pair를 먼저 할당하고 나머지를 capacity까지 round-robin으로 배분한다. 이 방식은 결정론적이고 starvation을 막으며 표현 가능한 최대 요청을 8 clip으로 만든다.

### 4.2 Source index를 timestamp의 기준으로 사용하기

일반 sampler는 긴 입력을 균등한 768 frame으로 제한할 수 있다. 이 상한 뒤에 `sampled_anchor / requested_fps`로 시간을 계산하면 수백 초 빠른 timestamp가 될 수 있다. 강화한 경로는 선택한 sampled position을 실제 source index로 매핑하고 `source_index / probed_source_fps`를 계산한다. 각 비디오는 자체 mapping을 가지며 독립된 시간 순서 prompt sequence를 시작한다.

### 4.3 Decode 전에 제한된 작업량을 검증하기

Loader는 frame extractor를 호출하기 전에 허용한 모든 source를 probe하고 모든 pair를 계획한다. 8 clip, 16 pair, 고유 선택 frame 32개, `width * height * 4 * selected_frames` 기준 512 MiB를 넘는 요청을 거부한다. Checked arithmetic으로 overflow도 오류로 변환한다. 기존 duration, resolution, source 제한은 probe 단계에서 계속 적용된다.

Extractor에는 정렬한 고유 source index만 전달한다. Clip별 map이 source index를 compact vector offset으로 바꾼다. Runtime 준비는 first/second entity에 `&DynamicImage`를 사용하므로 전체 sampled sequence와 선택 frame buffer 어느 쪽도 pair list 생성을 위해 복제하지 않는다.

### 4.4 Media를 현재 사용자 turn 안에 유지하기

공개 Inkling template은 user message, text content, image content, end-of-message 경계의 structural token을 출력한다. Runtime은 tokenizer에서 이 정확한 ID를 확인하고 마지막 `[message_user, content_text]` 경계를 찾는다. 이어지는 end marker가 있어야 하며, 뒤에 추가 user content가 있으면 거부하고, companion image placeholder가 질문보다 앞에 있어야 하며, 삽입 전후 placeholder cardinality가 정확해야 한다.

각 clip마다 완전한 text/image content part를 마지막 질문 바로 앞에 삽입한다. System message, 이전 user/model turn, 기존 companion still-image part, model-generation suffix는 원래 위치를 유지한다. 명시적 CLI `--no-chat-template` 요청에는 신뢰할 conversation 경계가 없으므로 별도 plain layout을 사용한다.

### 4.5 Video prefill과 merge된 MTP wrapper를 결합하기

Main commit `092d3dd0`은 `InklingVlModel.text`를 speculative target으로 사용하고 prepared-embedding의 sequence-aware 및 last-logit forwarding을 추가했다. Video 브랜치는 기존 이미지 embedding 준비 주변에 borrowed image preprocessing과 temporal slot 교체만 추가한다. Merge commit `2cade8d1`은 두 계약과 테스트를 모두 보존한다. 따라서 image/video 요청은 공통 text target으로 이어지기 전에 classic HMLP prepared prefill을 사용하며, video 경로는 merge된 embedding을 우회하거나 다시 정규화하지 않는다.

## 5. 리뷰와 보강

독립 리뷰에서 HIGH 세 건이 보고되었고 feature 브랜치에서 모두 수정했다.

- Prompt 삽입은 검증된 현재 사용자 경계를 대상으로 한다. 회귀 테스트는 system message, 이전 user/model history, companion still-image part, video clip 두 개, 현재 질문, generation tail을 포함한다.
- Pair 할당은 하나의 요청 전체 예산을 적용하면서 clip 경계와 독립 timestamp 원점을 보존한다.
- Video admission은 decode 전에 clip, pair, 선택 frame, decoded byte를 제한한다. Decoder와 processor는 deep clone 없이 compact selected frame만 처리한다.
- 실제 선택한 source index를 사용해 768-frame uniform sampling 상한 이후의 timestamp drift를 수정했다.
- Shape, tile layout, placeholder, FPS, 산술, decoded-frame cardinality 불일치는 HMLP 실행 전에 계속 실패한다.
- 서버 decode는 media admission이 확립한 fd 기반 source handle을 유지해 path reopen race를 피한다.
- Main의 일반 merge 뒤에도 native MTP wrapper dispatch, 정확한 state lifecycle, prepared-image prefill 동작과 video slot-1 준비가 함께 유지된다.

Merge 전에는 독립 재리뷰가 필요하다. 이 보고서는 집중 테스트 통과를 리뷰 승인으로 간주하지 않는다.

## 6. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo test -p mlxcel inkling --lib` | 통과, 57/57 |
| `cargo test -p mlxcel-core drafter::inkling_mtp --lib` | 통과, 7/7 |
| Main merge 전 `cargo test -p mlxcel inkling_ --lib` | 통과, 28/28 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | Main merge 후 통과 |
| `cargo check -p mlxcel --bin mlxcel --bin mlxcel-server` | Main merge 후 통과 |
| `cargo fmt --all -- --check` | Main merge 후 통과 |
| `git diff --check` 및 `git diff --cached --check` | 통과 |

결합된 57-test filter는 text graph, HMLP image 경로, clip-local video planning, 현재 사용자 prompt 삽입, temporal suffix 교체, Inkling VLM MTP adapter, prepared-embedding prefill 보존, target verification, 정확한 KV 및 네 convolution state restore/replay를 포함한다. Core 테스트 7개는 MTP config, detection, shard filtering, sanitization, forward shape, flat snapshot restoration을 추가로 검증한다.

Repository의 OpenXLA feature compile과 더 넓은 platform matrix는 PR CI가 담당한다. 이 호스트에는 end-to-end throughput 및 답변 품질 검증에 필요한 실제 checkpoint artifact와 Apple GPU hardware가 없다.

## 7. 검증 한계와 후속 작업

공개 Inkling-Small affine MLX checkpoint는 약 153.5 GB, native NVFP4 checkpoint는 약 170.7 GB, native MTP shard는 약 4.5 GB다. 검증 호스트에는 이 artifact와 의도한 실제 bouncing-ball fixture가 없었다. 실제 CLI/server video 답변 품질, 움직임 방향 정확도, peak unified memory, Apple GPU throughput, MTP throughput, MTP acceptance length는 주장하지 않는다.

결정론적 테스트는 host 의미론, reference pair index, temporal row 교체, prompt 위치, 제한된 planning, timestamp mapping, wrapper 결합, state restoration을 검증한다. 이후 checkpoint 기반 검증은 같은 짧은 motion clip을 공개 mlx-vlm의 2 fps 결과와 비교하고 확장된 visual-token cardinality를 확인하며, classic multimodal prefill 뒤의 MTP-capable text decode를 별도로 측정해야 한다.

## 참고

- 에픽 #1313, 이슈 #1323, 선행 이슈 #1327
- PR #1546, 선행 PR #1535, merge된 MTP PR #1540
- Main MTP commit `092d3dd0` 및 feature 통합 merge `2cade8d1`
- 공개 mlx-vlm Inkling `inkling.py`, `vision.py`, 이미지 processing, processor, video helper 구현
- `docs/supported-models.md`
