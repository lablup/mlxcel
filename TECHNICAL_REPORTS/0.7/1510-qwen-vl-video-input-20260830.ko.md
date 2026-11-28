# 기술 보고서: PR #1510 - qwen-vl-video-input

**작성일**: 2026-08-30

**상태**: 런타임 검증 한계가 있는 구현 완료

**언어**: Rust, Markdown

**위험도**: Medium

## 요약

PR #1510은 Qwen-VL 계열의 video input을 CLI, OpenAI-compatible server, Qwen processor, placeholder expansion, MRoPE position construction, visual embedding merge 경로에 연결한다. 구현은 fail-closed 원칙을 따른다. decoded media, rendered placeholder, visual grid, projected feature row, final token position이 정확히 맞아야 generation을 시작한다.

이 변경은 focused Rust tests, formatting, diff checks, non-workspace clippy, 그리고 report commit 이전에 관찰한 hosted static CI로 검증했다. 다만 이 worktree에는 `ffmpeg`/`ffprobe`와 local checkpoint가 없었으므로 실제 ffmpeg-backed video decoding 및 Qwen3.5/Qwen3.8 checkpoint generation은 실행하지 않았다.

## 1. 문제 정의

### 1.1 배경

Issue #1166은 Qwen3.8 qualification 작업에서 분리되었다. 해당 model family는 config와 chat template에 video contract를 이미 제공하지만, mlxcel은 Qwen-VL의 `--video` 요청을 거부하고 있었다. 기존 구현은 image/video token ID를 load 단계까지 전달했지만 runtime path에는 image preprocessing, image placeholder expansion, image-oriented Qwen MRoPE만 연결되어 있었다.

### 1.2 기존 문제점

- **CLI/server rejection**: Qwen-VL model은 checkpoint가 `video_token_id`를 선언하고 `<|video_pad|>` placeholder를 렌더링해도 video-capable embedding path로 라우팅되지 않았다.
- **Cardinality contract 부재**: 기존 Qwen path는 image block만 세었고, declared video, decoded frame, placeholder span, projected feature row, final embedding position이 일치하는지 엄격히 확인하지 않았다.
- **MRoPE drift risk**: 여러 Qwen wrapper가 중복된 visual-position builder를 가지고 있어 image/video semantics가 family별로 어긋날 위험이 있었다.
- **Silent media loss risk**: fail-closed check가 없으면 malformed mixed media prompt에서 video가 조용히 누락되거나 frame count가 clamp되거나 visual feature가 잘못된 token span에 들어갈 수 있었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|------|--------|-------------|
| Model이 지원하는데도 Qwen video request가 계속 실패함 | High | High |
| Placeholder와 feature count mismatch가 generation input을 오염시킴 | High | Medium |
| Unbounded 또는 silently clamped video sampling이 request failure를 숨김 | Medium | Medium |
| 실제 checkpoint behavior가 code-path test와 다름 | Medium | ffmpeg/checkpoint qualification 전까지 Medium |

## 2. 변경 요약

| 카테고리 | 변경 수 | 주요 내용 |
|----------|---------|-----------|
| CLI/server integration | 4 areas | CLI `--video`, server video request preparation, media-support detection, help/error text에 Qwen-VL variants를 추가했다. |
| Qwen processor/runtime | 5 areas | Processor sidecar parsing, mixed image/video patch preprocessing, strict media-token expansion, shared MRoPE, image/video embedding merge를 연결했다. |
| Tests | 7 focused groups | Qwen token ID, sidecar bounds, video padding, frame policy, media order, placeholder cardinality, CLI/server capability tests를 추가하거나 갱신했다. |
| Documentation | 1 file | `docs/supported-models.md`가 Qwen-VL video support를 설명하고 code-path support와 real video checkpoint qualification을 분리해서 적는다. |

### 통계

| 항목 | 값 |
|------|----|
| 변경된 파일 수 | 24 |
| 추가된 라인 | +1897 |
| 삭제된 라인 | -595 |
| 주요 커밋 | `78177cfb` `feat(models): add Qwen-VL video input` |

## 3. 기술적 선택과 그 이유

### 3.1 Family별 fork 대신 하나의 mixed-media Qwen path 사용

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL MoE, Qwen3.5-VL, Qwen3.5 MoE는 이 변경에 필요한 video contract를 공유한다. 즉 Qwen visual processor, `video_token_id`, visual grid metadata, temporal/spatial axis를 갖는 MRoPE가 공통이다. PR #1510은 placeholder expansion과 MRoPE construction을 `src/multimodal/qwen_vl.rs`에 집중시키고 각 family wrapper가 그 helper를 호출하게 했다.

이 선택의 trade-off는 shared helper가 일부 ad hoc prompt shape에 대해 더 엄격하다는 점이다. 이는 의도된 동작이다. Prompt/media ordering 문제는 visual embedding이 밀린 generation보다 invalid request로 실패하는 편이 안전하다.

### 3.2 Over-limit video sampling은 clamp가 아니라 error

Video loader는 이제 `FrameSamplingPolicy`를 받고 Qwen processor는 `processor_config.json`에서 나온 policy를 전달한다. Qwen policy에서 `max_frames`를 넘는 요청은 silently truncating/clamping하지 않고 named `SampledFramesTooMany` error를 반환한다.

이 방식은 checkpoint contract를 보존하고 validation failure를 사용자가 볼 수 있게 한다. 또한 runtime이 frame을 버렸는데 사용자는 model이 전체 clip을 봤다고 믿는 상황을 막는다.

### 3.3 Mixed media는 rendered prompt order 보존

Runtime은 placeholder가 존재할 때 image-before-video를 가정하지 않고 rendered token ID를 스캔해서 prompt가 기대하는 media order를 얻는다. Placeholder가 전혀 없을 때만 기존 insertion order인 images followed by videos로 fallback한다.

Scanner는 framing token 없이 서로 붙은 cross-kind visual run을 거부한다. Production Qwen MRoPE scanner가 이런 연속 visual span을 하나의 media block인지 두 개의 media block인지 구분할 수 없기 때문이다.

### 3.4 Video가 있으면 Qwen vision cache 비활성화

Image-only Qwen request는 기존 request-scoped vision cache behavior를 유지한다. Video request는 decoded frame sequence가 request-specific이고 크기가 클 수 있으므로 cache를 우회한다. Image key 아래에 video frame embedding을 저장하면 stale media reuse 또는 memory pressure 위험이 생긴다.

## 4. 구현 상세

### 4.1 End-to-end data flow

```text
CLI/server request
  -> render image/video content parts
  -> decode videos through ffmpeg-backed loader with Qwen frame policy
  -> preprocess mixed Qwen media into patch rows and visual grids
  -> expand or validate image/video token placeholders exactly
  -> compute shared Qwen MRoPE positions with T kept separate from H/W merge
  -> merge visual embeddings into image and video token positions
  -> generate with expanded prompt tokens
```

### 4.2 주요 파일

- `src/multimodal/video.rs`는 `FrameSamplingPolicy`, fail-closed over-max behavior, policy-aware video loading entry points를 추가한다.
- `src/vision/processors/qwen2_vl.rs`는 Qwen video sidecar parsing, mixed image/video preprocessing, temporal padding, video grid generation을 추가한다.
- `src/multimodal/qwen_vl.rs`는 strict mixed-media placeholder expansion과 shared Qwen image/video MRoPE position construction을 추가한다.
- `src/multimodal/vlm_runtime.rs`는 rendered prompt에서 mixed media order를 계산하고 Qwen media preprocessing을 embedding generation에 연결하며 Qwen preparation summary에 video count를 추가한다.
- `src/commands/generate_vlm.rs`와 `src/server/model_worker.rs`는 Qwen-VL video request를 새 shared helper로 라우팅한다.
- `src/vision/qwen*_vl*.rs` wrapper는 shared MRoPE helper를 사용하고 image/video token ID 모두에 visual embedding을 merge한다.
- `docs/supported-models.md`는 supported code path를 문서화하고, real ffmpeg-backed Qwen3.8 video Q&A는 정확한 run이 인용되기 전까지 별도 runtime qualification이라고 명시한다.

## 5. 기술적 검토 사항

### 5.1 Correctness

가장 중요한 correctness boundary는 exact visual cardinality다. 이 구현은 generation 전에 placeholder count, media ordering, expanded run length, grid divisibility by `spatial_merge_size`, positive grid dimension, checked token-count multiplication, processor frame bounds, temporal padding, adjacent visual-run ambiguity를 확인한다.

Finalizer review 중 lower-level insertion test는 있었지만 rendered-token scanner 자체를 직접 테스트하지 않는 gap을 발견했다. 그래서 runtime prompt-order scanner test를 추가했다.

### 5.2 Security

User input으로 shell string을 새로 구성하지 않았다. Video decoding은 기존 `ffmpeg` pipeline을 계속 사용하고, 새 Qwen entry point는 structured path와 policy value를 기존 loader API에 전달한다. 검토 결과 credential handling, authentication, SQL, web rendering, sensitive logging 변경은 없었다.

User-facing file-path exposure는 기존 media loader behavior와 같다. 실패한 local video load를 진단하기 위해 error에 canonical media source description이 포함된다. 이는 local/server operator surface에서는 적절하지만, server가 untrusted tenant에게 노출되는 배포에서는 일반적인 threat review 범위에 계속 포함해야 한다.

### 5.3 Performance

이 patch는 Qwen frame cap을 preprocessing 전에 강제하여 silent over-sampling을 피한다. 또한 video frame embedding을 image cache key 아래에 저장하지 않는다. 주 비용은 예상 가능한 비용이다. Qwen video preprocessing은 sampled frame마다 resize/normalize를 수행하고 request의 모든 media에 대해 하나의 patch tensor를 만든다.

Benchmark는 실행하지 않았다. 구현은 visual token count에 대해 bounded allocation check를 추가하고 existing per-request video limits를 사용해 decode size와 duration을 제어한다.

## 6. 학습 포인트

### 6.1 Qwen-VL MRoPE video semantics

Qwen-VL video에서 temporal grid size는 실제 axis로 유지된다. `spatial_merge_size`는 H/W axis에만 적용되고 T는 frame-group position으로 확장된다. Placeholder count가 맞더라도 T를 보존하지 않고 video를 flat image-token run처럼 처리하면 long clip의 position이 틀어진다.

### 6.2 Code-path support는 runtime qualification이 아니다

이 구현은 Rust integration contract가 compile되고 focused unit tests를 통과한다는 것을 보인다. 하지만 Qwen3.8 checkpoint가 video 질문에 올바르게 답한다는 것을 증명하지는 않는다. 그것은 ffmpeg, local checkpoint, 적절한 hardware에서의 실제 generation run이 필요하다.

## 7. 검증 기록

| Check | 결과 | 비고 |
|-------|------|------|
| `cargo fmt --all -- --check` | Pass | Rebase 후와 final test 추가 후 실행. |
| `git diff --check origin/main..HEAD` | Pass | 24-file implementation diff에 whitespace error 없음. |
| `cargo test --lib qwen_vl` | Pass | 13 passed, 11 ignored serial-MLX tests, 7276 filtered. |
| `cargo test --lib video_processor_config` | Pass | 2 passed. |
| `cargo test --lib preprocess_video_pads_frames_to_temporal_patch_grid` | Pass | 1 passed. |
| `cargo test --lib smart_nframes_policy` | Pass | 2 passed. |
| `cargo test --lib detect_model_media_support_recognises_qwen35_vlm_video` | Pass | 1 passed. |
| `cargo test --lib qwen35_vl_token_ids` | Pass | 8 passed. |
| `cargo test --lib qwen_media_order_from_prompt` | Pass | 2 passed. |
| `cargo test cli_video_content_part_count_enables_qwen_vl_videos` | Pass | Named `src/main.rs` test passed; filter 때문에 다른 enumerated target은 0 matching tests. |
| `cargo clippy --lib --tests -- -D warnings` | Pass | Non-workspace clippy, 1m29s. |
| `cargo check --lib --features xla-iree` | Environment skip | Local escalated run은 `mlxcel-xla` build script까지 도달했지만 `IREE_DIST`가 없어 중단. Report commit 전 hosted `OpenXLA feature compile`은 pass. |
| Hosted checks before report commit | Partial pass, one pending | Static setup, `cargo-fmt`, `cargo-deny`, `OpenXLA feature compile`은 pass. Hosted `cargo-clippy`는 report workflow를 진행할 때 아직 pending이었다. Report commit이 새 run을 트리거하기 때문에 기다리지 않았다. |
| `command -v ffmpeg` / `command -v ffprobe` | Unavailable | Real decode validation은 실행할 수 없었다. |
| `nvidia-smi` outside sandbox | Available | NVIDIA GB10, driver 580.173.02, CUDA 13.0 확인. |
| Local checkpoint lookup | Unavailable | Worktree에 `models/` 또는 `models/mlx`가 없어 real Qwen3.5/Qwen3.8 generation run은 수행하지 않았다. |

## 8. 후속 조치

### Real model qualification을 주장하기 전에 필요한 작업

- [ ] Validation host에 `ffmpeg`와 `ffprobe`를 설치하고 CLI/server path에서 real Qwen-VL video decode를 실행한다.
- [ ] Project model directory에 대상 Qwen3.5/Qwen3.8 checkpoint를 배치하고 fixed video prompt로 answer quality, placeholder expansion, prompt token accounting, MRoPE position을 확인한다.
- [ ] Concrete checkpoint와 media fixture로 동일한 mixed image/video request에 대해 CLI와 server prompt rendering을 비교한다.

### Merge 후 모니터링

- Qwen video request가 frame-bound error로 실패하는 사용자 보고를 관찰한다. Clip이 processor sidecar limit을 넘을 때 이는 의도된 fail-closed diagnostic이다.
- Video preprocessing은 sampled frame과 patch row를 request마다 materialize하므로 long clip에서 server memory usage를 관찰한다.

## 부록

### 관련 issue 및 PR

- Issue #1166: Qwen-VL video input support.
- PR #1510: 이 보고서에 설명한 code path와 tests를 구현한다.
