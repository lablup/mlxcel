# PR #2521: GLiNER2.5 CLI 옵션 노출, token-classification 잔여 수정

**작성일**: 2026-10-21
**상태**: 완료
**위험도**: Low
**언어**: Rust

---

## 요약

이슈 #2383/#2387/#2390/#2391의 token-classification 잔여 4개 항목을 하나의 PR로 구현했다. 헤드라인 항목은 CLI에서 도달할 수 없던 기존 기능이다: `Gliner2_5Model`은 `max_len`과 공백 없는 문자 스크립트용 `char` 워드 스플리터를 지원했지만 `mlxcel extract`가 둘 다 노출하지 않았다. 이제 두 플래그가 로드된 모델까지 연결된다. 나머지 세 항목: Viterbi 디코더가 업스트림의 f64 bias 추가에 맞춰 f64로 누적, Privacy Filter band 마스크가 레이어별 재빌드 대신 청크당 포워드당 1회 빌드, malformed 체크포인트 텐서 shape가 MLX 내부에서 abort하는 대신 텐서 이름을 명명하며 로드 시 에러. 검증 중 발견된 기존 결함 두 개도 함께 수정했다.

---

## 1. 문제 정의

### 1.1 배경

token-classification 서브시스템은 GLiNER2.5 지원(모델 측 `max_len` 검증, `set_word_splitter`)을 출시했지만 CLI 표면이 연결되지 않았다. Item 1은 출시된 기능이 실제 코드 흐름에서 도달 불가능함을 의미한다.

### 1.2 기존 문제점

- **Item 1**: `ExtractOptions`에 `include_confidence`/`include_spans`만 있었고 `--max-len`/`--word-splitter` 플래그가 없었다.
- **Item 2**: Viterbi 디코더가 업스트림이 f64 bias를 추가하는 동안 f32로 합산했다.
- **Item 3**: Privacy Filter band 마스크가 청크당 레이어별로 재빌드되었다.
- **Item 4**: malformed 텐서 shape가 로드 시 에러 대신 MLX 내부에서 abort했다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 출시된 기능이 CLI에서 도달 불가 | Medium (config 수정 없이는 CJK 추출 불가) | 결정론적 |
| malformed 체크포인트가 프로세스를 종료 | Low (사용자 대면 에러 품질) | malformed 체크포인트에서 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

CLEAN (pr-security-checker, CRITICAL/HIGH 0). CLI 입력 검증은 fail-fast: 비정수 `--max-len`은 범위 검사 전에 clap이 거부; 범위 검사(1..=32768)는 기존 에러 텍스트 재사용; `--word-splitter`는 정확히 `whitespace|char`만 수용. `validate_shapes`는 그래프 빌드 전에 `from_weights`에서 첫 번째 문장으로 실행되고 기술적 에러를 반환 (panic 없음, MLX abort 없음). band windows는 내부 가변성 없이 불변 `&[BandWindow]`로 흐름; 유일한 가변 static은 `BAND_MASKS_BUILT` 테스트 카운터 (AtomicUsize). 새 unsafe 코드 없음.

### 2.2 성능 관점

band 마스크 호이스팅은 엄격히 적은 작업: 청크당 레이어별 대신 청크당 포워드당 1회 빌드 (카운터 테스트: 7-쿼리 청크의 40 토큰, 2 레이어에서 6 vs 12). f64 레티스는 레이블 수 크기 배열에서 누산기 워드를 두 배로 늘리나 무시 가능. `validate_shapes`는 로드 시에만 해시 조회 + FFI shape 읽기 1회, 핫 패스 밖.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 새 플래그가 없을 때 기본 동작 불변 (기존 `load_span_extractor` 시그니처 유지, `SpanExtractorLoadOptions::default()`로 위임); 공개 `decode(&[Vec<f32>]) -> Vec<usize>` 시그니처 유지; `tag`/`extract`의 JSON 출력 shape 불변.
- **New Dependencies**: 없음.

### 2.4 코드 품질 관점

- Item-2 픽스처는 진짜 f64를 요구: V = 2^24에서 f32 ULP는 2이므로 f32 경로가 t2 E_X 증분 (2^24 + 1.0, round-half-even)을 2^24로 되돌리고 O-first 배경 경로를 tie해서 엔티티를 닫지 못함; f64는 16777217을 유지하고 닫음.
- Item-4 검증 목록은 그래프가 소비하는 모든 weight 키를 커버; `.scales` 형제 스킵은 UnifiedLinear의 자체 양자화 키잉과 일치 (`format!("{}.scales", prefix)`), 실제 8bit 체크포인트로 검증 (score.scales [33,10], packed score.weight [33,160] U32).
- token_classification 필터 87/0, extract 9/0, openai_privacy_filter 12/0 (CUDA, TF32=0, gpu-lock, 3회 필터링 런 통과); fmt와 clippy -D warnings clean.

---

## 3. 기술적 선택과 그 이유

### 3.1 모델 로드 전 fail-fast CLI 검증

두 플래그는 `resolve_model_source_with_override`와 `initialize_runtime_checked` 이전에 `gliner_load_options()`에서 검증되어, 범위 초과 `--max-len`이 모델 로드 전에 기존 config 에러 텍스트를 낸다.

### 3.2 `.scales` 스킵 키

독립 `score.weight` rank/width 검사는 양자화된 score 평면이 있으면 width 검사를 스킵하고, `.scales` 형제 키잉은 `.weight` 접미사를 제거 (MLX affine은 scales 형제를 projection 접두사에 저장: 키 `score.weight`의 `score.scales`). 기존 검사 `{key}.scales`는 8bit 체크포인트에서 false-positive를 내고 첫 런에서 두 테스트를 실패시켰다; 수정 후 실제 레이아웃으로 검증.

### 3.3 f64 누적, 시그니처 유지

레티스 산술이 모든 emission과 bias add에서 `f64::from`으로 f64 전환; `NEGATIVE_INFINITY`와 `best_of`는 f64; 공개 시그니처와 tie-break 순서 불변.

---

## 4. 구현 상세

**Item 1** — `src/commands/extract.rs` (플래그 + 파싱), `src/token_classification/loader.rs` (`SpanExtractorLoadOptions`, 연결), `src/token_classification/gliner2_5.rs` (config 경로와 공유하는 `check_max_len`, 도달 가능한 `set_max_len`/`set_word_splitter`), `src/commands/extract_tests.rs`.

**Item 2** — `src/token_classification/viterbi.rs`, `viterbi_tests.rs`.

**Items 3, 4** — `src/models/openai_privacy_filter_layers.rs` (포워드당 1회 `band_windows`, `BAND_MASKS_BUILT`), `openai_privacy_filter.rs` (`from_weights` 첫 문장 `validate_shapes`), `openai_privacy_filter_tests.rs`.

**문서** — `docs/token-classification.md`: `mlxcel extract` 옵션 문단에 두 플래그 문서화; GLiNER2.5 Input 문단의 낡은 "has no CLI flag yet" 서술 교체.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| token_classification 필터 (CUDA, TF32=0, gpu-lock, --test-threads=1) | 87 passed / 0 failed |
| extract 필터 (bin 하니스) | 9 passed / 0 failed |
| openai_privacy_filter 필터 | 12 passed / 0 failed |
| Fail-before (item 1) | 테스트 선작성; red 런이 "cannot find function `gliner_load_options`" 컴파일 실패 |
| 실체 체크포인트 회귀 | privacy_filter_8bit와 acceptance_sentence 통과; extract CJK 게이트가 mlx-vlm 픽스처와 일치 |
| --help 플래그 | tag_and_extract_are_discoverable_in_help |
| fmt / clippy -D warnings | clean / clean |
| pr-reviewer | 전 항목 PASS; MEDIUM docs 빈틈 하나 (17f35020으로 수정) |
| pr-security-checker | CLEAN. LOW 하드닝 노트 3개 기록 |

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 11 (코드/테스트 10 + 문서 1) |
| 추가 테스트 | 약 10 (항목별: CLI x4, f64 x1, band-mask 카운터 x1, malformed-shapes x1, help x1, 픽스처 업데이트 포함) |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|---------|
| `62b980e7` | feat | feat(token-classification): expose GLiNER2.5 CLI options, fix leftovers |
| `17f35020` | docs | docs(token-classification): document the GLiNER2.5 extract flags |

---

## 7. 후속 조치

### 향후 개선 사항 (기록된 LOW 노트)

- `validate_shapes`의 양자화 스킵이 존재 검사 전에 단락: `.scales`만 있고 `.weight`가 없는 체크포인트가 검증을 통과하고 UnifiedLinear의 "Weight not found" 에러로 하류에서 잡힘 (panic 없음, 메시지만 덜 구체적). 선택적 하드닝: 스킵 전 존재 검사.
- `--max-len`/`--word-splitter`는 비-GLiNER span extractor에서 조용히 무시됨 (--help에 "GLiNER2.5 only" 문서화됨). 런타임 경고가 플래그가 적용됐다고 믿는 사용자를 막음.
- config `max_len` 경로가 `check_max_len` 재사용 대신 범위 검사를 인라인으로 중복 (에러 텍스트 동일, 동작 위험 없음).

### 모니터링 필요

- 다음 릴리스 주기의 새 플래그를 통한 첫 실제 CJK 추출 런.
