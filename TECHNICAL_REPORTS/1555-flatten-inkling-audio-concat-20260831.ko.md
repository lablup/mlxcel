# 기술 보고서: PR #1555 - Inkling 오디오 청크 연결 평탄화

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 리뷰 중. 구현 및 검증 완료
**언어**: Rust, Python, JSON
**위험 수준**: 중간

---

## 요약

PR #1555는 Inkling audio tower의 left-deep concatenation graph를 제거한다. 평가가 끝난 chunk를 보관했다가 하나의 many-input MLX operation으로 연결하며, chunk가 하나뿐이면 새 graph node 없이 기존 array를 그대로 반환한다. 6,000개 chunk contract test가 허용된 요청 상한을 검증하고, 실제 256-array MLX test가 FFI 경계의 cardinality와 순서를 확인한다.

또한 자기 참조적이던 dMel test oracle을 고정된 NumPy 2.3.2 fixture로 교체한다. Fixture의 waveform과 전체 4 x 80 기대 출력은 mlxcel이나 mlx-vlm을 import하지 않고 production Rust FFT 또는 Slaney helper를 호출하지 않은 채 생성된다. Generator는 정확한 NumPy version, RNG recipe, tolerance, mlx-vlm PR #1767 revision을 기록하므로 수치 변화의 탐지와 재생성이 가능하다.

## 1. 문제 정의

### 1.1 Left-deep tower concatenation

Inkling audio tower는 compute chunk 하나에서 최대 256 frame을 처리하고 각 chunk를 독립적으로 평가한다. 기존 구현은 매 결과를 전체 prefix 뒤에 binary concatenate했다. Chunk가 `n`개이면 `n - 1`개의 concatenate node가 생기고 점점 커지는 prefix가 반복해서 연산에 참여해, 긴 요청에서 graph depth와 반복 data movement 위험이 증가했다.

모델은 요청 하나에서 최대 6,000개의 유효 audio-frame transform을 허용한다. `max_frames_per_chunk`는 1로 설정할 수 있으므로 지원해야 하는 경계는 256개가 아니라 6,000개 chunk다. 따라서 join 지점의 graph depth는 상수여야 하며 전체 허용 범위에서 chunk 순서를 보존해야 한다.

### 1.2 자기 참조적 수치 oracle

이전 reference test는 framing과 Slaney filterbank를 로컬에서 다시 만들었지만 production의 `crate::audio::fft::real_fft_magnitude`를 호출했다. FFT 경로에 결함이 생기면 구현과 기대값 계산에 동시에 반영되어 탐지되지 않을 수 있었다. Production-private Slaney 변환 helper도 재사용해 함께 발생하는 수치 변화를 발견하는 능력이 약했다.

대체 oracle은 Rust test 실행 중 정적이어야 하고 별도의 numerical stack으로 생성해야 한다. 전체 expected tensor를 비교하고 엄격한 tolerance를 고정하며, 미래 maintainer가 정확히 재생성할 수 있도록 provenance를 보존해야 한다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| Core MLX wrapper | Borrowed input의 lifetime을 보장하고 raw-pointer FFI를 한 번 호출하는 `concatenate_many(&[&MlxArray], axis)` 추가 |
| Tower 구성 | 정규화된 chunk를 평가·수집한 뒤 한 번의 many-array concatenate 수행. 단일 chunk no-concat fast path 보존 |
| 오류 처리 | 예상치 못한 null chunk를 조용히 누락하지 않고 명시적으로 거부 |
| 최대 범위 회귀 | 6,000개의 순서 있는 chunk를 production join seam에 전달해 arity 6,000의 단일 호출임을 증명 |
| FFI 회귀 | 실제 one-element MLX array 256개를 연결해 결과 shape와 전체 순서를 검증 |
| 독립 fixture | NumPy 2.3.2 waveform과 log-mel 기대값 320개 전체를 고정 tolerance `2e-6`과 함께 커밋 |
| 재현성 | 독립 generator와 SHA-stable output, RNG recipe, NumPy version, upstream revision 기록 |
| Provenance | 잘못된 mlx-vlm revision을 PR #1767 head `0d6805bb...`로 교정하고 merge commit `67bc41d...` 기록 |

## 3. 기술적 선택과 이유

### 3.1 기존 many-input MLX operation 사용

C++ bridge는 이미 `MlxArray` pointer slice를 받아 한 번의 `mlx::core::concatenate` 호출로 전달한다. Bridge를 확장하거나 Rust에서 balanced tree를 구성하면 최종 graph를 개선하지 않으면서 코드만 늘어난다. 선택한 Rust wrapper는 살아 있는 reference를 받고 FFI 호출 동안에만 raw pointer로 변환하며, unsafe block 바로 옆에 safety invariant를 문서화한다.

Tower는 각 chunk를 수집하기 전에 기존의 `eval(normalized)`를 유지한다. 따라서 원래의 제한된 per-chunk 평가 동작은 보존하면서 커지는 concatenate chain만 제거한다. Chunk 0개는 계속 오류이며, 1개는 그대로 이동하고, 2개 이상은 하나의 최종 operation을 사용한다.

### 3.2 객관적인 seam으로 graph 형태 검증

Wall-clock benchmark는 변동이 크고 graph depth를 직접 증명하지 못한다. Generic `join_chunks` helper는 concatenate operation을 주입한다. 최대 범위 회귀는 호출 횟수와 arity를 기록해 6,000-chunk 요청이 left-deep chain 대신 최종 join 하나를 만든다는 사실을 증명한다.

Pure contract test만으로 native bridge가 큰 pointer slice를 받아들이고 순서를 보존하는지는 증명할 수 없다. 따라서 두 번째 test는 실제 MLX operation을 256개 array로 호출한다. 두 test를 함께 사용하면 전체 6,000-frame tower model을 할당하지 않고도 construction shape와 FFI behavior를 검증한다.

### 3.3 전체 static NumPy oracle 커밋

Generator는 공개 mlx-vlm regression recipe를 따른다. `numpy.random.default_rng(5)`, float32 sample 2,401개, NumPy `rfft`, periodic Hann window, 독립 구현한 Slaney filterbank를 사용한다. Production Rust code나 mlx-vlm을 import하지 않는다. Rust test는 커밋된 JSON만 deserialize하며 Python을 실행하지 않는다.

Test는 모든 320개 값을 비교하기 전에 NumPy version, upstream revision, shape, waveform length, expected tensor 전체 길이, 정확한 `2e-6` tolerance를 검사한다. Rust에 tolerance를 고정했기 때문에 fixture 수정자가 허용 오차를 함께 늘려 oracle을 약화할 수 없다.

## 4. 학습 포인트

### 4.1 Chunk 크기와 chunk 수는 서로 다른 제한이다

`max_frames_per_chunk = 256`은 tower evaluation 하나의 작업량을 제한하지만 chunk 수를 제한하지 않는다. Request ceiling이 6,000 frame이고 유효한 chunk size가 1이면 최대 지원 chunk 수는 6,000이다. Batching code의 test는 chunk-size constant를 request cardinality로 취급하지 말고 admission과 configuration rule에서 두 경계를 각각 도출해야 한다.

### 4.2 독립 fixture에는 독립 계산이 필요하다

Reference function이 test 대상 production primitive를 호출하면 독립적이지 않다. 유효한 numerical oracle은 별도 구현에서 생성하고 static data로 커밋하며 정확한 provenance, shape, length, tolerance 검사로 보호해야 한다. Generator를 fixture 옆에 두면 Rust test dependency에 Python이나 NumPy를 추가하지 않고도 auditability를 확보할 수 있다.

## 5. 리뷰 및 검증

독립 품질 리뷰에는 남은 finding이 없었다. 보안·성능 리뷰에도 CRITICAL, HIGH, MEDIUM finding이 없었다. JSON에만 저장된 tolerance가 expected data와 함께 커질 수 있다는 LOW finding은 Rust test에 `2e-6`을 고정해 해결했다. Finalization에서도 correctness, coverage, provenance, file-size blocker를 발견하지 못했다.

| 게이트 | 결과 |
| --- | --- |
| `cargo test --lib audio::inkling_tower::tests` | 통과, 4/4 |
| `cargo test --lib audio::inkling_dmel::tests` | 통과, 7/7 |
| `cargo check --lib --tests` | 통과 |
| `cargo clippy --lib --tests -- -D warnings` | 통과 |
| `cargo clippy --workspace --all-targets -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` | 통과 |
| Fixture 재생성 | 통과. 동일 SHA-256 `17732d9a76d2a6d92ebf07b05a605cb34f3e1b864fb23115527ae78b63f00c78` |
| 보고서 커밋 전 GitHub CI | 필수 check 전부 성공, 플랫폼 비해당 job은 skip |

## 6. 변경 통계

| 항목 | 값 |
| --- | --- |
| 변경 파일 | 8 |
| 추가 line | 2,970 |
| 삭제 line | 86 |
| 새 회귀 시나리오 | 3개: 6,000-chunk graph contract, 256-array FFI 순서, 독립 full-tensor fixture |
| 보고서 전 구현 commit | `a8baa83c` - `refactor(audio): flatten Inkling chunk concatenation` |

추가 line 대부분은 실행 코드가 아니라 결정론적 JSON fixture data다.

## 7. 검증 한계와 후속 작업

이 PR은 synthetic MLX array와 NumPy oracle로 graph construction, FFI ordering, host dMel 수치를 검증한다. 공개된 153-171 GB Inkling checkpoint, Apple GPU memory use, 실제 transcription 품질, end-to-end throughput은 검증하지 않는다. 이 hardware-dependent 검증은 issue #1549에서 계속 추적한다.

적합한 Apple Silicon host에서 배포 후 long-audio peak memory와 prefill latency를 변경 전후로 비교하고, provenance를 추가 확인하기 위해 fixture waveform을 공개 mlx-vlm 구현에서도 실행해야 한다. Issue #1550을 위해 필요한 추가 코드 변경은 없다.

## 참고

- Epic #1313, issue #1550, prerequisite PR #1548
- Blaizzy/mlx-vlm PR #1767, head `0d6805bb7ef67998d8aeb655bc1df83854830d56`
- Fixture generator `tests/fixtures/generate_inkling_dmel_numpy.py`
