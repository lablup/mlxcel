# 기술 보고서: PR #1620 - fix(server): resolve absolute file:// URLs under --media-path

**작성일**: 2026-09-04
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (두 빌드로 라이브 서버에 A/B 검증, URL 표기 여섯 가지)
**언어**: Rust
**위험도**: Low-Medium (격리 경계를 건드리지만, 기존 거절을 수용이나 더 정확한 거절로 바꿀 뿐이고 격리 경로를 새로 만들지 않음)

---

## 요약

`--media-path`는 루트 기준 상대 경로로 해석되는 참조에 대해서만 `file://` 읽기를 허용했다. 그래서 다른 도구들이 다 받아 주는 절대 경로 형식만 유일하게 통하지 않았다. `--media-path /srv/media` 아래에서 `file:///srv/media/cat.png`는 `<root>/srv/media/cat.png`를 뒤지다가 `file does not exist or cannot be opened`로 거절됐다. 정작 그 파일은 존재하고 읽을 수도 있는데 말이다. 실제로 뒤진 경로는 응답에 실리지 않으니, 서버 밖에서는 원인을 짚을 방법이 없었다.

PR #1620은 b10621의 문자열 이어붙이기를 1차 해석으로 그대로 두고, 절대 경로 폴백을 얹어 같은 격리 검사를 통과시킨다. 이제 루트 안을 가리키는 절대 경로는 상대 표기와 똑같은 파일로 해석되고, 루트 밖을 가리키면 없는 파일이 아니라 이탈로 거절된다.

---

## 1. 문제 정의

### 1.1 배경

미디어 루트는 로컬 읽기를 가두려고 #1451에서 도입했다. llama-server b10621의 `handle_media`를 그대로 옮긴 것인데, 그쪽은 `media_path + file_path`를 계산한다. join이 아니라 순수 문자열 이어붙이기다. 이 구분이 실제로 무게를 진다. 절대 경로 조각으로 Rust `Path::join`을 하면 루트를 통째로 버리게 되고, 호환 기능이 임의 파일 읽기로 바뀐다. 그래서 `relative_component`가 앞쪽 구분자를 떼고 나서 join한다. 이 동작은 `an_absolute_looking_path_is_concatenated_not_joined`가 고정하고 있다.

### 1.2 무엇이 어긋났나

이어붙이기도 맞고 격리도 맞다. 문제는 그 규칙을 아무 데서도 말하지 않았다는 점이다. `--help`에는 "Directory that local `file://` media URLs are resolved against"뿐이었고, 호환 문서에는 "the request's path is resolved against the configured directory"라고만 적혀 있었다. 절대 경로가 대체되는 게 아니라 덧붙는다는 말은 어느 쪽에도 없다. 둘 중 무엇을 읽든 운영자는 절대 경로 형식을 쓸 테고, 존재하는 파일을 두고 "없다"는 오류를 받고서, 왜 그런지 짐작할 단서를 얻지 못한다.

저장소 자신의 벤치마크 하네스가 이미 여기에 걸렸다. `scripts/bench_embeddings.py`가 절대 경로인 `IMAGE`로 `f"file://{IMAGE}"`를 만들었고, 2026-09-04 임베딩 패스의 이미지 셀이 전부 HTTP 400을 받았다.

### 1.3 위험도 판단

여기를 손대는 일은 격리 경계를 손대는 일이다. 그래서 설계 제약을 하나 걸었다. 폴백이 두 번째 격리 경로를 만들면 안 된다. 새 검사를 추가하는 대신 기존 `canonical.starts_with(root)` 검사를 그대로 재사용했고, `Path::is_absolute`로만 진입하도록 좁혀서 상대 참조가 서버 작업 디렉터리 기준으로 canonicalize되는 일이 아예 없게 했다.

---

## 2. 기술 검토

### 2.1 근본 원인

`resolve_media_file_in`에는 해석 전략이 하나뿐이었다.

```rust
let relative = relative_component(raw);       // 앞의 '/'와 '\'를 뗀다
let canonical = tokio::fs::canonicalize(root.join(relative)).await?;
if !canonical.starts_with(root) { return Err(Escape); }
```

절대 참조가 들어오면 `relative_component`가 앞 구분자를 떼고, 남은 부분이 루트 뒤에 붙어서, 존재하지 않는 자리를 뒤진다. 우연한 실패가 아니라 구조적 실패다. 절대 참조가 성공할 수 있는 경로 자체가 없었다.

### 2.2 고칠 자리

`media_root::resolve_media_file`의 호출자는 `src/server/media.rs`의 `read_confined_bytes_with_limit` 하나뿐이고, 그 함수는 `try_read_image_url_with_limits`에서 불린다. 이 한 함수가 임베딩 라우트, rerank 라우트, 그리고 chat과 responses와 Anthropic 이미지 파트를 모두 담당한다. 폴백을 공유 리졸버 안에 넣으면 살아 있는 핸들러 전부에 한 번에 닿는다. 이슈의 마지막 합격 기준이 요구하는 바이고, 별도 헬퍼로 만들었다면 얻지 못했을 성질이다.

---

## 3. 기술적 결정

### 3.1 대체가 아니라 폴백

이어붙이기가 1차로 남는다. 폴백은 이어붙인 후보가 canonicalize에 실패하고 참조가 절대 경로일 때만 돈다. 이 순서 덕분에 전에 해석되던 것이 지금 다르게 해석되는 일이 없고, 기존 격리 테스트가 전부 그대로 성립하며, `<root>/<abs>`와 `<abs>` 양쪽에 파일이 있는 경우에도 b10621의 답을 유지한다.

### 3.2 루트 밖 절대 경로는 Unresolvable이 아니라 Escape

루트 밖으로 canonicalize되는 절대 경로는 이제 `MediaPathError::Escape`다. 관측 가능한 변화이고, 의도한 변화다. 업스트림은 루트 안이든 밖이든 모든 절대 표기에 `file does not exist or cannot be opened`로 답한다. 파일을 여는 일은 여전히 없으니 보안 성질은 그대로지만, 클라이언트가 두 경우를 구분할 수 있게 됐다.

`an_absolute_looking_path_is_concatenated_not_joined`의 `file:///etc/passwd` 기대값도 그에 맞춰 바꿨다. 이어붙이기가 여전히 1차임을 증명하는 절반, 즉 `file:///ok.png`가 `<root>/ok.png`로 해석된다는 단언은 손대지 않았다. 임의 파일 읽기를 막아 주는 단언이 그쪽이기 때문이다.

### 3.3 이탈 사항은 산문이 아니라 검사되는 필드에 적는다

관측 가능한 두 변화를 모두 `compat/llama-server/b10621/multimodal-and-audio.json`의 `--media-path` 항목 `divergence` 배열에 넣었다. `scripts/ci/check_llama_compat_manifest.py`가 검증하는 필드이고, 자유 서술인 `notes`가 아니다. 항목의 `state`는 `aliased`로 두어, 비어 있지 않은 `divergence`는 `supported`를 금지한다는 검사기 규칙을 만족한다.

이게 중요한 이유는 대안이 이미 한 번 실패했기 때문이다. b10621 호환 작업에서, state 필드는 `supported`라고 말하면서 이탈 사항은 자유 서술 노트에만 적어 둔 결과, 처음 검토한 59건 중 37건이 거짓 주장이었다. 사람만 읽을 수 있는 이탈 기록은 기록이 아니다.

### 3.4 루트를 흘리지 않으면서 스스로 설명하는 거절

`MediaPathError::Unresolvable`은 이제 경로를 무엇에 상대적으로 푸는지, 곧 `--media-path`를 지목하는 절을 뒤에 달고 나온다. 실제로 뒤진 후보는 `debug` 수준으로 서버 로그에만 가고 응답에는 절대 실리지 않는다. 인증 없는 호출자에게 설정된 루트를 그대로 알려 주는 문자열이기 때문이다.

### 3.5 경계 하나는 바꾸지 않고 고정만 했다

`validate_media_filename`은 b10621의 255바이트 상한을 경로 전체에 적용하고 두 해석 전략보다 먼저 돈다. 그래서 255바이트를 넘는 절대 경로는 `NotAllowed`로 거절되고 폴백까지 가지 않는다. 업스트림에 충실한 동작이라 그대로 두되, 다음에 읽는 사람이 이걸 버그로 착각하지 않도록 문서에 적었다.

---

## 4. 검증

`models/mlx/qwen2.5-vl-3b-instruct`를 `/v1/chat/completions`에, `--media-path <repo>/tests/fixtures`로 띄우고, `main` 빌드와 브랜치 빌드 두 개로 비교했다. 이미지 파트가 빠지면 chat 라우트가 400을 주므로 상태 코드만으로 수용과 거절이 갈린다.

| URL | 변경 전 | 변경 후 |
|---|---|---|
| `file://<repo>/tests/fixtures/test_image.png` (절대, 루트 안) | 400, `file does not exist or cannot be opened` | **200**, 이미지 소비됨 |
| `file://test_image.png` | 200 | 200 |
| `test_image.png` | 200 | 200 |
| `file:///test_image.png` | 200 | 200 |
| `file://<scratch>/outside/outside_image.png` (절대, 루트 밖) | 400, `file does not exist or cannot be opened` | 400, `file path escapes the --media-path root` |
| `file:///etc/passwd` | 400, `file does not exist or cannot be opened` | 400, `file path escapes the --media-path root` |
| `file://absent.png` | 400, `...: absent.png` | 400, `...: absent.png (paths are resolved relative to the --media-path root)` |

마지막 행에서는 `debug` 수준으로 로그에만 `local media reference did not resolve under the --media-path root reference="absent.png" probed=<repo>/tests/fixtures/absent.png`가 함께 나왔다.

게이트: `cargo test --workspace --profile test-fast --features metal,accelerate` 통과. `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_llama_compat_manifest.py` 모두 깨끗하다. PR CI도 `llama-compat manifest`를 포함해 전 항목 통과했다.

---

## 5. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 5 |
| 추가 줄 | 235 |
| 삭제 줄 | 21 |

### 분류별 변경

- `src/server/media_root.rs`: 절대 경로 폴백, 뒤진 후보를 로그로 보내고 오류에서는 빼는 비공개 `unresolvable()` 헬퍼, 늘어난 메시지, 모듈 문서.
- `src/server/media_root_tests.rs`: 새 케이스 다섯, 바뀐 기대값 하나, 메시지 단언 둘.
- `src/cli/multimodal_compat_args.rs`와 `docs/llama-server-compat.md`: 허용되는 절대 형식 하나와 상대 형식 하나를 곁들인 해석 규칙.
- `compat/llama-server/b10621/multimodal-and-audio.json`: 검사되는 이탈 사항 둘.

### 따로 반영한 부분

`docs/benchmark_results/embeddings-rerank-m5max-2026-09-04.md`는 `bench/0.7.0-refresh`(PR #1617)에 커밋 `c3a999b1e`로 갱신했다. 벤치마크 결과는 코드 PR이 아니라 그 브랜치에 속하기 때문이다. 해당 문서의 발견 사항은 그 패스가 측정한 시점 기준의 과거형으로 고쳐 쓰고, 그 뒤에 무엇이 달라졌는지를 이어 적었다.

### 관련 이슈

Closes #1612. 격리를 도입한 #1451의 후속이다.

---

## 6. 후속 작업

### 남은 잔여물

`scripts/bench_embeddings.py`의 `_image_data_uri()` 독스트링은 아직 #1612 이전 동작을 설명하고 있고, 같은 파일의 서버 명령줄에 달린 `--media-path` 주석도 마찬가지다. 둘 다 하네스가 왜 data URI를 보내는지에 대해서는 여전히 맞다. 서버 플래그가 필요 없어서 `--media-path`를 설정한 적 없는 호스트에서도 사다리를 재현할 수 있다는 이유다. 반면 `file://` URL이 어떤 형태까지 가능한지에 대해서는 이제 낡았다. 측정값에는 영향이 없다.

### 옮겨갈 만한 교훈

의도적이고, 테스트로 고정돼 있고, 문서에 없는 호환 동작은 경계에서 버그와 구별되지 않는다. 여기의 이어붙이기가 정확히 그 세 조건을 다 갖췄다. 비용을 치른 건 그 동작 자체가 아니라 아무도 그걸 말해 두지 않았다는 사실이다. 프로젝트 자신의 벤치마크 하네스가 하필 통하지 않는 그 형식을 보냈고, 오류 메시지는 읽을 수 있는 파일을 없다고 불렀다. 테스트로 고정할 만큼 무게가 있는 규칙이면 `--help`에 적을 만큼도 무게가 있다.
