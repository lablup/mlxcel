# 기술 리포트: PR #2023, mlxcel --help 서브커맨드 요약 통일

**작성일**: 2026-09-29
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

## 요약

`src/main.rs`의 `Commands` enum에는 두 가지 문서화 스타일이 공존하고 있었다. 일곱 개 서브커맨드 변형(`Run`, `Inspect`, `Detect`, `Embed`, `Rerank`, `Rm`, `Tune`)은 완전한 doc-comment 템플릿(마침표로 끝나는 요약, 짧은 부연 설명, `Examples:` 블록)을 갖추고 있었지만, 다섯 개(`Generate`, `Serve`, `List`, `Arch`, `Download`)는 마침표도 없고 예제도 없는 한 줄짜리 요약에 머물러 있었다. 이 PR은 다섯 개의 미완성 변형을 동일한 템플릿으로 끌어올려 `mlxcel --help`가 하나의 일관된 화면으로 읽히도록 했다.

## 1. 문제 정의

### 1.1 배경

모든 서브커맨드의 도움말 텍스트는 해당 `Commands` enum 변형의 doc comment에서 나온다. CLI가 성장하면서 동작이 자명하지 않은 일부 변형(`Inspect`의 메모리 추정기, `Tune`의 autotune 캐시 등)은 확장 설명과 실제 동작하는 예제를 포함한 더 완전한 템플릿을 갖추게 되었지만, 가장 자주 쓰이는 다섯 개 동사는 그 대우를 받지 못했다.

### 1.2 기존 문제점

- **문제 1**: `mlxcel --help`의 최상위 `Commands:` 목록에서 마침표로 끝나는 요약과 그렇지 않은 요약이 섞여 있어, 목록을 훑어보는 사람에게 일관성 없는 서식으로 보였다.
- **문제 2**: `mlxcel generate --help`, `mlxcel serve --help`, `mlxcel list --help`, `mlxcel arch --help`는 다른 모든 서브커맨드의 상세 도움말과 달리 실행 예제를 전혀 제공하지 않았다.

## 2. 기술적 검토 사항

### 2.1 코드 품질 관점

- **테스트 커버리지**: 변화 없음. `tests/cli_help_consistency.rs`(27개 테스트)는 공유 플래그 그룹 일관성(TurboQuant KV 캐시, 스펙큘레이티브 디코딩)과 바이너리 간 플래그 표면을 검증하는 것이지 서브커맨드 요약 문구를 검증하지 않으므로, 이번 변경은 그 범위 밖이다. 변경 전후 모두 27개 전부 통과했다.
- **검증**: 새로 추가한 모든 예제 실행은 해당 서브커맨드의 실제 clap `Args` 구조체(`GenerateArgs`와 그 안에 flatten된 `ModelOptions`/`GenerationOptions`/`SamplingOptions`, `ServeArgs`, `ListArgs`, `ArchArgs`)를 대조 확인했고, 빌드한 릴리스 바이너리의 실제 `--help` 출력과 다시 한 번 교차 검증했다. 기억에 의존하거나 다른 서브커맨드의 예제를 그대로 복사하지 않았다.

### 2.2 호환성 및 의존성 관점

- **호환성 파괴 여부**: 없음. doc comment만 변경했으며 `#[arg(...)]` 정의, 기본값, 파싱 동작은 전혀 바뀌지 않았다.
- **신규 의존성**: 없음.

## 3. 기술적 선택과 그 이유

### 3.1 Download: Examples 블록을 중복시키지 않고 요약만 확장

**컨텍스트**: 다른 네 개의 미완성 변형과 달리 `Download`의 `Args` 구조체(`src/downloader/cli.rs`의 `DownloadArgs`)는 이미 자체 `#[command(after_help = "...")]` 블록에 일곱 개의 실제 예제(기본 목적지, 베어 네임 org 확장, `--local-dir`, `--revision`, `--token`, `--force`, `--include`)를 갖고 있었다. 이 블록은 `mlxcel-server download`와도 공유되며, 이번 PR 이전에도 이미 "예제 실행이 최소 하나 이상 있어야 한다"는 기준을 충족하고 있었다.

**고려한 대안**:

| 옵션 | 장점 | 단점 |
|------|------|------|
| 다른 네 변형처럼 enum 변형에도 두 번째 `Examples:` 블록 추가 | 손댄 모든 변형에 `Examples:` 블록이 문자 그대로 존재한다는 일관성 | 같은 `--help` 출력 뒤쪽에 이미 렌더링되는 내용과 중복됨. 거의 동일한 예제 목록이 두 번 나오면 오히려 가독성이 나빠짐 |
| **채택: 마침표와 짧은 부연 설명만 추가, 새 Examples 블록은 추가하지 않음** | 중복이 없고, 기존 `after_help` 블록이 그대로 렌더링되며 새 문단이 그 블록을 가리킴 | 변형 자체의 doc comment에는 예제가 없어 다른 네 변형과 약간의 비대칭이 생김 |

**근거**: 인수 기준은 각 서브커맨드의 상세 도움말이 최소 하나의 예제 실행을 담고 있어야 한다는 것이지, 모든 변형의 doc comment가 문자 그대로 `Examples:` 헤딩을 포함해야 한다는 것이 아니다. `download --help`는 이미 그 기준을 충족하고 있었으므로, 두 번째 블록을 추가하는 것은 일관성을 높이기는커녕 도움말 품질을 떨어뜨렸을 것이다.

### 3.2 리터럴 예제 블록이 없는데도 `Download`에 `verbatim_doc_comment`를 적용한 이유

**컨텍스트**: 결과를 렌더링해보니 최상위 `Commands:` 목록에서 `download   Download a HuggingFace model repository snapshot`처럼 마침표 없이 표시되었다. 소스의 doc comment는 첫 줄을 마침표로 끝냈는데도 그랬다.

**근본 원인**: `#[command(verbatim_doc_comment)]`가 없으면 clap_derive의 비-verbatim 모드가 doc comment를 짧은 `about`과 긴 `long_about`으로 재구성한다. 요약 아래에 두 번째 문단(확장 설명)이 존재하는 순간, 자동으로 생성되는 `about`에서 첫 줄의 마침표가 조용히 제거된다. 이번에 손댄 다른 변형들은 각자의 리터럴 `Examples:` 블록 서식을 위해 이미 `verbatim_doc_comment`를 갖고 있었고, 그 부수 효과로 마침표도 보존되었다. `Download`에는 그런 블록이 없었기 때문에 이 버그가 유일하게 그 변형에서만 드러났다.

**근거**: 요약 줄을 온전하게 유지하기 위한 목적만으로 `Download`에도 `#[command(verbatim_doc_comment)]`를 추가했다. 이는 clap_derive 소스코드를 읽어서가 아니라, 바이너리를 빌드해 실제 렌더링된 `--help` 출력을 확인하는 과정에서 실증적으로 발견한 문제다.

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/main.rs`** (대표로 하나의 변형만 표시; 나머지 세 건의 미완성→완전 전환도 동일한 형태를 따른다)

```rust
// 변경 전
/// List downloaded models in the local store
#[command(visible_alias = "ls")]
List(ListArgs),

// 변경 후
/// List downloaded models in the local store.
///
/// Enumerates models downloaded into the global store (or
/// `--models-dir`), mirroring `ollama list`. The default table shows
/// NAME / SIZE / MODIFIED; the supported model-architecture catalog
/// lives under the separate `mlxcel arch` verb.
///
/// Examples:
///
///     mlxcel list
///     mlxcel list -v
///     mlxcel list --json
///     mlxcel list --sort modified
#[command(verbatim_doc_comment, visible_alias = "ls")]
List(ListArgs),
```

**변경 이유**: `Inspect`, `Detect`, `Embed`, `Rerank`, `Rm`, `Tune`가 이미 쓰고 있는 템플릿과 맞춘 것이다: 마침표로 끝나는 요약, 요약만으로는 자명하지 않은 동작을 설명하는 짧은 문단, 그리고 들여쓰기와 줄바꿈이 `verbatim_doc_comment`로 그대로 보존되는 리터럴 `Examples:` 블록.

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 수 | 1 (`src/main.rs`) |
| 추가된 줄 | +57 |
| 삭제된 줄 | -8 |
| 추가된 테스트 | 0 (기존 `tests/cli_help_consistency.rs`가 이 도움말 표면을 이미 다루고 있어 재실행 후 27/27 통과 확인) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 문서화 | 5개 변형 (`Generate`, `Serve`, `List`, `Arch`, `Download`) | 다른 7개 변형이 이미 쓰던 완전한 doc-comment 템플릿으로 통일 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `33c6df6` | docs | Unify Commands enum subcommand summaries with the full-style template |

## 8. 후속 조치

### 향후 개선 사항

- 이번 PR에서 손대지 않은 `Inspect`의 기존 `Examples:` 블록은 필수 플래그인 `-m`/`--model`을 빠뜨리고 있다. 표시된 예제(`mlxcel inspect models/llama-3.2-1b-instruct-4bit`)를 그대로 실행하면 `--model`이 포지셔널이 아니라 필수 옵션이기 때문에 clap의 "unexpected argument" 오류로 실패한다. 별도의 소규모 후속 수정이 필요하다.
- `src/bin/mlx_server.rs`는 `mlxcel-server` 바이너리의 `download` 서브커맨드용으로 자체 doc comment(`/// Download a HuggingFace model repository snapshot`, 마침표 없음)를 갖고 있는데, 이제 `src/main.rs`에서 새로 정리한 문구와 어긋난다. 이번 이슈는 `src/main.rs`의 `Commands` enum으로 범위가 한정되어 있어 범위 밖으로 남겨두었다. 같은 템플릿을 `mlxcel-server`의 서브커맨드로도 확장하는 후속 작업을 고려할 만하다.
