# 기술 보고서: PR #1269 - YAML bias 블록에서 결정적인 언어 우선순위

## 요약

`LangBiasSet.ordered`는 "pairs in priority order (index 0 = highest priority)"로 문서화돼 있고, 소비자 `to_token_bias`는 공유 토큰을 first-language-wins로 해소한다. 그런데 YAML 설정 경로가 그 순서를 `HashMap` 순회로 만들고 있었다. 즉 우선순위를 `RandomState`가 정했다. Han 문자는 `ja`, `zh`, `ko`가 공유하므로, CJK 언어를 둘 이상 적은 `--lang-bias-config` 파일은 공유 토큰마다 실행할 때마다 다른 바이어스를 배정했다. 조용히.

#1265와 같은 근본 원인 계열(`HashMap` 순회 순서가 순서 의존 상태로 새는 것)이지만, 테스트 픽스처가 아니라 프로덕션 코드다. 수정은 `bias:` 블록을 `MapAccess`로 훑어 순서 있는 `Vec`에 모으는 것이고, 그러면 작성자가 파일에 쓴 순서가 곧 우선순위가 된다. `--lang-bias`와 `LLAMA_ARG_LANG_BIAS` 폴백이 처음부터 해오던 것과 같아진다.

## 1. 문제

`LangBiasSet`을 만드는 진입점이 셋인데 틀린 것은 하나뿐이었다.

`parse_lang_bias_entries`는 `s.split(',')`를 문서 순서로 훑고, `seen` 맵을 멤버십 집합으로만 쓰며, 중복을 `CliError::DuplicateLanguageCode`로 거부한다. `env_fallback_lang_bias`는 `LLAMA_ARG_LANG_BIAS`를 같은 파서로 보낸다. YAML 경로만 `bias:`를 `Option<HashMap<String, BiasValueStr>>`로 역직렬화하고 맵의 순회 순서를 그대로 `ordered`에 밀어 넣었다.

결과가 둘인데 두 번째는 보이지도 않았다. 우선순위가 실행마다 무작위가 됐다. 그리고 `serde_yaml`이 반복된 키를 타입 지정 `HashMap`에 last-wins로 넣고 아무 진단도 내지 않기 때문에, resolve 루프의 중복 검사가 도달 불가가 됐다. YAML 경로가 `--lang-bias` 파서라면 계속 거부해 온 입력을 조용히 받아들이고 있었다는 뜻이다.

트리거는 이국적이지 않다. `LangBiasYamlConfig` 문서 주석의 스키마 예시 자체가 CJK 3종 설정이라, 문서화된 예시를 복사하는 것으로 충분했다.

## 2. 기술적 판단

### 2.1 순서 보존 맵 타입이 아니라 손으로 쓴 `Deserialize`

후보 셋을 놓고 골랐다. 필드 타입을 `IndexMap`으로 하면 순서는 지키지만 중복은 last-wins라 죽은 검사가 계속 죽어 있고 YAML 경로는 CLI가 거부하는 입력을 계속 받는다. `serde_yaml::Mapping`은 `IndexMap` 기반이라 중복을 거부하긴 하는데 `CliError::DuplicateLanguageCode`가 아니라 serde_yaml 자체 오류를 내므로 두 진입점이 내는 오류가 여전히 어긋난다.

택한 형태는 `Vec<(String, BiasValueStr)>` 위의 `BiasEntries` 뉴타입이고, `MapAccess::next_entry`로 모으는 `Deserialize`를 손으로 썼다. 문서 순서가 살아남고, 반복된 키는 두 번째 항목으로 도착해 기존 resolve 루프가 저장소 자체 오류로 거부하며, 새 의존성이 없다. `indexmap`은 `Cargo.lock`에는 있지만 `Cargo.toml`에는 없어서, 맵 타입 후보 둘은 전이 의존성을 직접 의존성으로 승격시켰을 것이다.

### 2.2 허용되는 YAML 스키마는 바뀌지 않는다

visitor가 `visit_map`을 구현하고 진입점이 `deserialize_map`이라 `bias:`는 여전히 평범한 매핑이다. `#[serde(deny_unknown_fields)]`는 그대로고, 블록이 없거나 비어 있으면 여전히 빈 집합으로 풀리며, 시퀀스 모양 블록은 여전히 파싱 오류다. 기존 설정 파일은 계속 동작한다. 사용자에게 `bias:`를 리스트로 다시 쓰라고 요구하는 수정이었다면 순서에 관한 버그 리포트에 대한 답으로는 형태가 틀렸을 것이다.

### 2.3 중복 오류 메시지가 두 표면을 모두 지목한다

YAML에서 검사가 도달 가능해지자, `--lang-bias`만 지목하던 기존 문구가 YAML 파일을 쓴 사용자에게는 적극적으로 틀린 말이 됐다. variant와 `code` 필드는 그대로 두고 메시지만 두 표면을 다 언급하도록 넓혔다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/lang_bias.rs` | `BiasEntries` 뉴타입과 `Deserialize`, `LangBiasYamlConfig::bias` 타입 변경, 중복 오류 메시지 확장, 신규 테스트 |
| `CHANGELOG.md` | 사용자 가시 동작 변경 2건에 대한 `## [Unreleased]` 항목 |

## 4. 리뷰 지적사항

가장 무겁게 잡은 요건은 리뷰 지적이 아니라 전제조건이었다. 회귀 테스트가 수정 전 코드에서 실제로 실패하는 것을 증명해야 했다. 이 버그 계열은 수정 전후로 다 통과하는 테스트를 만들어내는 데 특히 능하다. `resolve()` 한 번은 상당한 확률로 우연히 맞는 순서를 낸다.

증명은 필드 타입만 되돌리고(그리고 순서 있는 타입에서 컴파일되지 않는 기존 테스트 하나의 멤버십 단언만), 스위트를 돌린 뒤 `git stash`가 아니라 복사본에서 복원하는 방식으로 했다. 미추적 작업이 위험해지지 않는다. 테스트 4건이 실패했고, 출력에 같은 3키 파일의 서로 다른 순열이 **한 프로세스 안에서 세 가지** 나왔다.

```
iteration 0: [(Zh, -10.0), (Ko, 5.0), (Ja, -inf)]
iteration 2: [(Zh, -10.0), (Ja, -inf), (Ko, 5.0)]
           : [(Ja, -inf), (Zh, -10.0), (Ko, 5.0)]
```

#1267을 발행하며 잰 측정을 독립적으로 재현한 결과다. `RandomState`는 프로세스가 아니라 **`HashMap` 인스턴스마다** 무작위화한다(같은 다섯 키로 만든 맵 10개가 한 프로세스에서 고유 순서 9개). 각 순서 테스트가 `resolve()`를 1회가 아니라 32회 도는 이유도 그것이다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64)에서 실측.

- `cargo test --profile test-fast --features cuda --lib lang_bias`: 30 통과, exit 0. 수정 전 필드 타입 대비: 26 통과, 4 실패, exit 101.
- `cargo fmt --all -- --check`, `cargo clippy --lib --tests --features cuda -- -D warnings`, `cargo check --lib --tests --features cuda`, `cargo check --bins --features cuda`: 전부 exit 0.
- `make verify-test-cuda`: PR 스레드에 기록.

## 6. 관련 작업

- #1267: 이 PR이 닫는 이슈. PR #1268 리뷰 스윕에서 나왔다.
- #1265, PR #1266: 테스트 픽스처 네 곳의 같은 근본 원인이자 스윕의 출발점.
- #1277, #1276: 같은 스윕이 찾은 추가 인스턴스 둘. 분산 레지스트리 접근자와 RT-DETRv2 체크포인트 레이아웃 판별.

한 번의 스윕에서 같은 패턴이 독립적으로 넷 나온 것이 이 PR보다 오래 남을 발견이다. 패턴은 `HashMap` 순회 결과가 순서 있는 또는 순서에 민감한 결정이 되는 것이고, 툴체인 어느 것도 이걸 잡지 못한다. 타입은 맞고, 컴파일되고, 테스트는 대체로 통과한다.
