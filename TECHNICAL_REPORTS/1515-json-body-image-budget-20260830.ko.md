# 기술 보고서: PR #1515 - JSON body image budget

**작성일**: 2026-08-30
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

## 요약

PR #1515는 issue #1227을 해결한다. Main JSON route table에 남아 있던 Axum의 implicit 2 MiB extractor limit을 configured image payload 및 image count 설정에서 계산한 bounded limit으로 교체했다. 이제 2 MiB를 넘는 base64 image upload도 configured image budget 안에 있으면 실제 JSON handler까지 도달하고, extractor limit 초과는 OpenAI-compatible API의 다른 오류와 같은 structured error envelope로 반환된다.

## 1. 문제 정의

mlxcel은 기본 64 MiB encoded image payload budget과 request당 image block 최대 16개를 광고했지만, base64 `data:` image를 담은 JSON request는 handler가 request를 parse하기 전에 Axum의 default 2 MiB buffered-body limit에 걸렸다. Server-side HTTP image fetch path는 configured limit을 적용했지만, OpenAI-standard base64 JSON path는 해당 limit에 도달할 수 없었다.

Audio route에는 이미 더 큰 per-route extractor limit이 있었지만, 이 limit failure도 Axum의 bare 413 response를 사용했다. 필요한 동작은 하나의 structured boundary였다. Configured image limit이 JSON extractor budget을 결정하고, extractor overflow는 OpenAI-shaped 413으로 반환되어야 했다.

## 2. 기술적 선택과 그 이유

### 2.1 Configured image limit에서 JSON body ceiling 계산

Main JSON body limit은 startup이 `--max-image-payload-size`와 `--max-images`에서 설정한 process-wide `ImageInputLimits`에서 계산한다. 계산은 base64 expansion을 checked ceil arithmetic으로 구하고, configured simultaneous image count를 곱한 뒤 fixed 및 per-image JSON/data-URL overhead를 더한다.

### 2.2 기존 audio override 보존

Derived limit은 route table의 default extractor limit으로 적용한다. Audio route는 계속 명시적인 25 MiB per-route `DefaultBodyLimit`을 가진다. Axum의 request extension semantics에 따라 route-specific limit이 route-table default를 override한다.

### 2.3 Extreme configuration bound

Image가 0개인 configuration은 일반 JSON capacity를 줄이지 않고 Axum의 2 MiB default를 유지한다. Arithmetic overflow와 extreme operator value는 effectively unbounded buffered JSON allocation path를 만들지 않도록 2 GiB ceiling으로 clamp한다. Per-image payload, count, dimension, decoder allocation check는 parsing 이후에도 실행되어 exact configured media limit을 계속 enforce한다.

### 2.4 Extractor 413 response 정규화

가벼운 middleware가 `413 Payload Too Large` response를 project의 `ErrorResponse` JSON envelope와 `invalid_request_error` type으로 변환한다. 이 middleware는 extractor rejection 이후 적용되며 main JSON route와 audio upload route를 모두 커버한다.

## 3. 변경 요약

| 카테고리 | 변경 수 | 주요 내용 |
|---|---:|---|
| Body limit derivation | 1 | Main JSON extractor에 checked base64/JSON image-budget 계산 추가. |
| DoS guard | 1 | Extreme derived limit에 2 GiB hard ceiling 추가. |
| Error envelope | 1 | Extractor 413 response를 OpenAI-shaped JSON으로 정규화. |
| Documentation | 1 | Base64 JSON image와 image-limit flag의 상호작용 문서화. |
| Tests | 6 | Simultaneous input, above-2MiB acceptance, exact/oversized boundary, audio/main parity, zero/extreme config, error payload shape 검증. |

## 4. 검증

- `cargo test --lib server::app::tests`: 14 passed.
- `cargo clippy --lib --tests -- -D warnings`: passed.
- Hosted checks on implementation commit `1c357ccf`: cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compile, Detect changes, crate versions, cross-repo refs, kernel dtype keys, license/cla, llama-compat manifest passed. MLX pin extraction과 OpenXLA feature link는 change detection에 의해 skipped였다.

Broad workspace tests, serial all-tests, cold release build는 이 issue batch의 workflow guard가 금지하므로 실행하지 않았다.

## 5. 리뷰 메모

- **Correctness**: Test는 derived image budget이 허용하면 Axum default 2 MiB를 넘는 JSON request body가 accept되고, exact limit boundary는 accept, one-byte-over는 reject됨을 증명한다.
- **Security**: 모든 arithmetic은 checked 또는 saturating이며 extreme input은 2 GiB로 clamp된다. 이 변경은 extractor limit을 disable하지 않고, downstream image count, payload, dimension, decoder allocation validation도 bypass하지 않는다.
- **Performance**: Derivation은 app construction 시 한 번 실행된다. Runtime overhead는 기존 extractor limit과 response middleware의 status check로 제한된다.
- **Compatibility**: Body-limit error shape는 Axum bare 413에서 OpenAI-compatible `ErrorResponse` envelope로 바뀐다. 성공적인 JSON 및 audio route behavior는 그 외 변경이 없다.

## 6. 후속 조치

- 적절한 hardware가 있을 때 live VLM deployment에서 production configured payload ceiling에 가까운 실제 base64 image로 smoke test를 실행한다.
- Deployment가 더 큰 buffered JSON request를 의도적으로 필요로 하고 별도의 memory admission control을 갖춘 경우에만 2 GiB hard ceiling을 재검토한다.
