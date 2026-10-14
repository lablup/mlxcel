# 기술 보고서: PR #2313 — 네 가지 권고에 대응해 rustls 업데이트

**날짜:** 2026-10-15
**이슈:** #2312
**범위:** Lockfile-only 의존성 보안 패치. Manifest, feature, provider, API, error behavior, inference는 변경하지 않는다.

## 요약

PR #2313의 원래 source commit `e9604cba`는 유일하게 resolve된 `rustls` package를 0.23.45에서 RUSTSEC-2026-0346부터 RUSTSEC-2026-0349까지 수정한 upstream release 0.23.46으로 업데이트했다. 당시 전체 product diff는 `Cargo.lock`의 rustls version과 registry checksum이었으며, 나머지 resolve package 108개, manifest, 기존 provider feature 선택은 바뀌지 않았다.

**현재 상태:** 같은 rustls 0.23.46 lock update가 PR #2304를 통해 `main`에 반영되었다. 이 merged state를 통합한 뒤 PR #2313은 review 및 test 기록을 report-only 문서로 보존하며 추가 product change를 도입하지 않는다.

Configuration, command-line option, API, migration, 관측 가능한 error contract가 바뀌지 않으므로 사용자 문서 수정은 필요하지 않다. 보안 수정은 호환되는 upstream patch release가 제공한다.

## 권고 평가

| 권고 | Upstream 문제 | mlxcel 범위 판정 |
|------|---------------|-----------------|
| RUSTSEC-2026-0346 / GHSA-c8m5-c952-6cm4 | Derived `Debug` output이 custom client session store를 통해 TLS session secret을 노출할 수 있음 | mlxcel은 custom `ClientSessionStore`를 사용하지 않지만, 업데이트로 vulnerable implementation을 제거함 |
| RUSTSEC-2026-0347 / GHSA-vp88-798v-h67f | Opt-in ECH가 거절될 때 buffered application data가 인증된 outer server에 노출될 수 있음 | mlxcel은 ECH를 설정하지 않음 |
| RUSTSEC-2026-0348 / GHSA-h2v8-394m-p7vv | 인증되지 않은 Brotli certificate decompression이 최대 1 GiB를 할당할 수 있음 | Resolve된 rustls package는 Brotli를 활성화하지 않음 |
| RUSTSEC-2026-0349 / GHSA-w44r-93j7-qv28 | 인증되지 않은 ServerHello가 per-name key-share hint를 오염시켜 이후 resumable connection을 classical exchange로 downgrade할 수 있음 | Outbound client 때문에 문제 유형은 관련되지만 현재 rustls feature graph는 `ring` provider를 선택하며 product-level post-quantum coverage 또는 loss를 입증하지 않는다. Authentication bypass나 key compromise도 주장하지 않음 |

네 권고 모두 rustls 0.23.46을 patched version으로 지정한다. Advisory ignore나 policy exception은 추가하지 않았다.

## 검증

아래 검증은 원래 `e9604cba` source와 dependency graph에 대한 기록이며, 현재 report-only diff가 rustls pin을 변경한다는 주장이 아닌 historical evidence이다.

- `cargo deny check`는 advisories, bans, licenses, sources에서 로컬 통과했고 PR #2313의 hosted `cargo-deny` job도 통과했다.
- `cargo metadata --locked --no-deps --format-version 1`이 통과하고 lockfile을 변경하지 않은 채 workspace package 5개를 보고했다.
- `cargo tree --locked -i rustls -e features`는 rustls 0.23.46 하나를 resolve했다. Feature graph는 명시적인 `ring`, `std`, `tls12`, logging 선택을 유지하며 다른 rustls crypto provider를 활성화하지 않는다.
- Parsing한 diff 검사에서 version `0.23.45` → `0.23.46`과 checksum 교체라는 두 field만 바뀐 것을 확인했다. `git diff --check`도 통과했다.
- `MLXCEL_BUILD_METAL=OFF RUSTC_WRAPPER= cargo test --locked --profile test-fast --no-default-features --lib tls_tests:: -- --test-threads=1`에서 10개 test가 통과하고 ignored test는 없었다. 기존 TLS configuration, certificate, key, plaintext-fallback guard를 포함한다.
- 같은 방식의 `downloader::tests::` 실행에서 71개 test가 통과하고 optional integration test 1개가 ignored되었다. Live external download는 실행하지 않았다.
- `e9604cba`의 최초 default-macOS build는 test 실행 전에 MLX의 무관한 `gated_delta_update_nax.metal` MPP compile failure에 도달했다. Authoritative focused run은 무관한 Metal backend를 명시적으로 끄고 nonzero count로 완료했으며, 이 failure는 현재 `main`의 rustls pin에 대한 근거가 아니다.
- 독립 implementation 및 security review에서 lock-only patch나 제한을 명시한 advisory wording의 blocking issue를 찾지 못했다. Finalization 중 hosted CI의 cargo-deny와 관측된 repository contract check가 통과했으며, 남은 CI status는 merge 전에 별도로 확인한다.

## 위험과 한계

Cargo가 transitive update를 선택하지 않았고 source나 manifest가 바뀌지 않았으므로 patch-release resolution risk는 낮다. 네 advisory exploit을 독립적으로 재현하지 않았으며 보안 동작은 확인된 upstream release와 advisory record에 근거한다. Ring-only rustls provider 결과는 dependency qualification이지 mlxcel이 이전에 hybrid post-quantum exchange를 제공했다는 증거가 아니다. 이 report에서는 inference checkpoint, GPU path, performance benchmark, ECH flow, Brotli certificate, custom session store, live external downloader TLS session을 실행하지 않았다.

## 참고 자료

- Issue #2312 및 PR #2313.
- rustls 0.23.46 release: https://github.com/rustls/rustls/releases/tag/v/0.23.46
- RustSec record: RUSTSEC-2026-0346, RUSTSEC-2026-0347, RUSTSEC-2026-0348, RUSTSEC-2026-0349.
