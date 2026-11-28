# PR #2455: 로더 밖에서도 선언된 OptiQ vision sidecar 허용

**작성일**: 2026-10-19
**상태**: 실제 OptiQ 체크포인트로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

#2451을 닫는다. OptiQ 변환본은 vision 텐서를 `config.json`의 `optiq_vision.sidecar`가 선언하는 sidecar(`optiq/optiq_vision.safetensors`)에 인덱싱한다. #2446이 로더가 이 항목을 받아들이게 했지만, downloader의 완전성 검사, WebUI catalog, `weight_footprint_bytes`는 여전히 이 항목을 누락되었거나 안전하지 않은 shard로 취급했다. 이 PR은 config 읽기를 공유 helper 하나로 만들고 네 곳 모두에서 사용한다. 이슈는 세 항목을 한 PR에서 처리하고 검증은 마지막에 한 번만 수행하도록 요구한다.

## 문제 정의

- **스냅샷이 불완전으로 분류됨.** `missing_indexed_shards`가 이 항목을 `/`가 있으면 거부하는 `shard_present`로 보낸다. `model.safetensors`는 인덱싱되어 있으므로 `has_unindexed`는 false이고 스냅샷은 `Incomplete`가 된다. 그 결과 main에서는 OptiQ 체크포인트의 store 복사본이 `--offline` 실행에서 실패하고, 일반 실행에서는 다시 다운로드한다.
- **WebUI catalog가 unsafe shard path를 보고함.** `indexed_completeness`와 `push_index_shard_facts`가 여러 컴포넌트로 된 이름을 모두 거부하므로, catalog는 "safetensors index contains an unsafe shard path"라는 사유로 체크포인트를 불완전으로 표시한다.
- **Footprint에서 sidecar가 빠짐.** `weight_footprint_bytes`는 이 항목에서 `validate_index_shards`가 실패해 `model.safetensors` 헤더로 되돌아간다. `metadata.total_size`도 sidecar를 포함하지 않으므로 `mlxcel inspect`의 weight footprint에서 vision tower가 빠진다.

## 기술적 선택과 그 이유

- **공유 helper.** `mlxcel_core::weights::declared_optiq_vision_sidecar(dir)`는 #2446의 private helper에서 분리했다. Infinity/NaN을 허용하는 config 읽기와 `optiq_vision_sidecar_path` 검증을 수행하며, 하위 디렉터리 경로만 받는다. 로더의 `collect_shard_paths`와 새로 추가된 세 곳이 모두 이를 호출하므로 검증 로직을 복사하지 않는다.
- **config 읽기 크기 제한.** catalog가 통제할 수 없는 디렉터리에서 이 helper를 호출하고, catalog의 완전성 경로는 config를 읽기 전에 크기 검사가 없었다. 그래서 읽기를 16 MiB로 제한하고 그 이상이면 아무것도 선언하지 않은 것으로 처리한다. 두 번째 커밋에서 리뷰 후속 조치로 추가했다.
- **Downloader.** `missing_indexed_shards`에서 선언된 sidecar는 `dir.join(sidecar)`가 비어 있지 않으면 존재하는 것으로 보고, 아니면 상대 경로 이름으로 누락 보고한다. 선언되지 않은 하위 디렉터리 항목과 traversal 항목은 계속 누락으로 처리한다. config는 인덱스가 여러 컴포넌트 항목을 가질 때만 읽는다.
- **WebUI catalog.** `indexed_completeness`와 `push_index_shard_facts`는 선언된 sidecar에 한해서만 unsafe-path 거부를 건너뛴다. 이후 완전성은 위의 공유 규칙이 결정하고, content fingerprint는 이제 sidecar 파일도 포함한다.
- **Footprint.** `weight_footprint_bytes`는 인덱스 검증 전에 sidecar를 제외하고, 파일이 있으면 그 헤더 바이트를 더한다. sidecar가 선언된 경우 OptiQ가 sidecar 없이 기록하는 `metadata.total_size`를 믿지 않고 메인 shard 헤더를 합산한다. sidecar를 포함해 기록하는 변환기라면 이중 계산이 될 수 있기 때문이다. 빈 shard 목록 guard는 제거하여 비 OptiQ 체크포인트는 기존 경로를 그대로 탄다.

## 변경 내용

- `src/lib/mlxcel-core/src/weights.rs`: `declared_optiq_vision_sidecar`, 크기 제한이 있는 config 읽기, footprint 변경, 테스트.
- `src/downloader/completeness.rs`: `missing_indexed_shards`의 선언된 sidecar 처리, 테스트.
- `src/server/webui/catalog_fs.rs`, `src/server/webui/catalog_fs_tests.rs`: 선언된 sidecar에 대한 unsafe-path 거부 생략, fingerprint 반영, 테스트.

## 검증

GB10, 체크포인트는 `~/models_archive/mlx`.

| 검사 | main (수정 비활성화) | 이 PR |
|---|---|---|
| `mlxcel inspect -m lfm2.5-vl-3b-optiq-4bit` | 1958475776 bytes | 2811046368 bytes |
| `mlxcel inspect -m muse-glimmer-30b-optiq-4bit` | 20003849216 bytes | 23847540736 bytes |
| `mlxcel-server --offline -m mlx-community/<name> --model-store-root <store>` (각 OptiQ 체크포인트의 store 항목) | "no cached snapshot" 오류 | store에서 해석됨 (LFM은 `/health` 응답, Muse는 모델 로딩까지 진행) |
| `--models-dir ~/models_archive/mlx --webui`로 `GET /ui-api/v1/catalog` | 두 OptiQ 항목 모두 `complete: false`, unsafe shard path | 둘 다 `complete: true`, 사유 없음 |

기대 총합은 main 헤더와 sidecar 헤더의 합이며, safetensors 헤더에서 독립적으로 계산했다 (sidecar는 852570592 및 3843691520 bytes). 비 OptiQ 검사는 변하지 않았다. 나머지 191개 catalog 항목의 완전성과 fingerprint가 동일하고, gemma-3-4b-it-4bit, glm-4.5v-4bit (stale index), llama-3.2-1b-instruct-bf16, qwen3.5-0.8b-optiq-4bit (sidecar 없음)의 inspect footprint가 main과 일치하며, gemma-3-4b-it-4bit는 여전히 store에서 offline으로 해석된다.

새 테스트 다섯 개는 수정을 비활성화하면 실패하고 활성화하면 통과한다. 완전성 present/absent, catalog OptiQ 레이아웃, catalog fingerprint, footprint (sidecar를 이미 포함한 `total_size`와 stale index 포함), 크기 제한이 있는 config 읽기다. 선언되지 않은 경로와 traversal 경로를 다루는 테스트 두 개는 guard이며 양쪽 모두 통과한다. `downloader::`, `webui::catalog`, `execution::memory_estimate`, mlxcel-core `weights::`, clippy (`--lib --tests -D warnings`, 메인 crate와 mlxcel-core), fmt, contract 스크립트가 로컬에서 통과했다.

## 통계

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 4 |
| 추가된 줄 | 331 |
| 삭제된 줄 | 19 |

| 커밋 | 메시지 |
|------|--------|
| `1440ad55` | fix: accept the declared OptiQ vision sidecar outside the loader |
| `205cb110` | fix: bound the sidecar config read and stop trusting total_size |

## 관련 항목

- #2446: 로더가 선언된 sidecar를 받아들이게 했고 `mlxcel_core::weights`에 `optiq_vision_sidecar_path`를 도입했다.
- #2451: 이 PR이 닫는 이슈.
