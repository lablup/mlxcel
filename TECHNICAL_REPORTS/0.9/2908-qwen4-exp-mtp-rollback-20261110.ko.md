# 기술 보고서: PR #2908 - feat(speculative): wire the native MTP head and rollback for qwen4_exp

**작성일**: 2026-11-10
**리뷰어**: -
**상태**: 후속 조치 필요
**언어**: Rust
**위험도**: 중간

---

## 요약

`qwen4_exp`(Qwen3.8-Flash-Next)가 거부된 추측 블록을 되돌릴 수 있게 되어, 네이티브 `qwen4_exp_mtp` 드래프터와 이 패밀리의 프롬프트 룩업이 모두 동작한다. OptiQ 2-bit 체크포인트에서 temperature 0, 256토큰 기준으로 프롬프트 네 개 모두 일반 디코드와 바이트 단위로 동일했고, 외부 PLE 테이블에서도 MTP 출력이 동일했다. 쌍체 타이밍 벤치와 적응형 정책 판정이 남아 있어 MTP 기본값은 바꾸지 않았다.

---

## 1. 문제 정의

### 1.1 배경

이 패밀리는 KV 캐시, QSA 인덱스와 블록 요약, GatedDeltaNet 재귀 상태, 컨볼루션 윈도, PLE n-gram 컨텍스트 등 시퀀스별 상태를 모두 모델이 소유한다. 추측 디코딩은 이 모든 상태에서 거부된 토큰을 지워야 하는데 그 경로가 없었다. 그래서 `mtp.*` 텐서는 버려졌고 `--prompt-lookup`은 거부됐다.

### 1.2 기존 문제점

- **롤백 경로 없음**: GatedDeltaNet과 컨볼루션 상태는 잘라낼 수 없고, 드래프트됐다가 거부된 EOS는 n-gram 컨텍스트를 틀어지게 한다.
- **드래프터 계약 차이**: 헤드는 최종 믹서 이전의 `[B, L, hc_count * hidden]` 스트림을 입력받는다. Qwen 3.5 어댑터는 post-norm hidden을 넘기므로, 그대로 재사용하면 조용히 잘못된 입력이 들어간다.

---

## 3. 기술적 선택과 그 이유

### 3.1 잘라낼 수 없는 상태는 재생, 로그 깊이는 1

| 상태 | 방식 |
|------|------|
| KV, QSA raw key와 position | 거부된 개수만큼 자름 |
| QSA 블록 요약 | 불완전해진 블록을 버리고 지연 재계산 |
| GatedDeltaNet, 두 컨볼루션 윈도 | 수락된 prefix에 대해 재생 |
| PLE n-gram 컨텍스트 | 수락된 토큰으로 재계산 |

로그는 가장 최근의 추측 forward만 보관한다. 깊이 4는 시퀀스당 약 450 MB로 추정했다(실측 아님). 로그 범위를 넘는 되감기는 오류를 반환하고 상태를 건드리지 않는다.

### 3.2 디코드와 같은 산술 경로의 verify 블록

**컨텍스트:** 이것이 없으면 정확성 프로브가 실제 체크포인트의 블록 4에서 logit 바이트의 약 절반이 달라 거부됐다.

**선택 이유:** verify가 디코드와 같은 경로를 탄다. QSA 호출당 쿼리 하나, 행 단위 컨볼루션, 퓨즈드 GatedDeltaNet 프로젝션, 행 단위 dense gate 프로젝션, 비정렬 expert gather를 쓴다. GPU 세대 15 이상에 남은 차이의 원인은 `qmv_wide`이며 기존 정확성 게이트가 이를 끈다. 이 패밀리의 프롬프트 룩업도 이제 같은 게이트를 거친다.

**트레이드오프:** 추측 디코딩을 설정하면 `qmv_wide`가 프로세스 전체에서 꺼지므로, 해당 GPU의 일반 디코드 기준선은 플래그 상태를 명시해야 한다.

### 3.3 모델 소유 패밀리의 되감기는 옵트인

`LanguageModel::supports_speculative_verify_rewind`의 기본값은 `false`이고 `qwen4_exp`만 켠다. 초기 기본값은 Gemma 3, Baichuan 같은 다른 모델 소유 패밀리까지 프롬프트 룩업에 들여보냈다.

### 3.4 적응형 정책을 게이트로 유지

Qwen 3.5 MTP는 M1 Ultra에서 일반 디코드의 0.59~0.70배로 측정됐다. 다른 엔진의 좋은 결과만으로 기본값을 바꿀 근거가 되지 않는다. `mtp_policy.rs`는 변경하지 않았다.

---

## 4. 구현 상세

| 영역 | 파일 |
|------|------|
| 롤백 | `src/models/qwen4_exp_rollback.rs`, `Qwen4ExpCache`와 분리 |
| 드래프터 | `src/models/qwen4_exp_mtp_drafter.rs`, `src/models/qwen4_exp_mtp_config.rs`. 레이어가 타깃의 것이라 바이너리 크레이트에 둠 |
| 타깃 어댑터 | `src/models/qwen4_exp_mtp_target.rs` |
| 디스패치 | `LoadedModel::Qwen4ExpVLM`의 CLI, burst, slice, scheduler 분기 |

롤백은 시퀀스별 상태만 다루고 row 테이블은 건드리지 않으므로 #2881의 resident 저장소와 external 저장소에서 똑같이 동작한다.

검증 중 발견해 이 PR에서 수정한 결함:

- GPU 세대 15의 `qmv_wide`에서 프롬프트 룩업이 약 25토큰 뒤 발산했다.
- 타깃이 2048토큰 청크 대신 한 번에 prefill해서 긴 프롬프트의 MTP가 일반 디코드와 달라졌다.
- 서버의 `prefill_chunk_size`가 MTP에서 무시됐다.
- `hc_count` 불일치가 응답을 잘랐다. 이제 바인드 전에 거부한다.
- `generate --draft-model`이 빈 프롬프트에서 panic했다. 이제 이름을 밝혀 거부한다.

---

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 40 |
| 추가 줄 | +4456 |
| 삭제 줄 | -172 |
| 리베이스 후 `models::qwen4_exp` 테스트 | 147개 통과 |

---

## 8. 후속 조치

### 완료 필요

- [ ] 일반 디코드, 블록 크기 2/3/4의 MTP, 프롬프트 룩업의 쌍체 5회 벤치와 `/v1/internal/mtp-policy` 판정.

### 향후 개선 사항

- MTP prefill이 프롬프트 전체의 pre-mixer 스트림을 보관한다(bf16 기준 토큰당 약 20 KB).
- `MtpTarget`이 `Result`를 반환하지 않아 `verify_finalize`는 롤백 실패를 로그로만 남긴다.
- 드래프트 오류가 공유 MTP 루프에서 조용히 생성을 끝낸다(전 패밀리 공통).
- 드래프터 폭(bf16, 8-bit, 4-bit)별 비용 대비 효과는 측정하지 않았다. CUDA와 CPU 경로는 이 패밀리에 허용되지 않는다.

---

## 부록

### A. 테스트 결과

`21532270` 위로 리베이스한 뒤(HEAD `274dc165`): `cargo fmt --check`, `make verify-env-var-docs`, `cargo clippy --lib --tests --bins --features metal,accelerate -- -D warnings`, `cargo test --profile test-fast --features metal,accelerate --lib -- --test-threads=1 models::qwen4_exp`(147개 통과). 워크스페이스 전체 테스트는 CI와 최종 게이트에서 다룬다.

### B. 실제 모델 결과

타깃 `mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit`, 드래프터 `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`(`be6e2bc3`), M3 Ultra, temperature 0, 256토큰, 리베이스 전 커밋 `e2151399`.

| 프롬프트 | MTP와 일반 디코드 | MTP acceptance | 프롬프트 룩업과 일반 디코드 | 룩업 acceptance |
|----------|-------------------|----------------|-----------------------------|-----------------|
| 영어 에세이 | 동일 | 0.479 | 동일 | 0.229 |
| Rust 퀵소트 | 동일 | 0.630 | 동일 | 0.189 |
| 한국어 계절 | 동일 | 0.396 | 동일 | 0.206 |
| 영어 파일 재작성 | 동일 | 0.628 | 동일 | 0.411 |

리베이스 후(`274dc165`, 128토큰, Rust와 한국어 프롬프트): MTP와 프롬프트 룩업 모두 일반 디코드와 동일했다. Rust 프롬프트에서는 `MLXCEL_PLE_IO=mmap`의 외부 `safetensors_ranges` PLE 뷰로 돌린 MTP도 resident 체크포인트의 일반 디코드와 동일했다. 모든 추측 실행에서 거부된 드래프트가 있어 롤백이 실제로 수행됐다.

다른 유닛의 빌드가 겹친 상태의 디코드 속도는 벤치가 아니므로 싣지 않았다.

### C. 참고 자료

- 이슈 #2880
- Blaizzy/mlx-vlm#2040, Blaizzy/mlx-vlm#2046
- Strata `docs/MAC.md`, `146df3c` 기준
