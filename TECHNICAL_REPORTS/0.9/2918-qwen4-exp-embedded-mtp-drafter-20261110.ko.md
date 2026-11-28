# 기술 보고서: PR #2918 - feat(speculative): use a qwen4_exp checkpoint's own mtp.* head as the drafter

**작성일**: 2026-11-10
**리뷰어**: -
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

---

## 요약

`mtp.*` 텐서를 그대로 가진 `qwen4_exp`(Qwen3.8-Flash-Next) 대상 체크포인트를 자기 자신의 MTP 드래프터로 쓸 수 있게 했다. 대상 디렉터리를 `--draft-model`이나 `--model-draft`에 한 번 더 넘기면 된다. head가 든 샤드만 읽으므로 드래프터 메모리는 113 GB 체크포인트를 한 번 더 올리는 것이 아니라 1.61 GiB만 늘어난다. `mlx-community/Qwen3.8-Flash-Next-4bit-mtp`에서 출력은 classic decode와 바이트 동일했고, decode 속도는 프롬프트에 따라 약 46 tok/s에서 54~68 tok/s로 올랐다.

---

## 1. 문제 정의

### 1.1 배경

PR #2908(이슈 #2880)은 `qwen4_exp`의 native MTP 드래프터를 추가하면서, 대상 변환본이 `mtp.*`를 버리므로 head는 항상 별도 `qwen4_exp_mtp` 디렉터리로 온다고 가정했다. `mlx-community/Qwen3.8-Flash-Next-4bit-mtp`(리비전 `825df623`)는 head를 대상 안에 둔다. `mtp.*` 텐서 54개가 전용 샤드 `model-mtp-00001-of-00001.safetensors`에 있고, `mtp.` 접두사를 단 양자화 override가 11개 있다.

### 1.2 기존 문제점

- **드래프터 디렉터리 필수**: 로더가 `model_type: qwen4_exp_mtp`를 요구하고 디렉터리의 safetensors를 전부 읽어서, 내장 head를 쓰려면 직접 텐서를 뽑고 config를 만들어야 했다.
- **종류 판정**: `resolve_drafter_kind`가 `qwen4_exp` 디렉터리를 full-model 드래프터로 판정했다.
- **양자화 키**: 대상 config는 override 이름에 `mtp.` 접두사를 붙이지만, 드래프터는 접두사를 뗀 모듈 이름으로 찾는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 새 플래그 없음

| 대안 | 장점 | 단점 |
|------|------|------|
| `--draft-model` 없이 `--draft-kind mtp` | 명령이 짧다 | 기존 드래프터 경로 옆에 두 번째 관례가 생기고, 테스트할 인자 경로가 늘어난다 |
| **선택: 대상 디렉터리를 한 번 더 전달** | Inkling이 이미 쓰는 관례이고 서버와 offline 경로를 바꿀 필요가 없다 | 경로를 반복해서 쓴다 |

### 3.2 텐서 이름으로 판정, 기존 resolver 안에서

kind가 지정되지 않았거나 `mtp`일 때 `resolve_drafter_kind`가 `qwen4_exp` 디렉터리에서 `mtp.*`, `model.mtp.`, `language_model.mtp.` 텐서를(인덱스나 헤더로) 찾는다. head 없이 `--draft-kind mtp`를 주면 새 `DrafterError::NoEmbeddedMtpHead`로 실패한다. `--draft-kind`가 없으면 head 없는 디렉터리는 예전과 같이 판정된다.

**트레이드오프:** 내장 head가 있는 `qwen4_exp` 디렉터리를 `--draft-kind` 없이 드래프터로 넘기면 이제 full-model이 아니라 MTP로 판정된다. `--draft-kind full-model`로 예전 동작을 고를 수 있다.

### 3.3 대상 config에서 드래프터 config 유도

`Qwen4ExpMtpConfig::from_target_json`은 `model_type: qwen4_exp_mtp`와 명시적 `block_size: 4`를 넣는다(`from_json` 기본값은 `mtp_num_hidden_layers + 1 = 2`). `text_config`와 `tie_word_embeddings`를 옮기고, 양자화는 기본 쌍과 접두사를 뗀 `mtp.*` override만 남긴 뒤 기존 `from_json`을 실행한다. `from_json`을 재사용하면 체크포인트의 `full_attention` 표기를 `qwen_sparse_attention` 레이어 하나로 바꾸는 곳이 한 군데로 유지된다.

---

## 4. 구현 상세

- `src/lib/mlxcel-core/src/drafter/mod.rs`: `is_qwen4_exp_mtp_tensor_name`, `strip_qwen4_exp_mtp_prefix`, `has_qwen4_exp_mtp_tensors`(`dir_has_tensor_name` 기반), resolver 판정, `NoEmbeddedMtpHead`. 바이너리 크레이트가 생성하므로 core `load_drafter`는 `qwen4_exp`를 이름을 밝혀 거부한다.
- `qwen4_exp_sanitize::is_mtp_key`가 core 판정 함수에 위임해, 대상과 드래프터가 head의 범위를 똑같이 본다.
- `Qwen4ExpMtpDraftModel::from_target_checkpoint`는 `load_weights_from_dir_index_filtered(path, is_mtp_key)`로 읽어 head 텐서가 있는 샤드만 열고, 텐서 수와 상주 바이트를 로그로 남긴다.
- `drafter_loader::load_drafter`가 `qwen4_exp` + `Mtp`를 새 생성자로 보낸다. offline `generate`와 서버는 바뀌지 않고 그대로 도달한다.
- 메모리 추정: 변경 없음. 드래프터 바이트를 더하는 경로가 없고 인덱스 `total_size`가 head를 이미 한 번 센다(`mlxcel inspect` 113,245,755,466 바이트).

---

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 (이 보고서 제외) | 10 |
| 추가 라인 | +631 |
| 삭제 라인 | -25 |

| Hash | 유형 | 메시지 |
|------|------|--------|
| `628e86e5` | feat | use a qwen4_exp checkpoint's own mtp.* head as the drafter |
| `b03f722f` | fix | satisfy clippy and record the embedded-head validation |

---

## 8. 후속 조치

- 4bit 대상에서 드래프터 없는 classic `mlxcel-server`를 `MLXCEL_QMV_WIDE=0`로 돌리면 offline classic decode와 첫 답변 문장에서 갈라진다. 내장 head나 분할 드래프터를 단 서버는 offline classic과 같다. 이 차이는 classic 서버 decode의 문제이고 이 PR 이전부터 있었으므로 별도 이슈가 필요하다.
- `mtp.*`가 없는 `qwen4_exp` 디렉터리를 `from_target_checkpoint`에 직접 넘기면 일반 `WeightLoad` 오류가 난다. factory는 그런 디렉터리를 이 생성자로 보내지 않는다.
- 다른 패밀리의 내장 head는 다루지 않았다.

---

## 부록

### A. 테스트 결과

- 단위 테스트: 작은 fixture를 대상 체크포인트(인덱스가 mtp가 아닌 텐서를 무효 바이트 샤드로 가리킴)와 분할 디렉터리로 각각 썼을 때 프롬프트와 캐시된 step의 draft logits가 같고 둘 다 upstream과 일치한다. override 접두사 제거와 `block_size` 4, factory와 생성자의 `NoEmbeddedMtpHead`, `hc_count` 거부, `mtp.*` 없을 때의 기존 판정 유지. `mlxcel-core` drafter 테스트 257개, `qwen4_exp_mtp` 14개 통과.

### B. 실제 모델 (M3 Ultra, `Qwen3.8-Flash-Next-4bit-mtp`, `--temp 0`, reasoning 포함)

| 프롬프트 | MTP vs classic (`MLXCEL_QMV_WIDE=0`) | 수락률 | tok/s classic / MTP |
|---|---|---|---|
| 영어 | 바이트 동일, 256토큰 | 0.611 | 46.1 / 60.7 |
| 코드 | 바이트 동일, 214토큰 (EOS) | 0.731 | 45.2 / 67.8 |
| 한국어 | 바이트 동일, 256토큰 | 0.507 | 46.0 / 53.6 |

1회 실행이며 짝지은 벤치마크가 아니다. 드래프터 상주 크기 1,726,854,656 바이트, 텐서 54개. `mlxcel-server --model-draft <같은 디렉터리>`는 두 요청 모두 offline 출력을 돌려줬다. 분할 드래프터 `Qwen3.8-Flash-Next-MTP`와 OptiQ 2-bit 대상은 128토큰에서 classic과 바이트 동일했고(수락률 0.500), 4bit 서버에서는 내장 head와 같은 바이트를 돌려줬다.
