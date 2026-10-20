# PR #2507: PhiMoE norm 엡실론 키와 SAM antialiased bicubic a

**작성일**: 2026-10-21
**상태**: 완료
**위험도**: Low
**언어**: Rust

---

## 요약

#2465 종결. 머지된 모델 PR들이 남긴 두 개의 수치 패리티 갭을 닫는다: PhiMoE가 이제 체크포인트의 `rms_norm_eps` 표기를 존중하고 (Phi-3.5-MoE 체크포인트는 norm 엡실론을 `rms_norm_eps: 1e-5`로 쓰지만 `ModelArgs`는 `layer_norm_eps`만 읽어 체크포인트 값이 무시되고 norm이 잘못된 엡실론으로 돌았음), SAM 비전 인코더의 antialiased bicubic 리사이즈가 ATen의 `a = -0.5`를 쓴다 (Keys의 `-0.75`가 아니라). 둘 다 실제 파싱·전처리 경로에 반영돼 있고 커밋된 torch 생성 픽스처로 테스트가 고정돼 있다.

---

## 1. 문제 정의

### 1.1 배경

0.4.0 릴리스 스윕에서 발견된(PR #2457)과 그때 bit-identical로 남은(PR #2459) 두 개의 작은 수치 패리티 갭. 한 PR, 한 번의 검증 패스.

### 1.2 기존 문제점

- **PhiMoE 엡실론**: `phimoe::ModelArgs`는 `layer_norm_eps`(기본 1e-6)를 선언했지만 Phi-3.5-MoE 체크포인트는 `rms_norm_eps: 1e-5`로 표기한다. 체크포인트 값이 조용히 무시됐고(serde는 알 수 없는 필드를 버림), 모든 LayerNorm 블록이 1e-5 대신 1e-6으로 돌았다.
- **SAM 리사이즈 계수**: `resample_matrix.rs`의 antialiased bicubic 리사이즈가 `a = -0.75`(Keys, OpenCV 호환)를 썼지만 ATen의 antialiased 커널과 mlx-vlm은 `a = -0.5`를 쓴다. SAM 그리드 리샘플이 참조 전처리와 갈라졌다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| PhiMoE norm이 참조 체크포인트와 다른 엡실론 | Low (1e-5 vs 1e-6 norm 스케일) | High (모든 Phi-3.5-MoE 체크포인트) |
| SAM 전처리가 mlx-vlm/torch와 불일치 | Low (수치 패리티) | High (모든 SAM 이미지) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

클린. serde alias는 기존 f32 필드의 두 번째 표기를 받는 것뿐이다: 적대적 JSON도 panic 불가(NaN/Infinity 리터럴은 serde 파스 에러, 범위 밖 숫자는 panic 없이 degrade, 한 설정에 양쪽 표기가 있으면 serde duplicate-field Err). 공유·가변 상태 없음(모든 변경 줄을 static/once_cell/OnceLock/Mutex/RwLock으로 grep: 없음). 커밋된 `generate_reference.py`는 문서 전용이고 테스트가 실행하지 않는다. 픽스처 JSON은 `include_str!`로 `#[cfg(test)]` 게이트다.

### 2.2 성능 관점

핫 경로 변화 없음. `interp_matrix`는 모델 로드당(행렬 빌드)이지 토큰당이 아니고, 계수 변경은 비교 두 개. PhiMoE의 엡실론은 LayerNorm 생성 시 한 번 읽는다(phimoe.rs:514/524/614). 픽스처는 테스트 게이트 include라 프로덕션 바이너리에 영향 없음.

### 2.3 호환성/의존성 관점

- **호환성 깨짐**: 없음. 필드 이름 유지(`layer_norm_eps`)라 세 `LayerNorm::new` 호출 지점 불변, 기존 표기도 계속 파싱, 어느 키도 없으면 1e-6.
- **새 의존성**: 없음.

### 2.4 코드 품질 관점

- 테스트: phimoe 15 통과(신규: 체크포인트 표기 픽스처, 기존 표기 계속 수용, 어느 키도 없는 기본값), resample_matrix 9 통과(신규: torch AA 행렬 패리티에 16→24 업샘플 포함, SAM 그리드 패리티 vs torch 직접 2-D AA, 비 antialiased bit-identical 고정, rows-sum-to-one), deepseekocr 22 통과. resample 필터는 수정 전 코드에서 9개 중 3개 실패.
- fail-on-main 증명: 체크포인트 표기 config 테스트는 main에서 실패(알 수 없는 키 무시 → 1e-6); `-0.75` 계수는 커밋된 픽스처 대비 1.1e-02(다운샘플)~3.6e-02(업샘플) 벗어남.

---

## 3. 기술적 선택과 그 이유

### 3.1 체크포인트 표기의 serde alias

**컨텍스트:** `ModelArgs`는 원격 파생 경로(`impl_rope_lifting_deserialize!`) 뒤에 있다. 이 매크로는 rope 키를 JSON Value에서 들어올린 뒤 serde로 위임한다.

**선택:** 기존 필드에 `#[serde(default = "default_layer_norm_eps", alias = "rms_norm_eps")]`. alias는 원격 파생 위임을 통과하고(serde derive는 필드 속성을 처리), 필드 이름이 유지돼 호출 지점이 불변이며, 기본값은 어느 키도 없을 때만 적용된다. 한 설정에 양쪽 표기가 있는 것은 serde Err인데, 어떤 체크포인트도 만들지 않는 형태라 수용.

### 3.2 다운샘플일 때만이 아니라 antialiased면 항상 a = -0.5

**컨텍스트:** 이슈는 `-0.5`를 `scale > 1`(스트레치 서포트 경로)에 게이트할 것을 제안했다.

**선택:** 근거를 들어 이탈: torch 2.14.1은 antialiased 커널에 양방향 모두 `a = -0.5`를 쓴다. 프로브(커밋된 픽스처): 16→24 업샘플에서 `a=-0.5`는 torch와 6.1e-07 이내, `-0.75`는 3.6e-02 벗어남; 16→5 다운샘플: 1.0e-07 vs 1.1e-02. 이슈의 Implementation Notes도 업샘플 계수를 확인하고 거울로 맞추라고 지시했다. antialias로 업샘플하는 프로덕션 호출자는 없고(SAM은 1024→64 다운샘플), AA-다운샘플 경로 외 기존 동작 변화는 없음. 서포트는 여전히 다운샘플일 때만 스트레치(`filterscale = max(scale, 1.0)`), PIL/ATen과 동일.

### 3.3 비 antialiased 경로 불변

public `cubic()`는 여전히 `cubic_convolution(x, -0.75)`로 위임한다. kimi_vl의 MoonViT 에지 복제 그리드는 byte-identical 유지, 기존 `moonvit_bicubic_matrix_is_bit_identical`과 신규 `nonantialiased_cubic_matrix_is_unchanged`(16→5, 16→8 bit 단위 + 같은 크기 항등)로 고정.

---

## 4. 구현 상세

**파일: `src/models/phimoe.rs`** — `layer_norm_eps`에 `alias = "rms_norm_eps"` 추가, 모듈 문서가 체크포인트 표기(mlx-lm `phimoe.py`가 읽는 키) 기록.

**파일: `src/vision/encoders/resample_matrix.rs`** — `interp_matrix`가 `kernel == Cubic && antialias`일 때 `a = -0.5`, 아니면 `-0.75` 선택, 모듈 문서 갱신.

**테스트:** `src/models/phimoe_tests.rs` (신규 테스트 2개 + 체크포인트 표기 픽스처), `src/vision/encoders/resample_matrix_tests.rs` (신규 테스트 3개에 `include_str!` 픽스처 소비 포함).

**픽스처:** `tests/fixtures/deepseekocr_sam/{reference.json, generate_reference.py}` — 16→5/8/24 AA cubic 행렬, (5,5), (5,7), (24,24) SAM 그리드 리샘플; 기록된 교차 검증: mlx-vlm vs torch 최대 절대 차 9.76e-07, separable vs 직접 2-D 2.38e-07. torch 2.14.1+cpu와 mlx-vlm 0.7.8(커밋 73d366ecf9...)로 생성.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| cargo test phimoe (--test-threads=1) | 15 통과 / 0 실패 |
| cargo test resample_matrix (--test-threads=1) | 9 통과 / 0 실패 (수정 전 3/9 실패) |
| cargo test deepseekocr (--test-threads=1) | 22 통과 / 0 실패 |
| cargo fmt --all -- --check | 클린 |
| 스코프 clippy (lib+tests, -D warnings) | 클린 |
| 풀 게이트 (161 바이너리, HEAD 6ec45810) | build 0, clippy 0, doc 0, contract 0; 바이너리 green (mllama_parity 7/1 = 알려진 #2467 flake, 재실행 green) |
| mlx-lm 오라클 vs 수정 mlxcel (phi-3.5-moe-4bit) | 둘 다 "Paris."로 시작, 토큰 3에서 갈라짐 = 크로스 디바이스 수치 경로(오라클 CPU mlx 0.32.4 vs mlxcel CUDA), #932 계약대로 기록, token-exact 주장 아님 |
| 대조 (결정적) | rms_norm_eps를 지운 설정으로 돌린 수정 mlxcel == 동일 16토큰 스트림: 엡실론이 이 프롬프트의 생성을 바꾸지 않음, 수정은 유닛 테스트로 증명 |
| 보안 검토 | CLEAN, 모든 심각도 0건 |

업스트림 전제 검증: mlx-community/Phi-3.5-MoE-instruct-4bit config.json이 `rms_norm_eps: 1e-05`로 실제 표기.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 6 |
| 추가 | +3526 (픽스처 ~3.4k) |
| 삭제 | -16 |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|--------|
| `6ec45810` | fix | fix(models): PhiMoE norm epsilon key and SAM antialiased bicubic a |

---

## 7. 후속 조치

### 모니터링 필요

- 이 PR의 릴리스 모니터링 항목 없음.
- 로컬 gitignored 체크포인트 복사본 `models/mlx/phi-3.5-moe-4bit/config.json`이 현재 `rms_norm_eps`를 통째로 결여(업스트림과 달리); 로컬 모델 실행은 alias를 안 거치지만 커밋된 커버리지는 거침.

### 향후 개선 사항

- 이 PR에서 보류된 항목 없음.

---

## 부록

### A. 리뷰 판정

pr-reviewer: 6개 포커스 전부 PASS, 0건 (줄 이동 인용만, 선행 머지 때문). pr-security-checker: CLEAN, CRITICAL/HIGH/MEDIUM/LOW 전부 0 (serde alias는 적대적 JSON에도 panic 불가, 공유 상태 없음, 재정규화 합이 양수임을 테스트로 고정, 픽스처 테스트 게이트). pr-finalizer: 완료, 갭 없음; PR 본문 주장 전부 독립 재현(커밋된 픽스처 대비 순수 Python f32 프로브, 업스트림 config fetch, 스코프 clippy exit 0).
