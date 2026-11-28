# 기술 보고서: PR #2038 - nanochat 텍스트 모델

**날짜**: 2026-09-29

**상태**: M1 Ultra에서 HuggingFace 오라클 대비 로컬 구현 및 검증 완료, 머지 대기 중 (GB10 CI 러너 사용 불가).

**언어**: Rust

**위험도**: 낮음 (신규 모델 패밀리입니다. 공유 코드 변경은 `model_type == "nanochat"`에 한정된 로드 시점 dtype 제외 하나뿐입니다)

## 요약

이슈 #1368은 `nanochat` 스피드런 모델(d20 0.56B, d32 1.9B)을 요구했습니다. 구조는 GPT-2 형태이며, 네 가지 선택 중 하나라도 빠뜨리면 그럴듯하지만 틀린 텍스트가 나오므로 수용 기준은 유창함이 아니라 오라클 대비 greedy id 일치였습니다. 이 PR은 `src/models/nanochat.rs`를 추가하고 패밀리를 끝까지 등록합니다. 두 검증 체크포인트 모두에서 독립적인 HuggingFace `NanoChatForCausalLM` 오라클과 greedy id가 일치하며, 차이는 0.08 미만의 로짓 근소차뿐입니다.

## 1. 문제 정의

`model_type: "nanochat"`은 미지원 분기로 떨어졌습니다. 기존 패밀리의 설정 변형이 아닙니다. 노름 가중치가 전혀 없고, q와 k를 표준 RoPE와 반대 방향으로 회전하며, MLP에서 ReLU를 제곱하고, 출력 로짓에 소프트캡을 적용합니다.

## 2. 변경 요약

- `src/models/nanochat.rs`: 어텐션(`c_q` / `c_k` / `c_v` / `c_proj` 분리), `relu_squared` MLP, 가중치 없는 블록, 마지막 위치 로짓 오버라이드를 갖춘 `NanoChatModel`, `sanitize_weights`, `load`. `src/models/nanochat_config.rs`는 `ModelArgs`와 신뢰할 수 없는 `config.json`에 대한 범위 검증을 담습니다.
- 등록: `ModelType::NanoChat`, `LoadedModel::NanoChat`, `model_metadata.rs` 행, 레지스트리 패밀리 및 id 테이블, 메모리 추정기, 텐서 병렬 아키텍처 문자열(TP 미지원), `model_type` 기반 감지.
- `src/models/sanitize.rs`: nanochat을 bf16에서 f16으로의 로드 변환에서 제외하고, pre-Ampere CUDA에서 f16 취약 패밀리로 등재합니다.
- `docs/supported-models.md`: 네 가지 선택, 두 가지 레이아웃, BOS 요구 사항, 검증 체크포인트를 설명하는 항목.
- `nanochat_tests.rs`의 단위 테스트 11개와 감지 테스트 1개.

## 3. 기술적 결정

### 부호를 뒤집은 주파수 테이블로 구현한 미러 RoPE

`fast_rope_with_freqs`는 위치를 테이블 각 항목으로 나누므로, `traditional = false`와 함께 `-(base ** (arange(half) / half))`를 쓰면 절반 분할 레이아웃에서 각도 `-p / f_i`가 됩니다. 이는 HuggingFace의 `rotate_half`가 `[x2, -x1]`을 반환하는 것, 즉 일반 회전의 부호 반전과 일치합니다. 단위 테스트는 회전된 쌍을 닫힌 형태와 비교하고 `fast_rope`가 다른 값을 낸다는 점도 확인합니다.

### 두 소프트캡 키를 두 필드로

MLX 변환본은 `logits_soft_cap`과 `logits_softcap`을 함께 씁니다. serde `alias`는 이를 중복 필드 오류로 처리하므로 각각을 이중 옵션 필드로 두고, `soft_cap()`이 기본 키, 별칭, 15.0 순으로 선택합니다. 명시적 `null`은 캡을 끕니다.

### f16이 아니라 bf16이 필요

bf16 모델의 첫 실측 실행은 모든 위치에서 greedy id 0을 냈습니다. 레이어별 통계에서 레이어 15 이후 잔차 스트림 최대값이 64000이었고 레이어 16부터 NaN이 나왔습니다. 모든 노름에 가중치가 없고 서브레이어에만 입력을 주므로 잔차가 재정규화되지 않기 때문입니다. bf16은 f32의 지수 범위를 가집니다. 8비트 체크포인트는 양자화 모델이 원래 bf16을 유지하므로 영향이 없었습니다.

### QK-norm 순서는 크기로 검증할 수 없음

이슈는 RoPE 이후 norm이 RoPE 이전 norm과 다르다는 것을 확인하는 테스트를 요구합니다. 회전은 헤드별 L2 노름을 보존하므로 두 순서는 엡실론 오차 안에서 같습니다. 대신 테스트는 RoPE 후 norm의 CPU 기준값과 출력을 비교합니다. 순서 자체는 명세대로 구현했습니다.

## 4. 검증

오라클: CPU의 HuggingFace `transformers` 5.17 `NanoChatForCausalLM`, 동일한 safetensors에서 역양자화한 가중치. 프롬프트는 체크포인트의 채팅 템플릿 또는 리터럴 `<|bos|>` 접두사로 만들었고, mlxcel의 `/tokenize`가 동일한 프롬프트 id를 반환했습니다. 48토큰 greedy 디코드이며 id는 서버 `/completion`의 `tokens` 필드에서 읽었습니다.

| 체크포인트 | 프롬프트 | bf16 오라클 일치 | fp32 오라클 일치 |
|---|---|---|---|
| q8 (채팅 템플릿) | transformer | 45 중 43 | 45 중 43 |
| q8 | France | 7 / 7 | 7 / 7 |
| q8 | haiku | 41 / 41 | 17 (근소차 0.004) |
| bf16 (`<|bos|>`, 템플릿 없음) | transformer | 17 (fp32 오라클은 mlxcel과 일치) | 13 (0.008) |
| bf16 | France | 9 / 9 | 9 / 9 |
| bf16 | haiku | 48 / 48 | 30 (0.075) |

모든 불일치는 fp32 오라클에서 teacher-forcing으로 측정한 두 후보 토큰 사이의 로짓 차이가 0.08 미만입니다. 단위 테스트, `fmt`, `clippy -D warnings`, 세 가지 계약 테스트가 통과합니다. 다른 유닛이 GPU를 공유했으므로 처리량 수치는 보고하지 않습니다.

## 5. 보류 항목

d32 체크포인트(`karpathy/nanochat-d32`는 `.pt`만 제공), `<|python_start|>` 도구 루프, GB10 러너의 CI.
