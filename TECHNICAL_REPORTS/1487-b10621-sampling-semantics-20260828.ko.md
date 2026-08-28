# 기술 보고서: PR #1487 - b10621 샘플링 의미론과 윈도우

**작성일**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: High

## 요약

이슈 #1436이 소유한 b10621 샘플링 의미론 격차를 구현했다. `repeat_last_n` 페널티 윈도우가 드디어 샘플러에 도달하고, DRY 센티널이 업스트림과 일치하며, top-n-sigma, XTC, ignore-eos, reverse-prompt의 서버 전역·네이티브 표면이 두 바이너리에 들어가고, 샘플러 순서 플래그가 고정된 b10621 기본 체인에 대해 검증한다. 이슈가 소유한 매니페스트 엔트리 61건 중 26건이 `supported`, 3건이 검증된 거부와 함께 `not_applicable`이 되었고, 32건은 새 잔여 이슈 #1485로 재지정되어 pin.json 샤드 공동 소유자로 등록됐다.

## 1. 문제 정의

`--repeat-last-n`은 파싱되어 `/props`에 표시됐지만 반복·빈도·존재 페널티는 전체 히스토리를 스캔했다. DRY는 `0`을 "전체 스캔"으로 읽었는데 b10621은 `0`에서 비활성화한다. 정확히 `dry_allowed_length` 길이의 반복은 페널티를 받지 않았다(업스트림 `>=` 대비 `>`). top-n-sigma/XTC 서버 플래그, 네이티브 XTC·ignore-eos 필드, 샘플러 순서, reverse-prompt 등 여러 표면에는 mlxcel 철자가 아예 없었다.

## 2. 기술적 선택과 그 이유

### 2.1 fused 디스패치 앞의 Rust 윈도잉

`SamplingConfig::penalty_last_n`(`-1` 전체 히스토리이자 기본값, `0` 단계 비활성, `N > 0` 윈도우)이 기존 페널티 함수에 전달되는 히스토리를 슬라이스한다. 증분 `SamplerState`는 전체 히스토리 형태만 바이트 동일하게 담당하고, 양수 윈도우는 유계 재구성 경로를 탄다(b10621 기본 윈도우 64). `needs_token_history`가 윈도우를 게이트하므로 윈도우 0 설정은 fused batch 자격을 유지한다. 서버는 파싱만 되던 `repetition_context_size`(별칭 `repeat_last_n`)를 해석하거나 서버 기본값 64로 폴백한다. 필드를 생략한 페널티 사용 서버 요청의 동작이 바뀌는 의도된, 문서화된 변경이다.

### 2.2 리뷰로 교정한 업스트림 도메인

핀 고정된 b10621 소스 대조 리뷰가 첫 커밋의 오독 세 건을 교정했다. 요청 stop 문자열이 비어 있지 않으면 서버 전역 `--reverse-prompt` 목록을 병합이 아니라 통째로 대체한다(업스트림은 요청에 유효한 stop이 없을 때만 CLI antiprompt로 폴백). XTC 스키마 제한은 SOFT라 범위 밖 값은 400이 아니라 클램프된다. b10621 CLI는 음수 `--dry-penalty-last-n`을 거부하므로 두 서버 바이너리도 파싱 시점에 거부하고, 오프라인 generate CLI만 `-1`을 mlxcel 전용 전체 히스토리 철자로 유지한다. `dry_base`는 시작 시 정제되고 core·xla DRY 구현이 업스트림의 1.0 미만 조기 종료와 지수 상한을 반영한다.

### 2.3 EOG 바이어스로서의 ignore-eos, 정직한 범위 한정

`--ignore-eos` / `ignore_eos`는 enqueue 시점에 공유 token-bias 맵으로 병합 EOS 집합 전체를 `-inf`로 억제한다. 업스트림과 같은 메커니즘이다. OpenXLA 워커는 token-bias 경로가 없어 진단과 함께 거부하고, 분산 핸드오프는 token bias를 버리므로 기존 XTC 주석 옆에 격차를 기록했으며, 두 매니페스트 엔트리에 범위 한정을 명시했다.

### 2.4 검증된 불활성으로서의 고정 샘플러 순서

mlxcel의 체인 위치는 #1375/#1377이 b10621 기본 순서로 고정했으므로, `--samplers`, `--sampler-seq` / `--sampling-seq`, 네이티브 `samplers` 필드는 정확히 그 순서(불활성 설정, 업스트림 두 형태)만 받고 다른 순서는 모델 로드 전에 허용 형태를 안내하며 거부한다.

## 3. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경 파일 | 46개 (리뷰 후속 +15) |
| 라인 | 두 커밋 합산 +1335 / -311 |
| 매니페스트 | supported 26, not_applicable 3, #1485 재지정 32 |

검증: 코어 sampling 테스트 98개(윈도우 센티널, 커버링 윈도우 바이트 동일성, 윈도우 0 불활성과 fused 자격, DRY 센티널과 `base^0` 티어), xla 패리티 280개, 재빌드한 두 바이너리 대상 compat 통합, Qwen3-4B-4bit greedy 실검증(전체 히스토리 vs 커버링 윈도우 바이트 동일, 페널티 없는 윈도우 무영향, DRY `0` 무영향, 윈도우 64 처치군은 히스토리가 윈도우를 처음 넘는 76번째 단어에서 갈라짐). 추적되는 forward 경로는 변경 없음.

## 4. 후속 조치

- #1485가 잔여분을 소유한다: Mirostat, dynatemp, adaptive-p, grammar 표면, logit-bias, dry-sequence-breaker 문자열, min_keep, n_probs, post_sampling_probs, backend_sampling, temp/top-k/top-p 기본값 해석 차이, seed -1 미만 차이.
- 분산 와이어 프레임은 여전히 token bias(XTC, ignore-eos)를 버리고, 혼합 버전 핸드오프는 구버전 `dry_penalty_last_n: 0` 프레임을 비활성으로 읽는다. 모두 기록해 두었다.
