# PR #3491: Qwen 3.5 MTP drafter layer에 YaRN rope scaling 적용 (issue #3479)

**작성일**: 2026-11-27

**상태**: `944dbfc2d`로 머지. 파일 15개, +664 / -143. M3 Ultra에서 로컬 검증했고, 원격 CI는 메인테이너 결정으로 취소했다. YaRN Qwen 3.5 체크포인트가 없어 acceptance 향상은 측정하지 못했다.

**위험도**: Low (target이 모든 token을 검증하므로 출력은 정확하다. YaRN이 없는 config는 텍스트와 acceptance가 같음을 확인했다)

## 요약

Qwen 3.5 MTP drafter는 `rope_parameters`가 YaRN을 선언해도 Q와 K를 plain `rope_theta`로 회전했고, target은 YaRN divisor table과 attention factor를 썼다. 그래서 drafter는 target이 검증할 때와 다른 position encoding으로 token을 제안했고, 긴 context의 YaRN 체크포인트에서 acceptance가 떨어졌다. 이제 drafter는 target과 같은 함수로 만든 같은 table과 factor를 받는다.

## 1. 문제 정의

- **position encoding 불일치.** upstream의 MTP head는 target의 YaRN 대응 attention layer를 재사용한다. `mlxcel-core`의 `Qwen35MtpAttention`은 무조건 `fast_rope(.., rope_base, ..)`를 호출했다.
- **보이지 않는 실패.** speculative decoding은 구조상 정확하므로 greedy 텍스트는 맞았다. 비용은 acceptance 하락뿐이었고 이를 확인하는 테스트가 없었다.
- **계층 구조.** YaRN table 빌더(`Qwen35Config::yarn`)는 바이너리 crate에 있고, drafter는 바이너리 crate에 의존할 수 없는 `mlxcel-core`가 로드한다.

## 2. 기술적 선택과 그 이유

### 2.1 core가 drafter를 로드한 뒤 바이너리 crate가 table을 설치

| 선택지 | 장점 | 단점 |
|---|---|---|
| YaRN 빌더를 `mlxcel-core`로 이동 | drafter가 자체 완결 | `rope_utils`와 `--yarn-*` override 경로를 crate 사이로 옮겨야 한다 |
| core에 빌더 복제 | diff가 작다 | 어긋날 수 있는 YaRN 계산이 두 벌 생긴다 |
| **선택: core는 `set_yarn_rope`만 노출하고 바이너리의 `drafter_loader`가 table 설치** | 빌더는 하나. CLI, 서버 drafter slot, `speculative_bench`가 이미 `drafter_loader::load_drafter`를 거친다 | core `load_drafter`를 직접 부르면 YaRN이 빠진다 |

`Qwen35Config::yarn`의 본문은 free function `qwen35_yarn(rope_parameters, rope_dims, rope_theta, max_position_embeddings)`가 됐다. target은 자기 config에서, `drafter_loader::qwen35_mtp_yarn`은 drafter 자신의 geometry로 이를 호출하므로 `--yarn-*` override가 양쪽에 모두 닿는다. `yarn`을 지정했지만 table을 만들 수 없는 block은 조용히 fallback하지 않고 target과 같은 오류 메시지로 drafter 로드를 실패시킨다.

### 2.2 target과 drafter가 공유하는 lane scaling 함수

transformers의 YaRN은 cos와 sin에 attention factor를 곱하므로, partial rotary head에서는 회전하는 lane만 scale된다. `fast_rope_with_freqs`에는 그런 multiplier가 없어서 호출 측이 회전 전에 앞쪽 `rope_dims` lane을 scale한다. 이 코드는 `Qwen3NextAttention` 안에 private으로 있었고, 이제 `mlxcel_core::rope_lanes::scale_rotary_lanes`로 옮겨 target과 drafter가 같이 호출한다. factor가 정확히 1.0이면 연산 없이 copy를 돌려준다.

```rust
fn rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
    match &self.rope_freqs {
        Some(freqs) => ffi::fast_rope_with_freqs(
            &scale_rotary_lanes(x, self.rope_dims, self.rope_attention_factor),
            self.rope_dims, false, 1.0, offset, freqs),
        None => ffi::fast_rope(x, self.rope_dims, false, self.rope_base, 1.0, offset),
    }
}
```

### 2.3 drafter config가 `max_position_embeddings`를 읽음

YaRN table은 원래 context 길이에 의존한다. drafter text config가 이 값을 파싱하지 않았으므로 이제 읽는다. 테스트는 SenseNova YaRN config에서 drafter table이 target table과 비트 단위로 같은지 확인한다.

## 3. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core/src/rope_lanes.rs` (신규) | `scale_rotary_lanes` |
| `mlxcel-core/src/drafter/qwen3_5_mtp/{layer,model,config}.rs` | 선택적 YaRN table과 factor, `set_yarn_rope`, `yarn_rope`, `config()`, `max_position_embeddings` |
| `mlxcel-core/src/drafter/qwen3_5_mtp/yarn_tests.rs` (신규) | table 유무에 따른 회전, lane scaling |
| `src/models/qwen3_5.rs` | free function `qwen35_yarn`, `Qwen35Config::yarn`은 이를 위임 |
| `src/models/drafter_loader.rs` | `load_qwen35_mtp`, `qwen35_mtp_yarn` |
| `src/models/qwen3_next.rs` | 자체 복사본 대신 `scale_rotary_lanes` 사용 |
| `src/models/qwen3_5_mtp_yarn_tests.rs` (신규) | drafter table과 target table 일치, loader의 설치, 잘못된 YaRN block의 로드 실패 |

| 커밋 | 메시지 |
|---|---|
| `de04e847f` | fix: apply YaRN rope scaling in the Qwen3.5 MTP drafter layer |

## 4. 검증

| 확인 | 결과 |
|---|---|
| `cargo test -p mlxcel-core ... drafter::qwen3_5_mtp rope_lanes` | 26개 통과, 기존 케이스 변화 없음 |
| `cargo test -p mlxcel ...` `qwen3_5` / `qwen3_next` / `rope_utils` / `drafter_loader` | 62 / 23 / 36 / 3개 통과 |
| greedy MTP 생성, `qwen3.8-27b-4bit`와 `qwen3.8-27b-mtp-4bit` (YaRN 없음), 64 token | 전후 텍스트와 acceptance 동일 (28 round, 35 / 56 수락, 0.625) |
| 두 crate의 clippy `-D warnings`, fmt | 경고 없음 |

## 5. 후속 조치

- YaRN Qwen 3.5 체크포인트가 생기면 긴 prompt로 acceptance를 측정한다. 기대 향상은 아직 검증되지 않았다.
- `mlxcel_core::drafter::load_drafter`를 직접 호출하면 YaRN 없는 drafter를 받는다. 현재 호출처는 모두 바이너리 wrapper를 거치며, 새 진입점도 그래야 한다.
