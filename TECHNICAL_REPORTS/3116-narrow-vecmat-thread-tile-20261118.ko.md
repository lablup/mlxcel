# 기술 보고서: PR #3116 - 좁은 디코드 vecmat에 출력당 스레드 하나 부여

**작성일**: 2026-11-18
**상태**: 완료
**언어**: Rust (MLIR emitter 및 테스트), MLIR (튜닝 스펙), Bash (측정)
**위험도**: 중간

---

## 요약

`gfx1151`에서 기본 f16 상주 디코드 그래프의 레이어별 프로젝션은 같은 트레이스의 128256-wide LM head가 153 GB/s를 내는 동안 약 42 GB/s에 머물렀다. 컴파일러가 모든 rank-1 vecmat에 `thread = [8, 0]`, `workgroup = [256, 0]`을 적용해서, 프로젝션이 N/256개의 원-워프 워크그룹을 실행했다. 2048-wide q, o, down은 8개, gate/up은 32개다. 20-WGP 디바이스에서 한 웨이브보다 적고, 직렬 K 리덕션으로는 메모리 지연을 가릴 수 없다. 이제 ROCm 튜닝 스펙은 `128 <= N <= 8192`인 모든 단일 행 f16 x f16 -> f32 수축에 출력당 스레드 하나와 `reduction = [0, 32]`를 부여하며, K와 무관하다. 짝지은 6라운드 A/B에서 디코드 스텝은 41.05에서 73.98 tok/s로 측정됐고(+80.25%, 6라운드 모두 우세, `text EQUAL`), `N >= 8193`이면서 `K >= 4096`인 경우에는 기존 와이드 타일이 남아 LM head의 설정을 유지한다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #2873은 OpenXLA/IREE ROCm 백엔드의 남은 디코드 커널 효율 격차를 다뤘다. 그중 argmax 항목은 #2875의 2단계 리덕션으로 이미 처리됐고, 스텝당 0.030 ms로 측정된다. 기본 f16 경로(`MLXCEL_XLA_QUANT` 미설정)에 남은 것은 small-M vecmat lowering이었다. 같은 컴파일러와 같은 트레이스에서 레이어별 프로젝션이 LM head가 내는 대역폭에 크게 못 미쳤다.

프로토콜은 다른 분해와 같다. `rocprofv3 1.3.5 --kernel-trace -f csv`, llama-3.2-1b-instruct-4bit, greedy, 용량 256, `n=1`과 `n=128` 두 실행을 커널별로 차분해 127로 나누어 공유 프리필을 상쇄한다. 변경 전(main `b21825cf`)의 비계측 스텝은 24.06 ms(41.6 tok/s)였고, 스텝당 디스패치는 279개, 트레이스 커널 시간은 24.44 ms였다.

### 1.2 기존 문제점

- **문제 1**: 컴파일러의 기본 rank-1 vecmat 타일은 워크그룹마다 워프 하나를 실행하므로, 2048-wide 프로젝션은 워크그룹 8개, gate/up은 32개를 실행한다. 20 WGP 디바이스에서 한 웨이브에 못 미친다.
- **문제 2**: 그리드가 이렇게 작으면 스레드 내부의 직렬 K 리덕션이 메모리 지연을 가릴 수 없어, q/o가 42 GB/s, 같은 실행의 head가 153 GB/s로 측정된다.
- **문제 3**: 기존 튜닝 스펙은 장기 K 단일 행 수축(`K >= 4096`)만 매치해 리덕션 청크만 키웠으므로, 히든 크기가 2048인 모델의 q, o, gate/up에는 닿지 못했다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 매처가 파싱은 되지만 매치를 멈춤 | 높음 (기본 타일로 컴파일되고 처리량만 바뀜) | `838c19cf`가 추가한 컴파일러 기반 테스트로 처리 |
| 타일이 LM head에 닿아 activation 재읽기가 커짐 | 중간 (가장 큰 커널에서 회귀) | 낮음: N 상한이 `umax = 8192`이고 와이드 매처와 서로소 |
| 타일이 ragged 다중 행 그래프에 닿음 | 중간 (한 행용으로 측정한 설정이 N개 행에 적용) | 낮음: 배치 술어가 빈 배치 차원을 요구 |
| 리덕션 청크 변경이 합산 순서를 바꿈 | 높음 (greedy 출력 변경) | 낮음: 두 타일 모두 생성 어셈블리에서 오름차순 K로 직렬 누산하고, 모든 짝지은 실행에서 `text EQUAL` |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

새로운 입력, 네트워크, 파일 표면은 없다. 스펙은 `device == "hip"`이고 런타임 설정이 `RuntimeConfig::Dense(_)`일 때만 적용되는 정적 애셋이다. 잘못된 스펙은 조용히 넘어가지 않고 컴파일을 실패시킨다. 보안 리뷰에서는 보안 문제가 발견되지 않았다.

**체크리스트:**
- [x] 입력 검증 (변경 없음, 모델·요청·환경 입력이 매처에 닿지 않음)
- [x] 인증/인가 (해당 없음)
- [x] 데이터 암호화 (해당 없음)
- [x] 로깅 (모델 데이터 없음)

**발견 사항:** 없음.

### 2.2 성능 관점

`gfx1151`, llama-3.2-1b-instruct-4bit, 용량 256, `MLXCEL_XLA_QUANT` 미설정, `rocprofv3` 차분:

| 측정 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 디코드 스텝 (비계측) | 24.06 ms (41.6 tok/s) | 13.54 ms (73.5~74.5 tok/s) |
| 트레이스 커널 시간 | 24.44 ms | 12.48 ms |
| 스텝당 디스패치 | 279 | 279 |
| `vecmat_2048x2048` (q, o) | 42 GB/s | 219 GB/s |
| `vecmat_512x2048` (k, v) | 44 GB/s | 유효 81 GB/s (재읽기 포함 162 GB/s) |
| `vecmat_8192x2048` (gate, up) | 140 GB/s | 196 GB/s |
| `vecmat_2048x8192` (down) | 104 GB/s | 230 GB/s |
| `vecmat_128256x2048` (LM head, 설정 변경 없음) | 153 GB/s | 228 GB/s |
| argmax (2단계 리덕션) | 0.031 ms | 0.030 ms |

두 트레이스 사이 head의 153~228 GB/s 차이는 변경이 아니라 트레이스의 클록 상태다. head의 설정은 N 상한으로 제외된다. 같은 실행의 대조군인 head를 기준으로 읽으면 q/o는 head 대역폭의 0.27에서 0.96으로, down은 0.68에서 1.01로 이동한다. gate/up은 0.92에서 0.86으로 이동하는데, 이 변화는 변경되지 않은 head가 두 트레이스 사이에 기록한 1.49배보다 작다. 따라서 gate/up의 140~196 GB/s 절대 변화는 클록 이동과 분리할 수 없고, 보고서도 그렇게 적는다.

보고할 효과 크기는 짝지은 교차 A/B다(`scripts/xla/ab_decode_pair.sh`, 같은 모델·용량·토큰 예산, 6라운드). 41.05에서 73.98 tok/s, +80.25%, 6라운드 모두 새 arm이 우세했고 `text EQUAL`이었다.

untied qwen2.5-coder-7b-instruct-4bit(히든 3584, intermediate 18944, 28레이어)의 비계측 디코드는 12.28에서 15.44~15.46 tok/s로 측정됐고, greedy 출력은 48토큰 동안 동일하다. K를 2048로 제한한 별도 단기 K 매처를 둔 첫 버전은 이 체크포인트에서 down에만 닿아 12.94 tok/s였다. q와 o는 K = 3584라서 컴파일러 기본값에 남았다.

### 2.3 호환성/의존성 관점

- **호환성 파괴**: 없음. 튜닝 스펙은 모델 ABI의 일부가 아니고, 디스패치가 늘거나 줄지 않으며, 가중치 레이아웃과 리덕션 순서도 그대로다.
- **새 의존성**: 없음.
- **호환성**: 스펙은 이전과 같이 `hip` dense 그래프에만 적용된다. ragged 및 균일 배치 그래프는 빈 배치 술어로 컴파일러 기본값을 유지하므로 커밋된 골든도 바뀌지 않는다. 프리필 수축은 rank-1이 아니라 매치될 수 없다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 텍스트 테스트가 엔트리포인트 배선, 좁은 타일, 출력 폭 경계, 리덕션 축에 경계가 없음을 고정한다. 다른 테스트는 와이드 폴백의 두 술어와 두 술어의 서로소 관계를 고정한다. `#[ignore]` 처리된 컴파일러 기반 테스트는 단일 시퀀스 디코드 그래프를 생성해 고정된 스펙으로 컴파일하고, 실행 파일 덤프에서 선택된 설정을 읽는다. 버전이 맞는 ROCm `iree-compile`이 필요해서 ignore 상태이며, 보안 리뷰에서 정상 통과하고 `umax`를 64로 좁히면 실패함을 확인했다.
- **코드 복잡도**: 기존 매처 하나를 교체하고 장기 K 동작을 유지하는 매처 하나를 더했다. 두 술어는 서로소이므로 `transform.foreach_match`에서 순서가 결과를 바꾸지 않는다.
- **기술 부채**: 감소. 스펙 주석이 리덕션만 다루던 부분 설명 대신 측정된 메커니즘과 정확한 술어를 기술한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 리덕션 청크 확대가 아니라 출력당 스레드 하나

**맥락:** 프로젝션은 그리드 크기(원-워프 워크그룹 8~32개)에 제한됐고, 리덕션 청크는 스레드 내부 작업만 바꾼다.

**검토한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 리덕션 청크를 더 키움 | 출력 매핑 변경 없음 | 워크그룹이 늘지 않아 그리드가 한 웨이브 미만에 머묾 |
| **선택: `thread = [1, 0]`, `workgroup = [32, 0]`, `reduction = [0, 32]`** | N/256이 아니라 N/32 워프로 디바이스를 채움 | 출력당 스레드 하나가 출력마다 activation 행을 재읽기 |
| K를 워크그룹에 분할 | K 방향 병렬성 증가 | 표현 불가: `SetSplitReductionSizes`는 병렬 루프가 둘 이상인 matmul형 연산을 요구하고 rank-1 vecmat에는 하나뿐 |

**이유:** 제약은 리덕션 타일이 아니라 점유율이었다. 실행이 디바이스를 채우면 q/o와 down이 같은 실행 head의 대역폭에 도달한다.

**트레이드오프:** 스레드마다 activation 행을 재읽기한다. 이 비용은 N에 비례해 커지므로 매치를 8192에서 멈추고 LM head는 이전 설정에 남긴다.

### 3.2 K가 아니라 N을 경계로 둠

**맥락:** 타일의 점유율은 출력 개수가 정한다. 모든 출력은 여전히 K를 직렬로 순회하므로 타일은 리덕션 길이에 의존하지 않는다.

**검토한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 기존 매처처럼 `K >= 4096` 게이트 유지 | 보수적 | 히든 3584 모델의 q와 o가 기본 타일에 남아 첫 버전에서 0.66 tok/s 손실 |
| **선택: 모든 K, `128 <= N <= 8192`** | 히든 2048과 3584를 매처 하나로 처리 | head를 제외할 N 상한 필요 |

**이유:** 장기 K와 단기 K 매처가 같은 타일을 갖게 되면서 K 게이트는 하는 일이 없어졌다. N 하한 128은 매우 좁은 프로젝션을 기본값에 남기고, N 상한 8192는 activation 재읽기가 N에 비례해 커지는 LM head를 제외한다.

### 3.3 서로소 술어 뒤에 와이드 장기 K 설정 유지

**맥락:** 기존 `thread = [8, 0]`, `workgroup = [256, 0]` 설정은 넓은 장기 K head에 여전히 맞다.

**이유:** `N >= 8193`은 좁은 매처가 멈추는 지점에서 정확히 시작하고(IREE의 `umax`는 닫힌 구간), `K >= 4096`은 llama의 K = 2048 head에 와이드 타일이 닿지 않게 한다. 그 head는 의도적으로 컴파일러 기본값에 둔다. 두 구간이 서로소이므로 엔트리포인트 순서는 결과와 무관하다.

---

## 4. 구현 세부 사항

### 4.1 주요 코드 변경

**파일: `src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`**
```mlir
transform.named_sequence @match_single_row_vecmat(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
    : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
  %rhs = transform.get_operand %arg0[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.dim_bounds %rhs[1], umin = 128, umax = 8192 : !transform.any_value
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{reduction = [0, 32], thread = [1, 0], workgroup = [32, 0]}>, translation_info = <pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 1, 1] subgroup_size = 32>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
```
**변경 이유:** 이 타일은 수축의 출력 매핑을 일대일로 유지하면서 실행을 N/256 워프에서 N/32 워프로 늘린다.

**파일: `src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir`, 와이드 폴백**
```mlir
transform.iree.match.dim_bounds %rhs[0], umin = 4096, none : !transform.any_value
transform.iree.match.dim_bounds %rhs[1], umin = 8193, none : !transform.any_value
```
**변경 이유:** N 하한을 더해 두 vecmat 구간을 서로소로 만들었다. 와이드 타일은 좁은 출력에 닿지 않고, 좁은 타일은 head에 닿지 않는다.

**파일: `src/lib/mlxcel-xla/src/tuning.rs`, `src/lib/mlxcel-xla/src/iree.rs`**
애셋 주석, `ROCM_BIGK_VECMAT_TUNING` 문서 주석, 컴파일 지점 주석이 측정된 메커니즘과 새 구간을 기술하고, 테스트가 두 매처를 고정한다.

### 4.2 데이터 모델 변경

없음. 스펙은 텐서나 런타임 ABI가 아니라 컴파일러 입력을 바꾼다.

---

## 5. 배운 점

### 5.1 좁은 수축에서는 점유율이 첫 번째 타일 결정이다

**개념:** rank-1 수축에서 출력 개수가 그리드를 정한다. 워크그룹마다 워프 하나를 주는 타일은 N이 디바이스를 채울 만큼 커질 때까지 디바이스 일부를 놀린다. 스레드 내부의 리덕션 타일을 어떻게 잡아도 그리드는 바뀌지 않는다.

**이 PR에서:** q/o와 down은 실행이 모든 출력을 워크그룹의 레인으로 사용하면서 42와 104 GB/s에서 219와 230 GB/s로 이동했다. 128256-wide head는 기존 타일로도 이미 그 대역폭이었는데, 그리드가 이미 컸기 때문이다.

### 5.2 분리된 트레이스에는 같은 실행의 대조군이 필요하다

**개념:** 다른 시점에 찍은 두 트레이스는 클록 상태도 다르다. 설정이 바뀌지 않은 커널이 변경과 클록을 분리하는 대조군이 된다.

**이 PR에서:** 변경되지 않은 head는 한 트레이스에서 153 GB/s, 다른 트레이스에서 228 GB/s다. 커널별 열을 이 대조군으로 읽었기 때문에 gate/up을 40퍼센트 개선이 아니라 클록 이동과 분리할 수 없는 값으로 적었고, 보고할 효과 크기를 짝지은 A/B로 삼았다.

### 5.3 텍스트 테스트는 매치를 멈춘 매처를 볼 수 없다

**개념:** 튜닝 스펙은 파싱되고 적용되면서 아무것도 매치하지 않을 수 있다. 컴파일은 성공하고 처리량만 바뀌므로, 애셋을 텍스트로 읽는 단위 테스트로는 잡을 수 없다.

**이 PR에서:** 컴파일러 기반 테스트가 실제 디코드 그래프를 생성하고 스펙으로 컴파일한 뒤, 덤프된 설정을 단언한다. 컴파일러 호출 비용이 있어 `#[ignore]`로 두고 필요한 편집 지점을 문서에 적었다.

---

## 6. 변경 요약

### 통계
| 항목 | 값 |
|------|-------|
| 변경 파일 | 6 (코드와 문서, 보고서 파일 2개 제외) |
| 추가 줄 | +308 |
| 삭제 줄 | -29 |
| 추가 테스트 | 2 (하나는 컴파일러 기반이며 `#[ignore]`) |

### 범주별 변경

| 범주 | 개수 | 요약 |
|------|------|------|
| 성능 | 2 | 좁은 단일 행 타일, N 하한을 명시한 와이드 장기 K 폴백 |
| 테스트 | 2 | 좁은·와이드 술어 테스트, 실행 설정 관측 |
| 문서 | 3 | 크레이트 README 분해, 튜닝 스펙과 컴파일 지점 주석, 환경 변수 기준선 |
| 호환성 | 1 | 술어로 ragged 그래프와 프리필 제외, ABI 변경 없음 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|------|
| `051ffa9c` | perf | give the narrow decode vecmats one output per thread |
| `8beb4208` | test | pin the wide vecmat predicates and correct the tuning notes |
| `1a2814fa` | docs | name the authoritative figure for the #2873 breakdown |
| `838c19cf` | test | observe the vecmat tuning spec on the emitted decode graph |

---

## 7. 후속 조치

### 필수
- [ ] 없음.

### 모니터링 필요
- `hip` 기본 f16 경로의 장기 컨텍스트 디코드: q/o와 down이 LM head 대역폭 근처에 머물고, gate/up이 이전 비중 아래로 떨어지지 않아야 한다.

### 향후 개선
- k와 v는 N = 512라서 출력당 스레드 하나로는 16 워프에 그쳐 유효 81 GB/s에 머문다. 이를 올리려면 k/v 가중치를 업로드 시점에 연결하는 레이아웃이 필요하고, 스텝당 0.8 ms가 남는 여지를 약 0.5 ms로 제한한다.
- rank-1 수축에 대한 워크그룹 수준 K 분할은 튜닝 스펙으로 표현할 수 없다. 그래프나 연산자를 바꿔야 가능하다.

---

## 부록

### A. 테스트 결과

- `cargo test -p mlxcel-xla --lib tuning:: -- --include-ignored`: 5개 통과 (텍스트 4개, 컴파일러 기반 1개).
- `cargo test -p mlxcel-xla --lib emitter::`: 216개 통과.
- `cargo fmt -p mlxcel-xla -- --check`: 통과.
- `cargo clippy -p mlxcel-xla --all-targets -- -D warnings`: 통과.
- `scripts/xla/validate_arch.sh` (`hip`): PASS (구조, 토큰 동일, serve `REFERENCE-EXACT` 16/16).

### B. 성능 벤치마크

`scripts/xla/ab_decode_pair.sh`를 통한 짝지은 교차 A/B, `gfx1151`, llama-3.2-1b-instruct-4bit, 용량 256, 6라운드.

| arm | 중앙값 (tok/s) | 변화 | 우세 라운드 | 출력 |
|-----|----------------|-------|------------|------|
| before | 41.05 | | | `text EQUAL` |
| after | 73.98 | +80.25% | 6/6 | `text EQUAL` |

untied 체크포인트, 비계측 단일 실행: qwen2.5-coder-7b-instruct-4bit 12.28에서 15.44/15.46 tok/s(+26%), greedy 출력은 48토큰 동안 동일.

### C. 참조
- 이슈 #2873(디코드 커널 효율), #2875(2단계 argmax), #2921과 #2956(grouped-query 디코드 매처).
- PR #3106(native 경로 디코드 분해, 이 보고서가 재사용하는 측정 프로토콜).
- `src/lib/mlxcel-xla/README.md`의 "Decode step breakdown (the default f16 path, issue #2873)" 절.
