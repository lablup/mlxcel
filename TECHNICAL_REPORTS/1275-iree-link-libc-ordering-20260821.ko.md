# 기술 보고서: PR #1275 - IREE 아카이브 뒤에 libc 링크

## 요약

`--features cuda,xla-iree`로는 통합 테스트를 하나도 링크할 수 없었다. CUDA 호스트에서 OpenXLA 백엔드를 쓰는 가장 자연스러운 feature 조합인데도 그랬다. IREE 런타임 아카이브가 참조하는 `__stack_chk_guard`가 미해결로 남고, ld는 동적 링커를 "DSO missing from command line"으로 지목했다. 수정은 한 항목이다. `build.rs`의 `IREE_CUDA_HOME` 레시피에서 IREE 아카이브 **뒤에** `-lc`를 다시 놓는다.

이 보고서의 가치는 대부분 통하지 않은 것들에 있다. 진단을 세 번 연속으로 틀렸고, 최종 변경은 첫 시도보다 작다. 링크가 통과했다고 받아들이지 않고 첫 시도의 각 절반을 따로 제거해봤기 때문이다.

## 1. 문제

```text
/usr/bin/ld: libiree_runtime_unified.a(call.c.o): undefined reference to symbol '__stack_chk_guard@@GLIBC_2.17'
/usr/bin/ld: /lib/ld-linux-aarch64.so.1: error adding symbols: DSO missing from command line
```

`cargo check`는 링크를 하지 않으므로 같은 트리에서 `cargo check --features cuda,xla-iree --all-targets`는 통과했다. 실제로 테스트 바이너리를 만들 때만 드러났고, 그래서 `main`에 남아 있었다.

균일하지도 않았다. 같은 feature에서 `chat_template_kwargs`는 링크되고 `molmo2_xla_vision_parity`는 안 됐으며, 그 `molmo2_xla_vision_parity`도 `xla-diagnostics`에서는 링크됐다. 이 비균일성 때문에 앞선 두 진단이 그럴듯해 보였다.

## 2. 기술적 판단

### 2.1 에러 메시지가 해법의 반대편을 가리킨다

메시지가 `ld-linux-aarch64.so.1`을 지목하니 동적 링커를 링크하라는 뜻으로 읽힌다. `readelf`가 실제 위치를 확정해준다. `libc.so.6`은 `__stack_chk_guard`를 `UND`로 두고, 정의는 `ld-linux-aarch64.so.1`에 있다. 즉 libc는 심볼을 제공하지 않는다. 여기서 내린 최초 결론("`-lc` 추가로는 불가능하다")은 미묘하게 틀렸다. 수정에 필요한 것은 libc가 심볼을 정의하는 것이 아니라, ld가 libc의 `DT_NEEDED` 사슬을 따라갈 수 있는 위치에 libc가 놓이는 것이다.

### 2.2 라이브러리 누락이 아니라 순서

`rustc-link-arg`는 뒤에 붙이는 것만 가능하다. 따라서 IREE 아카이브는 rustc 자신의 `-lc` 뒤에 놓이고, `call.c.o`는 스택 보호기와 함께 컴파일되므로 `__stack_chk_guard` 참조가 뒤에 libc가 없는 지점에서 나타난다. 아카이브 뒤에 `-lc`를 반복하면 평범한 C 프로그램이 공짜로 얻는 순서가 복원된다.

### 2.3 정책 플래그는 ablation으로 걸러냈다

첫 시도는 ld가 전이적 해결을 거부한다는 가설로 `-Wl,--copy-dt-needed-entries`를 추가했다. 단독으로는 실패한다. rustc가 우리 인자를 자신의 `-lc` 뒤에 붙이므로 플래그는 80번에, `-lc`는 29번에 놓이고, 이 플래그는 자기 뒤에 오는 입력만 관장한다.

둘을 함께 넣으면 링크된다. 그대로 출하하는 대신 각 절반을 단독으로 시험했다.

| 구성 | 결과 |
| --- | --- |
| 뒤따르는 `-lc`만 | 링크됨 |
| `-Wl,--copy-dt-needed-entries`만 | 실패, 동일 에러 |
| 둘 다 | 링크됨 |

플래그는 불필요하고, 같은 링크 안에서 실제로 누락된 `-l`을 가릴 수 있는 전역 완화다. 둘을 함께 출하했다면 아무 이득 없이 그 위험만 늘었을 것이다. 이 보고서가 존재하는 주된 이유가 이것이다. **링크가 통과했다는 사실은 변경의 모든 부분이 제 몫을 했다는 증거가 아니다.**

### 2.4 형제 레시피는 건드리지 않는다

`IREE_DIST`는 거의 동일한 그룹을 내보내므로 같은 결함일 가능성이 높다. 그래도 바꾸지 않는다. 이 호스트에는 `IREE_DIST` 트리도 macOS도 없고, 링크 레시피는 아무도 링크해보지 않은 변경을 담아서는 안 된다. 이슈 #1274도 해당 레시피들을 변경하지 않거나 검증하거나 둘 중 하나로 명시했다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `build.rs` | `IREE_CUDA_HOME` 그룹에 `-lc` 추가, 순서 근거와 ablation 결과를 주석으로 기록 |
| `build.rs` | `link_args.insert(5, ...)`를 위치 기반 `push`로 교체. 앞에 항목을 추가해도 vendored printf 아카이브가 `--start-group` 밖으로 조용히 밀려나지 않는다 |
| `build.rs` | 그룹의 각 라이브러리에 존재 이유를 기록 |

## 4. 리뷰 지적사항

구현 중 자체 검토에서 커밋 전에 불필요한 정책 플래그를 걸러냈다. 외부 리뷰 지적사항은 없다.

매직 인덱스 제거는 이슈가 요구한 것이 아니다. 이번 변경이 같은 벡터에 항목을 추가했고, 만약 뒤가 아니라 앞에 추가했다면 기존 `insert(5, ...)`가 조건부 printf 아카이브를 조용히 옮겨놨을 것이기 때문에 함께 정리했다. 취향 문제가 아니라 다음 편집을 기다리는 결함이었다.

## 5. 검증

이미 통과하던 사례를 의도적으로 포함해 타깃 4개를 링크했다.

| 타깃 | feature | 이전 | 이후 |
| --- | --- | --- | --- |
| `molmo2_xla_vision_parity` | `cuda,xla-iree` | 실패 | 링크됨 |
| `cli_help_consistency` | `cuda,xla-iree` | 실패 | 링크됨 |
| `molmo2_xla_vision_parity` | `xla-diagnostics` | 링크됨 | 링크됨 |
| `chat_template_kwargs` | `cuda,xla-iree` | 링크됨 | 링크됨 |

여기에 `cargo check --features cuda --all-targets` 무경고와 `cargo fmt --all -- --check`를 더했다. 이미 통과하던 두 행은 구색이 아니다. 실패가 타깃마다 달랐으므로, 실패 타깃만으로 검증했다면 나머지를 망가뜨리고도 아무도 몰랐을 수 있다.

## 6. 관련 작업

이슈 #1274가 이 버그를 제기했고, 최초 주장 중 무엇이 살아남았는지 기록한 정정 절을 담고 있다. 이슈 #1270은 이것이 애초에 `main`에 도달하게 만든 CI 커버리지 공백을 추적한다. 어떤 워크플로도 XLA feature 조합을 컴파일하지 않는다.

미해결로 남긴 질문이 하나 있다. 실패가 왜 타깃별이고 feature별이었는가다. 작동하는 바이너리 둘은 `ld-linux-aarch64.so.1`을 명시적 `DT_NEEDED`로 갖고 실패하는 링크는 갖지 않으므로, 그 링크 라인들에서는 무언가가 이미 그것을 끌어오고 있었다. `surgery` 기본 feature가 의심됐지만 확인도 배제도 되지 않았다. 이번 수정이 모든 타깃에서 순서 의존성을 제거하므로 실무적 영향은 사라지지만, 구전으로 남기지 않기 위해 기록한다.
