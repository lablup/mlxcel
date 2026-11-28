# PR #2277: `split-mtp` 출력을 원자적·배타적으로 커밋하기

**날짜**: 2026-10-12
**상태**: 구현 및 리뷰 완료, 원본 GLM-4.7-Flash 체크포인트로 검증 완료
**위험도**: 중간 (체크포인트 출력 커밋 방식과 바이트 호환 safetensors writer 변경)

## 요약

PR #2277은 `mlxcel split-mtp`가 일부만 작성된 체크포인트를 노출하거나 출력 파일 심볼릭 링크를 따라가는 문제를 막습니다. 이제 명령은 출력과 같은 부모 아래의 비공개 staging 디렉터리에서 drafter 전체를 만들고, 모든 staging 파일을 배타적으로 생성한 뒤, 새 출력이나 빈 출력은 디렉터리 rename 한 번으로 게시합니다. 비어 있지 않은 출력에는 `--force`가 필요합니다.

내용이 있는 디렉터리는 이식성 있게 통째로 원자 교체할 수 없으므로 force 교체는 의도적으로 파일 단위로 유지합니다. 교체 직전에 목적지를 다시 검사하고, 외부 safetensors 파일과 심볼릭 링크인 drafter 경로를 거부하며, 오래된 safetensors index와 교체 대상인 알려진 drafter 파일만 다룹니다.

## 문제

기존 디렉터리 writer는 먼저 목적지 디렉터리를 만든 뒤 `model.safetensors`, `config.json`, companion 파일을 그 안에 바로 썼습니다. 직렬화, JSON 기록, 파일 복사에 기존 파일을 truncate하고 심볼릭 링크를 따라가는 API를 사용했습니다. 따라서 변환이 중단되면 불완전한 새 파일과 이전 파일이 섞였지만 체크포인트처럼 보이는 디렉터리가 남을 수 있었습니다. CLI 사전 검사 뒤에 다른 writer가 끼어들어 파일을 섞는 것도 가능했습니다.

출력 guard는 체크포인트 형태의 디렉터리는 인식했지만, 관련 없는 파일만 있는 비어 있지 않은 디렉터리는 `--force` 없이 허용했습니다. Staging을 도입하면서 `.`, `..`, Unix 비 UTF-8 이름, 변환 도중 조상 심볼릭 링크가 다른 곳을 가리키는 경우도 명시적으로 처리해야 했습니다.

## 설계

- Resolve한 출력과 같은 부모에 고유한 staging 디렉터리를 `create_dir`로 만듭니다. Unix에서는 mode를 `0700`으로 고정합니다. RAII guard가 모든 오류 뒤 staging을 제거합니다.
- Safetensors, config, companion 파일은 `create_new`를 사용합니다. Safetensors writer는 header와 tensor payload를 배타적으로 생성한 파일에 streaming하여 수 GB 크기의 두 번째 직렬화 버퍼를 만들지 않습니다.
- Writer는 safetensors 0.7의 dtype/name 정렬, offset, metadata, 8바이트 header padding을 그대로 재현합니다. 여러 이름과 f32, bf16, u32를 섞은 회귀 테스트가 `safetensors::serialize_to_file` 결과와 바이트 단위로 비교합니다.
- 출력이 없거나 비어 있으면 디렉터리 rename 한 번으로 커밋합니다. 그 사이 다른 writer가 비어 있지 않은 출력을 먼저 만들면 후발 writer는 실패하고 winner를 건드리지 않은 채 staging을 제거합니다.
- `--force`에서는 staged 파일을 개별 rename하기 전에 외부 weight, 7개 drafter 목적지, 목적지 타입, 오래된 index를 다시 검사합니다.
- 경로 해석은 Unix 비 UTF-8 이름을 보존하고 존재하지 않는 component를 만들지 않은 채 정규화합니다. 최초 resolve 결과를 staging guard에 저장하고, 커밋 시 resolve 결과가 달라지면 거부하며, 실제 기록은 저장한 target에만 수행합니다. 소스 체크포인트 alias 검사는 변환 전과 저장한 target을 대상으로 한 번 더 실행합니다.

## 사용자에게 보이는 변경

관련 없는 파일 하나만 있는 경우를 포함해 비어 있지 않은 모든 `--output`에는 이제 `--force`가 필요합니다. CLI help는 force를 사용해도 외부 safetensors 파일과 심볼릭 링크인 drafter 파일을 거부한다고 설명합니다. 새 출력과 빈 출력의 성공 결과는 같고, `.`, `..` 표기와 Unix 비 UTF-8 출력 이름도 계속 지원합니다.

## 검증

- Surgery 집중 단위 테스트: 26개 통과. 비공개 권한, cleanup, 동시 winner, force 교체, 정상·dangling 심볼릭 링크, 조상 심볼릭 링크 retarget, resolve되지 않은 `..`를 통한 source alias, 특수 경로 표기, 비 UTF-8 이름, 배타적 파일 생성, serializer 동등성을 포함합니다.
- CUDA feature CLI 집중 테스트: 10개 통과. 기본 non-empty 거부, force 규칙, 외부 weight, 심볼릭 링크, source alias 사전 검사, help 문구를 확인합니다.
- 포맷과 patch whitespace 검사가 통과했습니다. 집중 Clippy 결과는 PR test plan에 기록했습니다.
- 버전, kernel dtype, port dispatch, Llama 호환성, ROCm overlay, Python 3.12/3.13, WebUI 동작에 대한 이슈 contract 검사가 통과했습니다.
실제 체크포인트 검증에는 `zai-org/GLM-4.7-Flash` revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`을 사용했습니다. MTP 원본 tensor 212개가 든 shard 3개만 필요했습니다. `190b83c3` baseline과 최종 candidate 모두 `split-mtp --q-bits 4`를 실행해 출력 tensor 54개, quantized tensor 15개, 0.72 GB payload를 만들었습니다. `model.safetensors`와 `config.json`은 바이트 단위로 일치했습니다.

- weights SHA-256: `84018ae89ff88bf727f2552523748a21acb4e824d7f22bcf77d8a439d5f4dd61`
- config SHA-256: `7218072d09e5bb06e3eb005b2481ea0b1da08e7e7cd9ad59c69dbd132a754c24`

## 리뷰 결과

제출 전 리뷰에서 경로 관련 회귀 세 가지를 보완했습니다. Staging 이름은 더 이상 UTF-8을 요구하지 않고 `.`과 `..`를 올바르게 resolve합니다. Staging guard는 최초 resolve target에 커밋을 고정하고, 변환 도중 조상 심볼릭 링크가 다른 곳을 가리키면 이를 거부합니다. `model/new/..`처럼 아직 resolve되지 않은 경로도 `new`를 만들거나 소스 파일을 수정하지 않은 채 source alias로 인식합니다. 후속 보안 리뷰에서 이 범위의 추가 문제는 발견되지 않았습니다.
