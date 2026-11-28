# 기술 보고서: PR #1752 - feat(core): 어텐션 캐시 계열이 KV 스냅샷 직렬화를 공유하도록 정리

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 머지 전(테스트 44개 추가, clippy와 fmt 통과. M5 Max에서 `mlxcel-server`로 실제 체크포인트 세 개 실행. 기능은 들어가지만 512 MiB 스냅샷 버킷에 막히는데 Llama 4 두 턴이면 이미 다 찬다. #1761로 등록)
**언어**: Rust, Markdown
**위험도**: 중간(계열 셋이 없던 요청 간 상태 경로를 얻고 그 과정에서 Gemma 4와 Muse Glimmer가 이미 의존하던 코드의 결함 두 개를 고쳤다. 저장되는 텐서 레이아웃은 바뀌지 않는다)

---

## 요약

`mlxcel-server`의 프롬프트 캐시 재사용에는 생산자가 둘 있다. KV가 스케줄러의 `CachePool`에 사는 계열은 라딕스 접두 매칭과 부분 블록 채택을 쓰고 캐시를 `SequenceId` 단위로 직접 들고 있는 계열은 정확 접두 `ModelStateSnapshot` 복사만 쓸 수 있으며 그마저도 `LanguageModel::supports_snapshot_reuse()`로 참여를 선언해야 한다. 참여하려면 캐시 타입별 텐서와 스칼라 레이아웃을 계열마다 손으로 써야 했고 그래서 Gemma 4와 Muse Glimmer 둘만 해 두었다. Gemma 3, AFMoE, Llama 4는 평범한 `KVCache`, `RotatingKVCache`, `ChunkedKVCache` 상태를 `ModelOwnedSequenceState`에 들고 있으면서 프롬프트 캐시 경로 전체에서 빠져 있었다. 이 계열들에서는 멀티턴 요청마다 대화 전체를 토큰 0부터 다시 프리필했다는 뜻이다. 이 PR은 레이아웃을 `src/models/kv_snapshot.rs` 한 곳으로 옮기고 계열 셋을 참여시키며 Gemma 4와 Muse Glimmer는 예전부터 쓰던 텐서 이름 그대로 같은 코드를 호출하게 만든다.

이 모듈은 바이트 레이아웃만이 아니라 절단 계약까지 소유한다. 계열마다 베껴 쓸 때 틀리는 부분이 바로 절단 규칙이기 때문이다. `KVCache`는 언제나 절단할 수 있고 `RotatingKVCache`는 링이 감기지 않은 동안만, `ChunkedKVCache`는 앞쪽을 잘라내지 않은 동안만 절단할 수 있다. Gemma 3로 실제 대화를 끝까지 돌리는 과정에서 결함 두 개가 드러났고 둘 다 우회하지 않고 원인 자리에서 고쳤다. `RotatingKVCache`의 `CacheInterface::live_len`이 프리필 마스크 폭을 가시 윈도가 아니라 물리 링 버퍼에서 잡고 있었고 `restore_fp16_snapshot_state`가 윈도를 넘긴 첫 턴이 정상적으로 만들어내는 `idx > max_size` 모양을 거부하고 있었다. 앞의 것은 복원 이후 첫 append에서 MLX가 던지는 shape 예외이고 뒤의 것은 윈도를 넘긴 첫 턴을 전부 조용히 콜드 프리필로 보냈다. 슬라이딩 윈도가 1024인 4B Gemma 3에서는 실제 대화 대부분이 여기 걸린다.

머지되는 범위는 이 diff 밖으로 일부러 밀어낸 후속 과제 하나에 묶여 있다. 스냅샷 버킷은 여전히 순환 상태용으로 잡힌 고정 512 MiB 기본값을 쓴다. 순환 상태는 컨텍스트에 대해 O(1)이지만 어텐션 캐시는 O(컨텍스트)다. Llama 4 Scout는 토큰당 192 KiB쯤 저장하므로 검증 실행에서 실측한 엔트리는 1817토큰에서 384 MiB, 2352토큰에서 480 MiB였다. 같은 대화의 세 번째 턴은 저장 자체가 안 되고 거부는 아무 표시도 남기지 않는다. 이것이 #1761이고 그 전까지 이 기능은 짧은 대화에서 동작하고 정작 가속하려던 긴 대화에서 멈춘다.

---

## 1. 참여 선언의 비용

게이트는 양쪽에서 확인하는 트레이트 메서드 하나다. model-owned 계열의 기부는 `supports_snapshot_reuse()`가 참을 답하지 않으면 곧바로 반환하고 조회도 같은 이유로 스토어를 보기 전에 `SnapshotLookupOutcome::NoCandidate`를 낸다. 어느 쪽도 계열 이름을 부르는 거절 로그를 남기지 않으므로 Gemma 3 운영자가 `/v1/cache/stats`에서 본 것은 히트가 0인 멀쩡한 프롬프트 캐시였고 이유를 알려 주는 것은 없었다.

참을 답하려면 메서드 네 개가 더 필요하고 그 뒤에 캐시 타입별 직렬화기가 붙는다. Gemma 4가 `Standard`와 `Rotating`용으로 하나를 들고 있었고 Muse Glimmer가 텐서 이름만 다른 사본을 하나 더 들고 있었으며 `ChunkedKVCache`용 직렬화기는 트리 어디에도 없었다. 사본 둘은 에러 문자열이 이미 어긋나 있었고 `KVCacheMode` 태그 헬퍼는 `gemma4.rs`와 `recurrent_snapshot.rs`에 각각 private으로 두 벌 있었다. 계열을 하나 더 붙인다는 것은 같은 150줄을 세 번째로 베끼는 일이었고 거기에는 절단 판정도 포함된다. 절단 판정은 틀린 답이 실패가 아니라 생성 오염으로 나타나는 유일한 부분이다.

---

## 2. 공용 모듈

`src/models/kv_snapshot.rs`는 캐시 타입마다 함수를 네 개씩 연다. `snapshot_*`은 `layer{idx}` 접두 아래 텐서와 스칼라를 쓰고 `restore_*`은 되읽으며 `*_truncatable_to`는 살아 있는 캐시가 아니라 스냅샷 자신의 스칼라만 보고 짧은 복원이 타당한지 답하고 `truncate_*`이 실제로 잘라낸다. `kv_cache_mode_to_i32` / `kv_cache_mode_from_i32` 쌍은 `gemma4.rs`에서 빠져나와 이미 private 중복본을 갖고 있던 `recurrent_snapshot.rs`에서 `pub(crate)`로 승격됐다. #1335의 구현 계획은 이 쌍을 새 모듈에 두라고 했으니 작은 이탈인데, 있던 것을 승격하면 세 번째 거처를 만드는 대신 중복 하나가 사라진다. 옮기면서 에러 메시지의 Gemma 4 라벨도 떼어냈다.

### 2.1 캐시 타입 셋, 절단 규칙 셋

절단이 중요한 이유는 N토큰짜리 스냅샷이 그 N의 진부분 접두인 프롬프트 요청도 서빙할 수 있기 때문인데, 짧아진 캐시가 짧은 프롬프트를 콜드 프리필한 결과와 구분되지 않을 때만 그렇다. 세 타입의 답이 다르고 이유는 정책 선택이 아니라 기하다.

| 캐시 타입 | 절단 가능 조건 | 이유와 절단 방식 |
|---|---|---|
| `KVCache` | 항상, `0 <= target_len <= offset` 범위에서 | 토큰마다 자기 슬롯을 지키므로 꼬리를 버리는 것이 곧 되감기다. `KVCache::trim`이 `offset`을 논리적으로 되감고 물리 버퍼는 용량을 그대로 두어 다음 갱신이 덮어쓴다. |
| `RotatingKVCache` | `buffer_size == 0 && idx == offset && offset <= max_size` | 링이 한 번 감기면 논리 토큰 `t`가 더는 슬롯 `t`에 있지 않다. `RotatingKVCache::trim`은 `offset`과 `idx`만 되감으므로 슬롯과 위치가 일치하는 동안에는 맞고 어긋난 뒤에는 조용히 틀린다. 그래서 `truncate_rotating`은 `is_trimmable()`로 감긴 링을 아예 거부한다. |
| `ChunkedKVCache` | `start_position == 0`이고 `target_len <= offset` | 앞쪽을 잘라낸 뒤의 버퍼는 `[start_position, offset)`을 들고 있는데 `target_len`토큰을 콜드 프리필하면 `[target_len - chunk_size, target_len)`을 들게 된다. `start_position > target_len - chunk_size`인 동안 절단 복원은 콜드 실행보다 엄격히 적은 토큰을 어텐드한다. 절단 자체도 되감기가 아니라 물리 슬라이스다. `ChunkedKVCache`에는 `trim`이 없고 `update_and_fetch`가 버퍼 길이를 `get_buffer_size`로 되읽으므로 논리적 되감기만으로는 버려진 꼬리가 다음 성장 판단에 그대로 보인다. |

정확 접두 복원은 세 규칙 어디에도 걸리지 않는다. 전체 길이 복원은 감긴 링이든 앞이 잘린 버퍼든 캡처 당시 그대로 재현하고 버리는 것이 없기 때문이다. 거절하는 것은 짧은 접두 쪽뿐이다. Gemma 3가 링이 진작 감긴 3번째 턴에서도 복원에 성공하면서(6.2절) 같은 감긴 링으로의 절단은 거부하는(6.3절) 이유가 이것이다.

혼합 레이어 계열을 떠받치는 구현 세부가 하나 있다. `*_truncatable_to`는 자기 텐서가 없으면 참을 돌려주므로 `Cache::snapshot_truncatable_to`가 rotating과 standard 판정을 논리곱으로 묶어도 결과는 그 레이어가 실제로 저장한 쪽으로 줄어든다. Llama 4도 chunked와 standard 쌍으로 같은 일을 한다. 대안이었던 레이어별 타입 태그를 스냅샷에 넣어 되읽는 방식은 계열 둘이 이미 배포한 포맷에 스칼라를 더하는 일이 된다.

### 2.2 이름 파라미터가 지키는 것

텐서 이름은 계열의 스냅샷 계약 중 전송되는 부분이고 이 모듈보다 먼저 나간 두 계열은 같은 캐시 타입 둘을 다르게 적는다. Gemma 4는 `standard` / `rotating`을 쓰고 Muse Glimmer는 `full` / `sliding`을 쓴다. `KvSnapshotNames` 값이 세그먼트 이름 둘과 에러 메시지용 계열 라벨을 실어 나르므로 두 계열 모두 예전과 바이트 단위로 같은 스냅샷을 만들고 예전 빌드가 쓴 스냅샷도 그대로 복원한다. `chunked`는 파라미터가 아니라 상수다. Llama 4가 유일한 보유자이고 보존할 과거 표기가 없다.

AFMoE는 Gemma 3의 `Cache` 열거형을 재사용하면서도 자기 `KvSnapshotNames`를 넘긴다. 세그먼트 이름은 Gemma 3와 같으니 차이는 에러 문자열의 라벨뿐이고 계열 구분은 스냅샷의 family 태그가 복원 시점에 이미 한다. 라벨을 따로 넘기는 값은 계열당 상수 하나이며 그 대가로 AFMoE의 실패가 Gemma 3에서 온 것처럼 말하지 않는다.

하위 호환은 이름보다 한 겹 더 내려간다. `restore_rotating`의 모든 스칼라는 필드가 없으면 살아 있는 캐시의 값으로 물러서고 `snapshot_mode`는 태그가 없으면 `Fp16`을 기본으로 잡는다. 태그가 생기기 전에 쓰인 스냅샷은 전부 FP16이었기 때문이다. 그래서 예전 빌드의 스냅샷은 필드 누락으로 거부당하지 않고 복원된다.

### 2.3 양쪽 끝 모두 FP16만

양자화 모드는 스냅샷 컨테이너가 모델링하지 않는 사이드카 버퍼를 달고 다니므로 스냅샷과 복원 양쪽에서 `KVCacheMode::Fp16`이 아닌 것을 거부하고 복원은 저장된 모드와 살아 있는 캐시의 설정 모드가 일치할 것을 추가로 요구한다. 거부는 에러 문자열이고 호출자는 양자화 버퍼를 FP16으로 재해석하는 대신 콜드 프리필로 물러선다. `ChunkedKVCache`에는 양자화 변형이 아예 없어서 Llama 4의 chunked 레이어는 구조상 FP16이고 양자화 요청은 캐시 생성 시점에 한 번 경고한다.

`restore_rotating`은 윈도 폭도 같은 방식으로 다룬다. `max_size`는 시퀀스 상태가 아니라 모델 설정이므로 살아 있는 캐시와 다른 윈도를 실은 스냅샷은 조용히 윈도를 다시 여는 대신 거부한다. `restore_chunked`가 `chunk_size` 일치를 요구하는 이유도 같다.

---

## 3. 원인 자리에서 잡은 결함 둘

살아 있는 프리필 전용 시퀀스로는 어느 쪽도 도달할 수 없었고 그래서 둘 다 살아남았다. 앞의 것은 AFMoE가 함께 쓰는 Gemma 3의 마스크 경로에 있고 뒤의 것은 `mlxcel-core`의 복원 경로에 있다. 뒤쪽은 Gemma 4와 Muse Glimmer를 포함해 rotating 스냅샷을 쓰는 계열 전부가 지나간다.

### 3.1 `live_len`이 마스크 폭을 물리 링에서 잡았다

`RotatingKVCache`의 `CacheInterface::live_len`은 물리 버퍼 길이인 `seq_len()`을 돌려주었는데 슬라이딩 프리필 마스크는 `visible_len()`에서 폭을 잡아야 한다. `update_in_place`가 디코드 중 링을 `step`(256) 블록씩 키우므로 토큰을 하나라도 생성한 캐시는 `physical > offset` 상태가 되고 `update_concat`은 새 키 앞에 `visible_len()`개의 이전 키만 이어 붙인다. 물리 길이로 마스크를 잡으면 반환된 K/V보다 넓어져 디코드 이후 첫 다중 토큰 append에서 `broadcast_shapes`가 걸린다.

그 모양이 정확히 스냅샷 복원 다음에 오는 추가 토큰 프리필이고 서버의 다른 어떤 경로도 그 모양을 만들지 않는다. 살아 있는 프리필 전용 시퀀스는 concat이 과할당하지 않으므로 `physical == offset`으로 남는다. 접근자는 이제 `visible_len()`이고 기존 마스크 테스트가 볼 수 없던 디코드로 자란 경우는 `live_len_matches_the_keys_a_multi_token_append_returns`가 고정한다. 갓 concat한 캐시에서는 두 값이 같아서 기존 테스트로는 구분이 되지 않았다.

### 3.2 복원 경계가 윈도를 넘긴 첫 턴을 전부 거부했다

`restore_fp16_snapshot_state`는 `buffer_size == 0`일 때 `idx > max_size`를 거부했다. 손상된 쓰기 위치를 잡으려는 의도였고 rotating 캐시가 윈도 하나보다 많이 들고 있을 일이 없다면 맞는 경계다. 그런데 들고 있다. `update_concat`은 윈도보다 긴 프리필 뒤에 `max_size`보다 많이 저장하고 `idx`를 저장 길이에 맞춰 고정하므로, 그 프리필과 다음 단일 토큰 스텝 사이에 캡처된 스냅샷은 `idx == physical_len > max_size`를 정당하게 싣는다.

그래서 예전 검사는 윈도를 넘긴 첫 턴을 전부 콜드 프리필로 보냈고 복원 실패는 버그가 아니라 거절로 보고됐다. `models/gemma-3-4b-it-4bit`의 슬라이딩 윈도는 1024이고 검증 대화의 첫 턴은 프롬프트 1323토큰이므로 공용 직렬화기는 정작 그것을 위해 쓰인 계열에서 죽어 있었을 것이다. 경계는 이제 `idx <= physical_len`이다. 진짜로 손상된 `idx`는 여전히 걸리고 버퍼가 `max_size`를 향해 자라는 동안에는 이쪽이 더 빡빡한 검사다. `rotating_round_trip_survives_an_over_window_prefill`이 지킨다.

---

## 4. 범위를 한정한 거절 둘

### 4.1 패딩 프리필

Gemma 3, AFMoE, Llama 4는 `supports_padded_prefill()`에 `false`를 답하고 Gemma 4의 기존 거절에도 같은 설명이 붙었다. 이유는 이 모델들의 성질이 아니라 빠진 훅이다. 계약은 NA 타일 정렬 패딩 청크 뒤에 캐시를 실제 프롬프트 길이로 되잘라 달라고 요구하고 배치 스케줄러는 `CachePool`의 `Vec<KVCache>`를 잘라 그 요구를 지키는데 `model_owned` 계열의 풀 엔트리는 `caches` 벡터가 빈 `SequenceCacheSet::model_owned`다. 트림이 닿는 곳이 없으니 패드 위치가 모델 자신의 캐시에 남고 `offset`이 실제 토큰 수보다 패드 폭만큼 앞선다. 이 경로는 `should_align_prefill()`이 가드하므로 결함은 M5 전용이다.

결과는 스냅샷 계열에게 특히 나쁘고 그래서 거절이 별도 PR이 아니라 이 브랜치와 함께 들어왔다. 스냅샷의 키는 토큰 벡터이므로 캐시된 상태는 정확히 그 토큰들만 들고 있어야 한다. 패딩 프리필은 자기 키와 대응하지 않는 상태를 남기는데 이것은 품질 저하가 아니라 틀린 답을 내는 경로다.

일반적인 수리는 스케줄러가 model-owned 계열을 상대로 호출할 수 있는 시퀀스 인지 트림 훅이고 #1755로 등록했다. `trim_internal_caches`로는 안 된다. `SequenceId`를 받지 않고 CLI generate 경로에만 연결돼 있어서 배치 서버 안의 시퀀스 하나를 지목할 수 없다. #1755는 스냅샷이 없어 오염될 것도 없는 채로 같은 잠재 결함을 안고 기본값 `true`를 유지하는 model-owned 계열 셋도 함께 지목한다. `deepseek_v4`, `bailing_moe_linear`, `qwen3_next`다.

### 4.2 스냅샷 분기의 whole-entry 게이트

`try_adopt_cached_prefix`는 `require_whole_entry`를 KV 분기에서만 강제했다. 스냅샷 분기가 먼저 돌고 KV 분기에 닿기 전에 반환하므로 멀티모달 요청이 절단 스냅샷 복원에 도달할 수 있었고 `admission.rs`는 `prefill_start_offset > 0`이 되는 순간 준비해 둔 VLM 임베딩을 버린다. 그러면 접미부에 남은 플레이스홀더 토큰이 평범한 토큰 id로 forward된다.

이 빈틈은 이 브랜치보다 앞선다. `vision::gemma4_vl`과 `vision::gemma4_unified`는 Gemma 4가 처음 답한 이래로 `snapshot_truncatable_to`를 전달해 왔으므로 멀티모달 요청은 whole-entry 규칙을 한 번도 거치지 않고 절단 복원에 이미 닿을 수 있었다. 이 PR이 바꾸는 것은 도달 범위다. 스냅샷 훅 다섯 개를 공용 `vision::VisionLanguageModel`로 전달하면서 Gemma 3와 Llama 4의 VLM 체크포인트가 같은 경로에 올라온다. 게이트는 이제 두 분기 모두에 놓이고 시퀀스 슬롯을 할당하기 전에 거절해 엔트리를 나중의 정확 일치용으로 남겨 두며 KV 분기와 같은 `PromptCacheRejectReason::ModeMismatch`를 기록한다. 정확 접두 일치는 영향을 받지 않는다. `partial`이 거짓으로 남고 접미부에 플레이스홀더도 없다. #1756을 닫는다.

래퍼는 구현하지 않고 전달만 하며 왜 그것으로 충분한지가 문서 주석에 남아 있다. `VisionModule`은 인코더와 커넥터와 프로세서로 이루어져 있고 셋 다 요청 사이에 상태를 남기지 않으며 이미지 임베딩은 forward마다 프롬프트에 다시 계산해 넣는다. 그래서 텍스트 모델 스냅샷을 복원하면 래퍼가 요청 사이에 들고 있는 전부가 재구성된다. 미디어 페이로드가 복원을 타고 새어 나갈 수도 없다. 프롬프트 캐시 키가 요청의 멀티모달 다이제스트를 접어 넣으므로 텍스트만 있는 턴과 같은 토큰에 이미지가 붙은 턴은 다른 버킷에 떨어진다.

---

## 5. 복원 입력은 신뢰하지 않는다

`restore_*`에 도착한 `ModelStateSnapshot`은 이 프로세스가 만든 것이지만 다른 요청에서, 경우에 따라 다른 모델 설정에서 왔다. 그래서 이 브랜치는 그것을 설치할 상태가 아니라 검증할 입력으로 다룬다. rotating 경로는 대부분을 거저 얻는다. `restore_fp16_snapshot_state`가 넘겨받은 버퍼에 대고 링 기하를 검사하기 때문이다. full-attention과 chunked 경로는 스칼라를 직접 대입하므로 같은 부류의 검사가 `check_restored_buffers`에 들어 있다.

| 거부 | 무엇을 잡는가 |
|---|---|
| keys만 있고 values가 없거나 그 반대 | 절반만 쓰였거나 절반만 읽힌 엔트리, 스냅샷과 복원 양쪽에서 |
| 어느 한쪽이 rank 4가 아님 | 어텐션 캐시의 인덱싱 형태인 `[B, H_kv, T, D]`가 아닌 모든 것 |
| keys와 values의 배치·헤드·길이 불일치 | 서로 다른 캡처에서 온 버퍼 두 개 |
| 선언된 길이가 음수이거나 버퍼의 `T`보다 큼 | `offset`이 자기 버퍼를 넘어가는 상태 |
| `start_position < 0` 또는 `offset < start_position` | 뒤집힌 chunked 윈도 |
| `max_size`나 `chunk_size`가 살아 있는 캐시와 다름 | 다른 모델 설정에서 캡처된 스냅샷 |

이 중 실제로 무는 것은 선언 길이 검사다. 이것이 없으면 `offset`이 자기 버퍼를 넘어가는 상태도 깨끗하게 설치되고 다음 append가 시퀀스 축 끝을 넘어 슬라이스한다. `KVCache::update_fp16`이 버퍼를 키우기 전에 `buffer_idx()`로 정규화하기 때문이다. 그러면 복원 거절이 아니라 FFI 경계에서 MLX 예외로 튀어나온다. chunked 쪽은 `offset`이 아니라 `offset - start_position`을 검사한다. chunked 버퍼는 윈도 중 잘려 나가지 않은 부분만 들고 있다.

검사 못지않게 순서 두 가지가 중요하다. 검증이 대입보다 앞서므로 실패한 스냅샷은 갓 만든 캐시를 건드리지 않고 호출자는 콜드 프리필로 물러선다. 그리고 `truncate_chunked`는 두 슬라이스를 모두 만든 다음에 설치하므로 values 슬라이스가 실패해도 짧아진 keys 버퍼가 원래 길이의 values 버퍼 옆에 남지 않는다. 이쪽은 `chunked_truncate_does_not_shorten_keys_when_the_value_slice_fails`가 고정한다. 커밋 메시지에 적기는 쉽고 나중 수정에서 잃어버리기도 쉬운 종류의 보장이다.

`restore_sequence_state_truncated`는 계열 셋 모두에서 호출자를 믿지 않고 `snapshot_truncatable_to`를 다시 묻는다. 절반만 절단된 상태를 설치하면 생성이 조용히 오염되지만 여기서 에러를 내면 콜드 프리필 한 번이 비용의 전부다.

---

## 6. 검증

Apple M5 Max, 포트 19335의 `mlxcel-server`, 프롬프트 캐시 켬, 그리디(`temperature 0`, `seed 0`), 모든 갈래에서 `preserve_thinking` 고정.

### 6.1 성립하지 않는 비교

누구나 먼저 떠올리는 시험, 프롬프트 캐시 켬 대 `--no-prompt-cache`는 스냅샷을 분리하지 못하므로 여기서 쓰지 않았다. 캐시를 켜면 `capture_history_boundary_snapshot`(#1143)이 콜드인 첫 턴조차 대화 경계에서 쪼갠다. Gemma 3의 1323토큰 첫 턴은 1320과 3으로 forward되는 반면 캐시를 끈 갈래는 512/512/299로 forward한다. `docs/benchmarks.md`가 적어 둔 대로 forward 폭은 MLX가 어떤 양자화 matmul 커널을 디스패치할지 고르며 그 폭 변화가 89토큰 답변에서 근소하게 갈리던 토큰 하나를 재현 가능하게 뒤집었다. 곧은 아포스트로피와 굽은 아포스트로피다.

원인이 스냅샷이 아니라 분절이라는 것은 대조군 둘이 고정한다. 같은 갈래를 별도 프로세스에서 다시 돌리면 비트 단위로 같으므로 실행 간 잡음이 아니다. 그리고 스냅샷 지원이 없어 경계 분할도 일어나지 않는 `models/qwen2.5-7b-instruct-4bit`는 이 플래그를 건너 바이트 동일하다. 캐시 플래그를 바꾸는 비교는 프리필 분절도 함께 바꾸는 비교이고 거기서 나온 차이는 어느 쪽에도 귀속시킬 수 없다.

대체 설계는 두 갈래 모두 캐시를 켠 채로 두고 그 턴을 복원된 스냅샷으로 서빙할지 새 프로세스에서 콜드로 계산할지만 바꾸는 것이다. 프리필 분절이 맞춰지고 복원만 유일한 차이로 남는다.

### 6.2 웜 복원 대 콜드

**Gemma 3**(`models/gemma-3-4b-it-4bit`, rotating 1024와 standard). 첫 턴은 1024 윈도 위의 프롬프트 1323토큰으로 3.2절의 수정이 허용하게 된 바로 그 초과 윈도 모양이고 89토큰을 생성한다. 완료 기부가 `token_len=1412`를 저장하는데 이는 첫 턴 프롬프트에 생성분을 더한 값 그대로이며 워밍업이 이를 두 번째 턴 경계인 1417까지 늘린다. 두 번째 턴은 `restored 1417/1435, stored=1417, partial=false`로 채택하고 세 번째 턴은 링이 진작 감긴 상태에서 `restored 1547/1567, partial=false`로 채택한다. 2.1절의 면제 조항을 실제 가중치 위에서 실행한 셈이다.

| 턴 | 복원 | 콜드 대비 |
|---|---|---|
| 2 | 1435 중 1417 | 104토큰 전부 바이트 동일, 103개는 logprob까지 비트 동일, top-2 최소 격차 0.25 |
| 3 | 1567 중 1547 | 18토큰 전부 바이트 동일, 모든 logprob이 비트 동일 |

**Llama 4 Scout**(`models/llama-4-scout-17b-16e-instruct-4bit`, chunked와 standard). 첫 턴은 프롬프트 1305토큰에 생성 512토큰이고 완료 엔트리는 `1817 = 1305 + 512`이며 1822로 늘어난다. 두 번째 턴은 `restored 1822/1840, partial=false`로 채택하고 512토큰 전부가 콜드와 바이트 동일하며 그중 338개는 logprob까지 비트 동일하다. 실행 전체의 top-2 최소 격차가 0.0이므로 정확히 동점인 위치에서도 복원 갈래가 콜드와 같은 선택을 했다. 이 비교가 가질 수 있는 가장 엄격한 형태다.

**AFMoE**(`models/trinity-nano-preview-4bit`). 첫 턴은 프롬프트 1281토큰에 생성 512토큰이고 완료 엔트리는 `1793 = 1281 + 512`다. 두 번째 턴은 `restored 1280/1755, stored=1793, partial=true`로 채택한다. 답변이 저장된 꼬리의 접두로 재토크나이즈되지 않아 스토어가 첫 턴 프롬프트 끝까지로 잘라낸 것이다. 합성 캐시가 아니라 실제 가중치 위에서 절단 판정이 답한 경우이고 유닛 테스트가 공급할 수 없는 유일한 부분이다. 출력 동일성은 이 체크포인트에서 쓸 수 있는 신호가 아니다. `docs/supported-models.md`가 이 체크포인트는 mlx-lm 레퍼런스에서도 퇴화한 반복 텍스트를 낸다고 적어 두었고 두 갈래 모두 정확히 그렇게 한다. 스냅샷 기계는 대신 유닛 테스트가 덮으며 여기에는 ignore가 붙지 않은 `snapshot_restore_matches_cold_decode`가 포함된다.

### 6.3 제대로 거절하는가

계약은 거절도 해야 한다. Gemma 3의 1323토큰 프롬프트를 다시 보내면 1417토큰 엔트리를 찾아내고 감긴 링으로의 절단을 거부하며 잘못된 윈도를 복원하는 대신 `snapshot_diverged (context_len=1323, entry_len=1417)`로 거절한다. 참만 답하는 판정은 무언가를 검사하고 있다는 증거가 되지 못하고 둘을 가르는 것이 이 실행이다.

### 6.4 게이트

| 게이트 | 결과 |
|---|---|
| `--lib models::kv_snapshot` | 23 passed |
| `--lib models::gemma3` | 25 passed, `--ignored`로 6개 더 |
| `--lib models::llama4` | 5 passed, `--ignored`로 3개 더 |
| `--lib models::afmoe` | 28 passed |
| `--lib server::batch::scheduler::scheduler_model_owned_cache_tests` | 3 passed |
| `--lib models::gemma4_tests`, `--lib models::muse_glimmer` | 10과 13 passed. 위임 후에도 그대로다 |
| `cargo clippy --lib --tests --features metal,accelerate`, `cargo clippy -p mlxcel-core --lib` | `-D warnings`에서 clean |
| `cargo fmt --all -- --check` | clean |

모델 단위 스냅샷 테스트에는 `#[ignore = "requires serial MLX execution"]`이 붙어 있다. 합성 래퍼를 만들어 forward를 돌리기 때문인데 각각을 이름으로 지목해 단독 실행했고 모두 통과한다. PR 본문의 테스트 표는 그 이전 커밋 기준으로 쓰여 `models::kv_snapshot`을 16으로 적고 있다. 마지막 커밋이 일곱 개를 더 넣었고 트리에 있는 수는 23이다.

---

## 7. 기능을 제한하는 것과 입증되지 않은 것

### 7.1 스냅샷 버킷이 순환 상태 기준으로 잡혀 있다 (#1761)

`snapshot_family_is_model_aware`는 순환·하이브리드 계열만 나열하고 `gemma3`, `afmoe`, `llama4` 어느 것도 `Hybrid`나 `PureSsm`으로 분류되지 않으므로 `recommend_model_snapshot_capacity_from_config`가 셋 모두에 `None`을 돌려주고 셋 다 고정 512 MiB인 `DEFAULT_SNAPSHOT_CAPACITY_BYTES`로 떨어진다. 고정 버킷은 상태가 컨텍스트에 대해 O(1)인 순환 계열에는 맞는 정책이고 스냅샷이 O(컨텍스트)인 어텐션 캐시 계열에는 틀린 정책이다. 버킷을 넘긴 엔트리는 `Oversized`로 거부되고 프롬프트 캐시는 조용히 아무 일도 하지 않는다.

Llama 4 Scout는 토큰당 192 KiB쯤 저장하므로(레이어 48, KV 헤드 8, head dim 128, fp16, K와 V) 천장이 2731토큰 근처다. 위 실행에서 실측한 엔트리는 1817토큰 턴이 402,653,712바이트, 2352토큰 턴이 503,317,008바이트였고 각각 스텝 정렬 버퍼로 384 MiB와 480 MiB다. 두 턴짜리 검증이 상한 아래 6퍼센트 지점에 앉아 있고 세 번째 턴은 들어가지 않는다. Gemma 3는 1320토큰에서 188,253,128바이트, 1412토큰에서 341,346,192바이트로 자랐고 한 턴이 엔트리를 둘(`Boundary`와 `Completion`) 저장하므로 세 턴짜리 대화 하나가 이미 버킷을 상시 축출 상태로 밀어 넣는다.

같은 이슈가 그 함수의 두 번째 결함도 적어 두었다. `model_type()`이 `text_config.model_type`을 우선하는데 `models/` 아래 체크포인트들은 `gemma4_text`, `gemma4_unified_text`, `muse_glimmer_text`를 보고하며 어느 것도 목록의 `gemma4`나 `muse_glimmer`와 같지 않다. 모델 인지 경로가 기존 항목에서도 죽은 코드라는 뜻이다. 이 diff 밖에 둔 이유는 기본 메모리 정책을 바꾸는 변경이라 단언이 아니라 자체 측정이 필요해서다.

머지되는 것을 정직하게 적으면 이렇다. 계열 셋이 정확 접두 재사용을 얻고 Llama 4에서는 대략 2700토큰 아래 대화에서 쓸 수 있으며 그 위에서는 아무 표시 없이 사라진다.

### 7.2 프롬프트 전체 히트가 마지막 토큰을 다시 돌린다 (#1760)

같은 프로세스 안에서 동일한 프롬프트를 다시 보내면 시험한 모든 계열에서 첫 답변과 다른 답이 나온다. 채택이 프롬프트 전체를 덮고 `admission.rs`가 샘플러에게 새 로짓을 보여 주려고 프리필 커서를 한 토큰 물리는데 복원된 캐시는 되감지 않는다. 그래서 마지막 프롬프트 토큰이 그것을 이미 담고 있는 캐시 위로 forward된다.

이것은 여기서 생긴 것이 아니라 이전부터 있었고 대조군은 `models/gemma-4-12b-it-4bit`다. 이 브랜치 이전부터 같은 경로를 배포해 왔고 이 브랜치가 건드리지 않았는데 같은 발산을 그대로 재현한다. #1760은 설명 두 가지를 열어 두고 있으며(토큰이 실제로 중복되었거나, 6.1절의 forward 폭 효과이거나) 둘을 가르는 측정도 지목해 둔다. 클램프 직후 레이어별 캐시 offset을 `prompt_tokens.len() - 1`에 대고 단언하면 된다. 이 브랜치가 바꾸는 것은 영향권의 크기다. 계열 셋이 더 그 경로에 닿을 수 있게 됐다.

### 7.3 #1754의 off-by-one은 재현되지 않는다

#1754는 완료 기부가 `token_len = tokens.len()`을 기부하는데 캐시는 하나 적게 들고 있다고 본다. 디코드 스텝이 직전에 샘플된 토큰을 forward하므로 마지막 토큰은 끝내 모델을 통과하지 않는다는 것이다. 이슈는 그 빈틈을 EOS가 아닌 종료들로 한정하고 `Length`도 거기 들어간다.

위의 Llama 4 첫 턴이 바로 `length`로 끝났고 복원된 두 번째 턴은 512토큰 중 512개가 콜드와 바이트 동일했으며 top-2 최소 격차가 0.0이었다. 저장된 상태가 엔트리가 주장하는 것보다 토큰 하나 적게 들고 있었다면 접미부 토큰 전부가 RoPE 위치 하나씩 앞당겨 앉아 복원 실행이 발산할 수밖에 없다. Gemma 3도 같은 방향으로 읽힌다. 반대 증거는 이슈에 올려 두었고 가장 그럴듯한 화해도 함께 적었다. 채택된 엔트리가 1817토큰 완료 엔트리가 아니라 1822토큰 워밍업 엔트리였으므로 `prompt_cache.rs`의 워밍업 연장이 채택 전에 다시 맞춰 주고 있을 수 있다. 이슈에 남긴 권고는 기부 산술을 고치기 전에 캡처 시점의 레이어별 캐시 offset과 `snapshot.token_len()`을 직접 대조하라는 것이다. 있지도 않은 off-by-one을 겨냥한 수정은 없던 off-by-one을 만든다.

---

## 8. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 | 18 |
| 추가된 라인 | 3285 |
| 삭제된 라인 | 463 |
| 추가된 테스트 | 44개(모듈 23, Gemma 3 8, Llama 4 7, AFMoE 5, 스케줄러 1) |
| 프롬프트 캐시 재사용을 새로 얻은 계열 | 텍스트 3개와 그 VLM 래퍼 |

### 영역별 변경

- `src/models/kv_snapshot.rs`(신규 784줄): 직렬화 진입점 열두 개, `KvSnapshotNames` 어휘, 공용 검증기(`check_restored_buffers`, `restored_rank4_shape`, `check_truncate_target`, `slice_leading_tokens`).
- `src/models/gemma3.rs`, `src/models/afmoe.rs`, `src/models/llama4.rs`: 모듈로 위임하는 캐시 단위 메서드 네 개, `LanguageModel` 훅 다섯 개, `supports_padded_prefill() -> false` 거절. Gemma 3에는 `live_len` 수정도 들어간다.
- `src/models/gemma4.rs`(41+/252-), `src/models/muse_glimmer_cache.rs`(20+/143-): 손으로 쓴 직렬화기를 지우고 모듈 호출로 대체했다. 텐서 이름은 각자의 `KvSnapshotNames` 상수가 보존한다.
- `src/lib/mlxcel-core/src/cache.rs`: `restore_fp16_snapshot_state`의 idx 경계.
- `src/vision/mod.rs`: 텍스트 모델로 전달하는 훅 다섯 개, 그리고 무상태성과 멀티모달 다이제스트 근거 기록.
- `src/server/batch/scheduler/prompt_cache.rs`: whole-entry 게이트를 시퀀스 할당 앞으로 옮기고 두 분기 모두에 적용.
- `src/models/recurrent_snapshot.rs`: `KVCacheMode` 태그 쌍을 `pub(crate)`로 승격하고 에러 메시지에서 계열 라벨 제거.
- `docs/turbo-kv-cache.md`, `docs/supported-models.md`: 스냅샷 가능 계열 목록, 타입별 절단 규칙, 패딩 프리필 제외, 그리고 목록을 믿는 대신 다시 만들 수 있도록 `grep` 명령 한 줄.

### 커밋

| Hash | Type | Subject |
|---|---|---|
| `2dde058` | fix | bound the rotating snapshot idx by the physical buffer |
| `a40d421` | feat | share KV snapshot serialization, opt in three families |
| `ec96fa3` | fix | apply the whole-entry policy to the snapshot adopt branch |
| `9da46f4` | fix | bound KV snapshot restores by their own buffers |
| `3a5e506` | test | the model-owned gate test now expects a snapshot donation |
| `7647c23` | test | close snapshot validation and family-hook gaps for #1335 |
| `a0f9692` | docs | note snapshot prompt-cache reuse for #1335 families |

### 관련 이슈

#1335와 #1756을 닫는다. 이 브랜치의 리뷰에서 나온 후속 과제는 #1754(기부 산술, 반대 증거를 게시했다), #1755(시퀀스 인지 트림 훅. 이 계열들에 패딩 프리필을 되돌려 줄 것이다), #1760(프롬프트 전체 히트), #1761(스냅샷 버킷 크기)이다. 히트가 성립하게 해 주는 히스토리 경계 스냅샷은 #1143에, 절단 판정이 답하는 절단 복원 경로는 #1145에 의존한다.

---

## 9. 후속 조치

**#1761이 이 PR의 값을 좌우한다.** 나머지는 이미 있던 경로에 대한 정확성 작업이고 #1761은 그 경로가 캐시할 가치가 있는 대화에서 닿을 수 있는지를 정한다. 기본 메모리 정책을 바꾸므로 단언이 아니라 Llama 4 Scout와 Gemma 3에서 측정해야 하고 이슈가 지목한 `_text` 접미사 결함 때문에 기존 항목에도 체크포인트가 실제로 싣는 표기를 쓰는 테스트가 필요하다.

**패딩 프리필 거절은 만료일이 있는 우회책이다.** #1755는 여기서 거절한 계열 셋 중 최소 하나를 새 훅으로 패딩 프리필에 복귀시키라고 요구한다. 훅이 존재하기만 하는 것이 아니라 끝에서 끝까지 동작한다는 증거가 그것이다.

**model-owned 계열 셋은 여전히 기본값 `true`다.** `deepseek_v4`, `bailing_moe_linear`, `qwen3_next`는 오염될 스냅샷이 없을 뿐 같은 잠재 트림 결함을 안고 있다. #1755가 지목했고 고치거나 명시적으로 거절하게 해야 한다.

**같은 `seq_len()` 주장이 파일 하나 건너에 남아 있다.** `src/models/exaone_moe.rs`의 `AnyKVCache::live_len`은 rotating 캐시에 `seq_len()`을 돌려주면서 그것이 이미 가시 윈도를 보고한다는 주석을 달고 있다. 3.1절이 반증한 바로 그 주장이다. 이 계열은 스냅샷을 갖고 있지 않아 지금은 디코드 뒤의 다중 토큰 append를 만들어 낼 경로가 없고 여기서 넣은 가드는 Gemma 3의 사본만 덮는다. 어느 쪽으로든 단언하기 전에 한 번 읽어 볼 값이 있다.

**스케줄러 게이트 테스트가 카운터 단언 하나를 잃었다.** `prompt_cache_reject_model_owned_state`는 기부 쪽 거절을 세는데 Gemma 3가 이제 그 앞의 스냅샷 분기에서 반환한다. 그 단언이 지키던 보장은 K/V 버킷이 비어 있다는 형태로 여전히 직접 검사하지만 카운터 자체는 이 테스트 파일에 그것을 돌릴 모델이 없어졌다. 다음에 이 카운터를 커버리지로 신뢰하려 할 때 기억해 둘 일이다.

### 옮겨갈 교훈

여기서 잡은 결함 둘은 바꾸는 코드를 읽어서가 아니라 실제 대화 위에서 기능을 끝까지 돌려서 나왔다. `live_len`의 마스크 폭과 `idx > max_size` 경계는 서버가 그전까지 만들 수 있던 모든 모양에 대해 맞았고 복원된 캐시에 append하는 순간 둘 다 틀려졌다. 동작하는 사본 둘에서 뽑아낸 공용 추상은 그 둘의 동작과 함께 사각지대도 물려받는다. 그것을 드러내는 것은 사본들이 겨냥해 쓰인 두 호출자와 모양이 다른 세 번째 호출자뿐이다.

검증 설계가 나머지 절반이다. 누구나 먼저 잡는 비교인 기능 플래그 켬 대 끔은 여기서 성립하지 않았다. 그 플래그가 프리필 분절까지 바꾸기 때문이고 그 사실이 드러난 방식은 89토큰 답변에서 아포스트로피 하나가 뒤집힌 것이었다. 해법은 허용 오차가 아니라 다른 대조군이었다. 플래그를 고정하고 복원 여부만 바꾸면 된다. 두 가지를 함께 바꾸는 시험이 차이를 보고했다면 그 시험은 둘 중 어느 것에 대해서도 아무 말도 하지 않은 것이다.
