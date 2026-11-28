# 기술 보고서: PR #1280 - LLaVA와 Qwen2-VL의 이미지 context floor

## 요약

`xla_image_context_floor`는 설정만으로 이미지 하나가 확장될 수 있는 최대 프롬프트 토큰 수를 예측한다. OpenXLA 기동 가드가 이미지를 결코 받을 수 없는 그래프 shape을 거부하기 위해서다. Molmo2에만 공식이 있어 LLaVA와 Qwen2-VL은 가드 밖에 있었다.

이제 두 패밀리 모두 유도한다. 흥미로운 결과는 산술이 아니라 Qwen2-VL의 숫자가 드러낸 것이다. 이 패밀리의 실제 확장은 이미지 형태에 따라 64에서 16384토큰까지 **256배** 벌어지고, 기존 가드 메시지는 운영자가 따를 수 없는 조언을 하고 있었다.

## 1. 문제

`xla_image_context_floor`는 `Molmo2VLM`만 매칭하고 나머지는 `None`을 반환했다. `LlavaVLM`과 `Qwen2VL`은 둘 다 OpenXLA 이미지 경로 자격이 있으므로, 가드를 그대로 통과하고 가드가 막으려던 지연 admission 실패를 그대로 겪었다. 비전 타워가 돌고, 그 다음 admission이 정적 그래프 capacity 초과로 요청을 거부한다.

## 2. 기술적 판단

### 2.1 상위 패밀리가 아니라 이 구현에 맞춘다

이슈는 LLaVA가 `llava_next` 변형을 위해 anyres 그리드 순회를 필요로 하고, Qwen2-VL은 `preprocessor_config.json`에서 `max_pixels`를 읽는다고 서술했다. **둘 다 이 코드베이스와 다르다.**

`load_llava_host_preprocessor`는 `mm_tokens_per_image.unwrap_or((image_size / patch_size)^2)`를 계산하고 anyres 그리드를 적용하지 않는다. `llava_next` 체크포인트도 마찬가지이며, `get_model_type`은 텍스트 백본이 Granite가 아닌 한 이를 `LlavaVLM`으로 라우팅한다. `llava_token_block_info`는 `use_boi_eoi: false`에 접두·접미 목록이 비어 있으므로 프레이밍 토큰도 없다.

`qwen_vl_processor`는 `Qwen2VLProcessor::new`로 프로세서를 만드는데, 이 생성자는 `min_pixels`와 `max_pixels`를 기본값으로 설정하고 `preprocessor_config.json`을 전혀 읽지 않는다. 따라서 체크포인트의 `max_pixels` 키는 이 경로가 받아들이는 것에 아무 영향이 없고, 그 값에서 유도한 floor는 런타임이 지키지 않는 숫자가 된다.

가드는 런타임이 실제로 내놓을 값을 예측해야 하므로 두 유도 모두 코드를 따랐다. 이 차이는 이슈 텍스트만 봐서는 보이지 않고 HF 동작과 비교하는 사람에게는 버그로 보일 것이므로 기록한다.

### 2.2 픽셀 경계는 다시 적지 않고 공유한다

Qwen2-VL floor는 `max_pixels`에 의존하는데, 이 값은 두 생성자에 리터럴로 있었다. 유도 쪽에 다시 적으면 서로 일치해야 하지만 일치를 강제할 장치가 없는 숫자가 둘이 된다. `DEFAULT_MIN_PIXELS`, `DEFAULT_MAX_PIXELS`, `max_image_tokens`를 프로세서 모듈에 두고 양쪽이 쓰게 했다.

### 2.3 Qwen2-VL floor가 틀렸음을 증명한 가드 메시지를 다시 쓴다

기존 메시지는 "이미지를 서빙하려면" capacity를 floor로 설정하라고 했다. Molmo2의 1834에서는 따를 수 있다. Qwen2-VL의 16384에서는 불가능하다. capacity는 매 디코드 스텝이 어텐션하는 길이이고, 실측 디코드는 256에서 2048 사이에 이미 3.18에서 1.41 tok/s로 떨어진다.

메시지는 그 자체로도 틀렸다. floor보다 작은 capacity도 거기 들어가는 모든 이미지를 서빙한다. 이 패밀리에서는 그게 대부분의 실제 이미지다. 768x1024 사진은 999토큰으로 확장되므로 2048 capacity면 여유롭게 처리된다. 기존 문구는 그런 설정이 이미지를 하나도 못 서빙하는 것처럼 암시했다.

새 메시지는 운영자가 서빙하려는 최대 확장 크기를 묻고, 더 작은 값이 무엇을 하는지 밝힌다. 들어가는 것은 모두 받고 나머지는 비전 타워가 이미 돈 뒤에 admission에서 거부된다는 사실까지 포함한다. 명시 지정 탈출구도 유지한다. 이슈 번호는 넣지 않았다. 서버 기동 실패를 읽는 사람에게 GitHub 참조는 쓸모가 없다.

이 변경은 이슈가 out of scope로 둔 `ensure_xla_image_context_capacity`를 건드린다. 그 범위 선언은 256배 범위를 알기 전에 쓰인 것이고, 올바른 숫자를 불가능한 조언에 붙여 출하하는 것이 메우려던 공백보다 나빴을 것이다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/multimodal/host_preprocessor.rs` | `llava_image_context_floor`, `qwen2_vl_image_context_floor`, `read_config_json`, `xla_image_context_floor`의 새 arm, 가드 메시지 재작성 |
| `src/vision/processors/qwen2_vl.rs` | `DEFAULT_MIN_PIXELS`, `DEFAULT_MAX_PIXELS`, `max_image_tokens`, 생성자가 이를 사용 |
| `src/multimodal/host_preprocessor_tests.rs` | 패밀리별 floor 고정, 실제 프로세서 대상 타이트 경계 확인, 설정 누락·퇴화 사례, 경계 테스트, 무시 표시된 실제 체크포인트 테스트 |

## 4. 리뷰 지적사항

Qwen2-VL의 크기 문제는 PR을 열기 전에 제기했고 조용히 출하하지 않았다. 이전에는 기동되던 패밀리에 대해 가드가 하는 일을 바꾸기 때문이다. 세 가지를 검토했다. 진짜 최악값을 그대로 출하하거나, 메시지를 고쳐서 출하하거나, 도달 불가능한 floor는 없느니만 못하다는 판단으로 Qwen2-VL은 유도하지 않는 것이다. 두 번째를 택했다. 숫자는 정확하고 capacity 버킷이 오면 그대로 이미지 버킷 크기가 되므로, 결함은 숫자가 아니라 가드가 그 숫자에 대해 하는 말에 있었다.

## 5. 검증

단위 테스트: `host_preprocessor`에서 26개 통과, 기존 Qwen2-VL 프로세서 테스트도 유지.

실제 체크포인트, 각자의 `config.json`에서 floor 유도:

| 체크포인트 | 패밀리 | floor |
| --- | --- | --- |
| llava-1.5-7b-4bit | LlavaVLM | 576 |
| llava-interleave-qwen-0.5b-bf16 | LlavaVLM | 729 |
| llava-next-mistral-7b-4bit | LlavaVLM | 576 |
| qwen2-vl-2b | Qwen2VL | 16384 |
| qwen2-vl-2b-4bit | Qwen2VL | 16384 |

Qwen2-VL 검증은 유도식을 다시 계산하지 않고 실제 `smart_resize`와 `compute_grid_thw`를 극단 형태에 돌린다. 유도식을 다시 쓰면 공식이 자기 자신과 일치한다는 것만 증명된다.

```text
224x224   ->    64 토큰
768x1024  ->   999 토큰
4000x4000 -> 16384 토큰
8000x8000 -> 16384 토큰
20000x300 ->  7854 토큰
최악 = 16384, floor = 16384
```

최악 관측값이 floor와 정확히 일치하므로 이 경계는 안전한 과대 추정이 아니라 타이트하다. 방향이 중요하다. 진짜 최대보다 높은 floor는 더 큰 그래프 비용만 물리지만, 낮은 floor는 가드가 막으려던 실패를 되살린다.

## 6. 관련 작업

이슈 #1272가 이 PR로 닫히고, 확장된 가드는 이슈 #916에서 도입됐다. Qwen2-VL의 범위는 capacity 버킷의 가장 강한 근거다. 정적 shape 하나로는 floor가 모든 텍스트 전용 요청에 처리량 세금을 물리지만, 버킷이 있으면 이미지 버킷의 크기가 되어 텍스트 요청에 아무 비용도 물리지 않는다.

코드에 남긴 전망성 단서가 하나 있다. LLaVA 호스트 전처리기가 언젠가 anyres 그리드를 구현하면 LLaVA floor는 이미지당 고정 수가 아니게 되고 다시 유도해야 한다. 이 유도는 현재 존재하는 로더에 대해 옳은 것이지 패밀리 일반에 대해 옳은 것이 아니다.
