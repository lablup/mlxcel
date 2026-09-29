# 기술 보고서: PR #2054 - VoiceChat 스트리밍을 80 ms 프레임 예산 안으로

**날짜**: 2026-09-30

**상태**: Apple M1 Ultra에서 4-bit, 8-bit 체크포인트로 구현 및 검증 완료, 머지 대기

**언어**: Rust

**위험도**: 낮음. 출력은 이전 바이너리와 바이트 단위로 같고, 실행 비용은 상주 메모리 약 2.7 GB 증가뿐입니다.

## 요약

이슈 #2045의 목표는 유휴 상태의 M1 Ultra에서 Nemotron VoiceChat의 `realtime_factor`를 1.0 아래로 낮추는 것이었습니다. 이 PR 이전에는 80 ms 프레임 하나에 약 82 ms(4-bit), 88.6 ms(8-bit)가 걸렸고, 이후에는 69.0 ms(4-bit, 0.866), 75.3 ms(8-bit, 0.942)가 걸립니다. 개선은 전부 한 가지 정확한(exact) 변경에서 나옵니다. f32 활성값만 만나는 인코더, TTS 백본, MoG 헤드 가중치를 로드할 때 한 번 f32로 바꿔 두고, MLX가 매 프레임 모든 연산 안에서 캐스트하지 않게 했습니다. 원인을 찾는 데 쓴 호스트 동기화 횟수와 선택형 하위 단계 프로파일링도 함께 추가했습니다.

## 1. 문제 정의

`--stream --profile`은 단계 합계 다섯 개만 보고했고(4-bit 기준 perception 22.5 ms, RNNT 0.4, language 16.0, TTS 35.6, codec 7.3), 이슈의 후보 목록은 추정치로 정렬되어 있었습니다. 이슈는 프로파일링을 먼저 하고, 수치를 바꾸는 변경보다 출력을 보존하는 변경을 먼저 적용하라고 요구했습니다.

## 2. 변경 요약

- `src/audio/stage_probe.rs`(신규): 스레드 로컬 프레임별 호스트 동기화 카운터와 이름 붙은 하위 단계 타이머입니다. `FrameTiming`과 `ProfileSummary`에 `host_syncs`와 선택형 `sub_stages` 맵이 생겼습니다. 타이머는 `StreamingOptions::profile_stages`(CLI에서는 `MLXCEL_VOICECHAT_PROFILE_STAGES=1`)로 켜고, `MLXCEL_VOICECHAT_PROFILE_FRAMES=<path>`는 프레임별 타이밍을 파일로 씁니다.
- `src/audio/f32_weights.rs`(신규): `promoted_subset`은 접두사 아래 가중치를 돌려주되, 선택된 bf16/f16 항목은 f32로 캐스트하고 평가해 둡니다.
- `VoiceChatPerception::from_weights`는 인코더와 `proj` 가중치를 모두 승격합니다. `RvqEarTtsModel::from_weights`는 `promotes_to_f32`에 따라 백본의 어텐션과 MLP 프로젝션, MoG 헤드의 MLP 프로젝션과 출력 프로젝션, `embed_code`, fusion의 `audio_proj`를 승격합니다.
- `docs/nemotron-voicechat.md` Performance 절: 새 수치, 정상 상태 측정, 메모리 비용, 프로파일링 스위치를 반영했습니다.

## 3. 기술적 선택과 그 이유

### 캐스트가 병목이었던 이유

이슈에 올린 하위 단계 측정에서 conformer 24층이 25.7 ms, TTS 백본이 24.7 ms, MoG 헤드 5회 패스가 12.0 ms였습니다. 셋 모두 레퍼런스와 같이 bf16 가중치에 f32 활성값을 넣어 실행합니다. MLX의 `matmul`, `addmm`, `conv_general`, `fast::layer_norm`, `fast::rms_norm`은 그래프에 `astype(weight, out_type)`을 넣는 방식으로 승격합니다. 그래서 매 프레임 인코더 1.22 GB, 백본 1.19 GB, MoG 0.32 GB의 5배를 읽고, 그 두 배 크기의 f32 사본을 쓴 다음 다시 읽었습니다. 캐스트를 없애자 perception은 22.5 ms에서 17.0 ms로, TTS는 35.6 ms에서 28.1 ms로 줄었습니다.

### 변경이 정확한 이유

bf16에서 f32로의 변환은 손실이 없습니다. 미리 캐스트한 가중치는 같은 f32 값을 같은 레이아웃으로 같은 커널에 넘깁니다. 행 연속 배열의 전치 뷰는 어느 쪽이든 열 연속으로 남습니다. 단위 테스트는 matmul, addmm, conv1d(dense와 depthwise), layer_norm에서 승격 그래프와 사전 캐스트 그래프를 바이트 단위로 비교하고, bf16 perception 모듈로 같은 비교를 끝까지 한 번 더 합니다. 실제 체크포인트에서도 `--stream --seed 0` WAV md5가 그대로입니다. 4-bit는 `aad28bd67e52113444f52a2b4b49e2d0`, 8-bit는 `3223cf3627a874122fb2bb94b5b81ee8`입니다.

### bf16으로 남긴 가중치

가중치를 읽는 모든 연산이 f32 활성값에 맞춰 승격할 때만 그 가중치를 승격할 수 있습니다. Gemma RMSNorm은 `1 + weight`를 가중치 dtype으로 만들기 때문에 백본의 모든 norm과 `q_norm`, `k_norm`은 bf16으로 둡니다. subword 조건은 bf16으로 계산하므로 `embed_subword`와 fusion의 `text_proj`도 저장된 대로 둡니다. `proj_mus`와 `low_mat`은 행 단위로 모으는 테이블이라 호출당 비용이 작아서 역시 그대로 둡니다. 이 목록은 `promotes_to_f32`에 들어 있고, 단위 테스트가 실제 키 이름으로 고정합니다.

### 캐스트 위치

캐스트는 `NemotronVoiceChatModel::load`가 아니라 각 컴포넌트 로더 안에서 합니다. 그래서 컴포넌트를 직접 만드는 실가중치 패리티 테스트(`tts_real`, `front_real`, `stream_real`)도 승격 경로를 거칩니다. `gemma3_backbone.rs`와 `gemma3.rs`는 건드리지 않았으므로 Gemma 3 텍스트 모델의 동작은 바뀔 수 없습니다.

### 다른 후보를 적용하지 않은 이유

이슈는 목표를 달성하면 위험을 더하지 말라고 합니다. 이 변경만으로 4-bit는 0.866, 8-bit는 0.942가 되었고, 8-bit 추가 목표도 달성했습니다. 남은 exact 후보(T1 조건 메모이제이션, T2/T3 RVQ gather, P2/P3 위치 캐시, C1/L1 동기화 병합)는 하위 단계 수치로 보아 각각 1~2 ms 정도입니다. 더 작은 기기에서 여유가 필요해지면 후속 작업으로 진행합니다.

## 4. 검증

M1 Ultra(128 GB)에서 다른 작업 없이 `mlxcel generate -m <ckpt> --audio question.wav --stream --profile --seed 0`를 실행했습니다. 56 프레임 중 앞의 5개는 콜드 프레임으로 제외했고, 워밍업 1회 후 20초 간격으로 3회 측정했습니다.

| 체크포인트 | 빌드 | 측정 (realtime_factor) | 프레임 p50 | perception | language | tts | codec |
|---|---|---|---|---|---|---|---|
| 4-bit | 기준 (ac02b2dc) | 1.0313 / 1.0287 / 1.0300 | 82.2 ms | 22.5 | 16.0 | 35.6 | 7.3 |
| 4-bit | 이 PR | 0.8650 / 0.8673 / 0.8662 | 69.0 ms | 17.0 | 16.0 | 28.1 | 7.3 |
| 8-bit | 기준 | 1.1105 / 1.1114 / 1.1083 | 88.7 ms | 22.5 | 22.2 | 35.8 | 7.4 |
| 8-bit | 이 PR | 0.9411 / 0.9435 / 0.9422 | 75.3 ms | 16.9 | 22.2 | 28.1 | 7.3 |

RNNT는 모든 측정에서 0.44 ms입니다. 계측만 넣은 커밋은 1.0312 / 1.0301 / 1.0306으로, 카운터 비용은 측정되지 않을 만큼 작습니다. 호스트 동기화는 프레임당 8.2회로 그대로입니다.

긴 입력 측정에는 추가 디코딩 12초(168 프레임, 71 프레임 어텐션 창을 넘는 길이)를 썼습니다. 기준 바이너리는 전체 프레임 기준 1.036이었습니다. 이 PR은 전체 기준 0.871이고, 71번째 이후 프레임(102개)은 평균 69.8 ms, 계수 0.872입니다.

테스트:

- `cargo test --release --lib -- models::nemotron_voicechat audio::`: 219개 통과.
- `MLXCEL_VOICECHAT_MODEL`, `MLXCEL_VOICECHAT_REF`를 설정한 실가중치 테스트: `streaming_real`, `tts_real`, `codec_real`, `offline_real`, `front_real` 통과. `llm_real`과 `stream_real`은 이 기기에 `MLXCEL_VOICECHAT_PADDED`, `MLXCEL_VOICECHAT_STREAM_REF` 덤프가 없어서 건너뜁니다.
- Qwen3-4B 4-bit greedy 출력(60 토큰, `--show-reasoning`)이 기준 바이너리와 같습니다.
- fmt, clippy(`-p mlxcel --lib --tests --examples -D warnings`), `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest`, `realtime_ws` 통과.

## 5. 알려진 한계

- 상주 메모리가 약 2.7 GB 늘어납니다. 128 GB 기기에서 9 GB 체크포인트와 비교하면 작지만, 16~24 GB Mac에서는 의미가 있습니다.
- 수치는 기기 한 대에서만 측정했습니다. 여기서는 여유(0.87 / 0.94)가 넉넉하지만, 더 느린 Apple Silicon에서는 8-bit 체크포인트가 여전히 실시간을 못 맞출 수 있습니다.
- 기준 바이너리에는 프레임별 덤프 기능이 없어서, 기준의 정상 상태 수치는 전체 실행 요약(1.036)으로 대신했습니다.
- 하위 단계 타이밍은 모든 경계에서 평가를 강제하므로, 시간 분포를 설명하는 데는 쓸 수 있지만 그 합계를 실시간 계수로 볼 수는 없습니다.
