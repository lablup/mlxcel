# PR #3278: Inkling omni 실체크포인트 검증, NVFP4와 MTP 수정

**날짜**: 2026-11-23
**상태**: Draft PR 열림(#1549를 닫음). Apple M3 Ultra(512 GB, macOS 27.2.1)에서 실제 체크포인트로 검증.

## 요약

Inkling(`inkling_mm_model`)은 합성 픽스처 패리티만 있는 상태로 들어왔다. `docs/supported-models.md`에는 153.5 GB affine 타깃, 170.7 GB 네이티브 NVFP4 체크포인트, 4.5 GB MTP 샤드를 한 번도 돌려 보지 못했다고 적혀 있었다. 이 PR은 이들을 모두 M3 Ultra에서 돌리고 #1549의 항목별 근거를 기록한다.

검증 중 결함 두 개가 나왔다. 네이티브 NVFP4 체크포인트는 모든 greedy 토큰을 id 0으로 디코드했다. 양자화 정보가 `hf_quant_config.json`에만 있어서 로더가 bf16 dense 가중치를 f16으로 변환했기 때문이다. Inkling MTP는 exactness 검사 없이 켜졌고, draft 폭 2와 3에서 classic decode와 다른 greedy 스트림을 냈다. 둘 다 회귀 테스트와 함께 고쳤다. 4-bit와 NVFP4 체크포인트 모두에서 텍스트, 이미지, 오디오, 비디오 출력이 맞고, CLI와 서버 출력은 바이트 단위로 같다. MTP는 이제 게이트 아래에서 정확하지만 수용 길이가 라운드당 약 1.6 토큰(mlx-vlm drafter와 같은 수준)이라 Inkling-Small에서 속도 이득이 없다. 처리량 개선을 요구하는 기준은 충족하지 못했다.

## 1. 체크포인트와 설정

| 체크포인트 | 로컬 경로 | 비고 |
|---|---|---|
| `inference-optimization/Inkling-0.6B-A0.6B` | `models/inkling-0.6b-a0.6b` | HF config 방언, f32 |
| `mlx-community/Inkling-Small-mlx-4bit` | `models/inkling-small-mlx-4bit` | affine 4-bit, config.json에 `quantization` |
| `thinkingmachines/Inkling-Small-NVFP4` | `models/inkling-small-nvfp4` | ModelOpt NVFP4 expert, bf16 dense, `hf_quant_config.json`, `mtp.safetensors` 포함 |

모든 GPU 실행은 공유 GPU lock을 `/usr/bin/lockf`로 실행 동안만 잡았고, 대형 체크포인트는 한 번에 하나만 돌렸다. 피크 메모리는 `/usr/bin/time -l`의 peak memory footprint 값이다.

레퍼런스: 0.6B는 HF `transformers` 5.24.0 `InklingForConditionalGeneration`(CPU, float32)을 썼다. mlx-vlm은 HF 방언 config를 읽지 못한다(`dense_intermediate_size` 없음). Small 체크포인트의 레퍼런스는 `references/mlx-vlm`(d998a620)에 vendoring된 mlx-vlm이고, `mlx_vlm.split_mtp`와 `inkling_mtp` drafter를 포함한다.

미디어: `tests/fixtures/sam3/cats.png`(COCO 이미지 39769), macOS `say`로 만들어 정확히 80000 샘플로 맞춘 5초 16 kHz 모노 WAV, 60x60 빨간 사각형이 왼쪽에서 오른쪽으로 움직이는 4초 10 fps 320x240 클립.

## 2. 결함 1: NVFP4가 id 0만 디코드

### 2.1 증상

`Inkling-Small-NVFP4`에서 텍스트, 이미지, 오디오, 비디오 모든 프롬프트가 `!`(토큰 id 0)만 연달아 냈다. 오디오와 비디오 메타데이터(오디오 토큰 100개, 같은 프레임 쌍)는 맞았으므로 프런트엔드가 아니라 언어 모델 쪽 문제였다. 같은 디렉터리를 mlx-vlm으로 돌리면 정상 텍스트가 나왔다.

### 2.2 근본 원인

Apple Silicon에서 `bf16_to_f16_at_load(is_quantized, config)`는 양자화되지 않은 체크포인트의 bf16 가중치를 f16으로 바꾼다. VLM 로더는 config.json의 `quantization` 블록으로 양자화 여부를 판단한다. ModelOpt NVFP4 체크포인트에는 이 블록이 없고 양자화 선언이 `hf_quant_config.json`에만 있다. 그래서 bf16 dense 가중치(attention, norm, 공유 projection, embedding)가 f16으로 바뀌었고 Inkling residual stream에서 오버플로가 났다. 4-bit affine 체크포인트는 config에 `quantization`이 있어서 영향을 받지 않았다.

두 실험으로 원인을 좁혔다. `MLXCEL_INKLING_MOE_KERNEL=0`(fused expert 커널 끔)에서도 깨져서 NVFP4 expert 경로는 원인이 아니었다. `MLXCEL_KEEP_BF16=1`만으로 출력이 정상이 됐다.

expert sidecar 매핑은 따로 확인했고 올바르다. expert마다 `gate_scale = scale`(E4M3 블록 scale), `out_scale = scale * scale2`이고 mlx-vlm 로더와 같다.

### 2.3 수정

`bf16_to_f16_at_load`는 Inkling config(`is_inkling_config`: `model_type`이 `inkling` 또는 `inkling_mm_model`이거나 `text_config.model_type`이 `inkling_text`, `inkling2`는 제외)에 대해 false를 돌려준다. 한 곳에서 텍스트와 VLM 로더를 모두 덮는다. `is_model_quantized`가 sidecar를 읽게 고치는 대신 패밀리 전체를 제외한 이유는 변환으로 이득을 보는 Inkling 체크포인트가 없기 때문이다. affine 변환본은 이미 bf16을 유지하고, 0.6B 체크포인트는 f32다.

### 2.4 수정 후 결과

| 프롬프트 | NVFP4 응답 |
|---|---|
| 텍스트, 128 토큰 | 일관된 추론과 올바른 회문 함수. mlx-vlm NVFP4 출력과 처음 354자가 같고, 그 뒤 근소한 동률에서 단어 두 개 순서가 바뀜 |
| 이미지(고양이) | 태비 고양이 두 마리(왼쪽 새끼, 오른쪽 성묘), 분홍 담요 또는 소파, 리모컨 |
| 오디오(5초 WAV) | 단어 단위로 정확한 전사, 오디오 토큰 100개 |
| 비디오(빨간 사각형) | "오른쪽으로 이동". 8 프레임, 4 쌍, 타임스탬프는 4-bit와 동일 |

로드 후 resident 154.7 GB, peak footprint 273~274 GB, 텍스트 디코드 약 30.7 tok/s.

## 3. 결함 2: Inkling MTP가 exactness 게이트를 건너뜀

### 3.1 증상

`mtp_capable_target`은 `Inkling`과 `InklingVLM`에 대해 검사 없이 true를 돌려줬다. 4-bit 타깃과 네이티브 MTP 헤드로 `--draft-block-size 2` 128 토큰 greedy를 돌리면 중간에 classic decode와 갈라졌고(단어 하나 삽입), 폭 3도 갈라졌다. 기본 폭 10(적응형 3~10)은 우연히 일치했다.

### 3.2 원인

M >= 2 행의 verify forward는 Metal `qmv_wide` 커널을, 단일 토큰 디코드는 `qmv`를 쓴다. Apple GPU 15세대 이후에서 둘의 결과는 비트 단위로 같지 않다. `MLXCEL_QMV_WIDE=0`에서는 폭 2와 3 모두 classic decode와 정확히 일치했다. 다른 MTP 패밀리는 `src/models/speculative_exactness.rs`의 측정 probe로 이미 이를 막고 있었는데 Inkling만 우회돼 있었다.

### 3.3 수정

- `InklingModel::probe_block_chain_exactness(block)`은 새 내부 캐시 위에서 chain 쪽(단일 토큰 forward 반복)과 block 쪽(`T = block` forward 한 번)을 짧은 draw 세 개로 만들고, `compare_block_against_chain`으로 위치별 logit 원시 바이트를 비교한다.
- `InklingModel::mtp_exactness_allows(block)`은 이 probe를 `mtp_exactness_gate`에 넘긴다. 게이트는 프로세스 단위로 결과를 저장하고, `qmv_wide`를 끈 상태로 재시도하며, `MLXCEL_MTP_ALLOW_INEXACT=1`을 존중한다.
- `mtp_capable_target`(서버)과 `run_offline_mtp`(`mlxcel generate --draft-kind mtp`)가 이를 호출한다. 오프라인 경로는 override 이름을 담은 오류를 돌려준다.
- Inkling MTP drafter는 공유 drafter의 bf16→f16 정책을 더 이상 적용하지 않는다. 타깃이 bf16일 때 f16 drafter는 공유 embedding, norm, LM head 호출마다 dtype 경계를 만들었고 MLX는 그 혼합을 float32로 승격했다. 폭 2에서 128 토큰이 같은 수용률로 11.6초에서 5.1초가 됐다.

이 호스트에서 probe는 `qmv_wide`를 끈 block forward도 위치 0에서 chain과 다르다고 보고한다(logit 바이트의 약 47%). 짧은 convolution과 attention이 `T = 1`에서 다른 커널을 쓰기 때문이다. 그래서 override가 없으면 게이트가 이 호스트에서 Inkling MTP를 거절한다. 의도한 보수적 결과다. 정확함을 증명하지 못하는 speculative 경로는 게이트가 막는다.

### 3.4 MTP 측정(4-bit 타깃, NVFP4 디렉터리를 drafter로, 128 greedy 토큰)

| 실행 | classic과 일치 | 평균 수용 길이 | 처리량 |
|---|---|---|---|
| classic decode | 해당 없음 | 해당 없음 | 41.6 tok/s |
| 게이트 켬, 모든 폭 | MTP 거절 | 해당 없음 | 해당 없음 |
| override, 폭 10(적응형 3..10) | 예 | 1.608 | 10.8 tok/s |
| override, 폭 2 | 아니오 | 1.494 | 26.6 tok/s |
| override, 폭 3 | 아니오 | 1.587 | 24.2 tok/s |
| override, 폭 2, `MLXCEL_QMV_WIDE=0` | 예 | 1.512 | 26.4 tok/s |
| override, 폭 3, `MLXCEL_QMV_WIDE=0` | 예 | 1.608 | 24.1 tok/s |

같은 타깃과 split drafter로 돌린 upstream mlx-vlm도 라운드당 1.59 토큰(draft 토큰의 6.7%)을 수용한다. 낮은 수용률은 이 헤드와 타깃 조합의 성질이지 포팅 문제가 아니다. chain index를 바꾼 변형(seed 다음 block 1 사용)도 개선이 없었다(6.9%). 라운드당 약 1.6 토큰이면 verify forward와 drafter 비용이 절약되는 토큰보다 커서 어떤 폭도 classic decode를 이기지 못한다.

## 4. #1549 항목별 검증 결과

| 항목 | 근거 |
|---|---|
| 4-bit 텍스트, CLI와 `/v1/chat/completions`, chat template, 64 토큰 | 바이트 단위 동일(prompt 25 토큰, completion 64, finish `length`). 기본 effort에서는 전부 reasoning이고, `reasoning_effort: "none"`이면 content가 올바른 레일리 산란 설명. 128 토큰 출력이 mlx-vlm과 정확히 같음. 피크 약 154.5 GB |
| 0.6B 패리티, RAW, greedy, 64 토큰, `MLXCEL_FUSED_MOE=0` | 프롬프트 9개 모두 레퍼런스의 첫 `<|endoftext|>`까지 HF와 토큰 단위로 일치(토큰 수 9, 22, 52, 45, 21, 64, 45, 53, 48). 64 토큰 프롬프트는 EOS 없이 끝까지 감. 피크 3.6 GB. 모델이 암기된 텍스트를 내뱉어 출력 원문은 싣지 않음 |
| grounded 정지 이미지, CLI와 서버 | 태비 고양이 두 마리, 분홍 담요, 리모컨. 두 경로 동일, prompt 246 토큰. 피크 161.5 GB |
| 5초 16 kHz WAV, CLI와 서버 | `audio_tokens=100` = ceil(80000 / 800). 두 경로 모두 단어 단위 정확, WER 0. 피크 155.0 GB |
| 합성 좌→우 비디오 | 샘플 프레임 8, 쌍 [0,1], [2,3], [4,5], [6,7], 앵커 0.0, 1.1, 2.2, 3.3초, 비디오 토큰 216, prompt 319 토큰. 응답 "오른쪽으로", CLI와 서버 동일. 피크 157~158 GB |
| MTP: 네이티브 샤드, 128 토큰 일치, 수용률, 처리량 | `mtp.safetensors`에서 로드. 게이트 조건에서 일치, 평균 수용 길이 1.49~1.61, **처리량 이득 없음(미충족)**, 3.4 참조 |
| NVFP4 품질과 expert sidecar scale | 수정됨(2절). sidecar 매핑은 mlx-vlm과 동일 |

## 5. 테스트

| 테스트 | 고정하는 동작 |
|---|---|
| `models::sanitize::tests::inkling_configs_keep_checkpoint_dtype` | MLX 양자화 블록이 없는 Inkling config도 bf16 유지, `inkling2`는 해당 없음 |
| `drafter::inkling_mtp::tests::drafter_load_keeps_bf16_checkpoint_dtype`(mlxcel-core) | MTP 헤드가 bf16으로 로드됨 |
| `server::batch::speculative_burst_tests::mtp_capable_target_gates_inkling_on_the_exactness_verdict` | `mtp_capable_target`이 Inkling에 대해 seed된 게이트 판정을 따름 |
| `models::inkling::tiny_tests::exactness_probe_runs_on_the_tiny_model` | probe가 폭 3에서 실행되고 폭 1에서는 NotRun |

`models::inkling`, `models::speculative_exactness`, sanitize와 speculative-burst 테스트, `drafter::inkling_mtp`의 좁은 selector가 통과하고, `coverage_tests`, `-D warnings` clippy(lib, tests), `cargo fmt --check`, 저장소 계약 검사도 통과한다.

## 6. 한계와 후속 과제

- Inkling-Small에서 MTP는 속도 이득이 없고, M3 Ultra에서는 게이트가 기본으로 끈다. 수용률 개선은 모델이나 drafter의 문제이지 포팅 결함이 아니다.
- `T > 1` Inkling forward를 단일 토큰 디코드와 바이트 단위로 같게 만드는 작업(`T = 1`의 convolution tail과 attention 커널)은 이번에 다루지 않았다.
- 검증 호스트는 하나다. 메모리가 작은 Mac은 Small 체크포인트를 올릴 수 없다.
