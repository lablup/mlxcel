# PR #2910: Qwen3-VL 영상 타임스탬프 보강

**작성일**: 2026-11-11
**이슈**: #2402

## 요약

샘플링 메타데이터, 타임스탬프 확장, 이미지·영상 순서, MRoPE 시간 그룹 위치, 모델 계열 분기, 실제 CLI/서버 준비 경로를 검토했다. Qwen3-VL, Qwen3-VL-MoE, Qwen3.5 dense/MoE VLM에 타임스탬프를 적용하고 Qwen2/2.5-VL의 기존 단일 블록은 보존한다.

## 발견 사항과 수정

- 시간 계산에서 곱셈 오버플로와 패치 크기만큼의 패딩 할당을 제거했다. 그룹 수를 비교하고 원래 청크의 처음·마지막 시간을 사용한다. 역순 인덱스와 비유한 결과를 거부하고 `usize::MAX`, 극소 fps 등의 회귀 테스트를 추가했다.
- 외부에서 전달된 그룹 시간이 음수 또는 비유한 값이면 토큰화 전에 거부한다.
- mlx-vlm 파생 코드의 라이선스 고지와 적용 범위 설명을 보완했다. 원래 PR 설명과 달리 Qwen의 native video는 CLI 이미지 fallback으로 가지 않으며 `generate --video`도 공통 서버 준비 경로를 호출한다.
- 새로운 URL 해석, 프로세스 실행, 공유 가변 상태, 무제한 요청 캐시는 추가하지 않았다. 기존 프레임 제한을 유지하고 시간 계산은 프레임 수에 선형이며 패딩 복사본을 만들지 않는다.

## 실제 체크포인트 근거

Apple M1 Ultra 128 GiB, macOS 27.2, Metal+Accelerate release 빌드. 이전 버전은 main `261080ba`, 수정 버전은 원래 PR에 main `1b63a947`까지 통합하고 보강 변경을 적용한 상태다. 동일한 로컬 `mlx-community/Qwen3-VL-4B-Instruct-4bit` 체크포인트와 greedy decoding, `--seed 1 -n 96 -p "Describe the motion."`을 사용했다.

영상은 224×224 RGB 40프레임, 10 fps이며 검은 배경에 32×32 흰 사각형 하나가 i번째 프레임의 `(12 + 4*i, 96)`에 있다. ffmpeg/libx264, yuv420p로 인코딩했다. MP4 SHA-256은 `ab433dbcf0ab530dbc9e0fe9b45b6bf46d67848bd8a66037f829cbfb56a240b3`이다. 기본 2 fps로 샘플링했다. 영상은 로컬 생성이며 바이너리 결과물로 커밋하지 않았다.

**이전:** 96토큰 상한까지 답하며 영상을 “motion blur effect”라고 해석하고 흰 사각형의 수평 블러와 잔상, 속도 표현을 설명했다. 전체 원문은 짝을 이루는 영문 보고서에 보존했다.

**수정 후:** 24토큰에서 EOS로 종료했다.

> A white square moves from the left side of the screen to the center, then to the right side of the screen.

두 번째 계열인 `llama-3.2-11b-vision-instruct-4bit`에는 같은 영상과 `--video-max-frames 2 -n 48 --seed 1`을 사용했다. 전후 모두 중단 없이 동일한 문장을 생성하여 이미지 프레임 fallback의 동작 보존을 확인했다. 두 버전 모두 펜과 손을 환각했으므로 해당 모델의 품질 통과가 아닌 회귀 근거로만 해석한다.

## 검증과 한계

Metal 활성화 test-fast로 `multimodal::qwen_vl`, `multimodal::vlm_runtime`, `multimodal::video`의 새 경계 테스트를 포함해 126개가 통과했고 13개는 ignored였다. 실제 체크포인트 출력은 정확성 근거이며 공유 머신에서 처리량 개선을 주장하지 않는다. Qwen3-VL-MoE와 Qwen3.5 VLM 실물 체크포인트는 실행하지 않았고 계열 플래그와 프롬프트 계산은 코드 구조·테스트로 검증했다. 전체 workspace 테스트는 실행하지 않았다.

ffmpeg 활성화 `multimodal::video` 검증은 `--include-ignored --skip bench_single_pass_768_frames --test-threads=1`로 53개가 통과했다. 제외한 테스트는 명시적인 비게이트 시간 벤치마크다. Metal+Accelerate Clippy(`--profile test-fast -p mlxcel --lib --tests -- -D warnings`), 포맷, diff 공백 검사도 통과했다. 기존 C++ unused-capture 경고는 남아 있다.
