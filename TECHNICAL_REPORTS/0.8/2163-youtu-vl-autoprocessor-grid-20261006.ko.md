# PR #2163: Youtu-VL 패치 그리드를 AutoProcessor와 일치시킴

**Date**: 2026-10-06
**Status**: CUDA(GB10)에서 구현 및 검증 완료. 상한에서의 Metal 메모리는 미검증.
**Risk**: Medium. 큰 이미지는 이전 4096 패치 상한보다 최대 9배 많은 비전 토큰을 만듭니다.

## 요약

체크포인트가 문서화한 진입점인 `processing_youtu_vl.py`의 `YoutuVLProcessor.__call__`은 이미지 프로세서를 `max_num_patches=36864`로 호출하고, `get_image_size_for_patches`로 크기를 정합니다. 각 변을 32의 배수로 올림하고(최소 한 블록), 그리드가 상한 안에 들어올 때까지 `scale`을 0.02씩 줄입니다. mlxcel은 세 군데가 달랐습니다. 상한을 `vision_config.num_patches`(4096)로 두었고, 변을 올림 대신 가장 가까운 배수로 반올림했으며, Qwen2-VL의 `min_pixels`/`max_pixels` 경계를 적용했습니다. 이 때문에 330 픽셀 변, 아주 작은 이미지, 한 변이 약 1024 픽셀을 넘는 이미지는 transformers와 다른 그리드를 받았습니다. Closes #1611.

## 변경 사항

- `DEFAULT_MAX_PATCHES_PER_IMAGE = 36864`이며 출처로 `processing_youtu_vl.py:53`을 적었습니다. 이 값은 Python 기본 인자로만 존재하므로 상수로 둡니다.
- `smart_resize`는 `get_image_size_for_patches`를 이식했습니다. Python과 같이 `scale`을 f64에서 제자리 감소시키고(`1.0 - 0.02*k`는 반올림 결과가 다름), 캐스트 전에 f64 변 길이를 클램프합니다. 상한이 4 패치 미만이면 레퍼런스는 끝없이 반복하지만, 여기서는 반복 횟수를 제한해 변마다 한 블록에서 멈추고 `TooManyPatches`가 이를 거부합니다.
- `build_processor`는 `vision_config.num_patches`, 이미지 프로세서 단독 기본값인 `max_num_patches: 256`, 쓰이지 않는 `num_patches` 키, 픽셀 경계를 모두 무시합니다. `min_pixels`, `max_pixels`, `with_pixel_bounds`, `effective_max_pixels`는 제거했습니다.

## GB10 검증

- AutoProcessor(transformers 4.56.0, remote code)가 11개 크기에 대해 돌려주는 `spatial_shapes`는 224 14x14, 330 22x22, 336 22x22, 448 28x28, 512 32x32, 2048 128x128, 1080x1920 68x120, 3000x4000 166x220, 330x500 22x32, 100x3000 8x188, 32x32 2x2입니다. 테이블 테스트가 11개를 모두 고정합니다. 합성 로더 테스트가 36864 상한을 확인하고, ignored 테스트가 실제 체크포인트에서 330, 2048, 3000x4000에 대해 `build_processor`를 실행합니다.
- `tests/youtu_vl_parity.rs --ignored`: 3/3 통과. 224, 336, 448 픽스처의 그리드는 바뀌지 않았습니다.
- CLI와 `mlxcel-server` 모두에서 이미지 토큰 수는 121, 4096, 9130(병합 그리드 = 패치 / 4)이고, 세 이미지 모두 올바르게 설명합니다.
- 48토큰 생성 한 번의 호스트 메모리 최대 증가량은 484 패치 12.6 GB, 16384 패치 18.6 GB, 36520 패치 30.1 GB입니다. OOM은 없었고 `NV_ERR_NO_MEMORY`는 기준값 그대로였습니다. 비전 head_dim 72는 cuDNN flash SDPA 대상이며, 그렇지 않은 경우에는 1024 MiB 쿼리 청크 폴백이 전체 어텐션 레이어의 메모리를 제한합니다.

## 후속 과제

- Metal: MLX의 fused SDPA가 head_dim 72를 받지 않으면 상한에서 전체 어텐션 레이어 네 개가 약 43 GB의 점수 행렬을 만듭니다. Metal 측정이 필요합니다.
- 윈도우 어텐션은 윈도우별 출력을 순차 fold로 이어 붙이므로, 복사량이 윈도우 수(상한에서 144개)의 제곱에 비례합니다. 상한에서 19초는 아직 허용 범위지만, concatenate 한 번으로 바꾸면 이 비용이 없어집니다.
