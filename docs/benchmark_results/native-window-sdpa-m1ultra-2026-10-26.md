# Native sliding-window SDPA for Metal prefill (issue #2621)

Upstream MLX takes a `window_size` argument at `fast::scaled_dot_product_attention` since ml-explore/mlx#4734, and the pin bump in #2614 (`7ca40033132dae444d45f7fc3daae57e24c13f6c`) kept every mlxcel call site at `window_size = -1`. This report measures what routing a causal multi-query sliding-window prefill through that argument does on a pre-M5 Metal host, against the explicit full-width windowed mask mlxcel built before, and records the enablement decision.

- **Date:** 2026-10-26
- **Hardware:** Apple M1 Ultra, 128 GB unified memory (pre-M5: no Neural Accelerator; Apple GPU generation 13)
- **Software:** macOS 27.0.1 (26A434), Xcode 27.0 (27A266a), Rust 1.97.1
- **mlxcel:** branch `feature/issue-2621-native-window-sdpa` on top of `0ee8f03d`; MLX pin `7ca40033132dae444d45f7fc3daae57e24c13f6c`
- **Build:** `cargo build --release --features metal,accelerate`
- **Not run:** M5 / Neural Accelerator hosts (none available in this run), CUDA, ROCm. Nothing below is evidence for them.

## What the route does

`mlxcel_core::window_sdpa` decides, for a causal multi-query prefill whose returned keys outgrow the sliding window and that carries no softcap, sink or extra mask, between two SDPA entries:

- **legacy:** `create_causal_mask_with_window_full(q_len, k_len - q_len, Some(W))` plus the array-mask SDPA (the pre-#2621 over-window branch of `causal_attention`, kept byte for byte);
- **native:** upstream `scaled_dot_product_attention(q, k, v, scale, W, "causal")`, no mask array.

The semantic mapping is checked in `window_sdpa_tests.rs`: queries are the suffix of the returned keys, row `r` sees key `c` iff `c <= r && r - c < W` (the current position and up to `W - 1` earlier ones), on both entries, against an f64 host reference. The selector is `MLXCEL_NATIVE_WINDOW_SDPA` (`0`, `1`, `auto`), the evidence `MLXCEL_LOG_WINDOW_SDPA`; both are documented in `docs/environment-variables.md`.

## Eligibility on the pinned MLX

Read from [`mlx/backend/metal/scaled_dot_product_attention.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp) at the pin, and confirmed by the `sdpa_uses_fused_kernel` probe (MLX's own `ScaledDotProductAttention::use_fallback`), which the dispatch log prints for both arms.

| Call | Static route | Fused kernel on M1 Ultra (pre-NAX) |
| --- | --- | --- |
| Causal window prefill, `q_len > 8`, `q_len <= k_len`, `k_len > W`, D in {64, 72, 80, 96, 128, 192, 256}, f16/bf16/f32, no softcap/sink/mask | native eligible | D=64..128: yes, any width. D=192/256: no, except a fresh (`q_len == k_len`) f16/bf16 prefill of 2048 rows or more at D=256. The legacy array-mask arm at D=192/256 is never fused (the D=256 carve-out requires causal mode). |
| Same, keys within the window | legacy (plain causal, `"causal"` mode, unchanged) | n/a |
| `q_len <= 8` (decode, speculative verify blocks) | legacy (unchanged) | vector kernels |
| Softcap (Gemma 2), sinks (gpt-oss), padding / segment / per-head / cache-validity masks | legacy (unchanged) | n/a |
| CPU, CUDA, ROCm | legacy (unchanged; a positive window there selects MLX's graph fallback) | n/a |

Every cached sliding-window checkpoint on this host has head dim 256 (`gemma-3-1b-it-4bit` 4/1 heads, W=512; `gemma-4-e2b-it-4bit` 8/1, W=512; `gemma-4-e4b-it-4bit` 8/2, W=512; `gemma-4-31b-it-4bit` 32/16, W=1024). So on this host the fused window kernel is reached by a real model only for a fresh prefill of at least 2048 tokens; every chunked continuation and every shorter prefill of these models is the graph fallback on both arms. Ministral 3 (head dim 128, where the fused window kernel is reached at any width) has no cached checkpoint, so its model-level numbers are not in this report; the D=128 rows of the microbenchmark are the kernel-level evidence for that geometry.

## Protocol, fixed before the first timing

- **Noise band:** the 2026-09-24 report on this host saw about 2% run to run under load. Each timed cell here is the median of 5 samples of `iters` calls, arms interleaved in alternating order, GPU synchronized at both ends of every sample, 5 warmup calls per arm excluded. A baseline-against-itself repeat is reported first; its spread is the noise band.
- **Repeatable benefit:** the candidate median is at least 3% faster than the baseline median, the two arms' min/max ranges do not overlap, and no shape in the same eligibility class is more than 3% slower.
- **Default enablement:** the route is enabled by default (`auto`) only if (a) every correctness check in `window_sdpa_tests.rs` passes at the tolerances stated there (f32 3e-5, f16 2e-3, bf16 1.6e-2 against the host reference; invisible keys bitwise inert), (b) the teacher-forced logit trace at the width the route runs at shows no disagreement on decided positions (`--decided 2`), and (c) the full-model prefill tokens/s shows a repeatable benefit on at least one eligible shape with no eligible shape regressing. `auto` means MLX's fused verdict gates the native entry, so shapes that would take the graph fallback stay legacy regardless; a neutral result keeps the legacy default.
- **Correctness first:** no timing below was read before the correctness suite passed.

## Results

### Kernel level: `examples/window_sdpa_microbench`

Raw data: `native-window-sdpa-m1ultra-2026-10-26-microbench.csv` (one row per arm and shape: median, min and max of 5 samples, `iters` calls per sample, 5 warmup calls, peak bytes of one call). Command: `./target/release/examples/window_sdpa_microbench > microbench.csv`, run alone on the GPU. Three arms per shape, interleaved: `legacy, mask + SDPA` (the pre-#2621 path: `create_causal_mask_with_window_full` plus the array-mask SDPA, both per call), `legacy, prebuilt mask` (the same SDPA with the mask built once, so the attention cost is separated from the mask construction), `native` (upstream `window_size` with `"causal"`). `native fused` and `legacy fused` are MLX's `use_fallback` verdicts for the two arms over that shape. Continuation rows hold `W - 1` prior keys plus the chunk, the layout `RotatingKVCache` returns (#678). Windows equal to or larger than the visible keys were not timed: both arms run plain causal attention there (`keys_within_window` in the eligibility matrix), so there is no window route to compare.

| Geometry | Kind | dtype | q_len | k_len | W | native fused | legacy fused | legacy, mask + SDPA (us) | legacy, prebuilt mask (us) | native (us) | native / legacy | peak MiB legacy -> native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 2048 | 2048 | 512 | true | false | 2395 (2377 to 2876) | 2258 (2245 to 2261) | 1256 (1244 to 1283) | 0.52x | 94 -> 26 |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 2048 | 2048 | 1024 | true | false | 2392 (2366 to 2599) | 2253 (2252 to 2275) | 2117 (2111 to 2160) | 0.89x | 94 -> 26 |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 4096 | 4096 | 512 | true | false | 8336 (8322 to 8467) | 8060 (8050 to 8181) | 2090 (2081 to 2169) | 0.25x | 348 -> 84 |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 4096 | 4096 | 1024 | true | false | 8374 (8280 to 8395) | 8052 (8042 to 8208) | 3653 (3630 to 3816) | 0.44x | 348 -> 84 |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 8192 | 8192 | 512 | true | false | 30808 (30656 to 31030) | 29871 (29821 to 29897) | 3871 (3841 to 3906) | 0.13x | 1336 -> 296 |
| gemma3-1b H4/KV1/D256 | fresh | bf16 | 8192 | 8192 | 1024 | true | false | 30943 (30809 to 31103) | 29810 (29795 to 29933) | 7086 (6985 to 7107) | 0.23x | 1336 -> 296 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 2048 | 2559 | 512 | false | false | 2909 (2848 to 3810) | 2723 (2714 to 3409) | 2867 (2814 to 3271) | 0.99x | 115 -> 105 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 2048 | 3071 | 1024 | false | false | 3385 (3361 to 3465) | 3209 (3204 to 3222) | 3299 (3293 to 3319) | 0.97x | 135 -> 123 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 4096 | 4607 | 512 | false | false | 9696 (9622 to 9748) | 9398 (9273 to 9435) | 9699 (9632 to 9739) | 1.00x | 389 -> 352 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 4096 | 5119 | 1024 | false | false | 10522 (10488 to 10553) | 10154 (10120 to 10266) | 10543 (10536 to 10587) | 1.00x | 429 -> 389 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 8192 | 8703 | 512 | false | false | 32890 (32853 to 32946) | 31253 (31152 to 31286) | 32837 (32747 to 32899) | 1.00x | 1416 -> 1280 |
| gemma3-1b H4/KV1/D256 | continuation | bf16 | 8192 | 9215 | 1024 | false | false | 34751 (34661 to 34889) | 32974 (32923 to 33145) | 34646 (34621 to 34685) | 1.00x | 1497 -> 1353 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 2048 | 2048 | 512 | true | false | 2534 (2484 to 2740) | 2363 (2283 to 2437) | 1279 (1230 to 1336) | 0.50x | 94 -> 26 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 2048 | 2048 | 1024 | true | false | 1997 (1973 to 2138) | 1870 (1804 to 1890) | 1694 (1657 to 1701) | 0.85x | 94 -> 26 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 4096 | 4096 | 512 | true | false | 6650 (6623 to 6672) | 6366 (6342 to 6369) | 1621 (1618 to 1630) | 0.24x | 348 -> 84 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 4096 | 4096 | 1024 | true | false | 6632 (6618 to 6722) | 6365 (6344 to 6403) | 2865 (2831 to 2898) | 0.43x | 348 -> 84 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 8192 | 8192 | 512 | true | false | 24944 (24897 to 24962) | 23404 (23394 to 23464) | 3030 (3014 to 3059) | 0.12x | 1336 -> 296 |
| gemma3-1b H4/KV1/D256 | fresh | f16 | 8192 | 8192 | 1024 | true | false | 24935 (24907 to 25025) | 23445 (23391 to 23530) | 5533 (5507 to 5569) | 0.22x | 1336 -> 296 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 2048 | 2559 | 512 | false | false | 3009 (2656 to 3238) | 2856 (2515 to 2958) | 2790 (2716 to 3178) | 0.93x | 115 -> 105 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 2048 | 3071 | 1024 | false | false | 2919 (2852 to 3258) | 2717 (2674 to 2897) | 2861 (2785 to 3092) | 0.98x | 135 -> 123 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 4096 | 4607 | 512 | false | false | 7967 (7944 to 8054) | 7655 (7596 to 7756) | 7951 (7938 to 8077) | 1.00x | 389 -> 352 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 4096 | 5119 | 1024 | false | false | 8659 (8641 to 8769) | 8300 (8265 to 8336) | 8691 (8680 to 8714) | 1.00x | 429 -> 389 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 8192 | 8703 | 512 | false | false | 26920 (26869 to 27009) | 25224 (25202 to 25250) | 26960 (26944 to 27017) | 1.00x | 1416 -> 1280 |
| gemma3-1b H4/KV1/D256 | continuation | f16 | 8192 | 9215 | 1024 | false | false | 28401 (28068 to 28511) | 26670 (26596 to 26697) | 28419 (28081 to 28446) | 1.00x | 1497 -> 1353 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 2048 | 2048 | 512 | true | false | 11968 (11927 to 12013) | 11864 (11837 to 11910) | 5143 (5128 to 5182) | 0.43x | 432 -> 112 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 2048 | 2048 | 1024 | true | false | 11951 (11933 to 12815) | 11876 (11840 to 11967) | 8501 (8472 to 8602) | 0.71x | 432 -> 112 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 4096 | 4096 | 512 | true | false | 45674 (45582 to 45706) | 45495 (45482 to 45558) | 10389 (10352 to 10442) | 0.23x | 1472 -> 256 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 4096 | 4096 | 1024 | true | false | 45424 (45398 to 45456) | 45355 (45345 to 45401) | 18415 (18385 to 18478) | 0.41x | 1472 -> 256 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 8192 | 8192 | 512 | true | false | 189531 (189202 to 192464) | 188355 (188273 to 269618) | 20893 (20857 to 20974) | 0.11x | 5376 -> 640 |
| gemma4-31b H32/KV16/D256 | fresh | f16 | 8192 | 8192 | 1024 | true | false | 189845 (189188 to 192354) | 188471 (188237 to 188489) | 38422 (38390 to 38453) | 0.20x | 5376 -> 640 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 2048 | 2559 | 512 | false | false | 15022 (14979 to 15074) | 14873 (14838 to 17390) | 15015 (14990 to 15305) | 1.00x | 516 -> 506 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 2048 | 3071 | 1024 | false | false | 17853 (17816 to 17853) | 17711 (17696 to 17718) | 17843 (17802 to 17859) | 1.00x | 600 -> 588 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 4096 | 4607 | 512 | false | false | 54964 (54890 to 54969) | 54555 (54528 to 65278) | 55072 (55056 to 55082) | 1.00x | 1632 -> 1596 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 4096 | 5119 | 1024 | false | false | 60070 (60050 to 60584) | 59653 (59629 to 59701) | 60204 (60161 to 60247) | 1.00x | 1792 -> 1752 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 8192 | 8703 | 512 | false | false | 203936 (203854 to 204574) | 203270 (203106 to 204562) | 212750 (212725 to 212866) | 1.04x | 5688 -> 5552 |
| gemma4-31b H32/KV16/D256 | continuation | f16 | 8192 | 9215 | 1024 | false | false | 215335 (215235 to 215629) | 215725 (214344 to 218070) | 224610 (224567 to 224666) | 1.04x | 6000 -> 5856 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 2048 | 2048 | 512 | true | false | 3444 (3384 to 4032) | 3270 (3254 to 3405) | 1623 (1598 to 1769) | 0.47x | 140 -> 36 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 2048 | 2048 | 1024 | true | false | 3410 (3401 to 3418) | 3270 (3261 to 3286) | 2772 (2754 to 2816) | 0.81x | 140 -> 36 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 4096 | 4096 | 512 | true | false | 12224 (12188 to 12300) | 11944 (11916 to 11985) | 2970 (2954 to 3022) | 0.24x | 504 -> 104 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 4096 | 4096 | 1024 | true | false | 12214 (12155 to 12220) | 11936 (11905 to 11961) | 5103 (5064 to 5213) | 0.42x | 504 -> 104 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 8192 | 8192 | 512 | true | false | 47465 (47112 to 47495) | 45831 (45718 to 47022) | 5621 (5583 to 5677) | 0.12x | 1904 -> 336 |
| gemma4-e4b H8/KV2/D256 | fresh | f16 | 8192 | 8192 | 1024 | true | false | 47451 (47052 to 47458) | 45825 (45747 to 45890) | 10182 (10138 to 10235) | 0.21x | 1904 -> 336 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 2048 | 2559 | 512 | false | false | 4223 (4221 to 4955) | 4108 (4088 to 4560) | 4197 (4192 to 4219) | 0.99x | 169 -> 159 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 2048 | 3071 | 1024 | false | false | 5047 (5030 to 6110) | 4888 (4878 to 4926) | 4989 (4981 to 5017) | 0.99x | 198 -> 186 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 4096 | 4607 | 512 | false | false | 14697 (14654 to 14826) | 14330 (14291 to 14434) | 14592 (14560 to 14635) | 0.99x | 561 -> 525 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 4096 | 5119 | 1024 | false | false | 16087 (16001 to 16135) | 15663 (15619 to 15738) | 15932 (15913 to 15982) | 0.99x | 618 -> 578 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 8192 | 8703 | 512 | false | false | 51143 (50830 to 51213) | 49460 (49401 to 49537) | 50959 (50936 to 50998) | 1.00x | 2017 -> 1881 |
| gemma4-e4b H8/KV2/D256 | continuation | f16 | 8192 | 9215 | 1024 | false | false | 53979 (53589 to 54054) | 52151 (52070 to 71257) | 53762 (53553 to 53839) | 1.00x | 2130 -> 1986 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 2048 | 2048 | 512 | true | true | 6504 (6482 to 6722) | 6364 (6347 to 6408) | 1776 (1771 to 1800) | 0.27x | 88 -> 56 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 2048 | 2048 | 1024 | true | true | 6517 (6499 to 6632) | 6375 (6367 to 6579) | 2806 (2794 to 2826) | 0.43x | 88 -> 56 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 4096 | 4096 | 512 | true | true | 24688 (24654 to 24741) | 24371 (24329 to 24416) | 3398 (3370 to 7435) | 0.14x | 272 -> 144 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 4096 | 4096 | 1024 | true | true | 24675 (24667 to 24696) | 24373 (24288 to 24374) | 5908 (5880 to 5980) | 0.24x | 272 -> 144 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 8192 | 8192 | 512 | true | true | 97444 (97387 to 98896) | 95952 (95880 to 96294) | 6699 (6681 to 6836) | 0.07x | 928 -> 416 |
| upstream-ref H32/KV8/D128 | fresh | f16 | 8192 | 8192 | 1024 | true | true | 97259 (97192 to 97462) | 95994 (95807 to 96189) | 12112 (12004 to 12139) | 0.12x | 928 -> 416 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 2048 | 2559 | 512 | true | true | 9539 (9520 to 9580) | 9410 (9391 to 9427) | 2134 (2126 to 2144) | 0.22x | 102 -> 62 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 2048 | 3071 | 1024 | true | true | 11336 (11299 to 11346) | 11184 (11145 to 11212) | 3831 (3818 to 3956) | 0.34x | 116 -> 68 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 4096 | 4607 | 512 | true | true | 33215 (33162 to 33483) | 32792 (32747 to 32889) | 3989 (3980 to 3992) | 0.12x | 298 -> 154 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 4096 | 5119 | 1024 | true | true | 36956 (36905 to 37012) | 36417 (36379 to 36473) | 7436 (7402 to 7485) | 0.20x | 324 -> 164 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 8192 | 8703 | 512 | true | true | 124322 (124140 to 124573) | 122985 (122776 to 123062) | 7765 (7685 to 7786) | 0.06x | 978 -> 434 |
| upstream-ref H32/KV8/D128 | continuation | f16 | 8192 | 9215 | 1024 | true | true | 131604 (131564 to 131714) | 130052 (129735 to 130182) | 14397 (14372 to 14433) | 0.11x | 1028 -> 452 |

Three things the table says, all repeatable well outside the sample spread:

- Where the native arm reaches the fused kernel and the legacy arm does not (every fresh D=256 prefill of 2048 rows or more on this pre-NAX GPU), the native entry runs the attention in 0.50x to 0.11x of the legacy time at W=512 and 0.89x to 0.20x at W=1024, with the gain growing with the key length because the steel kernel skips the key blocks outside the band while the graph fallback materializes and masks the whole `[H, L, L]` score matrix. Peak memory of the call drops with it, 94 MiB to 26 MiB for Gemma 3 1B at 2048 rows and 5376 MiB to 640 MiB for the 31B geometry at 8192 rows.
- Where both arms are the graph fallback (every D=256 continuation chunk here), the native entry is within the noise of the legacy arm at 2048 and 4096 rows (0.93x to 1.00x) and 4% slower at the 31B geometry at 8192 rows, where the in-graph window mask MLX's fallback builds costs more than the mask mlxcel builds. `auto` therefore keeps the legacy route on these shapes; `1` is the arm that measured them.
- Where both arms are fused (the upstream reference geometry at D=128), the native entry is 0.43x to 0.06x of the legacy time on fresh and continuation shapes alike, the band skipping that ml-explore/mlx#4734 reports. No cached checkpoint has this geometry, so it is kernel-level evidence only.

Mask construction itself is under 1% (31B geometry) to 7% (2048 rows) of the legacy call (the gap between the two legacy columns), so the speedup is the kernel, not the saved mask.

Query lengths 1, 4 and 8 run the production route for those shapes (`causal_attention`, unchanged by #2621: a single query over a rolled-over window slices K/V to the window; a small block keeps the vector kernel with an explicit mask) against the forced native entry:

| Geometry | dtype | q_len | k_len | W | production route (us) | forced native (us) | forced / production |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b H4/KV1/D256 | bf16 | 1 | 513 | 512 | 354 (343 to 358) | 284 (270 to 290) | 0.80x |
| gemma3-1b H4/KV1/D256 | bf16 | 4 | 516 | 512 | 404 (357 to 431) | 340 (320 to 360) | 0.84x |
| gemma3-1b H4/KV1/D256 | bf16 | 8 | 520 | 512 | 423 (421 to 426) | 360 (357 to 363) | 0.85x |
| gemma3-1b H4/KV1/D256 | f16 | 1 | 513 | 512 | 336 (333 to 516) | 264 (258 to 280) | 0.79x |
| gemma3-1b H4/KV1/D256 | f16 | 4 | 516 | 512 | 383 (350 to 387) | 322 (315 to 327) | 0.84x |
| gemma3-1b H4/KV1/D256 | f16 | 8 | 520 | 512 | 385 (382 to 390) | 321 (321 to 328) | 0.84x |
| gemma4-31b H32/KV16/D256 | f16 | 1 | 513 | 512 | 360 (354 to 375) | 291 (273 to 294) | 0.81x |
| gemma4-31b H32/KV16/D256 | f16 | 4 | 516 | 512 | 392 (382 to 533) | 331 (328 to 350) | 0.85x |
| gemma4-31b H32/KV16/D256 | f16 | 8 | 520 | 512 | 519 (509 to 524) | 464 (456 to 475) | 0.89x |
| gemma4-e4b H8/KV2/D256 | f16 | 1 | 513 | 512 | 356 (354 to 362) | 290 (286 to 291) | 0.81x |
| gemma4-e4b H8/KV2/D256 | f16 | 4 | 516 | 512 | 352 (336 to 359) | 302 (291 to 305) | 0.86x |
| gemma4-e4b H8/KV2/D256 | f16 | 8 | 520 | 512 | 383 (376 to 394) | 323 (316 to 327) | 0.84x |
| upstream-ref H32/KV8/D128 | f16 | 1 | 513 | 512 | 352 (348 to 683) | 276 (274 to 290) | 0.78x |
| upstream-ref H32/KV8/D128 | f16 | 4 | 516 | 512 | 389 (376 to 550) | 319 (307 to 323) | 0.82x |
| upstream-ref H32/KV8/D128 | f16 | 8 | 520 | 512 | 450 (437 to 453) | 374 (370 to 375) | 0.83x |

The vector kernels take the window too, 15% to 22% faster than the production route at these sizes, which is a finding for a later issue and not a change here: the route keeps `q_len <= 8` on its existing path, so decode and the speculative verify blocks (whose byte identity with decode is measured at startup by the MTP probe) are untouched.

### Full model: `mlxcel generate`

Raw data: `native-window-sdpa-m1ultra-2026-10-26-full-model.csv`, one row per run, with the per-run `window-sdpa` dispatch counts parsed from stderr. Command per run (`ARM` in `0`, `1`, `auto`; `N` in 512, 2048, 4096, 8192):

```bash
MLXCEL_NATIVE_WINDOW_SDPA=$ARM MLXCEL_LOG_WINDOW_SDPA=all /usr/bin/time -l \
  ./target/release/mlxcel generate -m ~/.cache/mlxcel/models/mlx-community/$MODEL \
  -p "$(cat prompt-$N.txt)" -n 64 --temp 0 --no-chat-template --profile
```

Prompts are the first 2120, 7420, 14360 and 29000 characters of `docs/benchmarks.md` followed by `docs/CONTINUOUS_BATCHING.md` (502, 2105, 4129 and 8187 Gemma tokens); the same bytes go to every arm. Arms alternate per repetition (`0 1 auto`, `auto 1 0`, `1 0 auto`), each run is a fresh process with its own model load, GPU idle otherwise. `Prefill` is the `--profile` line (prompt tokens over the time from the first forward to the first sampled token, so it is also the TTFT without tokenizer and load time); `Decode` covers the 64 generated tokens. Max RSS is `/usr/bin/time -l`, dominated by the weights. Checkpoints: `mlx-community/gemma-3-1b-it-4bit` (config.json sha256 `8a92da31`, 4-bit group 64, f16 activations), `mlx-community/gemma-4-e4b-it-4bit` (`780ccb3a`, 4-bit, bf16 activations), `mlx-community/gemma-4-31b-it-4bit` (`a01be3d4`, 4-bit, bf16 activations); the Hub revision is not recorded in the cache, so the config hash is the identity.

| Model | Prompt tokens | Arm | Runs | Prefill ms, median (min to max) | Prefill tok/s, median | Decode tok/s, median | Native dispatches (of them fused) | Max RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-1b-it-4bit | 502 | `0` | 3 | 175 (171 to 180) | 2864 | 193.7 | 0 (0) | 1248 |
| gemma-3-1b-it-4bit | 502 | `auto` | 3 | 174 (173 to 176) | 2887 | 193.7 | 0 (0) | 1249 |
| gemma-3-1b-it-4bit | 502 | `1` | 3 | 176 (175 to 178) | 2860 | 193.9 | 0 (0) | 1254 |
| gemma-3-1b-it-4bit | 2105 | `0` | 3 | 432 (423 to 432) | 4873 | 197.4 | 0 (0) | 1308 |
| gemma-3-1b-it-4bit | 2105 | `auto` | 3 | 425 (417 to 435) | 4952 | 196.9 | 22 (22) | 1240 |
| gemma-3-1b-it-4bit | 2105 | `1` | 3 | 428 (428 to 428) | 4918 | 195.8 | 44 (22) | 1308 |
| gemma-3-1b-it-4bit | 4129 | `0` | 3 | 759 (747 to 760) | 5438 | 197.1 | 0 (0) | 1244 |
| gemma-3-1b-it-4bit | 4129 | `auto` | 3 | 752 (745 to 760) | 5492 | 197.4 | 22 (22) | 1243 |
| gemma-3-1b-it-4bit | 4129 | `1` | 3 | 772 (765 to 775) | 5350 | 197.6 | 66 (22) | 1243 |
| gemma-3-1b-it-4bit | 8187 | `0` | 3 | 1393 (1385 to 1425) | 5877 | 190.2 | 0 (0) | 1244 |
| gemma-3-1b-it-4bit | 8187 | `auto` | 3 | 1405 (1375 to 1428) | 5825 | 187.0 | 22 (22) | 1308 |
| gemma-3-1b-it-4bit | 8187 | `1` | 3 | 1398 (1382 to 1430) | 5858 | 190.0 | 88 (22) | 1311 |
| gemma-4-31b-it-4bit | 2105 | `0` | 2 | 15363 (15281 to 15445) | 137 | 22.1 | 0 (0) | 36484 |
| gemma-4-31b-it-4bit | 2105 | `auto` | 2 | 15114 (15082 to 15147) | 139 | 22.4 | 50 (50) | 36484 |
| gemma-4-31b-it-4bit | 4129 | `0` | 2 | 38775 (30456 to 47093) | 112 | 16.7 | 0 (0) | 33361 |
| gemma-4-31b-it-4bit | 4129 | `auto` | 2 | 33478 (31963 to 34992) | 124 | 20.0 | 50 (50) | 36486 |
| gemma-4-31b-it-4bit | 8187 | `0` | 2 | 63949 (60427 to 67471) | 128 | 20.6 | 0 (0) | 36485 |
| gemma-4-31b-it-4bit | 8187 | `auto` | 2 | 60264 (60234 to 60294) | 136 | 20.9 | 50 (50) | 36485 |
| gemma-4-e4b-it-4bit | 502 | `0` | 2 | 955 (642 to 1269) | 589 | 85.7 | 0 (0) | 12125 |
| gemma-4-e4b-it-4bit | 502 | `auto` | 2 | 631 (617 to 645) | 796 | 85.8 | 0 (0) | 12118 |
| gemma-4-e4b-it-4bit | 502 | `1` | 2 | 617 (616 to 618) | 814 | 85.5 | 0 (0) | 12118 |
| gemma-4-e4b-it-4bit | 2105 | `0` | 6 | 2327 (2311 to 2539) | 904 | 81.6 | 0 (0) | 12117 |
| gemma-4-e4b-it-4bit | 2105 | `auto` | 6 | 2257 (2229 to 2276) | 932 | 81.0 | 35 (35) | 12116 |
| gemma-4-e4b-it-4bit | 2105 | `1` | 2 | 2233 (2229 to 2238) | 942 | 82.8 | 70 (35) | 12121 |
| gemma-4-e4b-it-4bit | 4129 | `0` | 6 | 4545 (4481 to 4845) | 908 | 78.9 | 0 (0) | 12119 |
| gemma-4-e4b-it-4bit | 4129 | `auto` | 6 | 4496 (4425 to 4541) | 918 | 81.2 | 35 (35) | 12119 |
| gemma-4-e4b-it-4bit | 4129 | `1` | 2 | 4437 (4436 to 4438) | 931 | 81.8 | 105 (35) | 12120 |
| gemma-4-e4b-it-4bit | 8187 | `0` | 2 | 8885 (8852 to 8919) | 921 | 71.9 | 0 (0) | 12119 |
| gemma-4-e4b-it-4bit | 8187 | `auto` | 2 | 8796 (8795 to 8796) | 931 | 72.1 | 35 (35) | 12120 |
| gemma-4-e4b-it-4bit | 8187 | `1` | 2 | 8818 (8809 to 8826) | 928 | 72.3 | 140 (35) | 12118 |

How to read the dispatch column: the production prefill is chunked at `DEFAULT_PREFILL_CHUNK = 2048` tokens, so a 2105-token prompt is one fresh 2048-row chunk plus a 57-row continuation, 4129 tokens is one fresh chunk plus two continuations, and so on. Under `auto` the native route engages on the fresh 2048-row chunk of every sliding layer (22 of Gemma 3 1B's 26 layers, 35 of E4B's 42, 50 of the 31B's 60) and MLX reports every one of those dispatches fused; the continuation chunks are head-dim 256 at `q_len < k_len`, which MLX serves from its graph fallback, so `auto` leaves them on the model-built mask (they do not appear in the count because `causal_window_attention` is never reached). Under `1` every chunk takes the native entry (44, 66, 88 dispatches on Gemma 3 1B), of which only the first 22 are fused. The 502-token prompts never outgrow the 512 window, so no arm has a window dispatch there; the E4B `0` row at 502 tokens holds the first run of the whole sweep (1269 ms, Metal shader compilation) and is not a route effect.

Prefill time, `auto` against `0`:

- Gemma 3 1B: 2105 tokens -1.6%, 4129 tokens -0.9%, 8187 tokens +0.9%; all inside the 2% to 3% run-to-run band of the `0` arm against itself. The saved attention time per the microbench (about 1.2 ms per layer, 22 layers) is about 6% of a 432 ms prefill, and that much does not show above the noise at n = 3.
- Gemma 4 E4B (n = 6 at 2105 and 4129 tokens): 2105 tokens -3.0% (2327 ms to 2257 ms; the `0` arm's fastest run, 2311 ms, is slower than the `auto` arm's slowest, 2276 ms), 4129 tokens -1.1% (ranges overlap), 8187 tokens -1.0% (n = 2).
- Gemma 4 31B (n = 2): 2105 tokens -1.6%, 4129 tokens -13.7% and 8187 tokens -5.8%, where the `0` arm spreads from 30.5 s to 47.1 s and 60.4 s to 67.5 s between its two runs while `auto` stays within 3 s and 0.1 s; the 31B's legacy arm materializes a `[32, 2048, 2048]` bf16 score matrix per sliding layer per chunk (the 432 MiB peak in the microbench), and the run-to-run spread is consistent with allocator pressure from those transients rather than with attention arithmetic. Two runs cannot settle the size of the 31B gain; they can say its direction and that `auto` removes the variance.

Decode tokens/s is unchanged on every row (the decode step is a single query over a window-sized ring and never reaches the route), and max RSS does not move at the process level because the weights dominate it; the call-level peak reduction is in the microbench table.

### Output: greedy text and the teacher-forced logit trace

The 64 greedy tokens of every run were compared across arms. Gemma 3 1B (f16): all three arms are byte-identical to each other at every prompt size, and each arm is identical across its repetitions. Gemma 4 E4B and 31B (bf16): each arm is identical across its repetitions, `1` and `auto` are identical to each other at every size, and both differ from `0`: E4B diverges at generated word 24 of 44 (2105 tokens), 12 of 39 (4129) and 24 of 35 (8187), both arms continuing the document coherently; the 31B, prompted with raw document text and no chat template, falls into a repetition loop (`same same same`, `-// a`) in every arm at every size, and inside that loop the arms part at word 2 to 5, which is a run of near ties rather than a prefix the route changed. `1` equals `auto` even though `1` also sends the continuation chunks through the native entry, so the difference comes from the fused kernel on the fresh chunk, not from the graph-fallback window mask.

The teacher-forced trace (`examples/logit_trace`, `scripts/compare_logit_traces.py --decided 2`; full outputs with the route evidence lines in `native-window-sdpa-m1ultra-2026-10-26-logit-trace.txt`) quantifies that. Every chunk is a fresh forward of the stated width (`PREFILL = 0`, BOS anchored), which is the shape the route runs at; chunk widths 2048 and 4096 reach the fused kernel (the candidate's stderr shows `route=native ... native_fused=true` for `q=[1, 4, 2049, 256]` and `[1, 4, 4097, 256]`), width 512 does not (`native_fused=false`, both arms on the graph fallback), so the 512 trace is the arm where only the mask expression changes.

| Model | Width | Chunks | Positions | Reference | Candidate | Top-1 disagreement | At decided positions (gap >= 2) | Largest gap at a disagreement | Rank of the reference's pick, worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 1B | 2048 | 4 | 8192 | `0`, run a | `0`, run b | 0 | 0 / 3537 | none | identical |
| Gemma 3 1B | 2048 | 4 | 8192 | `0` | `1` | 24 (0.29%) | 0 / 3537 | 0.5 | 2 |
| Gemma 3 1B | 4096 | 2 | 8192 | `0` | `1` | 21 (0.26%) | 0 / 3387 | 0.5 | 2 |
| Gemma 3 1B | 512 | 4 | 2048 | `0` | `1` | 0 | 0 / 895 | none | identical |
| Gemma 4 E4B | 2048 | 3 | 6144 | `0` | `1` | 119 (1.94%) | 0 / 3150 | 0.75 | 4 |

Gemma 3 1B: every disagreement sits where the reference's top two were less than 0.5 logits apart, and the candidate picked the reference's runner-up at all of them; perplexity moves by 0.013% and -0.003%. Gemma 4 E4B (bf16): 117 of the 119 disagreements are in the under-0.5 bin (12.2% of those positions), two are between 0.5 and 1, none above 1; the reference's pick is the candidate's runner-up at 93% of them, perplexity moves by -0.064%. MLX's graph fallback scales the query and forms the scores and the probabilities in the activation dtype before the value matmul, while the steel kernel keeps them in float32, so the two arms round differently and bf16, with its 8-bit mantissa, rounds more; near-tie positions flip, decided positions do not. The greedy-text differences above are these flips compounding through a free-running generation. The reverted arm (`0`) traced against itself is byte-identical in effect, so the measurement would have reported a behaviour change had there been one.

### Correctness suite

`cargo test --release -p mlxcel-core --features metal,accelerate window_sdpa_tests::` (13 tests): native against legacy and against the f64 host reference within the stated tolerances over all 8 shape cases at f32, f16 and bf16; keys outside every window and keys after a row's position are bitwise inert on the native entry while a visible key moves its row; the eligibility matrix and mode plan as pure functions; `causal_attention`'s over-window branch takes the planned route and its legacy arm is byte-identical to the pre-#2621 masked call; a softcapped over-window prefill stays on the softcap composite; the rotating cache's unmasked prefill, continuation after a wrap, 20 decode steps and a post-trim chunk match the explicit-mask reference with cursors in agreement; and the fused probe matches the pinned dispatch rules on this GPU. The existing `attend_tests::` (18), the `causal_attention` and window-mask tests of `layers` and `utils` (84 selected) pass; `attend_tests::batched_paged_one_row_is_one_pooled_launch` fails under the default parallel runner which is consistent with a shared-counter race (it reads a process-global paged launch counter that the sibling paged tests also increment), and passes with `--test-threads=1` on this branch; the paged path is not touched by this change and the parallel failure was not reproduced on `main` in this run. `cargo clippy --release -p mlxcel-core --features metal,accelerate --lib --tests -- -D warnings` is clean.

## Decision

`MLXCEL_NATIVE_WINDOW_SDPA` defaults to `auto`.

Against the criterion fixed above: (a) the correctness suite passes at the stated tolerances; (b) the logit trace at the widths the route runs at shows zero disagreement on decided positions for the f16 and the bf16 family, with the reverted arm byte-identical to itself; (c) Gemma 4 E4B at 2105 tokens is a repeatable 3.0% prefill gain with non-overlapping ranges at n = 6, the 31B rows point the same way at n = 2, and no shape regresses beyond its noise band under `auto`. Criterion (c) is met at its margin: one shape (E4B at 2105 tokens, 3.0% against a 3% bar) carries it, and the 31B rows are directional evidence only. The kernel-level gain is 2x to 8x with 3.5x to 8x less peak memory on the shapes `auto` selects, and those shapes are exactly the ones where MLX's own dispatch reports a fused kernel; everything else keeps the legacy path, which is byte-identical to the previous release.

What `auto` does not do on this host: it never engages on a continuation chunk or a prefill shorter than 2048 rows for a head-dim 256 family, because MLX serves those from the graph fallback and the measured native arm there is neutral to 4% slower. A head-dim 128 sliding-window family (Ministral 3) would engage at every width, and the microbench says that is where the large full-model gains would be; no such checkpoint was on this host, so that remains unmeasured at the model level. Raising `DEFAULT_PREFILL_CHUNK` so that more of a long prompt runs as one fresh chunk would widen the engaged region; that is a separate trade-off against the chunking's memory purpose (#672) and is not changed here.

The experimental `1` mode is retained for A/B work; `0` is the rollback. Users of bf16 Gemma 4 checkpoints should expect greedy output to differ from the previous release at near-tie positions, as the trace section documents.

## Not run

- M5 and later (Neural Accelerator) hosts: none available. On NAX the pinned dispatch fuses head-dim 256 at every width and takes a different kernel, so neither the eligibility column nor the speedups above transfer; the probe-gated `auto` would engage on more shapes there and needs its own measurement.
- CUDA and ROCm: unchanged by construction (the route is Metal-only); not measured.
- Ministral 3 or any other head-dim 128 sliding-window checkpoint at the model level.
- Chunked continuation with the native fused kernel (does not exist at head dim 256 on this GPU).
- Quantized KV cache modes (`--kv-cache-mode` TurboQuant variants) were not timed. The model-level plan reads the K/V dtype the sliding cache returns (`RotatingKVCache::fetched_kv_dtype`): `fp16+turbo4` returns FP16 K/V, which under bf16 activations (Gemma 4) MLX promotes to float32 and serves from its graph fallback at head dim 256, so the plan is legacy and the mask is built once per forward as before; a mix of sliding cache modes that return different dtypes (`--turbo-boundary-v` keeps the outer layers FP16) also plans legacy. `model_plan_shape_matches_the_keys_every_cache_mode_returns` checks that the planned shape equals the one each cache mode returns. The other modes store the activation dtype on a rotating cache and plan exactly as FP16 does.
- On an M5 host the native entry bypasses `layers::attention`, so the `MLXCEL_LOG_NA_ATTENTION` dispatch lines and the `MLXCEL_METAL4_ATTENTION` kill switch no longer cover the window-prefill call; `MLXCEL_NATIVE_WINDOW_SDPA=0` is the switch for it there.
