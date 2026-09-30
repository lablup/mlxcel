# fix(rocm): track header dependencies of HIP objects

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-track-header-dependencies-of-HIP-objects.patch`](0001-fix-rocm-track-header-dependencies-of-HIP-objects.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 26 |
| Reproduction | `repro.sh` (needs a configured build directory) |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

`mlx/backend/rocm/CMakeLists.txt` compiles each `.hip` file with its own `add_custom_command`, whose only dependency was the `.hip` source. CMake never learned which headers an object includes, so an incremental build recompiled no HIP object when only a header changed (`kernel_utils.hpp`, `device.h`, `lru_cache.h`, `reduce/reduce.hpp` and so on). Fresh builds were correct; reused build trees could link kernels compiled against an old header, which also makes a header-only bisect unreliable. The `.cpp` sources added through `target_sources(mlx ...)` were never affected, because CMake scans those itself.

Each command now passes `-MD -MF <object>.d` to `hipcc` and `DEPFILE <object>.d` to `add_custom_command`, under `if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)` (the first version with `DEPFILE` for the Makefile generators), with policy `CMP0116` set to `NEW` when it exists. Below 3.20 every HIP object instead depends on a `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` list of every `*.h`, `*.hpp` and `*.cuh` under `mlx/` (the HIP sources include MLX headers outside the backend too), which rebuilds every HIP object on any header change. The top-level `CMakeLists.txt` requires CMake 3.25, which also implies `CMP0116` NEW, so that branch only guards a lower minimum. The first build after this change recompiles every HIP object once, because the command line changed.

Measured on gfx1151 with CMake 3.31.6 and Unix Makefiles, on a build of this backend with two more `.hip` files than this branch (41 HIP objects): before the change, editing `kernel_utils.hpp` recompiled none of them; after it, the same edit recompiled the 39 objects whose depfiles name the header, an edit to `reduce/reduce.hpp` recompiled exactly its 7 includers (both sets agree with a walk of the `#include` graph), reverting each recompiled the same set, and a rebuild with no change recompiled none. The same mechanism was checked under Ninja 1.13 on a small probe project; the full build was not run under Ninja. The depfile comes from the host compilation, so it lists ROCm and system headers too but not headers included only on the device side; `rocwmma/rocwmma.hpp` in `flash_attention_wmma.hip` is the one such include today.

## Reproduction

`repro.sh <mlx-source-dir> <build-dir>` edits `kernel_utils.hpp`, rebuilds, and counts the `Compiling HIP source` lines, then restores the header. Without the change the edited build compiles no HIP object.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
