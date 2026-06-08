# Unified Docker Build Matrix — Design

**Date:** 2026-06-08
**Status:** Approved (brainstorming)

## Goal

Produce every LuminousMiner build variant from Docker with **zero local tooling**,
from a homogeneous toolchain. A single backend selector chooses an **AMD-only**,
**NVIDIA-only**, or **combined AMD+NVIDIA** binary, for both Linux (ELF) and
Windows (PE32+, cross-compiled from Linux).

## Background / Driving Constraint

A single binary can contain only **one C++ ABI**. The current cross builds use
two incompatible ABIs:

- `Dockerfile.windows-amd-cross` → GNU ABI (llvm-mingw, `x64-mingw-static`, `-static`).
- `Dockerfile.windows-nvidia-cross` → MSVC ABI (clang-cl + xwin, `x64-windows-clangcl`).

CUDA host code **forces** the MSVC ABI (its headers reject libc++/mingw). Therefore a
combined AMD+NVIDIA binary requires AMD to move onto the MSVC-ABI clang-cl+xwin
toolchain. The miner **source already supports both backends at once** (independent
`BUILD_AMD` / `BUILD_NVIDIA` blocks, `AMD_ENABLE` / `CUDA_ENABLE` defines), so this is
purely a build-toolchain convergence — no device-layer code restructuring.

## Decisions (settled during brainstorming)

1. **Retire the mingw AMD path.** AMD converges onto the MSVC-ABI clang-cl+xwin
   toolchain. `Dockerfile.windows-amd-cross`, `cmake/toolchain-mingw.cmake`, and the
   `x64-mingw-static` usage are removed. The benchmarked-green mingw output is replaced
   and the new MSVC-ABI AMD build is re-validated (see Validation).
2. **Scope = Windows cross + Linux**, both delivered as Docker images. Native-local
   (non-Docker) builds are not a target.
3. **OpenCL on MSVC ABI:** retry vcpkg's `opencl` port under `x64-windows-clangcl`
   (xwin now supplies `windows.h`, which was the original `.rc` failure cause). If it
   still fails, fall back to compiling the Khronos OpenCL-ICD-Loader from source with
   clang-cl. The fallback lives in the implementation plan, not the happy path.
4. **Two images, one homogeneous toolchain.** AMD-only builds on a lean `ubuntu:24.04`
   base; NVIDIA/both build on `nvidia/cuda:13.1.2-devel-ubuntu24.04`. Both are Ubuntu
   24.04, so the toolchain setup is identical lines on either base. The base is chosen
   by the selector; CUDA-only setup steps are shell-gated.
5. **Bump Ubuntu 22.04 → 24.04.** The only genuinely-old piece. CMake 3.30.5,
   llvm-mingw (latest), and vcpkg (master) are already current.
6. **Retire the native-Windows-container Dockerfiles** (`windows-amd`,
   `windows-nvidia`). The cross path now covers both backends with no Windows host and
   no local tooling.

## Architecture

### Build matrix (2 Dockerfiles × `GPU` selector)

| | `GPU=amd` | `GPU=nvidia` | `GPU=both` (default) |
|---|---|---|---|
| `docker/Dockerfile.linux`         | ELF, OpenCL    | ELF, CUDA    | ELF, both    |
| `docker/Dockerfile.windows-cross` | PE32+, OpenCL  | PE32+, CUDA  | PE32+, both  |

All six variants build in **Linux container mode** (Windows binaries are
cross-compiled). No Docker engine mode switching.

### Selector mechanism

- `ARG GPU=both` (values: `amd` | `nvidia` | `both`).
- `ARG BASE_IMAGE` resolved from `GPU`:
  - `amd` → `ubuntu:24.04`
  - `nvidia` | `both` → `nvidia/cuda:13.1.2-devel-ubuntu24.04`
- The configure step derives `-D` flags from `GPU`:
  - `BUILD_AMD` = ON when `GPU ∈ {amd, both}`
  - `BUILD_NVIDIA` = ON when `GPU ∈ {nvidia, both}`
  - `USE_CLANG_CUDA` = ON when NVIDIA is included **and** target is Windows-cross
    (Linux uses native nvcc/FindCUDA, `USE_CLANG_CUDA=OFF`)
  - `VCPKG_MANIFEST_FEATURES` = `opencl` when AMD is included; OpenSSL is supplied
    prebuilt on Windows-cross (never from vcpkg there) and from vcpkg/system on Linux
- CUDA-only Docker steps (`cuda_*` redist libs, the cmath forward-declares patch) are
  shell-gated `if [ "$GPU" != amd ]`, so they are written once and skipped for AMD-only.

### `docker/Dockerfile.windows-cross` (replaces both cross Dockerfiles)

Common toolchain block (runs on either base, identical lines):
- CMake 3.30.5, llvm-mingw (clang/clang++/clang-cl symlink/lld-link/llvm-*),
  `ml64`→`llvm-ml` shim (boost-context `.asm`), xwin splat (MSVC CRT + Windows SDK),
  prebuilt FireDaemon MSVC OpenSSL (pinned URL+SHA256), vcpkg bootstrap.

GPU-gated (skipped when `GPU=amd`):
- Base flips to the CUDA image, CUDA redist libs (`cuda_cudart`, `cuda_nvrtc`), the
  `__clang_cuda_math_forward_declares.h` cmath patch.

vcpkg additionally builds **opencl** under `x64-windows-clangcl` when AMD is included.

Configure/build via the hidden `windows-cross-base` preset (triplet + toolchain) plus
`-D` overrides computed from `GPU`. Runtime DLLs (OpenSSL always; CUDA cudart/nvrtc when
NVIDIA) copied next to `miner.exe` in a separate `RUN` so a build failure is never masked.

### `docker/Dockerfile.linux` (replaces `linux-amd` + `linux-nvidia`)

- `ARG BASE_IMAGE` same selection rule.
- Native clang + (when NVIDIA) nvcc from the CUDA base; OpenCL headers + ICD loader
  (when AMD). `USE_CLANG_CUDA=OFF` (FindCUDA works natively on Linux).
- Same `GPU` selector and vcpkg feature logic; OpenSSL from vcpkg/system here.

### CMake / presets

- A hidden `windows-cross-base` configure preset holds the `x64-windows-clangcl` triplet
  and `cmake/toolchain-clang-cl-xwin.cmake`. The Dockerfile passes
  `--preset windows-cross-base` plus the `GPU`-derived `-D` overrides — avoids a
  six-way preset explosion while keeping toolchain config DRY.
- `CMakeLists.txt`: the `USE_CLANG_CUDA` env guard (currently FATAL_ERROR'ing on
  `OPENSSL_WIN_ROOT`/`CUDA_WIN_LIB`) must continue to hold for the Windows-cross
  NVIDIA/both path and must **not** trip on the AMD-only Windows-cross path (which sets
  `USE_CLANG_CUDA=OFF`). OpenCL find/`AMD_ENABLE` already works independent of ABI.
- `vcpkg.json`: `opencl`/`openssl` remain optional features; presets/Docker set
  `VCPKG_MANIFEST_FEATURES` explicitly per (os, gpu) with
  `VCPKG_MANIFEST_NO_DEFAULT_FEATURES=ON`.

### `scripts/docker-build.ps1`

New signature:
- `-Os {linux | windows-cross | all}`
- `-Gpu {amd | nvidia | both}` (default `both`)
- `-VcpkgRef` (unchanged, default `master`)

All targets build in Linux container mode — the Windows-container mode-switch logic
(`Assert-EngineMode`, `Build-WindowsTarget`, `docker create`/`docker cp`) is deleted.
Artifacts export via BuildKit `-o` to `dist/<os>-<gpu>/` (e.g. `dist/windows-cross-both/`).

## Data Flow (a build)

1. User runs `scripts/docker-build.ps1 -Os windows-cross -Gpu both` (or `docker build
   -f docker/Dockerfile.windows-cross --build-arg GPU=both --target artifact -o dist/...`).
2. `BASE_IMAGE` resolves to the CUDA base; common toolchain layers build; CUDA layers
   build (not gated out).
3. vcpkg builds boost + opencl for `x64-windows-clangcl`; OpenSSL prebuilt; CUDA redist
   staged.
4. CMake configures with `BUILD_AMD=ON BUILD_NVIDIA=ON USE_CLANG_CUDA=ON` + opencl
   feature; `.cu` compiled via `lm_clang_cuda_library`, `.cl` kernels staged to
   `bin/kernel/`, host/AMD TUs compiled by clang-cl.
5. Link → `miner.exe`; OpenSSL + CUDA runtime DLLs copied beside it.
6. `dist/windows-cross-both/` holds `miner.exe`, `kernel/`, and required DLLs.

## Error Handling

- **Missing env on the CUDA path:** the existing `USE_CLANG_CUDA` env guard
  (FATAL_ERROR when `OPENSSL_WIN_ROOT`/`CUDA_WIN_LIB` unset) is preserved.
- **vcpkg opencl failure under clangcl:** plan includes the ICD-loader-from-source
  fallback; the spike decides which branch ships.
- **DLL copy masking:** runtime-DLL copy stays in its own `RUN` after the build `RUN`.
- **Combined binary on an AMD-only host:** `miner.exe` links a cudart **loader stub**
  (`LoadLibrary cudart64_13.dll` at startup). On a host with no NVIDIA GPU,
  `cudaGetDeviceCount` returns 0; DeviceManager must skip NVIDIA cleanly and mine on
  AMD. Verified as part of validation.

## Validation / Testing

1. **vcpkg-opencl-under-clangcl spike** — confirm the port builds; otherwise switch to
   the ICD-loader fallback.
2. **AMD-on-MSVC-ABI re-benchmark on the RX 9070 XT** — compare hashrate/codegen against
   the retired mingw output; confirm no regression (the real risk of dropping mingw).
3. **Combined `miner.exe` runs on the AMD-only host** — graceful 0-CUDA-devices path;
   mines on AMD. (This makes the combined binary runtime-testable on existing hardware,
   unlike NVIDIA-only.)
4. **All six variants configure + build green** in CI/local Docker.
5. **AMD-only image stays lean** — no CUDA base, no `cuda_*` layers.

## Out of Scope

- Native (non-Docker) local builds.
- Runtime validation on NVIDIA Windows hardware (no such host available; build-validated
  only, as today).
- Algorithm/device-layer code changes.

## Affected Files

- **New:** `docker/Dockerfile.windows-cross`, `docker/Dockerfile.linux`.
- **Removed:** `docker/Dockerfile.windows-amd-cross`, `docker/Dockerfile.windows-nvidia-cross`,
  `docker/Dockerfile.windows-amd`, `docker/Dockerfile.windows-nvidia`,
  `cmake/toolchain-mingw.cmake`.
- **Modified:** `CMakePresets.json` (collapse to a `windows-cross-base` + `linux-base`
  hidden preset model), `scripts/docker-build.ps1` (new `-Os`/`-Gpu` signature),
  `vcpkg.json` (feature comments), `CMakeLists.txt` (AMD-on-Windows-cross gating
  alongside `USE_CLANG_CUDA`), possibly `triplets/` (drop the mingw triplet).
</content>
</invoke>
