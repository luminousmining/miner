# Unified Docker Build Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce every LuminousMiner build variant (Linux/Windows × AMD/NVIDIA/both) from Docker with zero local tooling, via one homogeneous MSVC-ABI cross toolchain and a single `GPU` backend selector.

**Architecture:** AMD moves off mingw (GNU ABI) onto the existing clang-cl+xwin MSVC ABI so AMD and CUDA can co-link in one binary. Two unified Dockerfiles (`windows-cross`, `linux`) pick their base image and gate CUDA-only steps from a `GPU={amd,nvidia,both}` build-arg. CMake's prebuilt-OpenSSL provisioning is decoupled from the CUDA flag so the AMD-only Windows path still links prebuilt OpenSSL.

**Tech Stack:** Docker/BuildKit, clang-cl + xwin (MSVC-ABI cross), llvm-mingw (LLVM 22), vcpkg (manifest mode), CMake presets, CUDA 13.1 redist, OpenCL ICD, PowerShell orchestrator.

---

## Reference: current state (read before starting)

- `CMakeLists.txt`
  - line 18: `option(USE_CLANG_CUDA ... OFF)`
  - lines 257–322: `if (BUILD_NVIDIA AND USE_CLANG_CUDA)` block — env guard + `lm_clang_cuda_library`
  - lines 405–410: `find_package(OpenCL)` (when `BUILD_AMD`), `find_package(CUDA)` (native NVIDIA)
  - lines 411–424: **OpenSSL** — `if (USE_CLANG_CUDA)` uses prebuilt IMPORTED targets, `else()` `find_package(OpenSSL)`  ← this is the coupling to fix
  - lines 431–433: AMD `find_library(OpenCL_LIBRARIES ...)` fallback
  - lines 434–457: CUDA import-lib resolution (clang vs native)
- `sources/CMakeLists.txt`
  - lines 5–22 / 96–103: `if (USE_CLANG_CUDA) lm_clang_cuda_library(...) else() cuda_add_library(...)`
  - lines 262–272: `if (NOT WIN32)` GNU static-libstdc++ link opts (skipped on all Windows targets)
  - lines 433–447: `if (WIN32 AND NOT MSVC)` explicit Windows syslibs (mingw-only; clang-cl is `MSVC`, uses `#pragma comment(lib)` auto-link)
- `CMakePresets.json`: `windows-amd-cross` (mingw), `windows-nvidia-cross` (clangcl), `linux-amd`, `linux-nvidia`, plus native-Windows `windows-amd`/`windows-nvidia`
- `cmake/toolchain-clang-cl-xwin.cmake`, `triplets/x64-windows-clangcl.cmake` (keep)
- `cmake/toolchain-mingw.cmake`, `triplets/x64-mingw-static.cmake` (to remove)
- `docker/`: `Dockerfile.{linux-amd,linux-nvidia,windows-amd,windows-nvidia,windows-amd-cross,windows-nvidia-cross}`
- `scripts/docker-build.ps1`: `-Target` enum, Windows-container mode-switch logic

**Build cost note:** these Docker builds are multi-GB and take many minutes (vcpkg builds Boost from source, xwin splats the SDK, the CUDA base is ~6 GB). Verification steps are real builds — budget time accordingly. Run from the repo root with `DOCKER_BUILDKIT=1` and Docker in **Linux** container mode.

**Branch:** work on `work` (carries all the cross-build infra). Commit in the user's name only — **no `Co-Authored-By` trailer**.

---

## Task 1: Spike — can vcpkg build OpenCL under the clang-cl triplet?

This de-risks the one unknown before touching the real Dockerfiles. The original failure was a `.rc` needing `windows.h`; xwin now supplies it.

**Files:**
- Create (throwaway, do not commit): `docker/Dockerfile.opencl-spike`

- [ ] **Step 1: Write a minimal spike Dockerfile that installs only enough to ask vcpkg for opencl under the clangcl triplet**

Create `docker/Dockerfile.opencl-spike` by copying the toolchain prologue of the current `docker/Dockerfile.windows-nvidia-cross` (lines 18–62: base, apt, CMake, llvm-mingw + `clang-cl` symlink, xwin splat) — **stop after the `xwin --accept-license splat` line** — then append:

```dockerfile
ENV VCPKG_ROOT=/opt/vcpkg
ENV VCPKG_FORCE_SYSTEM_BINARIES=1
RUN git clone https://github.com/microsoft/vcpkg "$VCPKG_ROOT" \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
COPY triplets /tmp/triplets
RUN /opt/vcpkg/vcpkg install opencl:x64-windows-clangcl \
        --overlay-triplets=/tmp/triplets \
    && find /opt/vcpkg/installed -iname 'OpenCL.lib' -o -iname 'libOpenCL*'
```

- [ ] **Step 2: Run the spike**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.opencl-spike -t lm-opencl-spike .
```
Expected (success): the final `find` prints a path ending in `installed/x64-windows-clangcl/lib/OpenCL.lib` and the build exits 0.
Expected (failure): a vcpkg error during the `opencl` port build (e.g. a `.rc`/`windows.h` or `llvm-rc` failure).

- [ ] **Step 3: Record the verdict**

If success → Task 5 uses `find_package(OpenCL)` straight from vcpkg (the default path already in `CMakeLists.txt`). If failure → Task 5 uses the **ICD-loader fallback** (its Step is written inline in Task 5). Note the outcome in the PR description.

- [ ] **Step 4: Delete the throwaway Dockerfile (do not commit it)**

Run:
```bash
rm docker/Dockerfile.opencl-spike
```
Expected: file gone; `git status` shows nothing to commit from this task.

---

## Task 2: Decouple prebuilt-OpenSSL from `USE_CLANG_CUDA` in CMake

So AMD-only and combined Windows-cross builds (where `USE_CLANG_CUDA` may be OFF) still link the prebuilt MSVC OpenSSL the Dockerfile stages.

**Files:**
- Modify: `CMakeLists.txt` (add `LM_WIN_CROSS` derivation; re-gate OpenSSL block at lines 411–424)

- [ ] **Step 1: Add an `LM_WIN_CROSS` signal right after the `USE_CLANG_CUDA` CUDA block**

In `CMakeLists.txt`, immediately **before** the `### FIND PACKAGE | LIBRARIES` banner (currently line 379), insert:

```cmake
################################################################################
### WINDOWS MSVC-ABI CROSS PROVISIONING (clang-cl + xwin)                   ####
################################################################################
# The windows-cross Docker image stages prebuilt MSVC OpenSSL (vcpkg can't
# cross-build it) and exports OPENSSL_WIN_ROOT. Its presence — NOT USE_CLANG_CUDA
# — is what means "link prebuilt OpenSSL", so the AMD-only cross build (which sets
# USE_CLANG_CUDA=OFF) still gets it. CUDA import libs remain gated on NVIDIA.
if (DEFINED ENV{OPENSSL_WIN_ROOT})
    set(LM_WIN_CROSS ON)
else()
    set(LM_WIN_CROSS OFF)
endif()
```

- [ ] **Step 2: Re-gate the OpenSSL provisioning block**

In `CMakeLists.txt`, change the OpenSSL block condition (currently line 411) from:

```cmake
if (USE_CLANG_CUDA)
    # Prebuilt MSVC OpenSSL staged by the Dockerfile (vcpkg can't cross-build it).
    set(_ssl "$ENV{OPENSSL_WIN_ROOT}")
```

to:

```cmake
if (LM_WIN_CROSS)
    # Prebuilt MSVC OpenSSL staged by the Dockerfile (vcpkg can't cross-build it).
    set(_ssl "$ENV{OPENSSL_WIN_ROOT}")
```

(The `else() find_package(OpenSSL ...)` arm is unchanged.)

- [ ] **Step 3: Loosen the `USE_CLANG_CUDA` env guard so it only requires the CUDA staging var**

In `CMakeLists.txt`, the guard inside the `if (BUILD_NVIDIA AND USE_CLANG_CUDA)` block (currently lines 263–268) requires both `OPENSSL_WIN_ROOT` and `CUDA_WIN_LIB`. OpenSSL is now handled by `LM_WIN_CROSS`, so this block only needs `CUDA_WIN_LIB`. Replace:

```cmake
    if (NOT DEFINED ENV{OPENSSL_WIN_ROOT} OR NOT DEFINED ENV{CUDA_WIN_LIB})
        message(FATAL_ERROR
            "USE_CLANG_CUDA expects the windows-nvidia-cross Docker environment "
            "(OPENSSL_WIN_ROOT and CUDA_WIN_LIB must be set). Build via "
            "docker/Dockerfile.windows-nvidia-cross, not a bare configure.")
    endif()
```

with:

```cmake
    if (NOT DEFINED ENV{CUDA_WIN_LIB})
        message(FATAL_ERROR
            "USE_CLANG_CUDA expects the windows-cross Docker environment "
            "(CUDA_WIN_LIB must point at the staged CUDA import libs). Build via "
            "docker/Dockerfile.windows-cross, not a bare configure.")
    endif()
```

- [ ] **Step 4: Sanity-check the edit parses (configure dry-run, expected to fail fast for the right reason)**

Run:
```bash
cmake -P /dev/stdin <<'EOF'
file(READ "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" _c)
if(_c MATCHES "if \\(LM_WIN_CROSS\\)" AND _c MATCHES "set\\(LM_WIN_CROSS ON\\)")
  message(STATUS "OK: LM_WIN_CROSS wired")
else()
  message(FATAL_ERROR "LM_WIN_CROSS not wired")
endif()
EOF
```
Expected: `-- OK: LM_WIN_CROSS wired`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(cmake): gate prebuilt MSVC OpenSSL on LM_WIN_CROSS, not USE_CLANG_CUDA

So AMD-only/combined Windows-cross builds (USE_CLANG_CUDA may be OFF) still link
the prebuilt OpenSSL the Docker image stages; CUDA libs stay NVIDIA-gated."
```

---

## Task 3: Collapse the cross/linux presets to one selectable preset each

The Docker images invoke one preset and override the backend flags per `GPU`. Keep the invariant toolchain/triplet config in the preset; pass `BUILD_AMD/BUILD_NVIDIA/USE_CLANG_CUDA/VCPKG_MANIFEST_FEATURES` as `-D` overrides from the Dockerfile.

**Files:**
- Modify: `CMakePresets.json`

- [ ] **Step 1: Replace the two cross presets and two linux presets with `windows-cross` + `linux`**

In `CMakePresets.json`, remove the `windows-amd-cross`, `windows-nvidia-cross`, `linux-amd`, `linux-nvidia`, `windows-amd`, and `windows-nvidia` configure presets and their matching build presets. Add these configure presets (defaults are "both"; the Dockerfile always passes explicit overrides, so defaults only matter for ad-hoc use):

```json
{
  "name": "linux",
  "displayName": "Linux (native clang; GPU selected via -D)",
  "inherits": "vcpkg-base",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/linux",
  "cacheVariables": {
    "BUILD_AMD": "ON",
    "BUILD_NVIDIA": "ON",
    "BUILD_CPU": "OFF",
    "VCPKG_MANIFEST_NO_DEFAULT_FEATURES": "ON",
    "VCPKG_MANIFEST_FEATURES": "opencl;openssl"
  }
},
{
  "name": "windows-cross",
  "displayName": "Windows cross (clang-cl + xwin, MSVC ABI; GPU selected via -D)",
  "inherits": "vcpkg-base",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/windows-cross",
  "cacheVariables": {
    "BUILD_AMD": "ON",
    "BUILD_NVIDIA": "ON",
    "BUILD_CPU": "OFF",
    "USE_CLANG_CUDA": "ON",
    "VCPKG_MANIFEST_NO_DEFAULT_FEATURES": "ON",
    "VCPKG_MANIFEST_FEATURES": "opencl",
    "VCPKG_TARGET_TRIPLET": "x64-windows-clangcl",
    "VCPKG_OVERLAY_TRIPLETS": "${sourceDir}/triplets",
    "VCPKG_CHAINLOAD_TOOLCHAIN_FILE": "${sourceDir}/cmake/toolchain-clang-cl-xwin.cmake",
    "VCPKG_APPLOCAL_DEPS": "OFF"
  }
}
```

And the matching build presets array becomes:

```json
"buildPresets": [
  { "name": "linux",         "configurePreset": "linux",         "configuration": "Release" },
  { "name": "windows-cross", "configurePreset": "windows-cross", "configuration": "Release" }
]
```

- [ ] **Step 2: Validate the JSON parses and exposes exactly the two presets**

Run:
```bash
python3 -c "import json;d=json.load(open('CMakePresets.json'));print(sorted(p['name'] for p in d['configurePresets'] if not p.get('hidden')))"
```
Expected: `['linux', 'windows-cross']`.

- [ ] **Step 3: Commit**

```bash
git add CMakePresets.json
git commit -m "build(presets): collapse to selectable linux/windows-cross presets

Backends are chosen via -DBUILD_AMD/-DBUILD_NVIDIA overrides from the Docker
images; presets hold only the invariant toolchain/triplet/feature config."
```

---

## Task 4: Unified `docker/Dockerfile.windows-cross`

Replaces both cross Dockerfiles. Base chosen from `GPU`; CUDA-only steps shell-gated. Common toolchain block written once.

**Files:**
- Create: `docker/Dockerfile.windows-cross`

- [ ] **Step 1: Write the unified cross Dockerfile**

Create `docker/Dockerfile.windows-cross`:

```dockerfile
# syntax=docker/dockerfile:1
#
# Cross-compile LuminousMiner for WINDOWS from a Linux container, MSVC ABI
# (clang-cl + xwin). One toolchain for AMD (OpenCL), NVIDIA (CUDA), or BOTH in a
# single miner.exe. Select with --build-arg GPU=amd|nvidia|both (default both).
#
#   GPU=amd     -> lean ubuntu:24.04 base, no CUDA layers
#   GPU=nvidia  -> nvidia/cuda devel base, CUDA only
#   GPU=both    -> nvidia/cuda devel base, AMD + CUDA in one binary
#
# AMD .cl kernels compile at runtime; NVIDIA .cu compile at build time via
# clang-CUDA (CMake can't drive clang+CUDA on Windows, issue #20776 -> see
# lm_clang_cuda_library in CMakeLists.txt). Boost et al. cross-build for the MSVC
# ABI via vcpkg + x64-windows-clangcl. OpenSSL is prebuilt MSVC libs; OpenCL is
# vcpkg's ICD loader. nvcc CANNOT do this; clang-CUDA can.
#
# Usage:
#   docker build -f docker/Dockerfile.windows-cross --build-arg GPU=both \
#       --target artifact -o dist/windows-cross-both .

ARG GPU=both
# Base is selected from GPU. ubuntu:24.04 for amd-only; the CUDA devel image
# (also Ubuntu 24.04, so the toolchain layers below are identical) otherwise.
ARG BASE_AMD=ubuntu:24.04
ARG BASE_CUDA=nvidia/cuda:13.1.2-devel-ubuntu24.04

FROM ${BASE_CUDA} AS base-nvidia
FROM ${BASE_AMD}  AS base-amd
# Resolve the real base from GPU: "amd" -> base-amd, anything else -> base-nvidia.
# (BuildKit picks the stage named by the GPU-derived arg.)
FROM base-${GPU_BASE:-nvidia} AS build
ARG GPU
ARG DEBIAN_FRONTEND=noninteractive
ARG VCPKG_REF=master

# --- Common toolchain (identical on either Ubuntu 24.04 base) ---------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl zip unzip tar xz-utils git pkg-config python3 \
        ninja-build perl bison file libicu74 \
    && rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.30.5
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
        | tar xz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/

# llvm-mingw: clang/clang++/clang-cl/lld-link/llvm-* (LLVM 22). MSVC target, not
# the mingw sysroot. Symlink clang-cl.
RUN set -eux; \
    url="$(curl -fsSL https://api.github.com/repos/mstorsjo/llvm-mingw/releases/latest \
        | grep -oP '"browser_download_url":\s*"\K[^"]*ucrt-ubuntu-[0-9.]+-x86_64\.tar\.xz' | head -1)"; \
    curl -fsSL "$url" -o /tmp/llvm.tar.xz; mkdir -p /opt/llvm-mingw; \
    tar xJf /tmp/llvm.tar.xz -C /opt/llvm-mingw --strip-components=1; rm /tmp/llvm.tar.xz; \
    ln -sf /opt/llvm-mingw/bin/clang /opt/llvm-mingw/bin/clang-cl
ENV PATH="/opt/llvm-mingw/bin:${PATH}"

# boost-context .asm assembles with ml64 (MSVC MASM) -> shim to llvm-ml.
RUN printf '#!/bin/sh\nexec /opt/llvm-mingw/bin/llvm-ml -m64 "$@"\n' > /usr/local/bin/ml64 \
    && chmod +x /usr/local/bin/ml64

# xwin: MSVC CRT + Windows SDK headers/libs.
RUN set -eux; \
    url="$(curl -fsSL https://api.github.com/repos/Jake-Shadle/xwin/releases/latest \
        | grep -oP '"browser_download_url":\s*"\K[^"]*x86_64-unknown-linux-musl\.tar\.gz' | head -1)"; \
    curl -fsSL "$url" -o /tmp/xwin.tgz; tar xzf /tmp/xwin.tgz -C /tmp; \
    mv /tmp/xwin-*/xwin /usr/local/bin/xwin; rm -rf /tmp/xwin*
RUN xwin --accept-license splat --output /opt/xwin
ENV XWIN_ROOT=/opt/xwin

# Prebuilt MSVC OpenSSL (FireDaemon). vcpkg can't cross-build OpenSSL to
# windows-msvc from Linux. Linked via import libs; DLLs shipped in the artifact.
ARG OPENSSL_WIN_URL="https://download.firedaemon.com/FireDaemon-OpenSSL/openssl-3.5.6.zip"
ARG OPENSSL_WIN_SHA256="31de50c939a40564961988d20383aff7c5086e0bdf5449569c5d5c0d8db30b5c"
RUN set -eux; cd /tmp; \
    curl -fsSL "$OPENSSL_WIN_URL" -o ossl.zip; \
    echo "$OPENSSL_WIN_SHA256  ossl.zip" | sha256sum -c -; \
    python3 -c "import zipfile;zipfile.ZipFile('ossl.zip').extractall('ossl')"; \
    mkdir -p /opt/openssl-win/lib /opt/openssl-win/include /opt/openssl-win/bin; \
    sslimp="$(find /tmp/ossl -ipath '*x64*' -iname 'libssl*.lib'    ! -iname '*static*' | head -1)"; \
    cryimp="$(find /tmp/ossl -ipath '*x64*' -iname 'libcrypto*.lib' ! -iname '*static*' | head -1)"; \
    cp "$sslimp" /opt/openssl-win/lib/libssl.lib; \
    cp "$cryimp" /opt/openssl-win/lib/libcrypto.lib; \
    osslv="$(find /tmp/ossl -ipath '*x64*' -path '*include/openssl/opensslv.h' | head -1)"; \
    cp -r "$(dirname "$(dirname "$osslv")")/openssl" /opt/openssl-win/include/; \
    find /tmp/ossl -ipath '*x64*' \( -iname 'libssl*.dll' -o -iname 'libcrypto*.dll' \) -exec cp {} /opt/openssl-win/bin/ \; ; \
    ls /opt/openssl-win/lib /opt/openssl-win/bin
ENV OPENSSL_WIN_ROOT=/opt/openssl-win

# --- CUDA-only staging (skipped for GPU=amd) -------------------------------
# Windows CUDA import libs + runtime DLLs from the redist, and the clang-CUDA
# cmath forward-declares patch. Guarded so AMD-only images stay lean and never
# require the CUDA layers.
RUN set -eux; \
    if [ "$GPU" != "amd" ]; then \
      H="$(echo /opt/llvm-mingw/lib/clang/*/include/__clang_cuda_math_forward_declares.h)"; \
      sed -i 's|__DEVICE__ float fma(float, float, float);|__DEVICE__ float fma(float, float, float);\n__DEVICE__ long double fma(long double, long double, long double);|' "$H"; \
      grep -q 'long double fma' "$H"; \
      mkdir -p /opt/cuda-win/lib/x64 /opt/cuda-win/bin; cd /tmp; \
      base=https://developer.download.nvidia.com/compute/cuda/redist; \
      j="$(curl -fsSL $base/redistrib_13.1.2.json)"; \
      for comp in cuda_cudart cuda_nvrtc; do \
        rel="$(echo "$j" | grep -oE "${comp}/windows-x86_64/[^\"]+\.zip" | head -1)"; \
        curl -fsSL "$base/$rel" -o c.zip; \
        python3 -c "import zipfile;zipfile.ZipFile('c.zip').extractall('x_'+'${comp}')"; \
      done; \
      find /tmp/x_* -path '*/lib/x64/*' -name '*.lib' -exec cp {} /opt/cuda-win/lib/x64/ \; ; \
      find /tmp/x_* -path '*/bin/*'    -name '*.dll' -exec cp {} /opt/cuda-win/bin/ \; ; \
      ls /opt/cuda-win/lib/x64; \
    else \
      echo "GPU=amd: skipping CUDA staging"; \
      mkdir -p /opt/cuda-win/lib/x64 /opt/cuda-win/bin; \
    fi
ENV CUDA_WIN_LIB=/opt/cuda-win/lib/x64

# --- vcpkg ------------------------------------------------------------------
ENV VCPKG_ROOT=/opt/vcpkg
ENV VCPKG_FORCE_SYSTEM_BINARIES=1
RUN git clone https://github.com/microsoft/vcpkg "$VCPKG_ROOT" \
    && git -C "$VCPKG_ROOT" checkout "$VCPKG_REF" \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
ENV PATH="$VCPKG_ROOT:${PATH}"

WORKDIR /src
COPY vcpkg.json CMakePresets.json CMakeLists.txt ./
COPY cmake ./cmake
COPY triplets ./triplets
COPY sources ./sources

# Derive the -D overrides from GPU and configure+build. opencl feature only when
# AMD is in; USE_CLANG_CUDA only when NVIDIA is in.
RUN --mount=type=cache,target=/root/.cache/vcpkg \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    set -eux; \
    case "$GPU" in \
      amd)    AMD=ON;  NV=OFF; CC=OFF; FEAT="opencl" ;; \
      nvidia) AMD=OFF; NV=ON;  CC=ON;  FEAT="" ;; \
      both)   AMD=ON;  NV=ON;  CC=ON;  FEAT="opencl" ;; \
      *) echo "GPU must be amd|nvidia|both" >&2; exit 2 ;; \
    esac; \
    cmake --preset windows-cross \
        -DBUILD_AMD=$AMD -DBUILD_NVIDIA=$NV -DUSE_CLANG_CUDA=$CC \
        -DVCPKG_MANIFEST_FEATURES="$FEAT"; \
    cmake --build --preset windows-cross

# Ship runtime DLLs next to miner.exe (separate RUN so a build failure above is
# never masked). OpenSSL always; CUDA DLLs only exist when NVIDIA was staged.
RUN set -eux; \
    cp /opt/openssl-win/bin/*.dll /src/build/windows-cross/bin/; \
    if [ "$GPU" != "amd" ]; then cp /opt/cuda-win/bin/*.dll /src/build/windows-cross/bin/; fi

FROM scratch AS artifact
COPY --from=build /src/build/windows-cross/bin /
```

- [ ] **Step 2: Resolve the base-stage selection**

BuildKit cannot interpolate `${GPU}` directly into `FROM base-${GPU}` without the arg being declared before the stage. The line `FROM base-${GPU_BASE:-nvidia}` above needs `GPU_BASE` set to `amd` only when `GPU=amd`. Implement that by replacing the three `FROM` lines (the `base-nvidia`/`base-amd`/`build` selector) with a single direct selection driven by `GPU`, using an `ARG` default map. Change:

```dockerfile
FROM ${BASE_CUDA} AS base-nvidia
FROM ${BASE_AMD}  AS base-amd
FROM base-${GPU_BASE:-nvidia} AS build
ARG GPU
```

to:

```dockerfile
# Map GPU -> concrete base image. amd is lean ubuntu; nvidia/both use the CUDA
# devel base. Both are Ubuntu 24.04 so the toolchain layers are identical.
FROM ${BASE_CUDA} AS base-both
FROM ${BASE_CUDA} AS base-nvidia
FROM ${BASE_AMD}  AS base-amd
ARG GPU=both
FROM base-${GPU} AS build
ARG GPU
```

This works because `GPU` is one of `amd|nvidia|both`, each matching a `base-<GPU>` stage name, and `ARG GPU` is declared before the selector `FROM`.

- [ ] **Step 3: Verify AMD-only builds on the lean base (no CUDA layers, no CUDA base)**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=amd --target artifact -o dist/windows-cross-amd .
```
Expected: exits 0; `dist/windows-cross-amd/miner.exe` exists; `dist/windows-cross-amd/` contains `libssl*.dll` and `libcrypto*.dll` but **no** `cudart*.dll`/`nvrtc*.dll`; `kernel/` present.
Verify the binary type:
```bash
file dist/windows-cross-amd/miner.exe
```
Expected: `PE32+ executable (console) x86-64, for MS Windows`.

- [ ] **Step 4: Verify the combined build links AMD + CUDA into one exe**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=both --target artifact -o dist/windows-cross-both .
```
Expected: exits 0; `dist/windows-cross-both/` contains `miner.exe`, `libssl*.dll`, `libcrypto*.dll`, `cudart64_13.dll`, `nvrtc64_130_0.dll`, and `kernel/`.

- [ ] **Step 5: Verify NVIDIA-only still builds (parity with the retired path)**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=nvidia --target artifact -o dist/windows-cross-nvidia .
```
Expected: exits 0; `dist/windows-cross-nvidia/miner.exe` exists; CUDA DLLs present; **no** OpenCL-related kernel-load failure at link.

- [ ] **Step 6: If Task 1 found vcpkg-opencl FAILS under clangcl, apply the ICD-loader fallback instead**

Only if Task 1 Step 3 recorded a failure. Insert this stage **before** the vcpkg section and drop `opencl` from `FEAT` (set `FEAT=""` in all branches), pointing CMake at the built loader via `-DOpenCL_LIBRARY`/`-DOpenCL_INCLUDE_DIR`:

```dockerfile
# OpenCL ICD loader built from source with clang-cl (vcpkg's opencl port can't
# cross-build under clangcl). Produces OpenCL.lib; the AMD driver's OpenCL.dll on
# the target dispatches to the GPU at runtime.
RUN set -eux; \
    if [ "$GPU" != "nvidia" ]; then \
      git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader /tmp/icd; \
      git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers /tmp/oclh; \
      cmake -S /tmp/icd -B /tmp/icd/build -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-clang-cl-xwin.cmake \
        -DOPENCL_ICD_LOADER_HEADERS_DIR=/tmp/oclh \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencl-win; \
      cmake --build /tmp/icd/build --target install; \
      cp -r /tmp/oclh/CL /opt/opencl-win/include/ 2>/dev/null || true; \
    fi
ENV OPENCL_WIN_ROOT=/opt/opencl-win
```

…and add to the `cmake --preset windows-cross` line (only meaningful when AMD is in; harmless otherwise):
```
        -DOpenCL_INCLUDE_DIRS=/opt/opencl-win/include \
        -DOpenCL_LIBRARIES=/opt/opencl-win/lib/OpenCL.lib
```
Re-run Step 3 to confirm AMD-only links against the source-built loader.

- [ ] **Step 7: Commit**

```bash
git add docker/Dockerfile.windows-cross
git commit -m "build(docker): unified windows-cross image (GPU=amd|nvidia|both)

One MSVC-ABI clang-cl+xwin toolchain; base is lean ubuntu for amd-only and the
CUDA devel image for nvidia/both. AMD now links into the same binary as CUDA."
```

---

## Task 5: Unified `docker/Dockerfile.linux`

Replaces `linux-amd` + `linux-nvidia`. Native clang; nvcc from the CUDA base when NVIDIA is in.

**Files:**
- Create: `docker/Dockerfile.linux`

- [ ] **Step 1: Write the unified linux Dockerfile**

Create `docker/Dockerfile.linux`:

```dockerfile
# syntax=docker/dockerfile:1
#
# Build LuminousMiner natively for Linux (ELF) in a container. One image for AMD
# (OpenCL), NVIDIA (CUDA), or BOTH. Select with --build-arg GPU=amd|nvidia|both
# (default both). No GPU needed to build.
#
#   GPU=amd     -> lean ubuntu:24.04 base
#   GPU=nvidia  -> nvidia/cuda devel base (nvcc), CUDA only
#   GPU=both    -> nvidia/cuda devel base, AMD + CUDA in one binary
#
# Usage:
#   docker build -f docker/Dockerfile.linux --build-arg GPU=both \
#       --target artifact -o dist/linux-both .

ARG BASE_AMD=ubuntu:24.04
ARG BASE_CUDA=nvidia/cuda:13.1.2-devel-ubuntu24.04

FROM ${BASE_CUDA} AS base-both
FROM ${BASE_CUDA} AS base-nvidia
FROM ${BASE_AMD}  AS base-amd
ARG GPU=both
FROM base-${GPU} AS build
ARG GPU
ARG DEBIAN_FRONTEND=noninteractive
ARG VCPKG_REF=master

# clang is the compiler the project forces on Linux (and the CUDA host compiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl zip unzip tar git pkg-config \
        build-essential ninja-build python3 \
        clang lld \
        linux-libc-dev libgnutls28-dev \
    && rm -rf /var/lib/apt/lists/*

# CMakeLists FORCE-sets clang-15/clang++-15 (a no-op after project()); alias the
# distro clang to that name so the forced compiler resolves on Ubuntu 24.04.
RUN ln -sf "$(command -v clang)"   /usr/local/bin/clang-15 \
    && ln -sf "$(command -v clang++)" /usr/local/bin/clang++-15
ENV CC=clang-15 CXX=clang++-15

ARG CMAKE_VERSION=3.30.5
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
        | tar xz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/

ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone https://github.com/microsoft/vcpkg "$VCPKG_ROOT" \
    && git -C "$VCPKG_ROOT" checkout "$VCPKG_REF" \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
ENV PATH="$VCPKG_ROOT:${PATH}"

WORKDIR /src
COPY vcpkg.json CMakePresets.json CMakeLists.txt ./
COPY cmake ./cmake
COPY sources ./sources

RUN --mount=type=cache,target=/root/.cache/vcpkg \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    set -eux; \
    case "$GPU" in \
      amd)    AMD=ON;  NV=OFF; FEAT="opencl;openssl" ;; \
      nvidia) AMD=OFF; NV=ON;  FEAT="openssl" ;; \
      both)   AMD=ON;  NV=ON;  FEAT="opencl;openssl" ;; \
      *) echo "GPU must be amd|nvidia|both" >&2; exit 2 ;; \
    esac; \
    cmake --preset linux \
        -DBUILD_AMD=$AMD -DBUILD_NVIDIA=$NV \
        -DVCPKG_MANIFEST_FEATURES="$FEAT"; \
    cmake --build --preset linux

FROM scratch AS artifact
COPY --from=build /src/build/linux/bin /
```

- [ ] **Step 2: Verify AMD-only Linux build on the lean base**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=amd --target artifact -o dist/linux-amd .
file dist/linux-amd/miner
```
Expected: exits 0; `file` reports `ELF 64-bit LSB ... x86-64`; `dist/linux-amd/kernel/` present.

- [ ] **Step 3: Verify combined Linux build (nvcc + OpenCL in one binary)**

Run:
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=both --target artifact -o dist/linux-both .
```
Expected: exits 0; `dist/linux-both/miner` is an ELF that linked both the CUDA runtime and the OpenCL loader (no unresolved-symbol errors in the build log).

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile.linux
git commit -m "build(docker): unified linux image (GPU=amd|nvidia|both)

Lean ubuntu base for amd-only; CUDA devel base (nvcc) for nvidia/both. Native
clang; one selector mirrors the windows-cross image."
```

---

## Task 6: Rewrite `scripts/docker-build.ps1` for the `-Os` / `-Gpu` matrix

**Files:**
- Modify: `scripts/docker-build.ps1`

- [ ] **Step 1: Replace the script body**

Overwrite `scripts/docker-build.ps1` with:

```powershell
#requires -Version 5.1
<#
.SYNOPSIS
    Build LuminousMiner via Docker for a chosen OS and GPU backend, extracting
    binaries into dist/<os>-<gpu>/.

.DESCRIPTION
    Everything builds in LINUX container mode -- Windows binaries are
    cross-compiled (clang-cl + xwin). No Docker engine mode switching, no local
    toolchain. Backends: amd (OpenCL), nvidia (CUDA), or both in one binary.

.PARAMETER Os
    linux | windows-cross | all   ('all' builds both for the chosen -Gpu.)

.PARAMETER Gpu
    amd | nvidia | both   (default: both)

.PARAMETER VcpkgRef
    git ref of vcpkg to pin inside the image (default: master).

.EXAMPLE
    scripts/docker-build.ps1 -Os windows-cross -Gpu both
.EXAMPLE
    scripts/docker-build.ps1 -Os all -Gpu amd
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('linux', 'windows-cross', 'all')]
    [string] $Os,

    [ValidateSet('amd', 'nvidia', 'both')]
    [string] $Gpu = 'both',

    [string] $VcpkgRef = 'master'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $repoRoot 'dist'

function Assert-LinuxEngine {
    $os = (& docker info --format '{{.OSType}}' 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Docker is not available. Is Docker Desktop running?" }
    if ($os.Trim() -ne 'linux') {
        throw "Docker is in '$($os.Trim())'-container mode; this build needs LINUX containers. Switch and re-run."
    }
}

function Build-One([string] $osName, [string] $gpu) {
    Assert-LinuxEngine
    $name = "$osName-$gpu"
    $out  = Join-Path $distRoot $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "==> Building $name (Linux container) -> $out" -ForegroundColor Cyan
    $env:DOCKER_BUILDKIT = '1'
    & docker build `
        -f (Join-Path $repoRoot "docker/Dockerfile.$osName") `
        --build-arg "GPU=$gpu" `
        --build-arg "VCPKG_REF=$VcpkgRef" `
        --target artifact `
        -o $out `
        $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "docker build failed for $name" }
    Write-Host "==> $name binaries in $out" -ForegroundColor Green
}

if ($Os -eq 'all') {
    foreach ($o in @('linux', 'windows-cross')) { Build-One $o $Gpu }
}
else {
    Build-One $Os $Gpu
}
```

- [ ] **Step 2: Verify the parameter contract (no build, just arg validation)**

Run:
```bash
pwsh -NoProfile -Command "& { . ./scripts/docker-build.ps1 -Os bogus -Gpu both } 2>&1 | Select-String 'ValidateSet|Cannot validate'"
```
Expected: a parameter-validation error mentioning the allowed set (`linux`, `windows-cross`, `all`) — confirms the enum is enforced and the script parses.

- [ ] **Step 3: Commit**

```bash
git add scripts/docker-build.ps1
git commit -m "build(scripts): docker-build.ps1 -Os/-Gpu matrix, Linux-container only

Drops the Windows-container mode-switch path; every target cross-builds in a
Linux container. Artifacts land in dist/<os>-<gpu>/."
```

---

## Task 7: Remove retired Dockerfiles, mingw toolchain, and stale references

**Files:**
- Delete: `docker/Dockerfile.windows-amd-cross`, `docker/Dockerfile.windows-nvidia-cross`, `docker/Dockerfile.linux-amd`, `docker/Dockerfile.linux-nvidia`, `docker/Dockerfile.windows-amd`, `docker/Dockerfile.windows-nvidia`
- Delete: `cmake/toolchain-mingw.cmake`, `triplets/x64-mingw-static.cmake`
- Modify: `vcpkg.json` (description/feature comments), `CLAUDE.md` if it references the old targets

- [ ] **Step 1: Confirm nothing still references the mingw toolchain/triplet or old Dockerfiles**

Run:
```bash
grep -rIn --exclude-dir=.git -e 'toolchain-mingw' -e 'x64-mingw-static' \
    -e 'windows-amd-cross' -e 'windows-nvidia-cross' \
    -e 'Dockerfile.linux-amd' -e 'Dockerfile.linux-nvidia' . || echo "NO REMAINING REFERENCES"
```
Expected: matches only inside `docs/superpowers/` (the spec/plan) — those are historical and fine. If any live config (`CMakePresets.json`, `scripts/`, `CMakeLists.txt`) still matches, fix it before deleting.

- [ ] **Step 2: Delete the retired files**

Run:
```bash
git rm docker/Dockerfile.windows-amd-cross docker/Dockerfile.windows-nvidia-cross \
       docker/Dockerfile.linux-amd docker/Dockerfile.linux-nvidia \
       docker/Dockerfile.windows-amd docker/Dockerfile.windows-nvidia \
       cmake/toolchain-mingw.cmake triplets/x64-mingw-static.cmake
```
Expected: each path staged for deletion. (If `triplets/x64-mingw-static.cmake` doesn't exist, drop it from the command — the mingw triplet may be a built-in vcpkg triplet rather than an overlay.)

- [ ] **Step 3: Refresh the `vcpkg.json` description to match the new feature usage**

In `vcpkg.json`, replace the `description` and the `openssl` feature note so they reference the unified images. Change the `"description"` value to:

```json
  "description": "Dependency manifest for LuminousMiner via vcpkg (Boost subset + optional OpenCL/OpenSSL). Enabled by -DUSE_VCPKG=ON + a vcpkg toolchain. OpenCL/OpenSSL are optional features selected per build by VCPKG_MANIFEST_FEATURES: the windows-cross image supplies OpenSSL prebuilt (no openssl feature) and adds opencl only for AMD; the linux image uses both.",
```

and the `openssl` feature description to:

```json
      "description": "OpenSSL via vcpkg (stratum TLS). Omitted on windows-cross (prebuilt MSVC libs); used on linux.",
```

- [ ] **Step 4: Verify the manifest still parses**

Run:
```bash
python3 -c "import json;json.load(open('vcpkg.json'));print('vcpkg.json OK')"
```
Expected: `vcpkg.json OK`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "build: remove mingw AMD cross path and per-target Dockerfiles

Superseded by the unified linux/windows-cross images on one MSVC-ABI toolchain.
Drops Dockerfile.{linux-amd,linux-nvidia,windows-amd,windows-nvidia,
windows-amd-cross,windows-nvidia-cross}, toolchain-mingw, and the mingw triplet."
```

---

## Task 8: Runtime validation on the AMD host (combined binary) + AMD perf parity

This is the acceptance gate for dropping mingw. Runs on the user's RX 9070 XT (Windows).

**Files:** none (validation only; produces notes for the PR)

- [ ] **Step 1: Confirm the combined miner.exe starts and selects AMD on an NVIDIA-less host**

Copy `dist/windows-cross-both/` to the Windows AMD machine. From that dir run:
```
miner.exe --help
```
Expected: prints usage and exits 0 — proves the cudart loader-stub (`cudart64_13.dll`, shipped) and OpenSSL DLLs resolve at process start even with no NVIDIA GPU/driver.

- [ ] **Step 2: Confirm it enumerates the AMD GPU and mines (short smoke run)**

Run the combined binary against the same pool/config used for the existing AMD benchmark, for a brief window. Expected log: the AMD device (RX 9070 XT) is detected and accepted; `cudaGetDeviceCount`→0 produces **no** NVIDIA devices and **no** crash (the DeviceManager skips the NVIDIA backend). If it crashes on 0 CUDA devices, file a follow-up to guard the CUDA enumeration path; the combined binary must degrade gracefully.

- [ ] **Step 3: Re-benchmark AMD on the MSVC-ABI build vs the retired mingw output**

Build the AMD-only benchmark through the new image (add `-DBUILD_EXE_BENCHMARK=ON` via an ad-hoc build, mirroring the existing AMD-benchmark workflow), run it on the RX 9070 XT, and compare hashrate to the recorded mingw baseline (`BASELINE_RDNA4.md` / the prior A/B results).
Expected: hashrate within noise of the mingw baseline. A material regression means the MSVC-ABI codegen differs and must be investigated before merge.

- [ ] **Step 4: Record results**

Append the combined-binary runtime result and the AMD MSVC-vs-mingw benchmark comparison to the PR description (and `BASELINE_RDNA4.md` if that's where AMD numbers live). No code commit unless Step 2/3 surfaced a defect.

---

## Task 9: Update the PR

**Files:** none (PR housekeeping)

- [ ] **Step 1: Push and refresh PR #149**

Push `work` to the fork branch backing PR #149 (`build/windows-cross-compile`). Update the PR body to describe the unified matrix (the `GPU` selector, two images, retired mingw), and note the Task 1 OpenCL verdict and Task 8 validation results. Keep the PR in **draft** until the user says otherwise (they asked to hold merging while optimizing).

```bash
git push <fork-remote> work:build/windows-cross-compile
gh pr edit 149 --repo luminousmining/miner --body-file <updated-body>
```
Expected: PR #149 reflects the unified design; still a draft.

---

## Self-Review notes

- **Spec coverage:** matrix (T3–T6), homogeneous toolchain/base selection (T4/T5), mingw retirement (T7), OpenCL-on-clangcl decision (T1, T4 S6), prebuilt-OpenSSL decoupling (T2), `-Os/-Gpu` script (T6), lean AMD image (T4 S3/T5 S2), combined-runs-on-AMD-host + AMD perf parity (T8), retire native-Windows Dockerfiles (T7). All spec sections map to a task.
- **Selector consistency:** `GPU` values `amd|nvidia|both` and the `AMD/NV/CC/FEAT` mapping are identical across the Dockerfile `case` (T4, T5) and the preset overrides (T3). `USE_CLANG_CUDA` is ON only for Windows-cross NVIDIA/both; OFF everywhere on Linux.
- **CMake consistency:** `LM_WIN_CROSS` (T2) is keyed on `OPENSSL_WIN_ROOT`, which both Windows-cross branches export (T4), so AMD-only and combined both get prebuilt OpenSSL; CUDA stays `BUILD_NVIDIA`-gated.
- **Open risk flagged in-task:** if clang-cl AMD code needs explicit Windows syslibs (the `WIN32 AND NOT MSVC` block at `sources/CMakeLists.txt:433` is skipped under clang-cl), T4 Steps 3–5 will surface unresolved symbols at link; the fix is to extend that syslib list to the MSVC clang-cl path. Noted here so the implementer recognizes it rather than treating it as mysterious.
</content>
