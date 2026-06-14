# Build — Linux (Docker, no local toolchain)

Build the Linux `miner` (ELF) in a container with **no local compilers or SDKs** —
only Docker (BuildKit) is required.

The Dockerfile exposes two independent axes — a `GPU` build-arg
(`amd` | `nvidia` | `both` | `none`, default `both`) and a `CPU` build-arg
(`ON` | `OFF`, default `OFF`):

| Dockerfile | Output | Backends |
|---|---|---|
| `docker/Dockerfile.linux` | `miner` (ELF) | AMD, NVIDIA, both, or none — plus the CPU resolver when `CPU=ON` |

`GPU=amd`/`none` use a lean `ubuntu:24.04` base; `GPU=nvidia`/`both` use
`nvidia/cuda:13.1.2-devel-ubuntu24.04`. `CPU=ON` folds the CPU resolver into
the same binary and can combine with any `GPU` value. `GPU=none CPU=ON` is a CPU-only
build; `GPU=none CPU=OFF` is rejected (nothing to build).

## Helper script (PowerShell)
```powershell
scripts/docker-build.ps1 -Os linux -Gpu amd          # AMD-only ELF
scripts/docker-build.ps1 -Os linux -Gpu amd -Cpu     # AMD + CPU in one ELF
scripts/docker-build.ps1 -Os linux -Gpu none -Cpu    # CPU-only ELF
```
Binaries are extracted to `dist/<os>-<gpu>[-cpu]/` (e.g. `dist/linux-amd/`,
`dist/linux-amd-cpu/`, `dist/linux-none-cpu/`).

## Direct docker build
```sh
# Linux, AMD only  ->  dist/linux-amd/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=amd --target artifact -o dist/linux-amd .

# Linux, AMD + CPU resolver  ->  dist/linux-amd-cpu/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=amd --build-arg CPU=ON --target artifact -o dist/linux-amd-cpu .

# Linux, CPU-only  ->  dist/linux-none-cpu/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=none --build-arg CPU=ON --target artifact -o dist/linux-none-cpu .
```
The artifact contains `miner`, the OpenCL `kernel/` directory, and the required
runtime files.
