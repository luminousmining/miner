# Build — Linux (Docker, no local toolchain)

Build the Linux `miner` (ELF) in a container with **no local compilers or SDKs** —
only Docker (BuildKit) is required.

The Dockerfile selects backends with a `GPU` build-arg (`amd` | `nvidia` | `both`,
default `both`):

| Dockerfile | Output | Backends |
|---|---|---|
| `docker/Dockerfile.linux` | `miner` (ELF) | AMD, NVIDIA, or both |

`GPU=amd` uses a lean `ubuntu:24.04` base; `GPU=nvidia`/`both` use
`nvidia/cuda:13.1.2-devel-ubuntu24.04`.

## Helper script (PowerShell)
```powershell
scripts/docker-build.ps1 -Os linux -Gpu amd    # AMD-only ELF
```
Binaries are extracted to `dist/<os>-<gpu>/` (e.g. `dist/linux-amd/`).

## Direct docker build
```sh
# Linux, AMD only  ->  dist/linux-amd/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=amd --target artifact -o dist/linux-amd .
```
The artifact contains `miner`, the OpenCL `kernel/` directory, and the required
runtime files.
