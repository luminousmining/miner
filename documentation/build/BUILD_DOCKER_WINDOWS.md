# Build — Windows (Docker, cross-compiled, no local toolchain)

Build a Windows `miner.exe` (PE32+) with **no local compilers or SDKs** — only Docker
(BuildKit) is required. Windows binaries are **cross-compiled** from a Linux container
(clang-cl + xwin), so Docker stays in **Linux container mode**; there is no engine-mode
switch and no Windows host needed.

The Dockerfile exposes two independent axes — a `GPU` build-arg
(`amd` | `nvidia` | `both` | `none`, default `both`) and a `CPU` build-arg
(`ON` | `OFF`, default `OFF`):

| Dockerfile | Output | Backends |
|---|---|---|
| `docker/Dockerfile.windows-cross` | `miner.exe` (PE32+) | AMD (OpenCL), NVIDIA (CUDA), both, or none — plus the CPU resolver when `CPU=ON` |

`CPU=ON` folds the CPU resolver into `miner.exe` and combines with any `GPU` value;
`GPU=none CPU=ON` is a CPU-only `miner.exe`. The CPU resolver parallelizes via an
in-process `std::thread` pool, so the clang-cl build is fully multicore (no OpenMP or
libomp needed). `GPU=none CPU=OFF` is rejected (nothing to build).

## Helper script (PowerShell)
```powershell
scripts/docker-build.ps1 -Os windows-cross -Gpu both          # combined AMD+NVIDIA miner.exe
scripts/docker-build.ps1 -Os windows-cross -Gpu none -Cpu     # CPU-only miner.exe
```
Binaries are extracted to `dist/<os>-<gpu>[-cpu]/` (e.g. `dist/windows-cross-both/`,
`dist/windows-cross-none-cpu/`).

## Direct docker build
```sh
# Windows, combined AMD+NVIDIA  ->  dist/windows-cross-both/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=both --target artifact -o dist/windows-cross-both .

# Windows, CPU-only  ->  dist/windows-cross-none-cpu/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=none --build-arg CPU=ON --target artifact -o dist/windows-cross-none-cpu .
```
The artifact contains `miner.exe`, the OpenCL `kernel/` directory, and the required
OpenSSL + CUDA runtime DLLs. The combined `miner.exe` runs on a host that has only one
vendor's GPU — it probes for the NVIDIA driver and skips NVIDIA cleanly when absent;
pass `--nvidia=false` or `--amd=false` to force a single backend.
