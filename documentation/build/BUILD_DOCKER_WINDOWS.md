# Build — Windows (Docker, cross-compiled, no local toolchain)

Build a Windows `miner.exe` (PE32+) with **no local compilers or SDKs** — only Docker
(BuildKit) is required. Windows binaries are **cross-compiled** from a Linux container
(clang-cl + xwin), so Docker stays in **Linux container mode**; there is no engine-mode
switch and no Windows host needed.

The Dockerfile selects backends with a `GPU` build-arg (`amd` | `nvidia` | `both`,
default `both`):

| Dockerfile | Output | Backends |
|---|---|---|
| `docker/Dockerfile.windows-cross` | `miner.exe` (PE32+) | AMD (OpenCL), NVIDIA (CUDA), or **both in one binary** |

## Helper script (PowerShell)
```powershell
scripts/docker-build.ps1 -Os windows-cross -Gpu both   # combined AMD+NVIDIA miner.exe
```
Binaries are extracted to `dist/<os>-<gpu>/` (e.g. `dist/windows-cross-both/`).

## Direct docker build
```sh
# Windows, combined AMD+NVIDIA  ->  dist/windows-cross-both/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=both --target artifact -o dist/windows-cross-both .
```
The artifact contains `miner.exe`, the OpenCL `kernel/` directory, and the required
OpenSSL + CUDA runtime DLLs. The combined `miner.exe` runs on a host that has only one
vendor's GPU — it probes for the NVIDIA driver and skips NVIDIA cleanly when absent;
pass `--nvidia=false` or `--amd=false` to force a single backend.
