# Build — macOS (Apple Silicon, CPU-only)

> **Note:** macOS has neither CUDA nor a usable OpenCL 3.0 GPU runtime, so this is a
> **CPU-only** build (both GPU backends OFF). It produces a Mach-O `miner` binary for
> build-system parity and artifact coverage; it does **not** perform GPU mining.
> A native macOS binary cannot be produced inside a container: Docker and podman both
> run a **Linux** VM on macOS (so they build the *Linux* miner, not a Mach-O one), and
> Apple's licensing forbids running macOS in a container regardless. This target therefore
> uses the same vcpkg + CMake-preset flow **natively** rather than a Dockerfile.

Requirements (via [Homebrew](https://brew.sh)):
```sh
brew install cmake ninja pkg-config   # pkg-config is needed by vcpkg's openssl port
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg
```

Configure and build with the `macos-cpu` preset (Boost + OpenSSL are resolved by vcpkg
from `vcpkg.json`):
```sh
cmake --preset macos-cpu
cmake --build --preset macos-cpu -j$(sysctl -n hw.ncpu)
```
The binary is written to `bin/miner`. CI builds this on a native
Apple Silicon runner via `.github/workflows/miner_macos_vcpkg.yml`.
