# Build — Linux ARM64 (CPU-only, in Docker)

A CPU-only Linux/aarch64 build. CUDA/OpenCL GPU backends are off (Docker has no GPU
passthrough on Apple Silicon, and there is no aarch64 OpenCL/CUDA mining path here).
On an Apple Silicon Mac the arm64 image builds and runs **natively** (no emulation):

```sh
docker buildx build --platform linux/arm64 -f docker/Dockerfile.linux \
    --build-arg GPU=none --build-arg CPU=ON --target runtime -t lm:linux-arm64 --load .
docker run --rm lm:linux-arm64 --help
```

The same `GPU=none CPU=ON` (CPU-only) mode works for x86-64 (`--platform linux/amd64`).
The `GPU=amd|nvidia|both` GPU builds are unchanged, and `CPU=ON` can be added to any of
them to fold in the CPU resolver. CI builds the arm64 image on a native arm64 runner via
`.github/workflows/miner_linux_arm64_vcpkg.yml`, and the x86-64 CPU path via
`.github/workflows/miner_linux_x64_vcpkg.yml`.
