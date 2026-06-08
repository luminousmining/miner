# Reusable dev harness for kHeavyHash correctness work.
# Lets us compile/run the GoogleTest suite and the Python oracle on a machine
# with no native C++ toolchain, and run OpenCL kernels on the CPU via POCL
# (no GPU required) so the GPU kernel can be checked bit-identical to the CPU ref.
#
#   docker build -t kheavyhash-dev sources/algo/kheavyhash/dev
#   docker run --rm -v "$PWD:/work" -w /work kheavyhash-dev <cmd>
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make libgtest-dev \
        python3 python3-pycryptodome \
        pocl-opencl-icd opencl-headers ocl-icd-opencl-dev clinfo \
    && rm -rf /var/lib/apt/lists/*
