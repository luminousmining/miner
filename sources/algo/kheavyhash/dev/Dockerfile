# Reusable dev harness for kHeavyHash CPU-reference correctness work.
# Lets us compile/run the GoogleTest suite and the Python oracle on a machine
# with no native C++ toolchain. CPU-only; no Boost/CUDA/OpenCL.
#
#   docker build -t kheavyhash-dev sources/algo/kheavyhash/dev
#   docker run --rm -v "$PWD:/work" -w /work kheavyhash-dev <cmd>
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make libgtest-dev \
        python3 python3-pycryptodome \
    && rm -rf /var/lib/apt/lists/*
