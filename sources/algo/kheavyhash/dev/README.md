# kHeavyHash dev harness (correctness oracle)

This directory is **build/test tooling only** — it is not compiled into the miner.

## What's here
- `Dockerfile` — minimal image (g++, CMake, GoogleTest, pycryptodome) so the
  CPU reference can be built and tested on a host with no native C++ toolchain.
- `oracle.py` — an **independent** re-implementation of kHeavyHash used purely to
  produce known-answer vectors. It (1) parses the rusty-kaspa reference files for
  their literal vectors and asserts agreement, (2) cross-checks the precomputed
  keccak initial-state constants against pycryptodome's cSHAKE256, then (3) emits
  `../tests/kheavyhash_test_vectors.hpp`. If any self-check fails it emits nothing.
- `ref_*.rs` — verbatim copies of rusty-kaspa source, kept so the oracle's
  self-checks are reproducible offline and provenance is auditable:
  - `ref_matrix.rs`, `ref_pow_hashers.rs`, `ref_xoshiro.rs`, `ref_lib.rs`
  - Source: https://github.com/kaspanet/rusty-kaspa (`master`), license **ISC**
    (permissive, GPL-3.0 compatible). These are reference/test-vector inputs only;
    the miner's implementation under `sources/algo/kheavyhash/*.{hpp,cpp}` is an
    independent rewrite.

## Regenerate vectors
```sh
docker build -t kheavyhash-dev sources/algo/kheavyhash/dev
docker run --rm -v "$PWD:/work" -w /work kheavyhash-dev \
    python3 sources/algo/kheavyhash/dev/oracle.py
```

## Build + run the KAT suite (no GPU toolchain needed)
```sh
docker run --rm -v "$PWD:/work" -w /work kheavyhash-dev bash -c \
  "cmake -S sources/algo/kheavyhash -B /tmp/bk && cmake --build /tmp/bk -j && \
   ctest --test-dir /tmp/bk --output-on-failure"
```
