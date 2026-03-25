# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Configure

```sh
# Linux (all platforms)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Linux (selective build)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_NVIDIA=ON -DBUILD_AMD=ON -DBUILD_CPU=OFF

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

### Build targets

```sh
cmake --build build --target miner -j$(nproc)
cmake --build build --target unit_test -j$(nproc)
cmake --build build --target benchmark -j$(nproc)
cmake --build build -j$(nproc)   # builds all enabled targets
```

### CMake build options

| Option | Default | Effect |
|---|---|---|
| `BUILD_NVIDIA` | `ON` | Compile CUDA resolvers |
| `BUILD_AMD` | `ON` | Compile OpenCL resolvers |
| `BUILD_CPU` | `ON` | Compile CPU resolver |
| `BUILD_EXE_MINER` | `ON` | Build `bin/miner` |
| `BUILD_EXE_UNIT_TEST` | `ON` | Build `bin/unit_test` (Google Test) |
| `BUILD_EXE_BENCHMARK` | `ON` | Build `bin/benchmark` |
| `TOOL_TRACE_MEMORY` | `OFF` | Developer memory tracing |
| `TOOL_MOCKER` | `OFF` | Device mocking (for tests without GPU) |
| `TOOL_ANALYZER` | `OFF` | Static analysis (cppcheck + clang-tidy) |

### Run tests

```sh
./bin/unit_test
# Run a specific test suite (Google Test filter)
./bin/unit_test --gtest_filter=AlgoTypeTest.*
```

### Code formatting

```sh
cmake --build build --target format         # Auto-format all sources
cmake --build build --target format-check   # Check formatting only (CI)
```

## Code Style

Enforced by `.clang-format` and `.clang-tidy`. See `CODING_STYLE.md` for the full spec. Key rules:

- **Naming**: `camelCase` for variables/functions/members, `UpperCase` for types/structs/enums, `UPPER_CASE` for `constexpr` constants, `lower_case` for namespaces
- **Brace style**: Allman (opening brace on its own line); always use braces even for single-line bodies
- **Column limit**: 120 characters
- **Pointers/references**: attached to type — `char** var`, `int& ref`
- **`const` placement**: right-side — `char const* const var`
- **Yoda conditions**: constant on left — `true == a`, `nullptr == ptr`
- **`auto`**: only in range-for loops or when the type is too complex to write out
- **Init**: use `{}` rather than `=` for initialization
- **2 blank lines** between function definitions

## Architecture Overview

LuminousMiner is structured as four vertical layers communicating through callbacks and atomic flags:

```
miner.cpp  (entry point)
     │
     ├── NETWORK LAYER    NetworkTCPClient  (Boost ASIO + SSL/TLS + SOCKS5)
     │         │
     ├── STRATUM LAYER    Stratum subclasses per protocol (EthereumV1/V2, EthProxy, SmartMining…)
     │         │  callbacks ──► DeviceManager
     │
     └── DEVICE MANAGER   DeviceManager (singleton)
               │
               └── DEVICE LAYER   Device per GPU (DeviceNvidia / DeviceAmd)
                         │
                         └── RESOLVER LAYER   ResolverAmd / ResolverNvidia per algorithm
```

### Threading model

- **Main thread**: initialization, joins all threads on exit
- **Network I/O thread** (1 per stratum): `boost::asio::io_context::run()` — async reads/writes
- **Device thread** (1 per GPU): `Device::loopDoWork()` — kernel launches, result reads, share submissions
- **Stats thread** (1): collects hashrates and displays dashboard every ~10 s

Synchronization uses `AtomicCounter<uint64_t>` to signal job/memory/constant updates from the stratum thread to device threads. The GPU TX queue is a `boost::lockfree::queue<string*>`.

### Key source directories

| Directory | Purpose |
|---|---|
| `sources/algo/` | Algorithm definitions, DAG context, hash types (`hash256`…`hash4096`) |
| `sources/common/` | Config singleton, CLI parser, logging, atomic helpers, kernel generator |
| `sources/network/` | `NetworkTCPClient` — Boost ASIO + SSL + SOCKS5 |
| `sources/stratum/` | Stratum protocol implementations (one per algorithm family) |
| `sources/device/` | `DeviceManager` singleton + `Device` base + `DeviceNvidia`/`DeviceAmd` |
| `sources/resolver/amd/` | OpenCL resolvers — one per algorithm |
| `sources/resolver/nvidia/` | CUDA resolvers — one per algorithm |
| `sources/resolver/cpu/` | OpenMP CPU resolver |
| `sources/profiler/` | NVML (NVIDIA) / ADL (AMD) GPU monitoring |
| `sources/statistical/` | Hashrate and share counters |
| `sources/api/` | HTTP REST API (default port 8080) backed by Boost.ASIO |
| `sources/benchmark/` | Benchmark executable framework |

### Adding a new algorithm

1. Define algorithm constants in `sources/algo/`
2. Add `ResolverAmd<Algo>` in `sources/resolver/amd/` (OpenCL kernel + DAG/constant management)
3. Add `ResolverNvidia<Algo>` in `sources/resolver/nvidia/` (CUDA kernel equivalent)
4. Add a `Stratum<Algo>` subclass in `sources/stratum/`
5. Register in `Device::setAlgorithm()` and `DeviceManager::connectToPools()`
6. Add unit tests under `sources/resolver/amd/tests/` and `sources/resolver/nvidia/tests/`

### Double-buffering strategy

Each GPU has two streams (CUDA) or two command queues (OpenCL). Iteration N runs on stream 0, iteration N+1 on stream 1; results from stream 0 are read while stream 1 is executing, hiding GPU→CPU transfer latency.

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| Boost | 1.90.0 | ASIO, JSON, Thread, LockFree |
| OpenSSL | 1.1.1 | TLS for stratum connections |
| CUDA | 13.1 | NVIDIA GPU kernels |
| OpenCL | 3.0.19 | AMD GPU kernels |
| Google Test | — | Unit tests |

See `BUILD.md` for full installation instructions per platform.
