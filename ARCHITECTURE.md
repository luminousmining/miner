# LuminousMiner — Architecture

> Version 0.12 · C++20 · NVIDIA (CUDA) · AMD (OpenCL) · CPU (OpenMP)

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Directory Structure](#2-directory-structure)
3. [Startup & Initialization](#3-startup--initialization)
4. [Connection Process](#4-connection-process)
5. [Mining Pipeline](#5-mining-pipeline)
6. [Threading Model](#6-threading-model)
7. [Component Deep-Dives](#7-component-deep-dives)
   - [Network Layer](#71-network-layer)
   - [Stratum Layer](#72-stratum-layer)
   - [Device Manager](#73-device-manager)
   - [Device Abstraction](#74-device-abstraction)
   - [Resolver Layer](#75-resolver-layer)
   - [Configuration](#76-configuration)
   - [Statistics & API](#77-statistics--api)
8. [Supported Algorithms](#8-supported-algorithms)
9. [Mining Profiles](#9-mining-profiles)
10. [Build System](#10-build-system)

---

## 1. High-Level Overview

LuminousMiner is a 0-fee, open-source GPU miner. It is structured around four independent vertical layers that communicate through callbacks and atomic flags:

```
┌──────────────────────────────────────────────────────────────┐
│                         miner.cpp                            │
│                      (entry point)                           │
└─────────────┬──────────────────────────────┬─────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────┐    ┌──────────────────────────────┐
│      NETWORK LAYER      │    │      DEVICE MANAGER          │
│  NetworkTCPClient       │    │  DeviceManager (singleton)   │
│  - Boost ASIO           │    │  - Enumerates all GPUs       │
│  - SSL/TLS              │    │  - Dispatches jobs           │
│  - SOCKS5 proxy         │    │  - Collects statistics       │
└─────────┬───────────────┘    └──────────────┬───────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────────┐    ┌──────────────────────────────┐
│     STRATUM LAYER       │    │       DEVICE LAYER           │
│  Stratum (base)         │◄───►  Device (NVIDIA / AMD / CPU) │
│  - EthereumV1/V2        │    │  - One thread per GPU        │
│  - EthProxy             │    │  - Double-buffered streams   │
│  - SmartMining          │    │  - Atomic job sync           │
└─────────────────────────┘    └──────────────┬───────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────┐
                               │      RESOLVER LAYER          │
                               │  ResolverAmd / ResolverNvidia│
                               │  - GPU kernels (OCL/CUDA)    │
                               │  - DAG management            │
                               │  - Result submission         │
                               └──────────────────────────────┘
```

---

## 2. Directory Structure

```
sources/
├── miner.cpp                  # Entry point
├── algo/                      # Algorithm definitions, hash types
├── api/                       # HTTP REST API server (port 8080)
├── benchmark/                 # Performance benchmarking framework
├── common/                    # Config, logging, CLI, atomics, kernels
├── device/                    # GPU abstraction + DeviceManager
├── network/                   # TCP socket (Boost ASIO + SSL)
├── profiler/                  # NVML (NVIDIA) / ADL (AMD) monitoring
├── resolver/
│   ├── amd/                   # OpenCL resolvers (one per algorithm)
│   ├── nvidia/                # CUDA resolvers (one per algorithm)
│   └── cpu/                   # OpenMP CPU resolver
├── statistical/               # Hashrate & share counters
├── stratum/                   # Stratum protocol implementations
└── web/                       # Web UI assets
```

---

## 3. Startup & Initialization

```
main()
 │
 ├─ 1. DeviceManager::instance()        ← Singleton construction
 ├─ 2. Config::instance().load(argc, argv)
 │        └─ Parse CLI flags
 │             (--host, --port, --wallet, --algo,
 │              --nvidia/--amd/--cpu, --threads, --blocks …)
 │
 ├─ 3. ServerAPI.bind(port)             ← Start HTTP API (default :8080)
 │
 ├─ 4. DeviceManager::initialize()
 │        ├─ Enumerate NVIDIA devices (CUDA + NVML)
 │        ├─ Enumerate AMD devices (OpenCL)
 │        └─ Build Device* list
 │
 └─ 5. DeviceManager::run()
          ├─ connectToPools()           ← Standard profile
          │    OR
          │   connectToSmartMining()    ← Smart Mining profile
          │
          ├─ device->run()  (×N GPUs)  ← Spawn one thread per device
          └─ loopStatistical()          ← Spawn stats thread
```

---

## 4. Connection Process

### 4.1 TCP / SSL Handshake

```
Miner                                   Pool (Stratum Server)
  │                                           │
  │── DNS resolve(host) ──────────────────────┤
  │                                           │
  │── TCP SYN ────────────────────────────────►
  │◄─ TCP SYN-ACK ────────────────────────────┤
  │── TCP ACK ────────────────────────────────►
  │                                           │
  │  [if SSL enabled]                         │
  │── TLS ClientHello ────────────────────────►
  │◄─ TLS ServerHello + Certificate ──────────┤
  │── TLS Finished ───────────────────────────►
  │◄─ TLS Finished ───────────────────────────┤
  │                                           │
  │  onConnect() callback fires               │
  │                                           │
```

### 4.2 Stratum Handshake (EthereumV1)

```
Miner                                   Pool
  │                                       │
  │── {"method":"mining.subscribe",       │
  │    "id":1, "params":[…]}  ───────────►│
  │◄─ {"id":1, "result":                  │
  │    [sessionId, extraNonce]}  ─────────┤
  │                                       │
  │── {"method":"mining.authorize",       │
  │    "id":2,                            │
  │    "params":["wallet","password"]} ──►│
  │◄─ {"id":2, "result":true}  ───────────┤
  │                                       │
  │◄─ {"method":"mining.set_difficulty",  │
  │    "params":[diff]}  ─────────────────┤
  │                                       │
  │◄─ {"method":"mining.notify",          │
  │    "params":[jobId, seedHash,         │
  │    headerHash, cleanJob]}  ───────────┤
  │                                       │
  │  onMiningNotify() ──► dispatchJob()   │
  │                                       │
  │  [share found]                        │
  │── {"method":"mining.submit",          │
  │    "id":1000+,                        │
  │    "params":[worker, jobId,           │
  │    nonce, headerHash, mixHash]}  ────►│
  │◄─ {"id":1000+, "result":true}  ───────┤  ← valid share
  │    OR                                 │
  │◄─ {"id":1000+, "result":false}  ──────┤  ← invalid share
  │                                       │
```

### 4.3 SOCKS5 Proxy (optional)

When `--socks5` is set, the TCP connection is first established to the SOCKS5 proxy, which then tunnels to the pool:

```
Miner ──► SOCKS5 Proxy ──► Pool
```

---

## 5. Mining Pipeline

### 5.1 Job Dispatch Flow

```
Pool
 │ mining.notify
 ▼
Stratum::onMiningNotify()
 │ Parse: jobId, seedHash, headerHash, difficulty, cleanJob
 │ Update: StratumJobInfo
 │
 ▼
callbackUpdateJob(stratumUUID, jobInfo)   ← function<void(...)>
 │
 ▼
DeviceManager::onUpdateJob()
 │ For each Device bound to this stratumUUID:
 │
 ▼
Device::update(memory, constants, jobInfo)
 │ Atomically increments:
 │   synchronizer.job      ← always
 │   synchronizer.memory   ← if epoch changed
 │   synchronizer.constant ← always
 │
 ▼
Device::loopDoWork()  [GPU thread wakes up]
```

### 5.2 GPU Work Loop (per device)

```
loopDoWork()
 │
 ├─ initialize()
 │   ├─ DeviceNvidia: cuCtxCreate, cudaStreamCreate×2
 │   └─ DeviceAmd:    clCreateContext, clCreateCommandQueue×2
 │
 ├─ waitJob()  ← block until first job arrives
 │
 └─ loop:
     │
     ├─ [job counter mismatch?]
     │    ├─ updateMemory()   ← epoch changed
     │    │    └─ resolver->updateMemory(jobInfo)
     │    │         ├─ AMD:    Build DAG on GPU via OpenCL
     │    │         └─ NVIDIA: Build DAG on GPU via CUDA
     │    │
     │    └─ updateConstants()
     │         └─ resolver->updateConstants(jobInfo)
     │              └─ Set nonce range, header, boundary, difficulty
     │
     ├─ resolver->executeAsync(jobInfo)
     │    ├─ NVIDIA: cudaLaunchKernel(stream[idx % 2])
     │    └─ AMD:    clEnqueueNDRangeKernel(queue[idx % 2])
     │
     ├─ miningStats.increaseKernelExecuted()
     │
     ├─ submit()
     │    └─ resolver->submit(stratum)
     │         ├─ Read result buffer from GPU
     │         ├─ [hash ≤ target?]
     │         └─ stratum->submit(nonce, mixHash, jobId)
     │              └─ Send mining.submit JSON to pool
     │
     └─ nonce += batchNonce    ← advance nonce window
```

### 5.3 Double-Buffering Strategy

Each GPU has two streams (CUDA) or two command queues (OpenCL). This avoids stalling the GPU:

```
Iteration N:   [Kernel on stream 0] ──────────┐
Iteration N+1: [Kernel on stream 1] ──────────┼──► GPU runs concurrently
                                               │
Read results:  [Read stream 0 result] ◄────────┘   (stream 0 already done)
```

```cpp
currentIndexStream = (currentIndexStream + 1) % 2;
// Launch on stream[currentIndexStream]
// Read  from stream[1 - currentIndexStream]
```

---

## 6. Threading Model

```
Process
 │
 ├── Main Thread
 │     ├─ Initialize config, devices, API
 │     └─ Join all threads on exit
 │
 ├── Stats Thread (×1)
 │     └─ loopStatistical()
 │           └─ Every ~10s: collect hashrates, display dashboard
 │
 ├── Network I/O Thread (×1 per Stratum)
 │     └─ boost::asio::io_context::run()
 │           ├─ async_read  → onReceive() → parse JSON → callbacks
 │           └─ async_write ← lockfree::queue<string*>
 │
 └── Device Thread (×1 per GPU)
       └─ Device::loopDoWork()
             ├─ GPU kernel launches
             ├─ Result reads
             └─ Share submissions
```

### Synchronization Primitives

| Primitive | Where | Purpose |
|---|---|---|
| `AtomicCounter<uint64_t>` | Device ↔ Stratum | Signal job/memory/constant updates |
| `boost::mutex` | `StratumJobInfo` | Protect job data during writes |
| `boost::lockfree::queue<string*>` | Network TX | Lock-free message sending |
| `boost::atomic<bool>` | `Device` | alive / computing state flags |
| `boost::thread` | Device, Stats | Manage work threads |

---

## 7. Component Deep-Dives

### 7.1 Network Layer

**File**: `network/network.hpp`

```
NetworkTCPClient
 ├─ boost::asio::io_context         ← Async I/O service
 ├─ boost::asio::ssl::stream<tcp>   ← Optional TLS socket
 ├─ boost::lockfree::queue<string*> ← Non-blocking TX queue
 └─ boost::thread runService        ← Dedicated I/O thread

Methods:
  connect()          → DNS resolve → TCP connect → [TLS handshake]
  send(json)         → Serialize → push to TX queue → async_write
  onConnect()        → virtual, called after handshake
  onReceive(string)  → virtual, called on each incoming line
```

### 7.2 Stratum Layer

**File**: `stratum/stratum.hpp` (base), `stratum/ethash.hpp`, `stratum/progpow.hpp`, `stratum/smart_mining.hpp`, …

```
NetworkTCPClient
 └─► Stratum  (base, owns StratumJobInfo)
      ├─► StratumEthash        (EthereumV1 / V2 / EthProxy)
      ├─► StratumProgPow
      ├─► StratumKawPow
      ├─► StratumAutolykosV2
      ├─► StratumBlake3
      └─► StratumSmartMining   (automatic algo/pool switching)

Key IDs:
  ID_MINING_SUBSCRIBE = 1
  ID_MINING_AUTHORIZE = 2
  ID_MINING_SUBMIT    = 1000 … (increments per share)

Callbacks exposed to DeviceManager:
  callbackUpdateJob(stratumUUID, StratumJobInfo)
  callbackShareStatus(isValid, requestID, stratumUUID)
```

### 7.3 Device Manager

**File**: `device/device_manager.hpp`

```
DeviceManager (singleton)
 │
 ├─ std::vector<Device*> devices
 ├─ std::map<uint32_t, Stratum*> stratums    ← UUID → pool
 │
 ├─ initialize()
 │    ├─ Enumerate NVIDIA (CUDA/NVML)
 │    └─ Enumerate AMD (OpenCL)
 │
 ├─ connectToPools()
 │    ├─ Create Stratum per pool config
 │    ├─ Assign devices to stratums (by UUID)
 │    └─ stratum->connect()
 │
 ├─ onUpdateJob(uuid, jobInfo)    ← called from Stratum thread
 │    └─ For each device with matching UUID:
 │         device->update(...)
 │
 └─ onShareStatus(isValid, id, uuid)
      └─ device->increaseShare(isValid)
```

### 7.4 Device Abstraction

```
Device  (abstract base)
 ├─ DeviceNvidia
 │    ├─ CUdevice, CUcontext
 │    └─ cudaStream_t stream[2]
 │
 └─ DeviceAmd
      ├─ cl::Device, cl::Context
      └─ cl::CommandQueue queue[2]

Lifecycle:
  run()          → Spawn loopDoWork() thread
  initialize()   → GPU context + resolver creation
  setAlgorithm() → Instantiate correct Resolver subclass
  update()       → Signal atomic counters
  kill()         → Set alive=false, join thread
```

### 7.5 Resolver Layer

```
Resolver  (abstract)
 │
 ├─ ResolverAmd  (OpenCL base)
 │    ├─ ResolverAmdEthash
 │    ├─ ResolverAmdEtchash
 │    ├─ ResolverAmdProgPow
 │    ├─ ResolverAmdProgPowZ
 │    ├─ ResolverAmdProgPowQuai
 │    ├─ ResolverAmdKawPow
 │    ├─ ResolverAmdMeowPow
 │    ├─ ResolverAmdFiroPow
 │    ├─ ResolverAmdEvrProgPow
 │    └─ ResolverAmdAutolykosV2
 │
 └─ ResolverNvidia  (CUDA base)
      ├─ ResolverNvidiaEthash
      ├─ ResolverNvidiaEtchash
      ├─ ResolverNvidiaProgPow
      ├─ ResolverNvidiaKawPow
      ├─ ResolverNvidiaBlake3        ← NVIDIA only
      └─ … (same set as AMD)

Interface:
  updateMemory(jobInfo)    → Build/rebuild GPU DAG
  updateConstants(jobInfo) → Push nonce range, header, target
  executeAsync(jobInfo)    → Enqueue kernel on current stream/queue
  executeSync(jobInfo)     → Synchronous variant (benchmarks)
  submit(stratum)          → Read result buffer, send share if valid
```

### 7.6 Configuration

**File**: `common/config.hpp`

Key configuration blocks:

| Block | Fields |
|---|---|
| `PoolConfig` | host, port, wallet, password, algo, stratumType, SSL, SOCKS5 |
| `DeviceOccupancy` | threads, blocks, internalLoop, cudaContext, isAuto |
| `DeviceEnableSetting` | nvidiaEnable, amdEnable, cpuEnable |
| `LogConfig` | level, file, intervalHashStats, showNewJob |

Loaded from CLI by `common::Cli`, supports per-device pool overrides:
```
--device_pool=0:host:port:wallet:algo
```

### 7.7 Statistics & API

**Statistics** (`statistical/statistical.hpp`):
- Tracks kernel executions, valid/invalid shares, elapsed time
- `getHashrate()` computes MH/s from kernel count × batch nonce / time
- Printed to console every ~10 seconds by the stats thread

**REST API** (`api/api.hpp`):
- Default port: `8080`
- Serves per-device hashrate, share counts, device health
- Backed by Boost.ASIO HTTP server

---

## 8. Supported Algorithms

| Algorithm | GPU | Notes |
|---|---|---|
| ETHASH | AMD, NVIDIA | Ethereum (legacy PoW) |
| ETCHASH | AMD, NVIDIA | Ethereum Classic |
| PROGPOW | AMD, NVIDIA, CPU | Ethereum ProgPoW |
| PROGPOWZ | AMD, NVIDIA | ProgPoW-Z variant |
| PROGPOWQUAI | AMD, NVIDIA | Quai Network |
| KAWPOW | AMD, NVIDIA | Ravencoin |
| MEOWPOW | AMD, NVIDIA | |
| FIROPOW | AMD, NVIDIA | Firo |
| EVRPROGPOW | AMD, NVIDIA | Evermore |
| AUTOLYKOS_V2 | AMD, NVIDIA | Ergo |
| BLAKE3 | NVIDIA only | |
| SHA256 | CPU | |

All DAG-based algorithms (Ethash family) share a common `algo::DagContext` and `algo::ethash::ContextGenerator`. The epoch is derived from the seed hash.

Hash types (`algo/hash.hpp`):

| Type | Size | Usage |
|---|---|---|
| `hash256` | 32 B | Job ID, header, seed, boundary |
| `hash512` | 64 B | DAG leaf |
| `hash1024` | 128 B | DAG node |
| `hash2048` | 256 B | Coinbase parts |
| `hash3072` | 384 B | BLAKE3 header blob |
| `hash4096` | 512 B | Large buffers |

---

## 9. Mining Profiles

### Standard Profile

```
User config (--host --algo --wallet)
       │
       ▼
DeviceManager::connectToPools()
       │
       ├─ Create Stratum for AMD devices
       └─ Create Stratum for NVIDIA devices
              │
              ▼
         Each GPU mines on its assigned pool
```

### Smart Mining Profile

```
User config (--sm_wallet=COIN:ADDR …)
       │
       ▼
DeviceManager::connectToSmartMining()
       │
       ▼
StratumSmartMining  (special protocol)
  ├─ Pool sends: set_algo  → Device switches algorithm dynamically
  ├─ Pool sends: set_job   → Update work
  └─ Pool sends: set_diff  → Adjust difficulty
       │
       ▼
  Miner automatically follows most profitable coin/algo
```

---

## 10. Build System

**CMake 3.22+** with the following options:

| Option | Default | Effect |
|---|---|---|
| `BUILD_NVIDIA` | `ON` | Compile CUDA resolvers |
| `BUILD_AMD` | `ON` | Compile OpenCL resolvers |
| `BUILD_CPU` | `ON` | Compile CPU resolver |
| `BUILD_EXE_MINER` | `ON` | Build `bin/miner` |
| `BUILD_EXE_UNIT_TEST` | `OFF` | Build `bin/unit_test` (Google Test) |
| `BUILD_EXE_BENCHMARK` | `OFF` | Build `bin/benchmark` |

**Key dependencies**:

| Library | Purpose |
|---|---|
| Boost.ASIO | Async network I/O |
| Boost.JSON | Stratum message parsing |
| Boost.Thread | Thread management |
| Boost.LockFree | Lock-free TX queue |
| CUDA Runtime | NVIDIA GPU kernels |
| OpenCL | AMD GPU kernels |
| NVML | NVIDIA GPU monitoring |
| ADL | AMD GPU monitoring |
| Google Test | Unit tests |
| PVS-Studio | Static analysis (CI) |

```bash
cmake -B build -DBUILD_NVIDIA=ON -DBUILD_AMD=ON
cmake --build build --target miner -j$(nproc)
./bin/miner --host pool.example.com --port 4444 \
            --wallet 0xYOUR_WALLET --algo ETHASH
```
