# kHeavyHash (Kaspa) — benchmark

Throughput of the kHeavyHash **OpenCL (AMD)** `search` kernel.

Two independent measurements are recorded and they agree:

1. the **`sources/benchmark` framework** (`bin/benchmark`), which times the bare
   kernel launch — no stratum, no share submission; and
2. the **end-to-end miner** (`bin/miner`) against a live pool, reading the
   steady-state dashboard hashrate.

> The kernel is the **correctness-only** implementation (one work-item per nonce,
> a full keccak-f[1600] per hash, the matrix re-read from global memory, no LDS
> caching or vectorisation). It is intentionally **not optimised** — these numbers
> are a baseline to tune against, not a peak.

## 1. `sources/benchmark` framework

The kHeavyHash benchmark lives at `sources/benchmark/amd/kheavyhash.cpp`
(`runAmdKHeavyHash`) and is wired into `runAmd()`. It generates the 64×64 nibble
matrix host-side (the same CPU reference the kernel is gated bit-identical
against), uploads matrix/header/target, then launches the `search` kernel in a
loop, timing each launch. The target is left all-zero so no work-item ever meets
it: the result buffer stays untouched and atomic contention never skews the
timing. Hashrate = (`blocks` × `threads`) nonces ÷ per-launch time.

Enable it in `sources/benchmark/config.json` (copied next to the binary) under
`amd.algorithms.kheavyhash` (`enabled: true`), then:

```bat
:: from the benchmark binary's directory
benchmark.exe config.json
```

The benchmark is **synchronous** (one `enqueue` + `finish` per launch, no
double-buffering), so the first ~30 launches measure the GPU ramping from idle
clocks; let it run ≥150 loops for a steady-state reading.

### Results

| GPU | Arch | Grid (blocks×threads) | Hashrate (steady) |
|---|---|---|---|
| AMD Radeon RX 9070 XT | RDNA4 (gfx1201) | 8192 × 256 | **~396 MH/s** (386–402, ~5.25 ms/launch) |

Cold-start ramp on the same run: launches climb 160 → 307 MH/s over the first
30 iterations (boost-clock spin-up) before settling at ~396 MH/s.

## 2. End-to-end miner (corroboration)

```bat
miner.exe --algo kheavyhash --host kheavyhash.auto.nicehash.com --port 9200 ^
          --wallet <addr> --workername bench --password x --api_port 38080
```
Read the steady-state `Hashrate` column of the `HASHRATE` dashboard (printed every
~10 s), or query the HTTP API: `GET http://127.0.0.1:<api_port>/api/get_stats` →
`hs` is the per-device hashrate in h/s. Any kHeavyHash pool works (unmineable /
NiceHash / 2miners); the hashrate is the kernel's throughput, independent of pool.

| GPU | Arch | Hashrate | Notes |
|---|---|---|---|
| AMD Radeon RX 9070 XT | RDNA4 (gfx1201) | **~389.8 MH/s** | two consecutive readings: 389.87, 389.80 MH/s |
| AMD Radeon 5700G iGPU | Vega (gfx90c) | 0 H/s | builds the kernel but reports no hashrate — separate iGPU follow-up |

The miner double-buffers (two command queues overlapping), so its sustained
number lands just under the framework's bare-launch peak — the two agree within
~2%, which is the expected gap between an overlapped pipeline and an isolated
`finish`-per-launch micro-benchmark.

Host: Windows 11; binary cross-compiled from Linux via `docker/Dockerfile.windows-cross`
(llvm-mingw + vcpkg `x64-mingw-static`); runtime OpenCL ICD from the AMD Radeon driver.
Correctness on the same run was confirmed end-to-end (pool-accepted shares).

## Caveats / follow-ups

- **Telemetry is flaky on this RDNA4 card.** The miner's stats thread floods
  `ADL cannot get activity`, which intermittently stalls hashrate reporting
  (dashboard blank, `api/get_stats` `hs:[0,0]`). The framework benchmark times the
  kernel directly and is unaffected. (See the ADL2/PMLog telemetry work; unrelated
  to the kernel.)
- The iGPU reporting 0 H/s is a device-2 follow-up, not a kernel correctness issue.
- For reference, optimised kHeavyHash miners reach ~1.3–1.7 GH/s on a 9070 XT, so the
  naive kernel is roughly a quarter of peak — the expected headroom for a later
  optimisation pass (LDS matrix tiling, fewer global re-reads, register pressure).
  When that pass lands it can add `search_lm1…lmN` kernel variants and compare them
  here through the same framework (the config's `kernels` list already supports it).
