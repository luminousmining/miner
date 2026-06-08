# kHeavyHash (Kaspa) — benchmark

Throughput of the kHeavyHash **OpenCL (AMD)** kernel, measured by running the real
miner against a live kHeavyHash pool and reading the steady-state hashrate.

> The kernel is the **correctness-only** implementation (one work-item per nonce,
> a full keccak-f[1600] per hash, the matrix re-read from global memory, no LDS
> caching or vectorisation). It is intentionally **not optimised** — these numbers
> are a baseline to tune against, not a peak.

## Method

```bat
:: from dist/windows-amd/ (the cross-built binary)
miner.exe --algo kheavyhash --host kheavyhash.auto.nicehash.com --port 9200 ^
          --wallet <addr> --workername bench --password x --api_port 38080
```
Read the steady-state `Hashrate` column of the `HASHRATE` dashboard (printed every
~10 s), or query the HTTP API: `GET http://127.0.0.1:<api_port>/api/get_stats` →
`hs` is the per-device hashrate in h/s.

Any kHeavyHash pool works (unmineable / 2miners / woolypooly); the hashrate is the
kernel's throughput and is independent of the pool. NiceHash was used here because
its dashboard rendered cleanly during the run.

## Results

| GPU | Arch | Hashrate | Notes |
|---|---|---|---|
| AMD Radeon RX 9070 XT | RDNA4 (gfx1201) | **~389.8 MH/s** | two consecutive readings: 389.87, 389.80 MH/s |
| AMD Radeon 5700G iGPU | Vega (gfx90c) | 0 H/s | builds the kernel but reports no hashrate — separate iGPU follow-up |

Host: Windows 11; binary cross-compiled from Linux via `docker/Dockerfile.windows-amd-cross`
(llvm-mingw + vcpkg `x64-mingw-static`); runtime OpenCL ICD from the AMD Radeon driver.
Correctness on the same run was confirmed end-to-end (pool-accepted shares).

## Caveats / follow-ups

- **Telemetry is flaky on this RDNA4 card.** The stats thread floods
  `ADL cannot get activity`, which intermittently stalls hashrate reporting (dashboard
  blank, `api/get_stats` `hs:[0,0]`). The two readings above are from a run where it
  rendered. (See the ADL2/PMLog telemetry work; unrelated to the kernel.)
- The iGPU reporting 0 H/s is a device-2 follow-up, not a kernel correctness issue.
- For reference, optimised kHeavyHash miners reach ~1.3–1.7 GH/s on a 9070 XT, so the
  naive kernel is roughly a quarter of peak — the expected headroom for a later
  optimisation pass (LDS matrix tiling, fewer global re-reads, register pressure).

## Why not the `sources/benchmark/` framework

That framework micro-benchmarks **kernel variants** (loop/threads/blocks `lm1..lmN`)
for tuning. With a single correctness-only kernel there are no variants to compare
yet, so the meaningful baseline is end-to-end miner throughput, recorded above. A
variant matrix belongs with the future optimisation work.
