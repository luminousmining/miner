# CPU Mining — Thread Count & Affinity Tuning

The CPU resolver scans nonces with a self-owned pool of persistent worker threads (one
batch of nonces fanned across the pool each iteration). Three flags control it:

| Flag | Default | Meaning |
|---|---|---|
| `--cpu` | `false` | Enable the CPU device. |
| `--cpu_threads` | all logical cores | Number of worker threads (pool size). |
| `--cpu_affinity` | none | Hex bitmask of logical cores to pin workers to (bit *i* = core *i*). |

`--cpu_affinity` accepts `0xFF` or `FF` (up to 64 cores). `--cpu_threads=0` is treated as
unset and falls back to all logical cores.

## How the two flags interact

| `--cpu_threads` | `--cpu_affinity` | Result |
|---|---|---|
| unset | unset | one worker per logical core, no pinning (default) |
| unset | set | N = popcount(mask); worker *k* is pinned to the *k*-th set bit |
| set | unset | N workers, no pinning |
| set | set | N workers; if N > set bits → pinned round-robin over them; if N < set bits → first N |

## Running CPU mining alongside a GPU

A GPU is not autonomous: its host-side feeder thread and the OpenCL/CUDA driver threads
(all inside the miner process) must launch kernels and read results continuously to keep
the GPU busy. If the CPU resolver spawns **one worker per core** (the default), those
workers saturate every core and starve the GPU's host threads — the GPU can drop to a
fraction of its solo hashrate while the CPU adds only a tiny amount.

> Example (16-core host + an RX 9070 XT, BLAKE3): GPU-only ≈ **1.70 GH/s**. With default
> all-core CPU mining also enabled, the GPU collapsed to ≈ **0.4 GH/s** — a far bigger
> loss than the ≈ 11 MH/s of CPU gained.

Lowering the thread *count* alone does **not** fix this: unpinned threads roam across all
cores and still bounce the GPU's feeder. The fix is to **reserve cores for the GPU with
affinity**:

```sh
# Pin the CPU pool to cores 0-7; leave cores 8-15 free for the GPU's host threads.
miner --algo=blake3 --host=<pool> --port=<port> --wallet=<addr> \
      --amd=true --cpu=true --cpu_threads=8 --cpu_affinity=0xFF
```

On the example host this restored the GPU to its full ≈ 1.70 GH/s while still mining
≈ 7 MH/s on the CPU.

## Recommendations

- On a GPU rig, **GPU-only is usually best** — a modern GPU out-hashes the CPU by orders
  of magnitude, so CPU mining rarely pays for the contention it adds.
- If you do want CPU hashes too, **reserve cores for the GPU** with `--cpu_threads` +
  `--cpu_affinity`; leave a few cores out of the CPU mask for the GPU's host/driver
  threads, and tune the split to taste.
- On a **CPU-only** machine, the defaults (all cores, no pinning) are correct.
- Process-level priority tweaks (e.g. forcing the whole miner to "below normal") do
  **not** relieve GPU+CPU contention — they lower the GPU feeder thread too. The
  reservation has to be per-core (affinity).
