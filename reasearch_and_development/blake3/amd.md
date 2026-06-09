# BLAKE3 (Alephium) — implementation notes

Status: **working & live-verified** on AMD (OpenCL). Live accepted shares (4/4, 0 rejected)
on RX 9070 XT vs HeroMiners (`alephium.herominers.com:1199`) and woolypooly
(`pool.woolypooly.com:3106`), 2026-06-09. **~1.71 GH/s** (full double-BLAKE3 PoW rate). BLAKE3
is **compute-bound** (no DAG), so unlike FishHash it likely has kernel-tuning headroom — no
dedicated optimization pass has been done yet.

> The displayed hashrate counts full PoWs/sec: `blocks × threads × internalLoop × kernels /
> elapsed`, with `internalLoop = 1` and one nonce per work-item. (An earlier "~248 MH/s" note
> was wrong — corrected after verifying both the counter and the live share cadence.)

## Algorithm

Alephium PoW = `BLAKE3(BLAKE3(nonce(24) || headerBlob(302)))`, 326 bytes total. No DAG, no
epochs.

- The 24-byte nonce is **prepended** (not overwriting the header): nonce = big-endian 8-byte
  search value + 16 zero bytes = exactly the 48-hex submit string. `headerBlob` (302 B) is
  left-aligned in the buffer (ubytes[0..301]) and ends with timestamp + target.
- Accept iff `digest <= targetBlob` (byte-wise from index 0) **and** the chain index
  `digest[31] % 16` maps to `(fromGroup, toGroup)` via `/4`, `%4` (CHAIN_NUMBER = 16).
- KAT vector (nonce `0x914544566c9a0a4d`):
  `394696ad2377a8ce8525032656e819183c0585d818ff1cffb52aca6acde2d095` (chainIndex 5).

## Key gotchas (hard-won — these once blocked all Alephium mining)

1. **Authorize parse.** A stratum authorize reply is `{result:true, error:null}` — no
   `params`. Reading `root.at("params")` threw `out_of_range`, leaving the session
   unauthenticated. Read `result` (bool), matching ethash/progpow/autolykos.
2. **Boundary from targetBlob.** Alephium pools ship the share target per-job (`targetBlob`)
   and send **no** `mining.set_difficulty`, so `jobInfo.boundary` stayed empty and
   `isValidJob()` dropped every job. Seed `boundary = targetBlob` (+ `boundaryU64`) in
   `onMiningNotify`.
3. **Nonce layout.** 24-byte prepend, big-endian (see above). The original CUDA path had the
   wrong layout (8-byte LE nonce overwriting header[0..7]) → pool "Invalid job chain index".
4. **Epoch thrash.** `onMiningNotify` must **not** bump `jobInfo.epoch` per job — a non-DAG
   algo rebuilds its kernel on epoch change (~1 s each), so a per-notify bump rebuilt ~2×/s
   and starved the miner. Keep the epoch constant (build once).

## Pools

- **HeroMiners** (plaintext, verified): `--host=alephium.herominers.com --port=1199`
- **woolypooly** (`pool.woolypooly.com:3106`) is **TLS-only**. The cross-built `.exe` cannot
  verify its certificate chain (no CA bundle ships with it), so SSL pools fail until a CA
  store is wired in (or a verify-skip flag is added). Use a plaintext pool until then.

Run: `miner --algo blake3 --host alephium.herominers.com --port 1199
--wallet <base58-ALPH-address> --workername <name>`

(`--algo` alone applies to every device; `--amd_algo`/`--nvidia_algo` are only needed
alongside `--amd_host`/`--amd_port` to point a vendor at a *different* pool+algo.)

> The wallet must be the **base58** Alephium address (≈45 chars), not a hex public key — pools
> reject the latter as "Invalid address".

## Test coverage

- CPU reference: `Blake3Ref` (double-BLAKE3 vs an independent lib).
- Shared primitive KAT: `Blake3SharedKat` (generic `blake3_hash_chunk`, incl. XOF).
- Mining-kernel KAT: `Blake3Cl` (`test_hash` + `search`, OpenCL).
- On-GPU AMD resolver: `ResolverBlake3AmdTest.findNonce` (compiles the kernel on the device,
  finds the winning nonce, submits).
- Offline stratum protocol: `sources/stratum/tests/blake3_protocol.cpp` (authorize result-bool,
  boundary-from-targetBlob, epoch held constant).

## Follow-ups (not done here)

- CUDA/NVIDIA resolver: a reworked CUDA blake3 path exists but is not registered/wired.
- Kernel tuning: compute-bound, so a profiling/optimization pass (cf. kHeavyHash) may lift the
  ~1.71 GH/s rate.
- TLS for SSL-only pools (woolypooly): ship/point to a CA bundle or add an insecure-verify flag.
