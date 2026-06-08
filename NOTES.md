# kHeavyHash (Kaspa) — Implementation Research Notes

Status: **Layers 1–3 verified; AMD integration + Layer-4 stratum code-complete
(full-miner compile pending).** Layer 5 (live accepted shares) pending.
Reference: **rusty-kaspa `master`** (fetched 2026-06-08), which is post-Crescendo.
Branch: `worktree-kheavyhash-kaspa`.

## Layer 4 — Kaspa stratum client
`stratum/kheavyhash.{hpp,cpp}` (`StratumKHeavyHash`) + `stratums.cpp` factory case,
modeled on the gostratum/kaspa-stratum-bridge NORMAL dialect (NOTES s4):
- `miningSubscribe` → base subscribe (non-BzMiner agent) then authorize.
- `onMiningNotify` parses `[jobIdStr, [4 LE u64], timestamp]` → `prePowFromWords`
  into `headerHash`, sets `timestamp`, restarts nonce at `startNonce`, pins
  `epoch=1` (so resolver `updateMemory` runs once; per-job `headerHash` change drives
  `updateConstants`), then `updateJob()`.
- `onMiningSetDifficulty` → `difficultyToTargetLe` → `boundary` (LE 256-bit).
- `onMiningSetExtraNonce`/subscribe-ack → `setExtraNonce`.
- `miningSubmit` emits `[wallet.worker, jobId, nonceHex]`.

The two error-prone primitives live dependency-free in
`algo/kheavyhash/stratum_math.{hpp,cpp}` and are **TDD-verified now** (4 KATs, run via
the dev image): `prePowFromWords` (4 LE u64 → 32 bytes, exact) and
`difficultyToTargetLe` (maxTarget=2^224-1, exact integer long-division, LE; diff=1 →
28×0xFF, diff=2 → 2^223-1, monotonic). ⚠ `maxTarget`/scaling is the most pool-specific
value — confirm against the chosen testnet pool before trusting acceptance (s4.3).

Not compile-verified in the full miner (needs Boost/the full build); the stratum is a
thin JSON wrapper over the verified math + the existing base-stratum plumbing.

## Integration into the main miner (AMD)
Wired KHEAVYHASH into the real miner, additive to the existing seams:
- `algo_type` enum + `toString`/`toEnum` ("kheavyhash"); `algo/kheavyhash/result.hpp`
  (`Result`/`ResultShare`, layout matching the `.cl` Result struct).
- `resolver/amd/kheavyhash.{hpp,cpp}` + `kheavyhash_kernel_parameter.hpp`, modeled
  on `ResolverAmdEthash` but **no DAG**: `updateMemory` only allocates buffers +
  builds the `search` kernel; `updateConstants` host-generates the matrix
  (`::kheavyhash::generateMatrix`) and uploads matrix/header/target; `executeSync`
  launches a 1-D `search` (global = blocks*threads, nonce = jobInfo.nonce + gid);
  `submit` emits `[jobId, nonceHex]`. header/target buffers typed `algo::hash256`.
- `StratumJobInfo.timestamp` (+ copy); `device.cpp` setAlgorithm KHEAVYHASH→AMD case
  (NVIDIA/CUDA path intentionally empty — no CUDA kernel yet).
- CMake: `algo/kheavyhash/CMakeLists.txt` guarded — standalone dev/POCL harness when
  built directly, else compiles the CPU reference into the miner; `opencl/` deploys
  `kheavyhash.cl` to `kernel/kheavyhash/`; CPU KATs fold into the main `unit_test`.

**Not yet compile-verified in the full miner** — needs the AMD cross-build toolchain
(clang-cl + OpenCL), which is on the `build/windows-cross-compile` branch, not this
worktree. The standalone CPU+OpenCL(POCL) KAT suites stay green (5/5 + 8/8). The
search kernel still has no live job source until Layer 4 (Kaspa testnet stratum)
populates headerHash/boundary/timestamp and signals the memory/constant counters.

## Progress (Layer 3 — OpenCL kernel, bit-identical to CPU oracle)
`sources/algo/kheavyhash/opencl/kheavyhash.cl` ports keccak-f[1600], PowHash,
KHeavyHash and the heavy matmul step to OpenCL, plus the per-nonce `search`
kernel (`pow <= target`, LE). Matrix is generated host-side (CPU `generateMatrix`)
and uploaded; the kernel never does rank-checking. Constants copied verbatim from
the CPU reference (not re-derived).

**5/5 OpenCL KATs pass** (`opencl/tests/opencl_kat_test.cpp`), run on the CPU via
**POCL** (no GPU needed) and gated on the *same* vectors as the CPU suite:
`test_pow_hash` == `POW_KAT_EXPECTED`, `test_kheavy` == `KHEAVY_EXPECTED`,
`test_heavy_hash` == `HEAVY_EXPECTED`, and the `search` kernel at exactly
`FP_NONCE` passing `FP_TARGET_PASS` (== `FP_FINAL`) while failing
`FP_TARGET_FAIL` (== `FP_FINAL`-1) — together pinning the end-to-end GPU pow to
`FP_FINAL` bit-for-bit. The shipped `.cl` is loaded at runtime (`KH_CL_PATH`), so
there is no second copy to drift. Build: `-DKHEAVYHASH_BUILD_OPENCL=ON`.

Not yet wired into `ResolverAmd`/`device.cpp` — the `.cl` + harness are additive
and isolated, exactly like the CPU reference. Wiring is part of a later layer.

## Progress (Layers 1–2 — CPU correctness oracle)
Implemented `sources/algo/kheavyhash/` (standalone, no Boost/CUDA/OpenCL):
`xoshiro.hpp` (xoshiro256++), `matrix.{hpp,cpp}` (gen + f64 rank), `keccak.{hpp,cpp}`
(f[1600]), `hashers.{hpp,cpp}` (PowHash/KHeavyHash via verbatim cSHAKE256 states),
`kheavyhash.{hpp,cpp}` (heavy step + `calculatePow` + `meetsTarget`).

**8/8 known-answer tests pass** (`sources/algo/kheavyhash/tests/`), covering:
matrix `compute_rank` (64 / rank-deficient 63), `generateMatrix` == rust
`expected_matrix`, `PowHash`, `KHeavyHash`, `heavy_hash` == rust `expected_hash`,
full `calculatePow`, and the LE target compare.

**Why the oracle is trustworthy** (`dev/oracle.py`): it is an independent
re-implementation that, before emitting any vector, asserts it reproduces
rusty-kaspa's own literal vectors (matrix-gen, heavy_hash, rank) **and** that the
precomputed keccak initial-state constants equal pycryptodome's independent
cSHAKE256. So the C++ is checked against rust literals where they exist, and
against a second independent cSHAKE impl where they don't (PowHash, full pipeline).

Build/run with no native toolchain: see `dev/README.md` (Docker + ctest).

> ⚠️ Correctness oracle = the rusty-kaspa source quoted below + the numeric
> known-answer vectors in §3. Do NOT trust prose paraphrase over the quoted
> constants. Copy the 25-lane keccak constant arrays **verbatim** and validate
> with the KATs before trusting anything.

---

## 0. Corrections to the original brief

| Brief said | Reality (per reference) | Impact |
|---|---|---|
| matrix built via `xoShiRo256**` | It is **xoshiro256++** (`XoShiRo256PlusPlus`), output `rotl(s0+s3, 23) + s0` | Different generator — must use ++ |
| "keccak/cSHAKE — confirm" | Both hashes are **cSHAKE256** with precomputed initial states; inner permutation is standard **keccak-f[1600]** (24 rounds) | Reuse repo's `keccak_f1600` |
| matrix is "4-bit values 0..15" | ✅ correct (nibbles); stored as `u16` per cell | — |
| compare hash2 `<= target` | ✅ correct, as **little-endian** 256-bit int | — |

Crescendo note: the file `crypto/hashes/src/pow_hashers.rs` and
`consensus/pow/src/matrix.rs` on current master are the live mainnet PoW.
Crescendo changed block cadence / DAA, **not** the kHeavyHash inner loop.
Residual uncertainty: **low**. Re-verify by diffing those two files across the
Crescendo release tag if paranoid.

---

## 1. The exact algorithm (current Kaspa mainnet PoW)

### 1.1 Inputs the miner gets from a pool (per job)
- `pre_pow_hash` — 32 bytes. (Pool sends it as **4 little-endian u64 words**.)
- `timestamp` — u64 (the block header timestamp).
- `target` — derived from stratum difficulty (see §4.3).
- `nonce` — u64, the search space.

`pre_pow_hash` itself = `blake2b-256(key="BlockHash", serialized header with
timestamp=0 AND nonce=0)`. The **miner does not build this** when pool-mining —
the pool provides it. (Only relevant if we ever solo-mine; see hasher.go
`SerializeBlockHeader`.)

### 1.2 Per-job host-side setup
1. Reconstruct `pre_pow_hash` 32 bytes from the 4 LE u64 words.
2. **Generate the 64×64 matrix** from `pre_pow_hash` (§1.3), regenerating until
   `rank == 64`.
3. Build the `PowHash` keccak state (§1.4) — pre-absorbs
   `pre_pow_hash || timestamp`. (Optional micro-opt; the naive path can just
   recompute from constants each nonce.)
4. Upload matrix + (pre_pow_hash, timestamp) + target to the device.

### 1.3 Matrix generation — `matrix.rs`

xoshiro256++ (`xoshiro.rs`), seeded from the hash's 4 LE u64 words:
```
s0,s1,s2,s3 = pre_pow_hash.to_le_u64()   // little-endian 8-byte words
u64():
    res = s0 + rotl(s0 + s3, 23)         // wrapping add; rotl = rotate_left
    t   = s1 << 17
    s2 ^= s0;  s3 ^= s1;  s1 ^= s2;  s0 ^= s3
    s2 ^= t
    s3  = rotl(s3, 45)
    return res
```

Candidate matrix fill (`rand_matrix_no_rank_check`), row-major:
```
for i in 0..64:                 // rows
    for j in 0..64:             // cols
        if j % 16 == 0: val = generator.u64()   // one u64 feeds 16 nibbles
        matrix[i][j] = (val >> (4 * (j % 16))) & 0x0F    // u16 cell, 0..15
```
=> 4 u64 draws per row, 256 draws per candidate.

Full-rank gate (`compute_rank`): convert cells to f64, Gaussian elimination,
`EPS = 1e-9`, accept iff `rank == 64`. **On failure, redraw the next candidate
continuing the SAME xoshiro stream** (the generator is NOT reseeded). Loop until
rank 64.

> Rank check uses f64 with EPS=1e-9. For the host (CPU reference) replicate
> exactly. The GPU never does rank checks — matrix arrives full-rank from host.

### 1.4 hash1 — `PowHash` (cSHAKE256 "ProofOfWorkHash") — `pow_hashers.rs`

Message conceptually absorbed by `cSHAKE256(N="", S="ProofOfWorkHash")`:
```
pre_pow_hash[32] || timestamp_u64_LE[8] || zero[32] || nonce_u64_LE[8]   // 80 bytes
```
Implemented with a **precomputed 25-lane state** (domain + cSHAKE trailing pad
already baked in). Lane layout (rate=136B=17 lanes; message=80B=10 lanes):
```
state = POW_INITIAL_STATE            // copy verbatim, see §2
state[0..=3] ^= pre_pow_hash words   // 4 LE u64 = 32 bytes
state[4]     ^= timestamp
// state[5..=8] = the 32 zero-padding bytes -> unchanged
state[9]     ^= nonce
keccak_f1600(state)                  // standard 24-round f[1600]
hash1 = state[0..=3] as little-endian bytes   // 32 bytes
```

### 1.5 heavy step + hash2 — `matrix.heavy_hash` + `KHeavyHash`

```
// 1. expand hash1 (32 bytes) to 64 nibbles, HIGH nibble first:
for i in 0..32:
    vec[2*i]   = hash1[i] >> 4
    vec[2*i+1] = hash1[i] & 0x0F

// 2. matrix-vector multiply, two rows -> one output byte:
for i in 0..32:
    sum1 = Σ_{j=0..63} matrix[2*i]  [j] * vec[j]   // fits u16 (max 14400)
    sum2 = Σ_{j=0..63} matrix[2*i+1][j] * vec[j]
    product[i] = ((sum1 >> 10) << 4) | (sum2 >> 10)   // each >>10 is 0..14

// 3. XOR with the ORIGINAL hash1 bytes:
for i in 0..32: product[i] ^= hash1[i]

// 4. hash2 = cSHAKE256("HeavyHash") over the 32 product bytes:
state = HEAVY_INITIAL_STATE          // copy verbatim, see §2
state[0..=3] ^= product words (4 LE u64 = 32 bytes)
keccak_f1600(state)
hash2 = state[0..=3] as little-endian bytes   // 32 bytes
```

### 1.6 Accept test — `lib.rs`
```
pow_value = uint256_from_le_bytes(hash2)
hit iff pow_value <= target
```

---

## 2. Constants to copy VERBATIM (from `pow_hashers.rs`)

`POW_INITIAL_STATE` = cSHAKE256("ProofOfWorkHash") with pad baked in:
```
1242148031264380989, 3008272977830772284, 2188519011337848018, 1992179434288343456, 8876506674959887717,
5399642050693751366, 1745875063082670864, 8605242046444978844, 17936695144567157056, 3343109343542796272,
1123092876221303306, 4963925045340115282, 17037383077651887893, 16629644495023626889, 12833675776649114147,
3784524041015224902, 1082795874807940378, 13952716920571277634, 13411128033953605860, 15060696040649351053,
9928834659948351306, 5237849264682708699, 12825353012139217522, 6706187291358897596, 196324915476054915,
```
`HEAVY_INITIAL_STATE` = cSHAKE256("HeavyHash") with pad baked in:
```
4239941492252378377, 8746723911537738262, 8796936657246353646, 1272090201925444760, 16654558671554924250,
8270816933120786537, 13907396207649043898, 6782861118970774626, 9239690602118867528, 11582319943599406348,
17596056728278508070, 15212962468105129023, 7812475424661425213, 3370482334374859748, 5690099369266491460,
8596393687355028144, 570094237299545110, 9119540418498120711, 16901969272480492857, 13372017233735502424,
14372891883993151831, 5171152063242093102, 10573107899694386186, 6096431547456407061, 1592359455985097269,
```
> These already include the cSHAKE domain absorb **and** the trailing pad
> (lane 10 ^= 0x04, lane 16 ^= 0x80<<56 for POW; lane 4 / lane 16 for HEAVY).
> Do not re-derive cSHAKE; just XOR the message lanes and permute once.

---

## 3. Known-answer vectors (extracted from the reference test suite)

### 3.1 cSHAKE equivalence (proves the keccak constants) — `pow_hashers.rs` tests
- `PowHash::new(pre_pow_hash=[42;32], timestamp=5435345234).finalize_with_nonce(432432432)`
  must equal `cSHAKE256(S="ProofOfWorkHash").update([42;32] || 5435345234_LE ||
  [0;32] || 432432432_LE)` read-32.
- `KHeavyHash::hash([42;32])` must equal `cSHAKE256(S="HeavyHash").update([42;32])`
  read-32.

### 3.2 heavy_hash full-step KAT — `matrix.rs` `test_heavy_hash`
Input hash (32 bytes):
```
82,46,212,218,28,192,143,92,213,66,86,63,245,241,155,189,
73,159,229,180,202,105,159,166,109,172,128,136,169,195,97,41
```
Matrix: the explicit 64×64 nibble matrix in `matrix.rs` lines 170–235 (copy from
source when building the test).
Expected output (32 bytes):
```
135,104,159,55,153,67,234,249,183,71,92,169,83,37,104,119,
114,191,204,104,252,120,153,202,235,68,9,236,69,144,195,37
```
This KAT exercises steps 1.5.1→1.5.4 (matrix multiply + XOR + KHeavyHash). It is
the single best oracle for the GPU-vs-CPU bit-identity test.

### 3.3 matrix-generation KAT — `matrix.rs` `test_generate_matrix`
`Matrix::generate` for a fixed seed → the explicit `expected_matrix`
(lines 245+). Use to validate §1.3 (xoshiro + fill + rank). NOTE: the test's
exact seed/use must be copied from source (the surrounding test body) — verify
when implementing.

> Action item before coding layer 2: pull `matrix.rs` test bodies verbatim into
> the unit test fixtures so vectors are exact, not transcribed.

---

## 4. Kaspa stratum protocol (de-facto: kaspa-stratum-bridge / gostratum)

JSON-RPC 2.0, newline-delimited. Our base `Stratum::onMethod` already routes
`mining.notify`, `mining.set_difficulty`, `mining.set_extranonce` — clean reuse.

### 4.1 Handshake
- Client → `mining.subscribe` (with user-agent string). **Keep the agent free of
  "BzMiner"** — the bridge's `bigJobRegex = .*BzMiner.*` switches job encoding
  based on it; we want the NORMAL encoding.
- Client → `mining.authorize` params `["wallet.worker", "x"]`.
- Server → `mining.set_difficulty` `[diffValue]` (float; sent once, vardiff may
  resend).
- Server → `mining.notify`.

### 4.2 `mining.notify` (NORMAL encoding) — `client_handler.go` + `hasher.go`
```
{ "method":"mining.notify", "id": <jobId int>,
  "params": [ "<jobId decimal string>",
              [ u64_0, u64_1, u64_2, u64_3 ],   // pre_pow_hash as 4 LE u64 words
              <timestamp u64> ] }
```
`GenerateJobHeader`: takes the 32-byte pre_pow_hash, reads 4 little-endian u64 →
the array. To rebuild the 32-byte hash: write each u64 back little-endian into
bytes[0:8],[8:16],[16:24],[24:32].

(BIG-job variant, BzMiner only: params = [jobIdStr, "<80 hex chars>"] where the
80 hex = 4×(u64 big-endian) + timestamp big-endian. We avoid this path.)

### 4.3 `mining.set_difficulty` → target — `hasher.go`
```
maxTarget = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF   // 56 hex F = 2^224 - 1
target    = floor(maxTarget / diffValue)        // big.Float division, truncated
hit iff pow_value(LE 256-bit) <= target
```
> ⚠ Copy the `maxTarget` literal verbatim and confirm it is 56 F's. Different
> POOLS may diverge here (some send target directly, some use a different diff1
> or apply a 2^32 scaling). **Confirm against the actual pool chosen for testing**
> — this is the most pool-specific value and a common cause of "all shares
> rejected as low-diff".

### 4.4 `mining.submit` — `share_handler.go`
```
{ "method":"mining.submit",
  "params": [ "wallet.worker", "<jobId decimal string>", "<nonce hex>" ] }
```
- nonce hex may carry `0x`; parsed as u64.
- If the pool assigned an **extranonce** (via `mining.set_extranonce`), the high
  bytes of the nonce are fixed by it: submitted nonce string is
  `extranonce_hex + (16 - len) padded remainder`. With extranonce length E bytes,
  the miner only searches the low `8 - E` bytes. Plan: support extranonce; start
  with E=0 (full 64-bit search) on a pool/testnet that allows it.

---

## 5. Repo integration seams (mapped from BLAKE3 — the only existing non-DAG algo)

End-to-end wiring of an algorithm (BLAKE3 is the cleanest template; it has no
multi-GB DAG, though it still calls `updateMemory`):

| Seam | File | What to add for KHEAVYHASH |
|---|---|---|
| Algo enum | `sources/algo/algo_type.{hpp,cpp}` | add `KHEAVYHASH`; `toString`/`toEnum` |
| Job fields | `sources/stratum/job_info.hpp` | `prePowHash` (hash256), `timestamp` (u64), `matrix` buffer (64×64 u16), reuse `boundary`/target |
| Stratum subclass | `sources/stratum/kheavyhash.{hpp,cpp}` | `onMiningNotify` (parse [jobIdStr,[4 u64],ts]), `onMiningSetDifficulty` (maxTarget/diff), `miningSubmit` ([worker,jobId,nonceHex]), `onResponse`, `miningSubscribe`/`miningAuthorize` reuse base |
| Stratum factory | `sources/stratum/stratums.cpp` `NewStratum` | add `case KHEAVYHASH -> NEW(StratumKHeavyHash)` |
| Resolver (kernel host) | `sources/resolver/amd/kheavyhash.{hpp,cpp}` (+ `_kernel_parameter.hpp`) | `updateMemory` = generate+upload matrix; `updateConstants` = upload pre_pow_hash+timestamp+target; `executeSync/Async` = launch; `submit` = push `[worker,jobId,nonceHex]` |
| OpenCL kernel | `sources/algo/kheavyhash/opencl/*.cl` | xoshiro not needed on GPU (matrix from host); kernel = PowHash → heavy multiply → KHeavyHash → compare. Reuse `algo/crypto/opencl/keccak_f1600.cl` |
| CPU reference | `sources/resolver/cpu/kheavyhash.{hpp,cpp}` or a `sources/algo/kheavyhash/` host lib | full pipeline = correctness oracle |
| Device wiring | `sources/device/device.cpp` `setAlgorithm` | add `case KHEAVYHASH` constructing AMD/NVIDIA resolver |
| Tests | `sources/algo/.../tests/`, `sources/resolver/amd/tests/`, `sources/stratum/tests/` | KAT (§3), kernel==CPU, stratum parse |

Resolver lifecycle (from `device.cpp` loop, seen in BLAKE3 resolver):
`updateMemory(job)` → `updateConstants(job)` → repeated `executeSync/executeAsync(job)` → `submit(stratum)`.
- `updateMemory` is the natural home for **per-job matrix generation** (not a DAG,
  just a 64×64×u16 = 8 KiB upload).
- Result path: kernel writes found nonces into a `Result` cache (like
  `algo::blake3::Result`); `submit` formats and calls `stratum->miningSubmit`.
- Staleness: `isStale(jobId)` guard already exists.

Keccak reuse: `sources/algo/crypto/opencl/keccak_f1600.cl` and
`.../cuda/keccak_f1600.cuh` already exist (used by ethash/progpow). Confirm it is
standard 24-round f[1600] on a `u64[25]` lane array, then reuse for both hashes.

### NOT reused (per architecture note): DAG generation, epoch handling, light-cache,
multi-GB buffers. `job_info.epoch` may be left at a dummy value (BLAKE3 does this).

---

## 6. Open questions to resolve before/within the plan
1. **GPU backend priority** — brief says OpenCL for the bit-identity layer; repo's
   newest algo (BLAKE3) is CUDA-only. Which first? (see plan).
2. **Test pool / testnet** — which Kaspa pool + testnet endpoint, and does it use
   the §4.3 target formula or send target directly?
3. **Extranonce** — start at E=0 if the pool allows; else implement E>0 search-space
   slicing.
4. **matrix.rs test_generate_matrix seed** — copy verbatim from source for the KAT.
