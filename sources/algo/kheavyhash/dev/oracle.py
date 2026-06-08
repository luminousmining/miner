#!/usr/bin/env python3
"""
kHeavyHash correctness oracle.

Independently re-implements Kaspa's kHeavyHash pipeline (xoshiro256++ matrix gen,
PowHash/KHeavyHash via cSHAKE256, heavy multiply) and:

  1. Parses the rusty-kaspa reference files (ref_*.rs) for their LITERAL known-answer
     vectors and asserts this implementation reproduces them exactly.
  2. Cross-checks the precomputed-keccak-state approach against pycryptodome's
     independent cSHAKE256 (the same primitive rusty-kaspa's own tests assert against).
  3. Emits ../tests/kheavyhash_test_vectors.hpp with all vectors as the C++ KAT data,
     INCLUDING PowHash (hash1) and full-pipeline vectors that the rust suite only
     covers transitively.

If any self-check fails, the script aborts non-zero and emits nothing.
"""
import re
import sys
from pathlib import Path

try:
    from Crypto.Hash import cSHAKE256  # pycryptodome (Crypto namespace)
except ModuleNotFoundError:
    from Cryptodome.Hash import cSHAKE256  # debian python3-pycryptodome (Cryptodome namespace)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "kheavyhash_test_vectors.hpp"

POW_DOMAIN = b"ProofOfWorkHash"
HEAVY_DOMAIN = b"HeavyHash"
MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------- parse refs
def read(name):
    return (HERE / name).read_text()


def ints(s):
    return [int(x) for x in re.findall(r"-?\d+", s)]


def parse_initial_states(src):
    # two `const INITIAL_STATE: [u64; 25] = [ ... ];` blocks: POW first, HEAVY second
    blocks = re.findall(r"const INITIAL_STATE:\s*\[u64;\s*25\]\s*=\s*\[(.*?)\];", src, re.S)
    assert len(blocks) == 2, f"expected 2 INITIAL_STATE blocks, got {len(blocks)}"
    pow_state = ints(blocks[0])
    heavy_state = ints(blocks[1])
    assert len(pow_state) == 25 and len(heavy_state) == 25
    return pow_state, heavy_state


def parse_matrix(src, name):
    m = re.search(rf"let {name}\s*=\s*Matrix\(\[(.*?)\]\);", src, re.S)
    assert m, f"matrix {name} not found"
    vals = ints(m.group(1))
    assert len(vals) == 64 * 64, f"{name}: expected 4096 ints, got {len(vals)}"
    return [vals[i * 64:(i + 1) * 64] for i in range(64)]


def parse_hash(src, label):
    m = re.search(rf"let {label}\s*=\s*Hash::from_bytes\(\[(.*?)\]\);", src, re.S)
    assert m, f"hash {label} not found"
    vals = ints(m.group(1))
    assert len(vals) == 32, f"{label}: expected 32 bytes, got {len(vals)}"
    return bytes(vals)


# ----------------------------------------------------------------------------- algorithm
class Xoshiro:
    def __init__(self, seed32):
        self.s = [int.from_bytes(seed32[i * 8:i * 8 + 8], "little") for i in range(4)]

    @staticmethod
    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & MASK64

    def next(self):
        s = self.s
        res = (s[0] + self.rotl((s[0] + s[3]) & MASK64, 23)) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self.rotl(s[3], 45)
        return res


def rand_matrix(gen):
    mat = []
    for _ in range(64):
        row = []
        val = 0
        for j in range(64):
            if j % 16 == 0:
                val = gen.next()
            row.append((val >> (4 * (j % 16))) & 0x0F)
        mat.append(row)
    return mat


def compute_rank(mat):
    EPS = 1e-9
    m = [[float(x) for x in row] for row in mat]
    rank = 0
    row_selected = [False] * 64
    for i in range(64):
        j = 0
        while j < 64:
            if not row_selected[j] and abs(m[j][i]) > EPS:
                break
            j += 1
        if j != 64:
            rank += 1
            row_selected[j] = True
            for p in range(i + 1, 64):
                m[j][p] /= m[j][i]
            for k in range(64):
                if k != j and abs(m[k][i]) > EPS:
                    for p in range(i + 1, 64):
                        m[k][p] -= m[j][p] * m[k][i]
    return rank


def generate_matrix(seed32):
    gen = Xoshiro(seed32)
    while True:
        mat = rand_matrix(gen)
        if compute_rank(mat) == 64:
            return mat


def cshake256(domain, data):
    h = cSHAKE256.new(data=data, custom=domain)
    return h.read(32)


def pow_hash(pre_pow_hash, timestamp, nonce):
    msg = (pre_pow_hash + timestamp.to_bytes(8, "little")
           + b"\x00" * 32 + nonce.to_bytes(8, "little"))
    return cshake256(POW_DOMAIN, msg)


def kheavy_hash(data32):
    return cshake256(HEAVY_DOMAIN, data32)


def heavy_hash(mat, hash1):
    vec = []
    for b in hash1:
        vec.append(b >> 4)
        vec.append(b & 0x0F)
    product = bytearray(32)
    for i in range(32):
        sum1 = sum(mat[2 * i][j] * vec[j] for j in range(64))
        sum2 = sum(mat[2 * i + 1][j] * vec[j] for j in range(64))
        product[i] = (((sum1 >> 10) << 4) | (sum2 >> 10)) & 0xFF
    for i in range(32):
        product[i] ^= hash1[i]
    return kheavy_hash(bytes(product))


def calculate_pow(pre_pow_hash, timestamp, nonce):
    mat = generate_matrix(pre_pow_hash)
    hash1 = pow_hash(pre_pow_hash, timestamp, nonce)
    return heavy_hash(mat, hash1)


# ----------------------------------------------------------------------------- self-checks
def main():
    pow_src = read("ref_pow_hashers.rs")
    mat_src = read("ref_matrix.rs")

    pow_state, heavy_state = parse_initial_states(pow_src)
    ref_test_matrix = parse_matrix(mat_src, "test_matrix")
    ref_expected_matrix = parse_matrix(mat_src, "expected_matrix")
    ref_heavy_expected = parse_hash(mat_src, "expected_hash")
    ref_heavy_input = parse_hash(mat_src, "hash")  # the 82,46,... input in test_heavy_hash

    checks = []

    # (A) matrix generation: generate([42;32]) == rust expected_matrix
    seed42 = bytes([42] * 32)
    gen_matrix = generate_matrix(seed42)
    assert gen_matrix == ref_expected_matrix, "matrix generation mismatch vs rust literal"
    checks.append("matrix.generate([42;32]) == rust expected_matrix")

    # (B) rank: full-rank matrix == 64, a rank-deficient copy < 64
    assert compute_rank(ref_expected_matrix) == 64
    deficient = [row[:] for row in ref_expected_matrix]
    deficient[0] = deficient[1][:]
    assert compute_rank(deficient) == 63
    checks.append("compute_rank: full==64, duplicated-row==63")

    # (C) heavy_hash: rust test_matrix + input == rust expected_hash
    assert heavy_hash(ref_test_matrix, ref_heavy_input) == ref_heavy_expected, "heavy_hash KAT mismatch"
    checks.append("heavy_hash(test_matrix, input) == rust expected_hash")

    # (D) precomputed keccak constants vs pycryptodome cSHAKE256 (independent)
    #     rebuild cSHAKE from the parsed INITIAL_STATE arrays and confirm equality.
    assert keccak_pow(pow_state, ref_heavy_input, 5435345234, 432432432) \
        == pow_hash(ref_heavy_input, 5435345234, 432432432), "POW_INITIAL_STATE mismatch"
    assert keccak_heavy(heavy_state, ref_heavy_input) == kheavy_hash(ref_heavy_input), "HEAVY_INITIAL_STATE mismatch"
    checks.append("precomputed INITIAL_STATE arrays == pycryptodome cSHAKE256")

    # ----- mint vectors the rust suite lacks as literals -----
    # PowHash KAT, matching rust test_pow_hash inputs exactly:
    pow_kat_pre = bytes([42] * 32)
    pow_kat_ts = 5435345234
    pow_kat_nonce = 432432432
    pow_kat_out = pow_hash(pow_kat_pre, pow_kat_ts, pow_kat_nonce)

    # Full-pipeline KAT: generate matrix from pre_pow, run pipeline.
    fp_pre = bytes(range(32))  # 0,1,2,...,31
    fp_ts = 0x0011223344556677
    fp_nonce = 0x89ABCDEF01234567
    fp_matrix = generate_matrix(fp_pre)
    fp_hash1 = pow_hash(fp_pre, fp_ts, fp_nonce)
    fp_out = heavy_hash(fp_matrix, fp_hash1)
    fp_value = int.from_bytes(fp_out, "little")  # Uint256 LE
    # a loose target the value passes, and a tight target it fails, to exercise compare:
    target_pass = fp_value
    target_fail = fp_value - 1

    # standalone KHeavyHash KAT (cSHAKE256 "HeavyHash" over 32 bytes), no matrix step:
    kheavy_out = kheavy_hash(ref_heavy_input)

    emit(pow_state, heavy_state, seed42, gen_matrix,
         ref_test_matrix, ref_heavy_input, ref_heavy_expected, kheavy_out,
         pow_kat_pre, pow_kat_ts, pow_kat_nonce, pow_kat_out,
         fp_pre, fp_ts, fp_nonce, fp_hash1, fp_out, fp_value, target_pass, target_fail)

    print("ORACLE SELF-CHECKS PASSED:")
    for c in checks:
        print("  [ok]", c)
    print(f"emitted {OUT}")


# precomputed-state keccak path (mirrors what the C++ will do), to validate constants
def keccak_f1600(state):
    RHO = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
    PI = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
    RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
          0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
          0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
          0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
          0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
          0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
    a = state
    for rnd in range(24):
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x + 4) % 5] ^ Xoshiro.rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(0, 25, 5):
                a[x + y] ^= d[x]
        cur = a[1]
        for i in range(24):
            j = PI[i]
            tmp = a[j]
            a[j] = Xoshiro.rotl(cur, RHO[i])
            cur = tmp
        for y in range(0, 25, 5):
            t = [a[y + x] for x in range(5)]
            for x in range(5):
                a[y + x] = t[x] ^ ((~t[(x + 1) % 5] & MASK64) & t[(x + 2) % 5])
        a[0] ^= RC[rnd]
    return a


def keccak_pow(pow_state, pre, ts, nonce):
    s = list(pow_state)
    words = [int.from_bytes(pre[i * 8:i * 8 + 8], "little") for i in range(4)]
    for i in range(4):
        s[i] ^= words[i]
    s[4] ^= ts
    s[9] ^= nonce
    s = keccak_f1600(s)
    return b"".join(s[i].to_bytes(8, "little") for i in range(4))


def keccak_heavy(heavy_state, data32):
    s = list(heavy_state)
    words = [int.from_bytes(data32[i * 8:i * 8 + 8], "little") for i in range(4)]
    for i in range(4):
        s[i] ^= words[i]
    s = keccak_f1600(s)
    return b"".join(s[i].to_bytes(8, "little") for i in range(4))


# ----------------------------------------------------------------------------- emit header
def u64arr(name, vals):
    body = ",\n    ".join(", ".join(f"0x{v:016x}ULL" for v in vals[i:i + 4]) for i in range(0, len(vals), 4))
    return f"inline constexpr std::array<uint64_t, {len(vals)}> {name}{{{{\n    {body}\n}}}};\n"


def bytearr(name, b):
    body = ", ".join(f"0x{x:02x}" for x in b)
    return f"inline constexpr std::array<uint8_t, {len(b)}> {name}{{{{ {body} }}}};\n"


def matrixarr(name, mat):
    rows = ",\n    ".join("{" + ", ".join(str(v) for v in row) + "}" for row in mat)
    return f"inline constexpr std::array<std::array<uint16_t, 64>, 64> {name}{{{{\n    {rows}\n}}}};\n"


def emit(pow_state, heavy_state, seed42, gen_matrix, test_matrix, heavy_in, heavy_exp, kheavy_exp,
         pow_pre, pow_ts, pow_nonce, pow_out,
         fp_pre, fp_ts, fp_nonce, fp_hash1, fp_out, fp_value, target_pass, target_fail):
    def u256(name, v):
        # little-endian 32 bytes
        b = v.to_bytes(32, "little")
        return bytearr(name, b)

    parts = [
        "// GENERATED by sources/algo/kheavyhash/dev/oracle.py — DO NOT EDIT BY HAND.",
        "// Known-answer vectors for the kHeavyHash CPU reference.",
        "// Provenance: rusty-kaspa master (matrix.rs / pow_hashers.rs) + pycryptodome cSHAKE256.",
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace kheavyhash::kat",
        "{",
        u64arr("POW_INITIAL_STATE", pow_state),
        u64arr("HEAVY_INITIAL_STATE", heavy_state),
        bytearr("GEN_SEED", seed42),
        matrixarr("GEN_EXPECTED_MATRIX", gen_matrix),
        matrixarr("HEAVY_TEST_MATRIX", test_matrix),
        bytearr("HEAVY_INPUT", heavy_in),
        bytearr("HEAVY_EXPECTED", heavy_exp),
        bytearr("KHEAVY_EXPECTED", kheavy_exp),
        f"inline constexpr uint8_t POW_KAT_PRE[32]{{ {', '.join(f'0x{x:02x}' for x in pow_pre)} }};",
        f"inline constexpr uint64_t POW_KAT_TIMESTAMP{{ 0x{pow_ts:016x}ULL }};",
        f"inline constexpr uint64_t POW_KAT_NONCE{{ 0x{pow_nonce:016x}ULL }};",
        bytearr("POW_KAT_EXPECTED", pow_out),
        bytearr("FP_PRE", fp_pre),
        f"inline constexpr uint64_t FP_TIMESTAMP{{ 0x{fp_ts:016x}ULL }};",
        f"inline constexpr uint64_t FP_NONCE{{ 0x{fp_nonce:016x}ULL }};",
        bytearr("FP_HASH1", fp_hash1),
        bytearr("FP_FINAL", fp_out),
        u256("FP_TARGET_PASS", target_pass),
        u256("FP_TARGET_FAIL", target_fail),
        "}",
        "",
    ]
    OUT.write_text("\n".join(parts))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("ORACLE SELF-CHECK FAILED:", e, file=sys.stderr)
        sys.exit(1)
