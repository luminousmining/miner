#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
argon2d_test.py
===============
Comparative tests between:
  - argon2-cffi (reference C implementation, official library)
  - argon2d.py  (from-scratch RFC 9106 implementation)

For each test vector we compare:
  * argon2d  (mode 0: data-dependent access)
  * argon2i  (mode 1: data-independent access)
  * argon2id (mode 2: hybrid)

Prerequisites:
    pip install argon2-cffi

Expected output: PASS for each case if both implementations
produce exactly the same bytes.
"""

import sys
import io
import os

# -- UTF-8 encoding on Windows console ------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- imports ----------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from argon2d import argon2d, argon2i, argon2id

try:
    from argon2.low_level import hash_secret_raw, Type as Argon2Type
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False
    print("[WARN] argon2-cffi not installed. Install via: pip install argon2-cffi")
    print("       Only hardcoded RFC vectors will be tested.")
    print()


# =========================================================================
# Test vectors
# =========================================================================
#
# Structure: (label, password, salt, time_cost, memory_cost, parallelism,
#              hash_len)
# Note: argon2-cffi does not expose the K (secret) and X (assoc_data) parameters
#       in its Python API. Comparative tests therefore use K=b'' and X=b''.
#       RFC vectors (with K and X) are tested separately.
#
TEST_VECTORS = [
    # --- minimal parameters ---
    ("Minimal (t=1, m=8, p=1, 32B)",
     b"password", b"somesalt", 1,  8, 1, 32),

    ("Minimal (t=1, m=8, p=1,  8B)",
     b"password", b"somesalt", 1,  8, 1,  8),

    # --- empty salt forbidden (8 bytes minimum) ---
    # (only salts >= 8 bytes are used)

    # --- password variations ---
    ("Empty pwd, salt 8B",
     b"", b"saltsalt", 1, 8, 1, 32),

    ("Pwd 1 byte 0x00",
     b"\x00", b"saltsalt", 1, 8, 1, 32),

    ("Pwd 32 bytes 0x01",
     bytes([0x01]*32), b"saltsalt", 1, 8, 1, 32),

    ("Pwd 64 bytes",
     bytes(range(64)), b"saltsalt", 1, 8, 1, 32),

    ("Long ASCII pwd",
     b"The quick brown fox jumps over the lazy dog", b"saltsalt", 1, 8, 1, 32),

    # --- salt variations ---
    ("Salt exactly 8 bytes",
     b"password", b"12345678", 1, 8, 1, 32),

    ("Salt 16 bytes",
     b"password", bytes([0x02]*16), 1, 8, 1, 32),

    ("Salt 32 bytes",
     b"password", bytes(range(32)), 1, 8, 1, 32),

    # --- time variations (t) ---
    ("t=1, m=16, p=1",
     b"password", b"saltsalt", 1, 16, 1, 32),

    ("t=2, m=16, p=1",
     b"password", b"saltsalt", 2, 16, 1, 32),

    ("t=3, m=16, p=1",
     b"password", b"saltsalt", 3, 16, 1, 32),

    # --- memory variations (m) ---
    ("t=1, m=8, p=1",
     b"password", b"saltsalt", 1,  8, 1, 32),

    ("t=1, m=16, p=1",
     b"password", b"saltsalt", 1, 16, 1, 32),

    ("t=1, m=32, p=1",
     b"password", b"saltsalt", 1, 32, 1, 32),

    # --- parallelism variations (p) ---
    ("t=1, m=16, p=2",
     b"password", b"saltsalt", 1, 16, 2, 32),

    ("t=1, m=32, p=4",
     b"password", b"saltsalt", 1, 32, 4, 32),

    # --- tag length variations ---
    ("hash_len=4",
     b"password", b"saltsalt", 1, 8, 1,  4),

    ("hash_len=8",
     b"password", b"saltsalt", 1, 8, 1,  8),

    ("hash_len=16",
     b"password", b"saltsalt", 1, 8, 1, 16),

    ("hash_len=32",
     b"password", b"saltsalt", 1, 8, 1, 32),

    ("hash_len=64",
     b"password", b"saltsalt", 1, 8, 1, 64),

    # --- RFC 9106 Annex B parameters (without K and X) ---
    ("RFC-like t=3, m=32, p=4, 32B",
     bytes([0x01]*32), bytes([0x02]*16), 3, 32, 4, 32),

    # --- binary data ---
    ("Pwd 0xFF*16, salt 0xAA*8",
     bytes([0xFF]*16), bytes([0xAA]*8), 1, 8, 1, 32),

    ("Pwd null*8, salt null*8",
     b"\x00"*8, b"\x00"*8, 1, 8, 1, 32),
]

# =========================================================================
# RFC 9106 vectors (hardcoded, with K and X)
# These vectors cannot be verified via argon2-cffi (limited API)
# but are verified against RFC 9106 specification Annexes B.1 and B.3
# =========================================================================
RFC_VECTORS = [
    {
        "label"      : "RFC 9106 Annex B.1 - Argon2d",
        "type"       : "d",
        "fn"         : None,   # filled later
        "password"   : bytes([0x01]*32),
        "salt"       : bytes([0x02]*16),
        "secret"     : bytes([0x03]*8),
        "assoc_data" : bytes([0x04]*12),
        "time_cost"  : 3,
        "memory_cost": 32,
        "parallelism": 4,
        "hash_len"   : 32,
        "expected"   : bytes.fromhex(
            "512b391b6f1162975371d30919734294"
            "f868e3be3984f3c1a13a4db9fabe4acb"
        ),
    },
    {
        "label"      : "RFC 9106 Annex B.3 - Argon2id",
        "type"       : "id",
        "fn"         : None,
        "password"   : bytes([0x01]*32),
        "salt"       : bytes([0x02]*16),
        "secret"     : bytes([0x03]*8),
        "assoc_data" : bytes([0x04]*12),
        "time_cost"  : 3,
        "memory_cost": 32,
        "parallelism": 4,
        "hash_len"   : 32,
        "expected"   : bytes.fromhex(
            "0d640df58d78766c08c037a34a8b53c9"
            "d01ef0452d75b65eb52520e96b01e659"
        ),
    },
]


# =========================================================================
# Reference functions (argon2-cffi)
# =========================================================================

def _cffi_hash(pwd, salt, t, m, p, hl, argon2_type):
    """Calls argon2-cffi to obtain the reference value."""
    return hash_secret_raw(
        secret=pwd, salt=salt,
        time_cost=t, memory_cost=m, parallelism=p,
        hash_len=hl, type=argon2_type
    )


# =========================================================================
# Test execution
# =========================================================================

def run_cffi_comparison() -> tuple:
    """Compares our implementation against argon2-cffi for all vectors."""
    if not HAS_CFFI:
        return 0, 0, []

    print("=" * 72)
    print("  PART 1: Comparison against argon2-cffi (official C reference)")
    print("=" * 72)
    print()

    cffi_types = [
        ("Argon2d",  Argon2Type.D,  argon2d),
        ("Argon2i",  Argon2Type.I,  argon2i),
        ("Argon2id", Argon2Type.ID, argon2id),
    ]

    col_label = 40
    col_mode  = 8

    print(f"  {'Vector':<{col_label}}  {'Mode':<{col_mode}}  Result")
    print(f"  {'-'*col_label}  {'-'*col_mode}  ---------")

    total   = 0
    passed  = 0
    failures = []

    for label, pwd, salt, t, m, p, hl in TEST_VECTORS:
        for name, cffi_t, our_fn in cffi_types:
            total += 1
            try:
                ref = _cffi_hash(pwd, salt, t, m, p, hl, cffi_t)
                our = our_fn(pwd, salt, t, m, p, hl)
                ok  = (ref == our)
            except Exception as exc:
                ok  = False
                ref = None
                our = None
                failures.append((label, name, ref, our, str(exc)))

            status = "PASS" if ok else "FAIL"
            print(f"  {label:<{col_label}}  {name:<{col_mode}}  {status}")

            if ok:
                passed += 1
            else:
                if ref is not None:
                    failures.append((label, name, ref, our, None))

    return total, passed, failures


def run_rfc_vectors() -> tuple:
    """Tests hardcoded RFC 9106 vectors (with K and X)."""
    print()
    print("=" * 72)
    print("  PART 2: RFC 9106 Annex B vectors (with secret + assoc_data)")
    print("=" * 72)
    print()

    fn_map = {"d": argon2d, "i": argon2i, "id": argon2id}

    total    = 0
    passed   = 0
    failures = []

    for v in RFC_VECTORS:
        total += 1
        fn = fn_map[v["type"]]
        try:
            got = fn(
                v["password"], v["salt"],
                time_cost   = v["time_cost"],
                memory_cost = v["memory_cost"],
                parallelism = v["parallelism"],
                hash_len    = v["hash_len"],
                secret      = v["secret"],
                assoc_data  = v["assoc_data"],
            )
            ok = (got == v["expected"])
        except Exception as exc:
            ok  = False
            got = None
            failures.append((v["label"], v["expected"], None, str(exc)))

        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {v['label']}")
        print(f"        Expected: {v['expected'].hex()}")
        if got:
            print(f"        Got     : {got.hex()}")
        print()

        if ok:
            passed += 1
        else:
            if got is not None:
                failures.append((v["label"], v["expected"], got, None))

    return total, passed, failures


def print_failures(failures: list) -> None:
    if not failures:
        return
    print()
    print("  FAILURE DETAILS")
    print("  " + "-" * 68)
    for item in failures:
        if len(item) == 5:
            label, name, expected, got, exc = item
        else:
            label, expected, got, exc = item
            name = ""

        header = f"[FAIL] {label}" + (f"  ({name})" if name else "")
        print(f"\n  {header}")
        if exc:
            print(f"         Exception: {exc}")
        elif expected and got:
            print(f"         Expected : {expected.hex()}")
            print(f"         Got      : {got.hex()}")
            for i, (e, g) in enumerate(zip(expected, got)):
                if e != g:
                    print(f"         1st diff : byte[{i}]  expected=0x{e:02x}  got=0x{g:02x}")
                    break


def show_reference_values() -> None:
    """Displays some reference values for the three modes."""
    print("=" * 72)
    print("  argon2-cffi reference values for the three modes")
    print("=" * 72)
    print()

    if not HAS_CFFI:
        print("  (argon2-cffi not available)")
        return

    samples = [
        (b"password", b"saltsalt", 1, 8, 1, 32),
        (bytes([0x01]*32), bytes([0x02]*16), 3, 32, 4, 32),
    ]

    for pwd, salt, t, m, p, hl in samples:
        print(f"  pwd={pwd[:8]}... salt={salt[:8]}... t={t} m={m} p={p} hl={hl}")
        for name, cffi_t, our_fn in [
            ("Argon2d",  Argon2Type.D,  argon2d),
            ("Argon2i",  Argon2Type.I,  argon2i),
            ("Argon2id", Argon2Type.ID, argon2id),
        ]:
            ref = _cffi_hash(pwd, salt, t, m, p, hl, cffi_t)
            print(f"    {name}  : {ref.hex()}")
        print()


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    show_reference_values()

    t1, p1, f1 = run_cffi_comparison()
    t2, p2, f2 = run_rfc_vectors()

    all_failures = f1 + f2

    print_failures(all_failures)

    total  = t1 + t2
    passed = p1 + p2

    print()
    print("=" * 72)
    print(f"  Overall results: {passed}/{total} PASS  |  {total - passed} FAIL")
    print("=" * 72)

    if not all_failures:
        print()
        print("  All implementations are identical to the reference.")
    print()
