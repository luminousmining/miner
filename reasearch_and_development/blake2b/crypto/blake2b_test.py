#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blake2b_test.py
===============
Comparative tests between:
  - hashlib.blake2b  (reference C implementation, bundled with Python)
  - blake2b.py       (from-scratch implementation, RFC 7693)

For each test vector we compare:
  * hash256 (digest_size=32)
  * hash512 (digest_size=64)

Expected output: PASS for each case if both implementations
produce exactly the same bytes.
"""

import sys
import io
import os
import hashlib

# -- UTF-8 encoding on Windows console ------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    # Python < 3.7: fallback via stream replacement
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- import our from-scratch implementation -------------------------------
# Add the current directory to the path so we can import blake2b.py
sys.path.insert(0, os.path.dirname(__file__))
from blake2b import hash256, hash512, blake2b as blake2b_scratch


# =========================================================================
# Test vectors
# =========================================================================

# Each entry is a tuple (label, data, key)
# key=b'' means "no key" (sequential mode)
TEST_VECTORS = [
    # --- edge cases ---
    ("Empty message (no key)",              b"",                                          b""),
    ("1 byte 0x00",                         b"\x00",                                      b""),
    ("1 byte 0xFF",                         b"\xFF",                                      b""),
    ("1 byte 'a'",                          b"a",                                         b""),

    # --- short ---
    ("'abc'",                               b"abc",                                       b""),
    ("'hello world'",                       b"hello world",                               b""),
    ("'RandomX'",                           b"RandomX",                                   b""),
    ("Bytes 0x00..0x0F (16 bytes)",         bytes(range(16)),                             b""),
    ("Bytes 0x00..0x1F (32 bytes)",         bytes(range(32)),                             b""),
    ("Bytes 0x00..0x3F (64 bytes)",         bytes(range(64)),                             b""),

    # --- block boundary (128 bytes) ---
    ("127 bytes (block - 1)",               bytes(range(127)),                            b""),
    ("128 bytes (exactly 1 block)",         bytes(range(128)),                            b""),
    ("129 bytes (block + 1)",               bytes(range(128)) + b"\x80",                 b""),
    ("192 bytes (1.5 blocks)",              bytes(i % 256 for i in range(192)),           b""),
    ("256 bytes (2 blocks)",                bytes(i % 256 for i in range(256)),           b""),
    ("384 bytes (3 blocks)",                bytes(i % 256 for i in range(384)),           b""),

    # --- long ---
    ("1000 bytes",                          bytes(i % 256 for i in range(1000)),          b""),
    ("Long ASCII string",                   b"The quick brown fox jumps over the lazy dog", b""),
    ("Repeated string x 100",              b"RandomX" * 100,                             b""),

    # --- with key (keyed mode) ---
    ("1-byte key, empty msg",               b"",                                          b"\x42"),
    ("Key 'key', msg 'abc'",               b"abc",                                       b"key"),
    ("32-byte key, msg 'hello'",            b"hello",                                     bytes(range(32))),
    ("64-byte key, empty msg",              b"",                                          bytes(range(64))),
    ("64-byte key, msg 256 bytes",          bytes(i % 256 for i in range(256)),           bytes(range(64))),

    # --- binary data ---
    ("All bytes 0x00 (64 b.)",             b"\x00" * 64,                                b""),
    ("All bytes 0xFF (64 b.)",             b"\xFF" * 64,                                b""),
    ("Alternating pattern 0xAA/0x55",      bytes([0xAA, 0x55] * 64),                    b""),
]


# =========================================================================
# Utility functions
# =========================================================================

def ref256(data: bytes, key: bytes = b"") -> bytes:
    """Blake2b-256 via hashlib (reference)."""
    return hashlib.blake2b(data, digest_size=32, key=key).digest()


def ref512(data: bytes, key: bytes = b"") -> bytes:
    """Blake2b-512 via hashlib (reference)."""
    return hashlib.blake2b(data, digest_size=64, key=key).digest()


def our256(data: bytes, key: bytes = b"") -> bytes:
    """Blake2b-256 via our from-scratch implementation."""
    return blake2b_scratch(data, digest_size=32, key=key)


def our512(data: bytes, key: bytes = b"") -> bytes:
    """Blake2b-512 via our from-scratch implementation."""
    return blake2b_scratch(data, digest_size=64, key=key)


def hex_short(b: bytes) -> str:
    """Shows first 8 + last 8 bytes if > 16, otherwise all."""
    h = b.hex()
    if len(b) > 16:
        return h[:16] + "..." + h[-16:]
    return h


def run_tests() -> None:
    print("=" * 72)
    print("  BLAKE2b from-scratch vs hashlib  --  Comparative tests")
    print("=" * 72)
    print()

    total    = 0
    passed   = 0
    failed   = 0
    failures = []

    col_label = 42
    col_mode  = 8

    print(f"  {'Vector':<{col_label}}  {'Mode':<{col_mode}}  Result")
    print(f"  {'-'*col_label}  {'-'*col_mode}  ---------")

    for label, data, key in TEST_VECTORS:
        for mode, rfn, ofn in [
            ("256-bit", ref256, our256),
            ("512-bit", ref512, our512),
        ]:
            total += 1
            expected = rfn(data, key)
            try:
                got = ofn(data, key)
                ok  = (got == expected)
            except Exception as exc:
                ok  = False
                got = None
                failures.append((label, mode, expected, None, str(exc)))

            status = "PASS" if ok else "FAIL"
            short  = f"  {label:<{col_label}}  {mode:<{col_mode}}  {status}"
            print(short)

            if ok:
                passed += 1
            else:
                failed += 1
                if got is not None:
                    failures.append((label, mode, expected, got, None))

    # ----- summary -----
    print()
    print("=" * 72)
    print(f"  Results: {passed}/{total} PASS  |  {failed} FAIL")
    print("=" * 72)

    if failures:
        print()
        print("  FAILURE DETAILS")
        print("  " + "-" * 68)
        for label, mode, expected, got, exc in failures:
            print(f"\n  [FAIL] {label}  ({mode})")
            if exc:
                print(f"         Exception : {exc}")
            else:
                print(f"         Expected  : {expected.hex()}")
                print(f"         Got       : {got.hex() if got else 'None'}")
                # Find the first differing byte
                if got:
                    for i, (e, g) in enumerate(zip(expected, got)):
                        if e != g:
                            print(f"         1st diff  : byte[{i}]  expected=0x{e:02x}  got=0x{g:02x}")
                            break
    else:
        print()
        print("  All implementations are identical to hashlib.")

    print()


# =========================================================================
# Display reference values (informational)
# =========================================================================

def show_reference_values() -> None:
    """Displays some known Blake2b values for visual reference."""
    print("=" * 72)
    print("  Reference values (RFC 7693 / official blake2 tool)")
    print("=" * 72)
    print()

    samples = [
        (b"",      b""),
        (b"abc",   b""),
        (b"The quick brown fox jumps over the lazy dog", b""),
    ]

    for data, key in samples:
        label = repr(data.decode('ascii', errors='replace'))
        print(f"  Input : {label}")
        h256 = ref256(data, key)
        h512 = ref512(data, key)
        print(f"  256   : {h256.hex()}")
        print(f"  512   : {h512.hex()}")
        print()


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    show_reference_values()
    run_tests()
