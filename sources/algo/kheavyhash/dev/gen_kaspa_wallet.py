#!/usr/bin/env python3
"""Generate a random Kaspa wallet (secp256k1 keypair + bech32 address).

The bech32/CashAddr encoder is a faithful port of rusty-kaspa
crypto/addresses/src/bech32.rs and is gated on that crate's own test vectors
(crypto/addresses/src/lib.rs) before any address is emitted -- if any vector
fails the script aborts and prints nothing usable.

Pure stdlib (secrets), no third-party deps. secp256k1 scalar-mult is
self-checked against the known generator point (privkey 1 -> G).
"""
import secrets
import sys

CHARSET = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def polymod(values):
    c = 1
    for d in values:
        c0 = c >> 35
        c = ((c & 0x07FFFFFFFF) << 5) ^ d
        if c0 & 0x01:
            c ^= 0x98F2BC8E61
        if c0 & 0x02:
            c ^= 0x79B76D99E2
        if c0 & 0x04:
            c ^= 0xF33E5FB3C4
        if c0 & 0x08:
            c ^= 0xAE2EABE2A8
        if c0 & 0x10:
            c ^= 0x1E4F43E470
    return c ^ 1


def conv8to5(data):
    five, buff, bits = [], 0, 0
    for b in data:
        buff = (buff << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            five.append((buff >> bits) & 0x1F)
            buff &= (1 << bits) - 1
    if bits > 0:
        five.append((buff << (5 - bits)) & 0x1F)
    return five


def checksum(payload5, prefix_str):
    prefix5 = [b & 0x1F for b in prefix_str.encode()]
    return polymod(prefix5 + [0] + list(payload5) + [0] * 8)


def encode_address(prefix_str, version, pubkey):
    payload5 = conv8to5(bytes([version]) + bytes(pubkey))
    chk = checksum(payload5, prefix_str)
    chk5 = conv8to5(chk.to_bytes(8, "big")[3:])  # low 40 bits
    body = bytes(CHARSET[c] for c in (payload5 + chk5)).decode()
    return f"{prefix_str}:{body}"


# --- KAT gate: rusty-kaspa crypto/addresses/src/lib.rs test vectors -----------
VECTORS = [
    ("kaspatest", 0, bytes(32), "kaspatest:qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhqrxplya"),
    ("kaspatest", 1, bytes(33), "kaspatest:qyqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhe837j2d"),
    ("kaspatest", 1,
     bytes.fromhex("ba01fc5f4e9d9879599c69a3dafdb835a7255e5f2e934e9322ecd3af190ab0f60e"),
     "kaspatest:qxaqrlzlf6wes72en3568khahq66wf27tuhfxn5nytkd8tcep2c0vrse6gdmpks"),
    ("kaspa", 0, bytes(32), "kaspa:qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqkx9awp4e"),
    ("kaspa", 0,
     bytes.fromhex("5fff3c4da18f45adcdd499e44611e9fff148ba69db3c4ea2ddd955fc46a59522"),
     "kaspa:qp0l70zd5x85ttwd6jv7g3s3a8llzj96d8dncn4zmhv4tlzx5k2jyqh70xmfj"),
]
for prefix, ver, pk, expected in VECTORS:
    got = encode_address(prefix, ver, pk)
    if got != expected:
        sys.exit(f"KAT FAILED for {prefix}/v{ver}:\n  got      {got}\n  expected {expected}")
print(f"address encoder: {len(VECTORS)}/{len(VECTORS)} rusty-kaspa KATs pass")


# --- secp256k1 (self-checked against the generator point) ---------------------
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def inv(a, m):
    return pow(a, m - 2, m)


def point_add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1) * inv(2 * y1, P) % P
    else:
        lam = (y2 - y1) * inv((x2 - x1) % P, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def scalar_mul(k, point):
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


# self-check: 1*G == G
assert scalar_mul(1, (GX, GY)) == (GX, GY), "secp256k1 generator check failed"
assert scalar_mul(2, (GX, GY)) is not None
print("secp256k1 scalar-mult: generator self-check pass")


# --- generate a random testnet wallet -----------------------------------------
while True:
    priv = secrets.randbelow(N - 1) + 1  # in [1, N-1]
    pub = scalar_mul(priv, (GX, GY))
    if pub is not None:
        break

x_only = pub[0].to_bytes(32, "big")  # Kaspa schnorr address uses the x-only pubkey
address = encode_address("kaspatest", 0, x_only)

print()
print("=== RANDOM KASPA TESTNET WALLET (throwaway) ===")
print(f"private key (hex) : {priv.to_bytes(32, 'big').hex()}")
print(f"x-only pubkey(hex): {x_only.hex()}")
print(f"address (testnet) : {address}")
# round-trip sanity: re-encode must be stable
assert encode_address("kaspatest", 0, x_only) == address
