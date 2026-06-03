#include <algo/random_x/argon2d.hpp>

#include <cstring>
#include <cstdint>


namespace
{

////////////////////////////////////////////////////////////////////////////////
// Blake2b-512 (CPU)
////////////////////////////////////////////////////////////////////////////////

static constexpr uint64_t B2B_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

// Blake2b uses 12 rounds (rows 10-11 repeat rows 0-1)
static constexpr uint8_t B2B_SIGMA[12][16]
{
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
};

static inline uint64_t b2b_rotr64(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}

struct Blake2bState
{
    uint64_t h[8];
    uint64_t t[2];
    uint64_t f;
    uint8_t  buf[128];
    uint32_t buflen;
    uint32_t outlen;
};


static void b2b_compress(Blake2bState& s, uint8_t const* block)
{
    uint64_t v[16];
    uint64_t m[16];

    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        v[i]     = s.h[i];
        v[i + 8] = B2B_IV[i];
    }
    v[12] ^= s.t[0];
    v[13] ^= s.t[1];
    v[14] ^= s.f;

    for (uint32_t i{ 0u }; i < 16u; ++i)
    {
        uint8_t const* const p{ block + i * 8u };
        m[i] = (uint64_t)p[0]        | ((uint64_t)p[1] << 8u)  |
               ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
               ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
               ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
    }

#define B2B_G(r, i, a, b, c, d) \
    v[a] += v[b] + m[B2B_SIGMA[r][2*(i)]]; \
    v[d] = b2b_rotr64(v[d] ^ v[a], 32u); \
    v[c] += v[d]; \
    v[b] = b2b_rotr64(v[b] ^ v[c], 24u); \
    v[a] += v[b] + m[B2B_SIGMA[r][2*(i)+1]]; \
    v[d] = b2b_rotr64(v[d] ^ v[a], 16u); \
    v[c] += v[d]; \
    v[b] = b2b_rotr64(v[b] ^ v[c], 63u);

    for (uint32_t r{ 0u }; r < 12u; ++r)
    {
        B2B_G(r, 0,  0,  4,  8, 12)
        B2B_G(r, 1,  1,  5,  9, 13)
        B2B_G(r, 2,  2,  6, 10, 14)
        B2B_G(r, 3,  3,  7, 11, 15)
        B2B_G(r, 4,  0,  5, 10, 15)
        B2B_G(r, 5,  1,  6, 11, 12)
        B2B_G(r, 6,  2,  7,  8, 13)
        B2B_G(r, 7,  3,  4,  9, 14)
    }
#undef B2B_G

    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        s.h[i] ^= v[i] ^ v[i + 8u];
    }
}


static void b2b_init(Blake2bState& s, uint32_t const outlen)
{
    memset(&s, 0, sizeof(s));
    s.outlen = outlen;
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        s.h[i] = B2B_IV[i];
    }
    s.h[0] ^= 0x01010000ULL | (uint64_t)outlen;
}


static void b2b_update(Blake2bState& s, void const* const in, uint32_t const inlen)
{
    uint8_t const* p{ static_cast<uint8_t const*>(in) };
    uint32_t rem{ inlen };

    while (rem > 0u)
    {
        uint32_t const free{ 128u - s.buflen };
        uint32_t const fill{ rem < free ? rem : free };
        memcpy(s.buf + s.buflen, p, fill);
        s.buflen += fill;
        p        += fill;
        rem      -= fill;

        if (s.buflen == 128u && rem > 0u)
        {
            s.t[0] += 128u;
            if (s.t[0] < 128u) { ++s.t[1]; }
            b2b_compress(s, s.buf);
            s.buflen = 0u;
        }
    }
}


static void b2b_final(Blake2bState& s, void* const out)
{
    s.t[0] += s.buflen;
    if (s.t[0] < s.buflen) { ++s.t[1]; }
    s.f = 0xFFFFFFFFFFFFFFFFULL;
    memset(s.buf + s.buflen, 0, 128u - s.buflen);
    b2b_compress(s, s.buf);

    uint8_t* p{ static_cast<uint8_t*>(out) };
    for (uint32_t i{ 0u }; i < s.outlen; ++i)
    {
        p[i] = static_cast<uint8_t>(s.h[i / 8u] >> ((i % 8u) * 8u));
    }
}


static void blake2b(void* const out, uint32_t const outlen,
                    void const* const in, uint32_t const inlen)
{
    Blake2bState s;
    b2b_init(s, outlen);
    b2b_update(s, in, inlen);
    b2b_final(s, out);
}


////////////////////////////////////////////////////////////////////////////////
// H' — Blake2b variable-length hash (Argon2 H' function)
////////////////////////////////////////////////////////////////////////////////

static void blake2b_long(void* const out, uint32_t const outlen,
                         void const* const in, uint32_t const inlen)
{
    uint8_t* pout{ static_cast<uint8_t*>(out) };
    uint8_t const len_bytes[4]
    {
        static_cast<uint8_t>(outlen),
        static_cast<uint8_t>(outlen >> 8u),
        static_cast<uint8_t>(outlen >> 16u),
        static_cast<uint8_t>(outlen >> 24u)
    };

    if (outlen <= 64u)
    {
        Blake2bState s;
        b2b_init(s, outlen);
        b2b_update(s, len_bytes, 4u);
        b2b_update(s, in, inlen);
        b2b_final(s, pout);
        return;
    }

    // First block: full 64-byte hash
    uint8_t A[64];
    {
        Blake2bState s;
        b2b_init(s, 64u);
        b2b_update(s, len_bytes, 4u);
        b2b_update(s, in, inlen);
        b2b_final(s, A);
    }
    memcpy(pout, A, 32u);
    pout += 32u;

    uint32_t remaining{ outlen - 32u };

    // Middle blocks: each produces 32 bytes out of a 64-byte Blake2b hash
    while (remaining > 64u)
    {
        uint8_t B[64];
        blake2b(B, 64u, A, 64u);
        memcpy(pout, B, 32u);
        memcpy(A, B, 64u);
        pout      += 32u;
        remaining -= 32u;
    }

    // Last block: produces exactly `remaining` bytes
    blake2b(pout, remaining, A, 64u);
}


////////////////////////////////////////////////////////////////////////////////
// Argon2 block type (128 × uint64_t = 1024 bytes)
////////////////////////////////////////////////////////////////////////////////

struct Block
{
    uint64_t v[128];
};


////////////////////////////////////////////////////////////////////////////////
// fBlaMka — Argon2 mixing function (Blamka variant of Blake2b G)
////////////////////////////////////////////////////////////////////////////////

static inline void fBlaMka(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a += b + 2u * (uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b;
    d  = b2b_rotr64(d ^ a, 32u);
    c += d + 2u * (uint64_t)(uint32_t)c * (uint64_t)(uint32_t)d;
    b  = b2b_rotr64(b ^ c, 24u);
    a += b + 2u * (uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b;
    d  = b2b_rotr64(d ^ a, 16u);
    c += d + 2u * (uint64_t)(uint32_t)c * (uint64_t)(uint32_t)d;
    b  = b2b_rotr64(b ^ c, 63u);
}


// P — permutation on 16 words (4×4 matrix: columns then diagonals)
static inline void argon2_P(
    uint64_t& v0,  uint64_t& v1,  uint64_t& v2,  uint64_t& v3,
    uint64_t& v4,  uint64_t& v5,  uint64_t& v6,  uint64_t& v7,
    uint64_t& v8,  uint64_t& v9,  uint64_t& v10, uint64_t& v11,
    uint64_t& v12, uint64_t& v13, uint64_t& v14, uint64_t& v15)
{
    // Columns
    fBlaMka(v0, v4, v8,  v12);
    fBlaMka(v1, v5, v9,  v13);
    fBlaMka(v2, v6, v10, v14);
    fBlaMka(v3, v7, v11, v15);
    // Diagonals
    fBlaMka(v0, v5, v10, v15);
    fBlaMka(v1, v6, v11, v12);
    fBlaMka(v2, v7, v8,  v13);
    fBlaMka(v3, v4, v9,  v14);
}


// G(X, Y) → Z — Argon2 block compression function
static void fill_block(Block const& x, Block const& y, Block& z, bool const with_xor)
{
    Block R;
    for (uint32_t i{ 0u }; i < 128u; ++i)
    {
        R.v[i] = x.v[i] ^ y.v[i];
    }

    Block Q{ R };

    // Row permutations: 8 rows of 16 consecutive words each
    for (uint32_t row{ 0u }; row < 8u; ++row)
    {
        uint64_t* const r{ Q.v + row * 16u };
        argon2_P(r[0],  r[1],  r[2],  r[3],
                 r[4],  r[5],  r[6],  r[7],
                 r[8],  r[9],  r[10], r[11],
                 r[12], r[13], r[14], r[15]);
    }

    // Column permutations: 8 columns — column i uses words at {2i, 2i+1} in each row
    for (uint32_t col{ 0u }; col < 8u; ++col)
    {
        argon2_P(Q.v[col * 2u +   0u], Q.v[col * 2u +   1u],
                 Q.v[col * 2u +  16u], Q.v[col * 2u +  17u],
                 Q.v[col * 2u +  32u], Q.v[col * 2u +  33u],
                 Q.v[col * 2u +  48u], Q.v[col * 2u +  49u],
                 Q.v[col * 2u +  64u], Q.v[col * 2u +  65u],
                 Q.v[col * 2u +  80u], Q.v[col * 2u +  81u],
                 Q.v[col * 2u +  96u], Q.v[col * 2u +  97u],
                 Q.v[col * 2u + 112u], Q.v[col * 2u + 113u]);
    }

    // Z = Q XOR R (optionally also XOR old z value)
    if (with_xor)
    {
        for (uint32_t i{ 0u }; i < 128u; ++i)
        {
            z.v[i] ^= Q.v[i] ^ R.v[i];
        }
    }
    else
    {
        for (uint32_t i{ 0u }; i < 128u; ++i)
        {
            z.v[i] = Q.v[i] ^ R.v[i];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Reference index computation (phi formula from Argon2 spec RFC 9106 §3.3)
////////////////////////////////////////////////////////////////////////////////

static uint32_t compute_ref_index(
    uint64_t const j1,
    uint64_t const ref_area_size,
    uint64_t const start_pos,
    uint32_t const lane_length)
{
    // Quadratic distribution: recent blocks are more likely to be chosen
    uint64_t x{ (uint32_t)j1 };
    x = (x * x) >> 32u;
    uint64_t const relative_pos{ ref_area_size - 1u - ((ref_area_size * x) >> 32u) };
    return static_cast<uint32_t>((start_pos + relative_pos) % lane_length);
}


////////////////////////////////////////////////////////////////////////////////
// Argon2d constants for RandomX
////////////////////////////////////////////////////////////////////////////////

static constexpr uint32_t ARGON2_MEMORY_BLOCKS  { 262144u };
static constexpr uint32_t ARGON2_ITERATIONS      { 3u };
static constexpr uint32_t ARGON2_SYNC_POINTS     { 4u };
static constexpr uint32_t ARGON2_LANES           { 1u };
static constexpr uint32_t ARGON2_LANE_LENGTH     { ARGON2_MEMORY_BLOCKS / ARGON2_LANES };
static constexpr uint32_t ARGON2_SEGMENT_LENGTH  { ARGON2_LANE_LENGTH / ARGON2_SYNC_POINTS };


////////////////////////////////////////////////////////////////////////////////
// Fill one segment of Argon2d memory
////////////////////////////////////////////////////////////////////////////////

static void fill_segment(
    Block* const memory,
    uint32_t const pass,
    uint32_t const lane,
    uint32_t const slice)
{
    uint32_t const starting_index{ (pass == 0u && slice == 0u) ? 2u : 0u };
    uint32_t curr_offset{ lane * ARGON2_LANE_LENGTH + slice * ARGON2_SEGMENT_LENGTH + starting_index };

    for (uint32_t i{ starting_index }; i < ARGON2_SEGMENT_LENGTH; ++i, ++curr_offset)
    {
        // Previous block wraps at lane boundary
        uint32_t const prev_offset{ (curr_offset % ARGON2_LANE_LENGTH == 0u)
            ? curr_offset + ARGON2_LANE_LENGTH - 1u
            : curr_offset - 1u };

        // Argon2d: pseudo-random value from first word of previous block
        uint64_t const pseudo_rand{ memory[prev_offset].v[0] };
        uint64_t const j1{ pseudo_rand & 0xFFFFFFFFULL };

        // Compute reference area size (same-lane path, p=1 means j2 % 1 == 0 always)
        uint64_t ref_area_size;
        if (pass == 0u)
        {
            if (slice == 0u)
            {
                ref_area_size = (uint64_t)i - 1u;
            }
            else
            {
                ref_area_size = (uint64_t)slice * ARGON2_SEGMENT_LENGTH + i - 1u;
            }
        }
        else
        {
            ref_area_size = (uint64_t)ARGON2_LANE_LENGTH - ARGON2_SEGMENT_LENGTH + i - 1u;
        }

        // Start position of the reference set
        uint32_t const start_pos{ (pass == 0u) ? 0u
            : ((slice + 1u) % ARGON2_SYNC_POINTS) * ARGON2_SEGMENT_LENGTH };

        uint32_t const ref_index{
            compute_ref_index(j1, ref_area_size, start_pos, ARGON2_LANE_LENGTH) };

        fill_block(memory[prev_offset], memory[ref_index], memory[curr_offset], pass != 0u);
    }
}

} // anonymous namespace


////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

void algo::random_x::buildCache(
    uint8_t* const       cache,
    uint8_t const* const key,
    uint32_t const       keyLen)
{
    // Argon2d salt for RandomX
    static constexpr uint8_t SALT[8]{ 'R', 'a', 'n', 'd', 'o', 'm', 'X', 0x03 };

    // ── Compute H0 ────────────────────────────────────────────────────────
    // H0 = Blake2b-64(LE32(p) || LE32(T) || LE32(m) || LE32(t) ||
    //                 LE32(v) || LE32(y) ||
    //                 LE32(|P|) || P || LE32(|S|) || S ||
    //                 LE32(|K|) || LE32(|X|))
    Blake2bState h0;
    b2b_init(h0, 64u);

    auto push_le32 = [&h0](uint32_t const val)
    {
        uint8_t const bytes[4]
        {
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8u),
            static_cast<uint8_t>(val >> 16u),
            static_cast<uint8_t>(val >> 24u)
        };
        b2b_update(h0, bytes, 4u);
    };

    push_le32(1u);                              // parallelism
    push_le32(0u);                              // tag length = 0 (RandomX omits the finalizer step)
    push_le32(262144u);                         // memory (KiB)
    push_le32(3u);           // iterations
    push_le32(0x13u);        // version
    push_le32(0u);           // type (Argon2d = 0)
    push_le32(keyLen);       // |password|
    b2b_update(h0, key, keyLen);
    push_le32(8u);           // |salt|
    b2b_update(h0, SALT, 8u);
    push_le32(0u);           // |secret| = 0
    push_le32(0u);           // |additional| = 0

    uint8_t H0[64];
    b2b_final(h0, H0);

    // ── Initialize first two blocks of each lane ──────────────────────────
    // B[lane][0] = H'(H0 || LE32(0) || LE32(lane), 1024)
    // B[lane][1] = H'(H0 || LE32(1) || LE32(lane), 1024)
    Block* const memory{ reinterpret_cast<Block*>(cache) };

    for (uint32_t lane{ 0u }; lane < ARGON2_LANES; ++lane)
    {
        // Input: H0 (64 bytes) || col (4 bytes LE) || lane (4 bytes LE)
        uint8_t input[72];
        memcpy(input, H0, 64u);
        // lane bytes at [68..71]
        input[68] = static_cast<uint8_t>(lane);
        input[69] = static_cast<uint8_t>(lane >> 8u);
        input[70] = static_cast<uint8_t>(lane >> 16u);
        input[71] = static_cast<uint8_t>(lane >> 24u);

        // col = 0
        input[64] = 0u; input[65] = 0u; input[66] = 0u; input[67] = 0u;
        blake2b_long(memory[lane * ARGON2_LANE_LENGTH + 0u].v, 1024u, input, 72u);

        // col = 1
        input[64] = 1u; input[65] = 0u; input[66] = 0u; input[67] = 0u;
        blake2b_long(memory[lane * ARGON2_LANE_LENGTH + 1u].v, 1024u, input, 72u);
    }

    // ── Fill all remaining blocks ─────────────────────────────────────────
    for (uint32_t pass{ 0u }; pass < ARGON2_ITERATIONS; ++pass)
    {
        for (uint32_t slice{ 0u }; slice < ARGON2_SYNC_POINTS; ++slice)
        {
            for (uint32_t lane{ 0u }; lane < ARGON2_LANES; ++lane)
            {
                fill_segment(memory, pass, lane, slice);
            }
        }
    }

    // RandomX spec: "The finalizer and output calculation steps of Argon2 are
    // omitted. The output is the filled memory array."
    // The raw 256 MiB memory blocks ARE the cache — nothing more to do.
}
