#include <algo/random_x/superscalar.hpp>

#include <cstdint>
#include <cstring>
#include <algorithm>


namespace
{

////////////////////////////////////////////////////////////////////////////////
// Blake2b-512 (CPU) — minimal version for BlakeGenerator
////////////////////////////////////////////////////////////////////////////////

static constexpr uint64_t BLAKE2B_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static constexpr uint8_t BLAKE2B_SIGMA[10][16]
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
};

static inline uint64_t b2b_ror64(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}


static void b2b_512(uint8_t* const out, uint8_t const* const in, uint32_t const inlen)
{
    uint64_t h[8];
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        h[i] = BLAKE2B_IV[i];
    }
    h[0] ^= 0x0000000001010040ULL;  // outlen=64, fanout=1, depth=1, no key

    // Process message blocks
    uint8_t  block[128];
    uint32_t t0{ 0u };
    uint32_t t1{ 0u };
    uint32_t rem{ inlen };
    uint32_t src{ 0u };

    while (rem > 128u)
    {
        memcpy(block, in + src, 128u);
        src += 128u;
        rem -= 128u;
        t0  += 128u;
        if (t0 < 128u) { ++t1; }

        uint64_t v[16];
        uint64_t m[16];
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            v[i]     = h[i];
            v[i + 8] = BLAKE2B_IV[i];
        }
        v[12] ^= t0;
        v[13] ^= t1;

        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            uint8_t const* const p{ block + i * 8u };
            m[i] = (uint64_t)p[0]        | ((uint64_t)p[1] << 8u)  |
                   ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
                   ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
                   ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
        }

#define BG(r, i, a, b, c, d) \
    v[a] += v[b] + m[BLAKE2B_SIGMA[r][2*(i)]]; \
    v[d] = b2b_ror64(v[d] ^ v[a], 32u); \
    v[c] += v[d]; \
    v[b] = b2b_ror64(v[b] ^ v[c], 24u); \
    v[a] += v[b] + m[BLAKE2B_SIGMA[r][2*(i)+1]]; \
    v[d] = b2b_ror64(v[d] ^ v[a], 16u); \
    v[c] += v[d]; \
    v[b] = b2b_ror64(v[b] ^ v[c], 63u);

        for (uint32_t r{ 0u }; r < 12u; ++r)
        {
            BG((r % 10u), 0,  0,  4,  8, 12)
            BG((r % 10u), 1,  1,  5,  9, 13)
            BG((r % 10u), 2,  2,  6, 10, 14)
            BG((r % 10u), 3,  3,  7, 11, 15)
            BG((r % 10u), 4,  0,  5, 10, 15)
            BG((r % 10u), 5,  1,  6, 11, 12)
            BG((r % 10u), 6,  2,  7,  8, 13)
            BG((r % 10u), 7,  3,  4,  9, 14)
        }
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            h[i] ^= v[i] ^ v[i + 8u];
        }
    }

    // Final block (padded)
    memset(block, 0, 128u);
    if (rem > 0u)
    {
        memcpy(block, in + src, rem);
    }
    t0 += rem;
    if (t0 < rem) { ++t1; }

    {
        uint64_t v[16];
        uint64_t m[16];
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            v[i]     = h[i];
            v[i + 8] = BLAKE2B_IV[i];
        }
        v[12] ^= t0;
        v[13] ^= t1;
        v[14]  = ~v[14];  // finalization flag

        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            uint8_t const* const p{ block + i * 8u };
            m[i] = (uint64_t)p[0]        | ((uint64_t)p[1] << 8u)  |
                   ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
                   ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
                   ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
        }

        for (uint32_t r{ 0u }; r < 12u; ++r)
        {
            BG((r % 10u), 0,  0,  4,  8, 12)
            BG((r % 10u), 1,  1,  5,  9, 13)
            BG((r % 10u), 2,  2,  6, 10, 14)
            BG((r % 10u), 3,  3,  7, 11, 15)
            BG((r % 10u), 4,  0,  5, 10, 15)
            BG((r % 10u), 5,  1,  6, 11, 12)
            BG((r % 10u), 6,  2,  7,  8, 13)
            BG((r % 10u), 7,  3,  4,  9, 14)
        }
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            h[i] ^= v[i] ^ v[i + 8u];
        }
    }
#undef BG

    // Write output (64 bytes, little-endian)
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        uint8_t* const p{ out + i * 8u };
        p[0] = static_cast<uint8_t>(h[i]);
        p[1] = static_cast<uint8_t>(h[i] >> 8u);
        p[2] = static_cast<uint8_t>(h[i] >> 16u);
        p[3] = static_cast<uint8_t>(h[i] >> 24u);
        p[4] = static_cast<uint8_t>(h[i] >> 32u);
        p[5] = static_cast<uint8_t>(h[i] >> 40u);
        p[6] = static_cast<uint8_t>(h[i] >> 48u);
        p[7] = static_cast<uint8_t>(h[i] >> 56u);
    }
}


////////////////////////////////////////////////////////////////////////////////
// BlakeGenerator — PRNG for SuperscalarHash program generation
// State: 64-byte Blake2b-512 output, read position
////////////////////////////////////////////////////////////////////////////////

struct BlakeGenerator
{
    uint8_t  state[64];
    uint32_t position{ 0u };

    void init(uint8_t const* const key, uint32_t const keylen)
    {
        // Reference Blake2Generator initializes the 64-byte buffer as:
        //   {key[0..min(keylen,60)-1], zeros, nonce_le32=0}
        // then sets dataIndex=64 so the first read triggers blake2b(data, 64).
        // We replicate this exactly: nonce=0 so the last 4 bytes stay zero.
        static constexpr uint32_t MAX_SEED_SIZE{ 60u };
        memset(state, 0, 64u);
        uint32_t const copyLen{ keylen < MAX_SEED_SIZE ? keylen : MAX_SEED_SIZE };
        memcpy(state, key, copyLen);
        // position = 64 so the first getByte/getUint32 triggers a refresh
        position = 64u;
    }

    // Get 1 byte
    uint8_t getByte()
    {
        if (position >= 64u)
        {
            b2b_512(state, state, 64u);
            position = 0u;
        }
        return state[position++];
    }

    // Get 4 bytes (uint32, little-endian) — refills if fewer than 4 bytes remain
    uint32_t getUint32()
    {
        if (position > 60u)
        {
            b2b_512(state, state, 64u);
            position = 0u;
        }
        uint32_t const v
        {
              static_cast<uint32_t>(state[position])
            | (static_cast<uint32_t>(state[position + 1u]) << 8u)
            | (static_cast<uint32_t>(state[position + 2u]) << 16u)
            | (static_cast<uint32_t>(state[position + 3u]) << 24u)
        };
        position += 4u;
        return v;
    }
};


////////////////////////////////////////////////////////////////////////////////
// Haswell execution port model
// Ports: P0=bit0, P1=bit1, P5=bit2
// Combined: P01=3, P05=5, P015=7
////////////////////////////////////////////////////////////////////////////////

namespace ExecutionPort
{
    using type = int;
    static constexpr type Null { 0 };
    static constexpr type P0   { 1 };
    static constexpr type P1   { 2 };
    static constexpr type P5   { 4 };
    static constexpr type P01  { P0 | P1 };
    static constexpr type P05  { P0 | P5 };
    static constexpr type P015 { P0 | P1 | P5 };
}

// Size of the port-busy table: RANDOMX_SUPERSCALAR_LATENCY + 4
static constexpr int CYCLE_MAP_SIZE     { static_cast<int>(algo::random_x::SUPERSCALAR_LATENCY) + 4 };
static constexpr int LOOK_FORWARD_CYCLES{ 4 };
static constexpr int MAX_THROWAWAY_COUNT{ 256 };

// r5 (x86 r13) cannot be the destination of IADD_RS due to LEA encoding limitations
static constexpr int REGISTER_NEEDS_DISPLACEMENT{ 5 };


////////////////////////////////////////////////////////////////////////////////
// Reference instruction types (matches reference enum 0-13)
// These are distinct from ScalarInstType (0-9) which collapses C7/C8/C9 variants
////////////////////////////////////////////////////////////////////////////////

enum class RefInstType : int
{
    ISUB_R   = 0,
    IXOR_R   = 1,
    IADD_RS  = 2,
    IMUL_R   = 3,
    IROR_C   = 4,
    IADD_C7  = 5,
    IXOR_C7  = 6,
    IADD_C8  = 7,
    IXOR_C8  = 8,
    IADD_C9  = 9,
    IXOR_C9  = 10,
    IMULH_R  = 11,
    ISMULH_R = 12,
    IMUL_RCP = 13,
    INVALID  = -1,
};

static inline bool refIsMultiplication(RefInstType const t)
{
    return t == RefInstType::IMUL_R
        || t == RefInstType::IMULH_R
        || t == RefInstType::ISMULH_R
        || t == RefInstType::IMUL_RCP;
}

static inline bool isZeroOrPowerOf2(uint32_t const x)
{
    return (x & (x - 1u)) == 0u;  // also true for x==0 since 0&(~0)==0 is false, but 0-1 wraps…
    // Actually: x==0 gives 0 & 0xFFFFFFFF == 0, true; power-of-2 gives 0, true
}


////////////////////////////////////////////////////////////////////////////////
// MacroOp — describes one x86 macro-operation
////////////////////////////////////////////////////////////////////////////////

struct MacroOp
{
    int                  size;
    int                  latency;
    ExecutionPort::type  uop1;
    ExecutionPort::type  uop2;
    bool                 dependent;  // true for the Imul_rr in IMUL_RCP (depends on depCycle)

    bool isEliminated() const { return uop1 == ExecutionPort::Null; }
    bool isSimple()     const { return uop2 == ExecutionPort::Null; }
    bool isDependent()  const { return dependent; }
    int  getLatency()   const { return latency; }
    int  getSize()      const { return size; }
};

// Individual macro-op definitions
static constexpr MacroOp MOP_Sub_rr    { 3,  1, ExecutionPort::P015, ExecutionPort::Null, false };
static constexpr MacroOp MOP_Xor_rr    { 3,  1, ExecutionPort::P015, ExecutionPort::Null, false };
static constexpr MacroOp MOP_Imul_r    { 3,  4, ExecutionPort::P1,   ExecutionPort::P5,   false };
static constexpr MacroOp MOP_Mul_r     { 3,  4, ExecutionPort::P1,   ExecutionPort::P5,   false };
static constexpr MacroOp MOP_Mov_rr    { 3,  0, ExecutionPort::Null, ExecutionPort::Null, false };  // eliminated
static constexpr MacroOp MOP_Lea_sib   { 4,  1, ExecutionPort::P01,  ExecutionPort::Null, false };
static constexpr MacroOp MOP_Imul_rr   { 4,  3, ExecutionPort::P1,   ExecutionPort::Null, false };
static constexpr MacroOp MOP_Ror_ri    { 4,  1, ExecutionPort::P05,  ExecutionPort::Null, false };
static constexpr MacroOp MOP_Add_ri    { 7,  1, ExecutionPort::P015, ExecutionPort::Null, false };
static constexpr MacroOp MOP_Xor_ri    { 7,  1, ExecutionPort::P015, ExecutionPort::Null, false };
static constexpr MacroOp MOP_Mov_ri64  { 10, 1, ExecutionPort::P015, ExecutionPort::Null, false };
static constexpr MacroOp MOP_Imul_rr_dep{ 4, 3, ExecutionPort::P1,   ExecutionPort::Null, true  };  // dependent


////////////////////////////////////////////////////////////////////////////////
// SuperscalarInstructionInfo — describes one logical RandomX instruction
// Maps instruction type to its macro-op sequence and operand roles
////////////////////////////////////////////////////////////////////////////////

struct SuperscalarInstructionInfo
{
    MacroOp const* ops;    // pointer to macro-op array
    int            nOps;   // number of macro-ops
    int            latency;
    int            resultOp;
    int            dstOp;
    int            srcOp;
    RefInstType    type;
};

// Macro-op arrays for multi-uop instructions
static constexpr MacroOp IMULH_R_ops[]  { MOP_Mov_rr, MOP_Mul_r,    MOP_Mov_rr };
static constexpr MacroOp ISMULH_R_ops[] { MOP_Mov_rr, MOP_Imul_r,   MOP_Mov_rr };
static constexpr MacroOp IMUL_RCP_ops[] { MOP_Mov_ri64, MOP_Imul_rr_dep };

// Single macro-op wrappers (to allow uniform ops[] pointer usage)
static constexpr MacroOp ISUB_R_ops[]   { MOP_Sub_rr  };
static constexpr MacroOp IXOR_R_ops[]   { MOP_Xor_rr  };
static constexpr MacroOp IADD_RS_ops[]  { MOP_Lea_sib };
static constexpr MacroOp IMUL_R_ops[]   { MOP_Imul_rr };
static constexpr MacroOp IROR_C_ops[]   { MOP_Ror_ri  };
static constexpr MacroOp IADD_C_ops[]   { MOP_Add_ri  };
static constexpr MacroOp IXOR_C_ops[]   { MOP_Xor_ri  };

// Instruction info table
static constexpr SuperscalarInstructionInfo INST_INFO[]
{
    // type        ops            nOps  latency  resultOp  dstOp  srcOp
    { ISUB_R_ops,   1, 1,  0, 0,  0, RefInstType::ISUB_R   },
    { IXOR_R_ops,   1, 1,  0, 0,  0, RefInstType::IXOR_R   },
    { IADD_RS_ops,  1, 1,  0, 0,  0, RefInstType::IADD_RS  },
    { IMUL_R_ops,   1, 3,  0, 0,  0, RefInstType::IMUL_R   },
    { IROR_C_ops,   1, 1,  0, 0, -1, RefInstType::IROR_C   },
    { IADD_C_ops,   1, 1,  0, 0, -1, RefInstType::IADD_C7  },  // IADD_C7
    { IXOR_C_ops,   1, 1,  0, 0, -1, RefInstType::IXOR_C7  },  // IXOR_C7
    { IADD_C_ops,   1, 1,  0, 0, -1, RefInstType::IADD_C8  },  // IADD_C8
    { IXOR_C_ops,   1, 1,  0, 0, -1, RefInstType::IXOR_C8  },  // IXOR_C8
    { IADD_C_ops,   1, 1,  0, 0, -1, RefInstType::IADD_C9  },  // IADD_C9
    { IXOR_C_ops,   1, 1,  0, 0, -1, RefInstType::IXOR_C9  },  // IXOR_C9
    { IMULH_R_ops,  3, 4,  1, 0,  1, RefInstType::IMULH_R  },
    { ISMULH_R_ops, 3, 4,  1, 0,  1, RefInstType::ISMULH_R },
    { IMUL_RCP_ops, 2, 4,  1, 1, -1, RefInstType::IMUL_RCP },
};

static inline SuperscalarInstructionInfo const& getInstInfo(RefInstType const t)
{
    return INST_INFO[static_cast<int>(t)];
}


////////////////////////////////////////////////////////////////////////////////
// Decoder buffer configurations
// Each buffer describes the slot sizes (in bytes) used in one decode cycle.
// Buffer index matches the reference: 0="4,8,4", 1="7,3,3,3", 2="3,7,3,3",
//   3="4,9,3", 4="4,4,4,4", 5="3,3,10"
////////////////////////////////////////////////////////////////////////////////

struct DecoderBuffer
{
    int const* counts;
    int        size;
    int        index;
};

static constexpr int BUF0_COUNTS[] { 4, 8, 4 };
static constexpr int BUF1_COUNTS[] { 7, 3, 3, 3 };
static constexpr int BUF2_COUNTS[] { 3, 7, 3, 3 };
static constexpr int BUF3_COUNTS[] { 4, 9, 3 };
static constexpr int BUF4_COUNTS[] { 4, 4, 4, 4 };
static constexpr int BUF5_COUNTS[] { 3, 3, 10 };

static constexpr DecoderBuffer DECODE_BUF[]
{
    { BUF0_COUNTS, 3, 0 },  // "4,8,4"
    { BUF1_COUNTS, 4, 1 },  // "7,3,3,3"
    { BUF2_COUNTS, 4, 2 },  // "3,7,3,3"
    { BUF3_COUNTS, 3, 3 },  // "4,9,3"
    { BUF4_COUNTS, 4, 4 },  // "4,4,4,4"
    { BUF5_COUNTS, 3, 5 },  // "3,3,10"
};

// fetchNext: select the decode buffer for the next cycle based on current instruction type,
// cycle number, multiplication count, and the PRNG.
// Mirrors the reference DecoderBuffer::fetchNext() exactly.
static DecoderBuffer const& fetchNext(
    RefInstType const    instrType,
    int const            decodeCycle,
    int const            mulCount,
    BlakeGenerator&      gen)
{
    // IMULH/ISMULH require buffer5 (3,3,10) because the 128-bit multiply
    // decodes to 2 uops and forces a specific alignment
    if (instrType == RefInstType::IMULH_R || instrType == RefInstType::ISMULH_R)
    {
        return DECODE_BUF[5];
    }

    // Ensure multiplication port stays saturated
    if (mulCount < decodeCycle + 1)
    {
        return DECODE_BUF[4];
    }

    // IMUL_RCP needs a 4-byte slot first (for the MOV_ri64 which is 10 bytes,
    // but the IMUL_RCP itself needs a 4-byte slot for the imul r,r that follows)
    if (instrType == RefInstType::IMUL_RCP)
    {
        return (gen.getByte() & 1) ? DECODE_BUF[0] : DECODE_BUF[3];
    }

    // Default: random selection from buffers 0-3
    return DECODE_BUF[gen.getByte() & 3];
}


////////////////////////////////////////////////////////////////////////////////
// Reciprocal computation for IMUL_RCP
// Returns floor(2^x / d) where x is chosen so that result fits in 64 bits
////////////////////////////////////////////////////////////////////////////////

static uint64_t compute_reciprocal(uint32_t const d)
{
    if (d == 0u || (d & (d - 1u)) == 0u)
    {
        return 0ull;
    }

    // Matches reference randomx_reciprocal: compute floor(2^(63+shift) / d)
    // where shift = 64 - clzll(d).
    uint64_t const p2exp63  { 1ULL << 63 };
    uint64_t const dw       { static_cast<uint64_t>(d) };
    uint64_t const q        { p2exp63 / dw };
    uint64_t const r        { p2exp63 % dw };
    uint32_t const shift    { 64u - static_cast<uint32_t>(__builtin_clzll(d)) };
    return (q << shift) + ((r << shift) / dw);
}


////////////////////////////////////////////////////////////////////////////////
// Register state tracking
////////////////////////////////////////////////////////////////////////////////

struct RegisterInfo
{
    int         latency     { 0 };
    RefInstType lastOpGroup { RefInstType::INVALID };
    int         lastOpPar   { -1 };
};


////////////////////////////////////////////////////////////////////////////////
// Port scheduling
// portBusy[cycle][port] — nonzero if the port is occupied at that cycle
// Ports: column 0=P0, 1=P1, 2=P5
////////////////////////////////////////////////////////////////////////////////

template<bool commit>
static int scheduleUop(
    ExecutionPort::type                          uop,
    ExecutionPort::type (&portBusy)[CYCLE_MAP_SIZE][3],
    int                                          cycle)
{
    // Priority: P5 -> P0 -> P1  (avoids overloading the multiply port P1)
    for (; cycle < CYCLE_MAP_SIZE; ++cycle)
    {
        if ((uop & ExecutionPort::P5) != 0 && !portBusy[cycle][2])
        {
            if (commit) { portBusy[cycle][2] = uop; }
            return cycle;
        }
        if ((uop & ExecutionPort::P0) != 0 && !portBusy[cycle][0])
        {
            if (commit) { portBusy[cycle][0] = uop; }
            return cycle;
        }
        if ((uop & ExecutionPort::P1) != 0 && !portBusy[cycle][1])
        {
            if (commit) { portBusy[cycle][1] = uop; }
            return cycle;
        }
    }
    return -1;
}

template<bool commit>
static int scheduleMop(
    MacroOp const&                               mop,
    ExecutionPort::type (&portBusy)[CYCLE_MAP_SIZE][3],
    int                                          cycle,
    int                                          depCycle)
{
    // Dependent macro-ops (Imul_rr in IMUL_RCP) must start no earlier than depCycle
    if (mop.isDependent())
    {
        cycle = std::max(cycle, depCycle);
    }

    // Eliminated macro-ops (MOV r,r register-rename) consume no execution port
    if (mop.isEliminated())
    {
        return cycle;
    }

    if (mop.isSimple())
    {
        // Single uop
        return scheduleUop<commit>(mop.uop1, portBusy, cycle);
    }

    // 2-uop macro-op: both uops must execute in the same cycle
    for (; cycle < CYCLE_MAP_SIZE; ++cycle)
    {
        int const c1{ scheduleUop<false>(mop.uop1, portBusy, cycle) };
        int const c2{ scheduleUop<false>(mop.uop2, portBusy, cycle) };
        if (c1 >= 0 && c1 == c2)
        {
            if (commit)
            {
                scheduleUop<true>(mop.uop1, portBusy, c1);
                scheduleUop<true>(mop.uop2, portBusy, c2);
            }
            return c1;
        }
    }
    return -1;
}


////////////////////////////////////////////////////////////////////////////////
// Current instruction state (mirrors SuperscalarInstruction from reference)
////////////////////////////////////////////////////////////////////////////////

struct CurrentInstruction
{
    SuperscalarInstructionInfo const* info    { nullptr };
    int                               src     { -1 };
    int                               dst     { -1 };
    int                               mod     { 0 };
    uint32_t                          imm32   { 0u };
    RefInstType                       opGroup { RefInstType::INVALID };
    int                               opGroupPar { -1 };
    bool                              canReuse         { false };
    bool                              groupParIsSource { false };

    void reset()
    {
        src              = -1;
        dst              = -1;
        canReuse         = false;
        groupParIsSource = false;
    }

    RefInstType getType() const
    {
        return (nullptr != info) ? info->type : RefInstType::INVALID;
    }

    // Create the instruction for a given slot size.
    // Exactly mirrors createForSlot() from the reference.
    void createForSlot(
        BlakeGenerator& gen,
        int const       slotSize,
        int const       fetchType,
        bool const      isLast,
        bool const      /*isFirst*/)
    {
        switch (slotSize)
        {
            case 3:
            {
                if (isLast)
                {
                    // Last slot: can also be IMULH_R or ISMULH_R
                    static constexpr RefInstType slot3L[4]
                    {
                        RefInstType::ISUB_R, RefInstType::IXOR_R,
                        RefInstType::IMULH_R, RefInstType::ISMULH_R
                    };
                    create(getInstInfo(slot3L[gen.getByte() & 3]), gen);
                }
                else
                {
                    // Non-last slot: only ISUB_R or IXOR_R
                    static constexpr RefInstType slot3[2]
                    {
                        RefInstType::ISUB_R, RefInstType::IXOR_R
                    };
                    create(getInstInfo(slot3[gen.getByte() & 1]), gen);
                }
                break;
            }
            case 4:
            {
                if (fetchType == 4 && !isLast)
                {
                    // 4-4-4-4 buffer: first 3 slots are multiplications
                    create(getInstInfo(RefInstType::IMUL_R), gen);
                }
                else
                {
                    static constexpr RefInstType slot4[2]
                    {
                        RefInstType::IROR_C, RefInstType::IADD_RS
                    };
                    create(getInstInfo(slot4[gen.getByte() & 1]), gen);
                }
                break;
            }
            case 7:
            {
                static constexpr RefInstType slot7[2]
                {
                    RefInstType::IXOR_C7, RefInstType::IADD_C7
                };
                create(getInstInfo(slot7[gen.getByte() & 1]), gen);
                break;
            }
            case 8:
            {
                static constexpr RefInstType slot8[2]
                {
                    RefInstType::IXOR_C8, RefInstType::IADD_C8
                };
                create(getInstInfo(slot8[gen.getByte() & 1]), gen);
                break;
            }
            case 9:
            {
                static constexpr RefInstType slot9[2]
                {
                    RefInstType::IXOR_C9, RefInstType::IADD_C9
                };
                create(getInstInfo(slot9[gen.getByte() & 1]), gen);
                break;
            }
            case 10:
            {
                create(getInstInfo(RefInstType::IMUL_RCP), gen);
                break;
            }
            default:
            {
                break;
            }
        }
    }

    // Initialize instruction state from info + generator.
    // Exactly mirrors create() from the reference.
    void create(SuperscalarInstructionInfo const& infoRef, BlakeGenerator& gen)
    {
        info = &infoRef;
        reset();

        switch (infoRef.type)
        {
            case RefInstType::ISUB_R:
            {
                mod              = 0;
                imm32            = 0u;
                opGroup          = RefInstType::IADD_RS;  // ISUB_R belongs to IADD_RS group
                groupParIsSource = true;
                break;
            }
            case RefInstType::IXOR_R:
            {
                mod              = 0;
                imm32            = 0u;
                opGroup          = RefInstType::IXOR_R;
                groupParIsSource = true;
                break;
            }
            case RefInstType::IADD_RS:
            {
                mod              = static_cast<int>(gen.getByte());  // consumes 1 byte
                imm32            = 0u;
                opGroup          = RefInstType::IADD_RS;
                groupParIsSource = true;
                break;
            }
            case RefInstType::IMUL_R:
            {
                mod              = 0;
                imm32            = 0u;
                opGroup          = RefInstType::IMUL_R;
                groupParIsSource = true;
                break;
            }
            case RefInstType::IROR_C:
            {
                mod = 0;
                do
                {
                    imm32 = static_cast<uint32_t>(gen.getByte()) & 63u;
                }
                while (imm32 == 0u);
                opGroup    = RefInstType::IROR_C;
                opGroupPar = -1;
                break;
            }
            case RefInstType::IADD_C7:
            case RefInstType::IADD_C8:
            case RefInstType::IADD_C9:
            {
                mod        = 0;
                imm32      = gen.getUint32();  // consumes 4 bytes
                opGroup    = RefInstType::IADD_C7;  // all IADD_C variants share group
                opGroupPar = -1;
                break;
            }
            case RefInstType::IXOR_C7:
            case RefInstType::IXOR_C8:
            case RefInstType::IXOR_C9:
            {
                mod        = 0;
                imm32      = gen.getUint32();  // consumes 4 bytes
                opGroup    = RefInstType::IXOR_C7;  // all IXOR_C variants share group
                opGroupPar = -1;
                break;
            }
            case RefInstType::IMULH_R:
            {
                canReuse         = true;
                mod              = 0;
                imm32            = 0u;
                opGroup          = RefInstType::IMULH_R;
                opGroupPar       = static_cast<int>(gen.getUint32());  // consumes 4 bytes
                break;
            }
            case RefInstType::ISMULH_R:
            {
                canReuse         = true;
                mod              = 0;
                imm32            = 0u;
                opGroup          = RefInstType::ISMULH_R;
                opGroupPar       = static_cast<int>(gen.getUint32());  // consumes 4 bytes
                break;
            }
            case RefInstType::IMUL_RCP:
            {
                mod = 0;
                do
                {
                    imm32 = gen.getUint32();  // consumes 4 bytes per attempt
                }
                while (isZeroOrPowerOf2(imm32));
                opGroup    = RefInstType::IMUL_RCP;
                opGroupPar = -1;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    // Select source register.
    // Mirrors selectSource() from the reference exactly.
    bool selectSource(int const cycle, RegisterInfo (&registers)[8], BlakeGenerator& gen)
    {
        // Build list of registers ready at or before this cycle
        int avail[8];
        int nAvail{ 0 };
        for (int i{ 0 }; i < 8; ++i)
        {
            if (registers[i].latency <= cycle)
            {
                avail[nAvail++] = i;
            }
        }

        // Special case for IADD_RS with exactly 2 available registers:
        // if r5 (RegisterNeedsDisplacement) is one of them, force it as source
        // because r5 cannot be the destination of IADD_RS
        if (nAvail == 2
            && info->type == RefInstType::IADD_RS
            && (avail[0] == REGISTER_NEEDS_DISPLACEMENT
                || avail[1] == REGISTER_NEEDS_DISPLACEMENT))
        {
            opGroupPar = src = REGISTER_NEEDS_DISPLACEMENT;
            return true;
        }

        if (nAvail == 0)
        {
            return false;
        }

        int idx{ 0 };
        if (nAvail > 1)
        {
            idx = static_cast<int>(gen.getUint32() % static_cast<uint32_t>(nAvail));
        }
        src = avail[idx];

        if (groupParIsSource)
        {
            opGroupPar = src;
        }

        return true;
    }

    // Select destination register.
    // Mirrors selectDestination() from the reference exactly.
    bool selectDestination(
        int const           cycle,
        bool const          allowChainedMul,
        RegisterInfo (&registers)[8],
        BlakeGenerator&     gen)
    {
        int avail[8];
        int nAvail{ 0 };

        for (int i{ 0 }; i < 8; ++i)
        {
            // Condition set from reference:
            // 1. register must be ready
            bool const ready        { registers[i].latency <= cycle };
            // 2. cannot be same as source unless canReuse (IMULH/ISMULH)
            bool const notSrc       { canReuse || i != src };
            // 3. chained multiplication check
            bool const chainOk      { allowChainedMul
                                      || opGroup != RefInstType::IMUL_R
                                      || registers[i].lastOpGroup != RefInstType::IMUL_R };
            // 4. avoid repeating same group/par on same register
            bool const notRepeat    { registers[i].lastOpGroup != opGroup
                                      || registers[i].lastOpPar  != opGroupPar };
            // 5. r5 cannot be destination of IADD_RS
            bool const notR5ForLea  { info->type != RefInstType::IADD_RS
                                      || i != REGISTER_NEEDS_DISPLACEMENT };

            if (ready && notSrc && chainOk && notRepeat && notR5ForLea)
            {
                avail[nAvail++] = i;
            }
        }

        if (nAvail == 0)
        {
            return false;
        }

        int idx{ 0 };
        if (nAvail > 1)
        {
            idx = static_cast<int>(gen.getUint32() % static_cast<uint32_t>(nAvail));
        }
        dst = avail[idx];

        return true;
    }

    // Convert to output ScalarInst.
    // Maps reference type (0-13) to ScalarInstType (0-9), extracts imm correctly.
    void toInstr(algo::random_x::ScalarInst& instr) const
    {
        // Map reference type to ScalarInstType
        using T = algo::random_x::ScalarInstType;
        switch (info->type)
        {
            case RefInstType::ISUB_R:   instr.type = T::ISUB_R;   break;
            case RefInstType::IXOR_R:   instr.type = T::IXOR_R;   break;
            case RefInstType::IADD_RS:  instr.type = T::IADD_RS;  break;
            case RefInstType::IMUL_R:   instr.type = T::IMUL_R;   break;
            case RefInstType::IROR_C:   instr.type = T::IROR_C;   break;
            case RefInstType::IADD_C7:
            case RefInstType::IADD_C8:
            case RefInstType::IADD_C9:  instr.type = T::IADD_C;   break;
            case RefInstType::IXOR_C7:
            case RefInstType::IXOR_C8:
            case RefInstType::IXOR_C9:  instr.type = T::IXOR_C;   break;
            case RefInstType::IMULH_R:  instr.type = T::IMULH_R;  break;
            case RefInstType::ISMULH_R: instr.type = T::ISMULH_R; break;
            case RefInstType::IMUL_RCP: instr.type = T::IMUL_RCP; break;
            default:                    instr.type = T::ISUB_R;   break;
        }

        instr.dst = static_cast<uint8_t>(dst);
        // Reference: src >= 0 ? src : dst
        instr.src = static_cast<uint8_t>(src >= 0 ? src : dst);

        // Extract immediate value based on type
        switch (info->type)
        {
            case RefInstType::IADD_RS:
            {
                // shift = (mod >> 2) & 3
                instr.imm = static_cast<uint32_t>((mod >> 2) & 3);
                break;
            }
            case RefInstType::IROR_C:
            case RefInstType::IADD_C7:
            case RefInstType::IADD_C8:
            case RefInstType::IADD_C9:
            case RefInstType::IXOR_C7:
            case RefInstType::IXOR_C8:
            case RefInstType::IXOR_C9:
            case RefInstType::IMUL_RCP:
            {
                instr.imm = imm32;
                break;
            }
            default:
            {
                instr.imm = 0u;
                break;
            }
        }
    }
};

// Null instruction (used as initial state, type INVALID with NOP info)
static SuperscalarInstructionInfo const NULL_INFO
{
    nullptr, 0, 0, 0, 0, -1, RefInstType::INVALID
};


////////////////////////////////////////////////////////////////////////////////
// Generate one SuperscalarHash program
// This is a faithful translation of generateSuperscalar() from the reference.
////////////////////////////////////////////////////////////////////////////////

static void generate_program(
    BlakeGenerator&                     gen,
    algo::random_x::SuperscalarProgram& prog)
{
    // Port-busy table: portBusy[cycle][port], port: 0=P0, 1=P1, 2=P5
    ExecutionPort::type portBusy[CYCLE_MAP_SIZE][3];
    memset(portBusy, 0, sizeof(portBusy));

    RegisterInfo registers[8];  // all initialized to latency=0, group=INVALID, par=-1

    // Start with a null instruction so fetchNext sees INVALID type on first cycle
    CurrentInstruction currentInstruction;
    currentInstruction.info = &NULL_INFO;

    int  macroOpIndex   { 0 };
    int  cycle          { 0 };
    int  depCycle       { 0 };
    int  retireCycle    { 0 };
    bool portsSaturated { false };
    int  programSize    { 0 };
    int  mulCount       { 0 };
    int  throwAwayCount { 0 };
    int  decodeCycle    { 0 };

    // Decode RANDOMX_SUPERSCALAR_LATENCY cycles or until ports saturate or program full
    for (decodeCycle = 0;
         decodeCycle < static_cast<int>(algo::random_x::SUPERSCALAR_LATENCY)
             && !portsSaturated
             && programSize < static_cast<int>(algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS);
         ++decodeCycle)
    {
        // Fetch decode buffer for this cycle
        DecoderBuffer const& decodeBuffer{ fetchNext(currentInstruction.getType(), decodeCycle, mulCount, gen) };

        int bufferIndex{ 0 };

        // Process all slots in this decode buffer
        while (bufferIndex < decodeBuffer.size)
        {
            // Save cycle at the top of this slot processing
            int const topCycle{ cycle };

            // If all macro-ops of the current instruction have been issued, create a new one
            if (macroOpIndex >= ((nullptr != currentInstruction.info) ? currentInstruction.info->nOps : 0))
            {
                if (portsSaturated
                    || programSize >= static_cast<int>(algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS))
                {
                    break;
                }

                bool const isLast  { bufferIndex == decodeBuffer.size - 1 };
                bool const isFirst { bufferIndex == 0 };

                currentInstruction.createForSlot(
                    gen,
                    decodeBuffer.counts[bufferIndex],
                    decodeBuffer.index,
                    isLast,
                    isFirst);
                macroOpIndex = 0;
            }

            int const instrSize{ currentInstruction.info->nOps };
            MacroOp const& mop{ currentInstruction.info->ops[macroOpIndex] };

            // Calculate earliest cycle this mop can be scheduled
            int scheduleCycle{ scheduleMop<false>(mop, portBusy, cycle, depCycle) };
            if (scheduleCycle < 0)
            {
                portsSaturated = true;
                break;
            }

            // Find a source register ready at scheduleCycle
            if (macroOpIndex == currentInstruction.info->srcOp)
            {
                int forward{ 0 };
                for (;
                     forward < LOOK_FORWARD_CYCLES
                         && !currentInstruction.selectSource(scheduleCycle, registers, gen);
                     ++forward)
                {
                    ++scheduleCycle;
                    ++cycle;
                }
                if (forward == LOOK_FORWARD_CYCLES)
                {
                    if (throwAwayCount < MAX_THROWAWAY_COUNT)
                    {
                        ++throwAwayCount;
                        // Discard this instruction; set macroOpIndex past end to trigger new creation
                        macroOpIndex = instrSize;
                        continue;
                    }
                    // Abort this decode buffer
                    currentInstruction.info = &NULL_INFO;
                    break;
                }
            }

            // Find a destination register ready at scheduleCycle
            if (macroOpIndex == currentInstruction.info->dstOp)
            {
                int forward{ 0 };
                for (;
                     forward < LOOK_FORWARD_CYCLES
                         && !currentInstruction.selectDestination(scheduleCycle, throwAwayCount > 0, registers, gen);
                     ++forward)
                {
                    ++scheduleCycle;
                    ++cycle;
                }
                if (forward == LOOK_FORWARD_CYCLES)
                {
                    if (throwAwayCount < MAX_THROWAWAY_COUNT)
                    {
                        ++throwAwayCount;
                        macroOpIndex = instrSize;
                        continue;
                    }
                    // Abort this decode buffer
                    currentInstruction.info = &NULL_INFO;
                    break;
                }
            }

            throwAwayCount = 0;

            // Commit the scheduling — actually reserve the port(s)
            scheduleCycle = scheduleMop<true>(mop, portBusy, scheduleCycle, scheduleCycle);
            if (scheduleCycle < 0)
            {
                portsSaturated = true;
                break;
            }

            // Update dependency chain
            depCycle = scheduleCycle + mop.getLatency();

            // If this is the result-writing macro-op, update register state
            if (macroOpIndex == currentInstruction.info->resultOp)
            {
                int const dst{ currentInstruction.dst };
                retireCycle            = depCycle;
                registers[dst].latency     = retireCycle;
                registers[dst].lastOpGroup = currentInstruction.opGroup;
                registers[dst].lastOpPar   = currentInstruction.opGroupPar;
            }

            ++bufferIndex;
            ++macroOpIndex;

            // Check termination: if we scheduled past the latency limit, mark saturated
            if (scheduleCycle >= static_cast<int>(algo::random_x::SUPERSCALAR_LATENCY))
            {
                portsSaturated = true;
            }

            // Restore cycle to topCycle — the cycle counter only advances once per decode cycle
            cycle = topCycle;

            // If instruction is complete, emit it to the program
            if (macroOpIndex >= instrSize)
            {
                currentInstruction.toInstr(prog.instructions[programSize++]);
                mulCount += refIsMultiplication(currentInstruction.getType()) ? 1 : 0;
            }
        }

        ++cycle;
    }

    // Calculate ASIC latency (assumes 1 cycle per op, unlimited parallelism)
    int asicLatencies[8]{ 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i{ 0 }; i < programSize; ++i)
    {
        algo::random_x::ScalarInst const& instr{ prog.instructions[i] };
        int const latDst{ asicLatencies[instr.dst] + 1 };
        int const latSrc{ (instr.dst != instr.src) ? asicLatencies[instr.src] + 1 : 0 };
        asicLatencies[instr.dst] = std::max(latDst, latSrc);
    }

    // addressReg = register with the highest ASIC latency
    int asicLatencyMax{ 0 };
    int addressReg    { 0 };
    for (int i{ 0 }; i < 8; ++i)
    {
        if (asicLatencies[i] > asicLatencyMax)
        {
            asicLatencyMax = asicLatencies[i];
            addressReg     = i;
        }
    }

    prog.size       = static_cast<uint32_t>(programSize);
    prog.addressReg = static_cast<uint32_t>(addressReg);
}

} // anonymous namespace


////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

void algo::random_x::buildSuperscalarPrograms(
    uint8_t const* const      cacheKey,
    SuperscalarProgram         programs[SUPERSCALAR_ITERS])
{
    BlakeGenerator gen;
    gen.init(cacheKey, 32u);

    for (uint32_t i{ 0u }; i < SUPERSCALAR_ITERS; ++i)
    {
        generate_program(gen, programs[i]);
    }
}


uint64_t algo::random_x::superscalarComputeReciprocal(uint32_t const divisor)
{
    return compute_reciprocal(divisor);
}


void algo::random_x::executeSuperscalarProgram(
    SuperscalarProgram const& prog,
    uint64_t                  r[8])
{
    for (uint32_t i{ 0u }; i < prog.size; ++i)
    {
        ScalarInst const& instr{ prog.instructions[i] };
        uint64_t const dst_val{ r[instr.dst] };
        uint64_t const src_val{ (instr.src < 8u) ? r[instr.src] : 0ull };

        switch (instr.type)
        {
            case ScalarInstType::ISUB_R:
            {
                r[instr.dst] = dst_val - src_val;
                break;
            }
            case ScalarInstType::IXOR_R:
            {
                r[instr.dst] = dst_val ^ src_val;
                break;
            }
            case ScalarInstType::IADD_RS:
            {
                r[instr.dst] = dst_val + (src_val << (instr.imm & 3u));
                break;
            }
            case ScalarInstType::IMUL_R:
            {
                r[instr.dst] = dst_val * src_val;
                break;
            }
            case ScalarInstType::IROR_C:
            {
                uint32_t const shift{ instr.imm & 63u };
                r[instr.dst] = (dst_val >> shift) | (dst_val << (64u - shift));
                break;
            }
            case ScalarInstType::IADD_C:
            {
                r[instr.dst] = dst_val + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(instr.imm)));
                break;
            }
            case ScalarInstType::IXOR_C:
            {
                r[instr.dst] = dst_val ^ static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(instr.imm)));
                break;
            }
            case ScalarInstType::IMULH_R:
            {
                // High 64 bits of unsigned 128-bit product
                __uint128_t const prod{ (__uint128_t)dst_val * (__uint128_t)src_val };
                r[instr.dst] = static_cast<uint64_t>(prod >> 64u);
                break;
            }
            case ScalarInstType::ISMULH_R:
            {
                // High 64 bits of signed 128-bit product
                __int128_t const prod
                {
                    (__int128_t)(int64_t)dst_val * (__int128_t)(int64_t)src_val
                };
                r[instr.dst] = static_cast<uint64_t>((unsigned __int128)prod >> 64u);
                break;
            }
            case ScalarInstType::IMUL_RCP:
            {
                uint64_t const rcp{ compute_reciprocal(instr.imm) };
                if (rcp != 0ull)
                {
                    r[instr.dst] = dst_val * rcp;
                }
                break;
            }
        }
    }
}
