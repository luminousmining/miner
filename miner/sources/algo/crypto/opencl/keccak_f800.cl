#define BC_INDEX_MAX 5u
#define OUT_INDEX_MAX 25u
#define KECCAK_INDEX_MAX 24u
#define KECCAK_F800_NUMBER_ROUND 22u


inline
void keccak_f800_round(
    uint const round,
    uint* const restrict dst)
{
    uint const F800_ROUND_CONSTANT[KECCAK_F800_NUMBER_ROUND] =
    {
        0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
        0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
        0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
        0x0000800a, 0x8000000a, 0x80008081, 0x00008080
    };

    uint const rotc[KECCAK_INDEX_MAX] =
    {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };

    uint const piln[KECCAK_INDEX_MAX] =
    {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };

    uint t;
    uint bc[BC_INDEX_MAX];

    // Theta
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < BC_INDEX_MAX; ++i)
    {
        bc[i] = dst[i]
            ^ dst[i + 5]
            ^ dst[i + 10]
            ^ dst[i + 15]
            ^ dst[i + 20];
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < BC_INDEX_MAX; ++i)
    {
        t = bc[(i + 4) % BC_INDEX_MAX] ^ rol_u32(bc[(i + 1) % BC_INDEX_MAX], 1u);
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < OUT_INDEX_MAX; j += 5)
        {
            dst[j + i] ^= t;
        }
    }

    // Rho Pi
    t = dst[1];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < KECCAK_INDEX_MAX; ++i)
    {
        uint j = piln[i];
        bc[0] = dst[j];
        dst[j] = rol_u32(t, rotc[i]);
        t = bc[0];
    }

    // Chi
    __attribute__((opencl_unroll_hint))
    for (uint j = 0u; j < OUT_INDEX_MAX; j += 5)
    {
        __attribute__((opencl_unroll_hint))
        for (uint i = 0u; i < BC_INDEX_MAX; ++i)
        {
            bc[i] = dst[j + i];
        }
        __attribute__((opencl_unroll_hint))
        for (uint i = 0u; i < BC_INDEX_MAX; ++i)
        {
            dst[j + i] ^= (~bc[(i + 1) % BC_INDEX_MAX]) & bc[(i + 2) % BC_INDEX_MAX];
        }
    }

    //  Iota
    dst[0] ^= F800_ROUND_CONSTANT[round];
}


inline
void keccak_f800(uint* const restrict dst)
{
    __attribute__((opencl_unroll_hint(1)))
    for (uint round = 0u; round < KECCAK_F800_NUMBER_ROUND; ++round)
    {
        keccak_f800_round(round, dst);
    }
}
