inline
void initialize_seed(
    __constant uint4 const* const restrict header,
    uint* const restrict seed,
    ulong const nonce)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 25u; ++i)
    {
        seed_result[i] = seed_init[i];
    }
    keccak_f800(seed);
}


inline
void sha3(
    uint const* const restrict seed_init,
    uint4 const* const restrict digest,
    uint* const restrict seed_result)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 25u; ++i)
    {
        seed_result[i] = seed_init[i];
    }

    keccak_f800(seed_result);
}
