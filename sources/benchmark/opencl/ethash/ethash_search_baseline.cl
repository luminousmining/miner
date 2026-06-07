#include "kernel/common/rotate_byte.cl"
#include "kernel/common/to_u4.cl"
#include "kernel/common/xor.cl"
#include "kernel/crypto/fnv1.cl"
#include "kernel/ethash/ethash_result.cl"


__constant
ulong ETHASH_KECCAK_ROUND[LEN_KECCAK] =
{
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};


inline
void ethash_keccak_f1600_round(
    ulong* const state,
    uint const round)
{
    ulong value;
    ulong C[5];
    ulong D[5];
    ulong tmp[25];

    // THETA
    C[0] = xor5(state, 0u);
    C[1] = xor5(state, 1u);
    C[2] = xor5(state, 2u);
    C[3] = xor5(state, 3u);
    C[4] = xor5(state, 4u);

    D[0] = rol_u64(C[0], 1u);
    D[1] = rol_u64(C[1], 1u);
    D[2] = rol_u64(C[2], 1u);
    D[3] = rol_u64(C[3], 1u);
    D[4] = rol_u64(C[4], 1u);

    value = D[1] ^ C[4];
    state[0] ^= value;
    state[5] ^= value;
    state[10] ^= value;
    state[15] ^= value;
    state[20] ^= value;

    value = D[2] ^ C[0];
    state[1] ^= value;
    state[6] ^= value;
    state[11] ^= value;
    state[16] ^= value;
    state[21] ^= value;

    value = D[3] ^ C[1];
    state[2] ^= value;
    state[7] ^= value;
    state[12] ^= value;
    state[17] ^= value;
    state[22] ^= value;

    value = D[4] ^ C[2];
    state[3] ^= value;
    state[8] ^= value;
    state[13] ^= value;
    state[18] ^= value;
    state[23] ^= value;

    value = D[0] ^ C[3];
    state[4] ^= value;
    state[9] ^= value;
    state[14] ^= value;
    state[19] ^= value;
    state[24] ^= value;

    tmp[1]  = rol_u64(state[1],  1u);
    tmp[2]  = rol_u64(state[2],  62u);
    tmp[3]  = rol_u64(state[3],  28u);
    tmp[4]  = rol_u64(state[4],  27u);
    tmp[5]  = rol_u64(state[5],  36u);
    tmp[6]  = rol_u64(state[6],  44u);
    tmp[7]  = rol_u64(state[7],  6u);
    tmp[8]  = rol_u64(state[8],  55u);
    tmp[9]  = rol_u64(state[9],  20u);
    tmp[10] = rol_u64(state[10], 3u);
    tmp[11] = rol_u64(state[11], 10u);
    tmp[12] = rol_u64(state[12], 43u);
    tmp[13] = rol_u64(state[13], 25u);
    tmp[14] = rol_u64(state[14], 39u);
    tmp[15] = rol_u64(state[15], 41u);
    tmp[16] = rol_u64(state[16], 45u);
    tmp[17] = rol_u64(state[17], 15u);
    tmp[18] = rol_u64(state[18], 21u);
    tmp[19] = rol_u64(state[19], 8u);
    tmp[20] = rol_u64(state[20], 18u);
    tmp[21] = rol_u64(state[21], 2u);
    tmp[22] = rol_u64(state[22], 61u);
    tmp[23] = rol_u64(state[23], 56u);
    tmp[24] = rol_u64(state[24], 14u);

    // PI
    state[0] = tmp[0];
    state[16] = tmp[5];
    state[7] = tmp[10];
    state[23] = tmp[15];
    state[14] = tmp[20];

    state[10] = tmp[1];
    state[1] = tmp[6];
    state[17] = tmp[11];
    state[8] = tmp[16];
    state[24] = tmp[21];

    state[20] = tmp[2];
    state[11] = tmp[7];
    state[2] = tmp[12];
    state[18] = tmp[17];
    state[9] = tmp[22];

    state[5] = tmp[3];
    state[21] = tmp[8];
    state[12] = tmp[13];
    state[3] = tmp[18];
    state[19] = tmp[23];

    state[15] = tmp[4];
    state[6] = tmp[9];
    state[22] = tmp[14];
    state[13] = tmp[19];
    state[4] = tmp[24];

    // CHI
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 5u; ++i)
    {
        uint const j = i * 5u;
        C[0] = state[j]      ^ ((~state[j + 1u]) & state[j + 2u]);
        C[1] = state[j + 1u] ^ ((~state[j + 2u]) & state[j + 3u]);
        C[2] = state[j + 2u] ^ ((~state[j + 3u]) & state[j + 4u]);
        C[3] = state[j + 3u] ^ ((~state[j + 4u]) & state[j]);
        C[4] = state[j + 4u] ^ ((~state[j])      & state[j + 1u]);

        state[j]      = C[0];
        state[j + 1u] = C[1];
        state[j + 2u] = C[2];
        state[j + 3u] = C[3];
        state[j + 4u] = C[4];
    }

    // IOTA
    state[0] ^= ETHASH_KECCAK_ROUND[round];
}


inline
void ethash_seed_last_round_keccak_f1600(
    ulong* const state)
{
    // theta
    ulong C[5];
    ulong D[5];

    // THETA
    C[0] = xor5(state, 0u);
    C[1] = xor5(state, 1u);
    C[2] = xor5(state, 2u);
    C[3] = xor5(state, 3u);
    C[4] = xor5(state, 4u);

    D[0] = rol_u64(C[0], 1u);
    D[1] = rol_u64(C[1], 1u);
    D[2] = rol_u64(C[2], 1u);
    D[3] = rol_u64(C[3], 1u);
    D[4] = rol_u64(C[4], 1u);

    state[0] ^= D[1] ^ C[4];
    state[10] ^= D[1] ^ C[4];

    state[6] ^= D[2] ^ C[0];
    state[16] ^= D[2] ^ C[0];

    state[12] ^= D[3] ^ C[1];
    state[22] ^= D[3] ^ C[1];

    state[3] ^= D[4] ^ C[2];
    state[18] ^= D[4] ^ C[2];

    state[9] ^= D[0] ^ C[3];
    state[24] ^= D[0] ^ C[3];

    // rho pi
    state[1] = rol_u64(state[6],  44u);
    state[6] = rol_u64(state[9],  20u);
    state[9] = rol_u64(state[22], 61u);
    state[2] = rol_u64(state[12], 43u);
    state[4] = rol_u64(state[24], 14u);
    state[8] = rol_u64(state[16], 45u);
    state[5] = rol_u64(state[3],  28u);
    state[3] = rol_u64(state[18], 21u);
    state[7] = rol_u64(state[10], 3u);

    // chi
    ulong const f = state[0];
    ulong const s = state[1];
    state[0] = state[0] ^ ((~state[1]) & state[2]);
    state[1] = state[1] ^ ((~state[2]) & state[3]);
    state[2] = state[2] ^ ((~state[3]) & state[4]);
    state[3] = state[3] ^ ((~state[4]) & f);
    state[4] = state[4] ^ ((~f)        & s);
    state[5] = state[5] ^ ((~state[6]) & state[7]);
    state[6] = state[6] ^ ((~state[7]) & state[8]);
    state[7] = state[7] ^ ((~state[8]) & state[9]);

    // iota
    state[0] ^= ETHASH_KECCAK_ROUND[MAX_KECCAK_ROUND];
}


inline
void keccak_f1600_first(
    ulong* const state)
{
    __attribute__((opencl_unroll_hint))
    for (uint round = 0u; round < MAX_KECCAK_ROUND; ++round)
    {
        ethash_keccak_f1600_round(state, round);
    }
    ethash_seed_last_round_keccak_f1600(state);
}


inline
void keccak_f1600_final_round(
    ulong* const state)
{
    ulong tmp[5];

    // theta
    tmp[0] = xor5(state, 0u);
    tmp[1] = xor5(state, 1u);
    tmp[2] = xor5(state, 2u);
    tmp[3] = xor5(state, 3u);
    tmp[4] = xor5(state, 4u);

    state[0]  = state[0]  ^ tmp[4] ^ rol_u64(tmp[1], 1u);
    state[6]  = state[6]  ^ tmp[0] ^ rol_u64(tmp[2], 1u);
    state[12] = state[12] ^ tmp[1] ^ rol_u64(tmp[3], 1u);

    // rho
    state[1] = rol_u64(state[6], 44u);
    state[2] = rol_u64(state[12], 43u);

    //chi
    state[0] = state[0] ^ ((~state[1]) & state[2]);

    // iota
    state[0] ^= ETHASH_KECCAK_ROUND[MAX_KECCAK_ROUND];
}


inline
void keccak_f1600_final(
    ulong* const restrict state)
{
    state[12] = 1ul;
    state[13] = 0ul;
    state[14] = 0ul;
    state[15] = 0ul;
    state[16] = 0x8000000000000000ul;

    __attribute__((opencl_unroll_hint))
    for (uint i = 17u; i < LEN_STATE; ++i)
    {
        state[i] = 0ul;
    }

    __attribute__((opencl_unroll_hint))
    for (uint round = 0u; round < MAX_KECCAK_ROUND; ++round)
    {
        ethash_keccak_f1600_round(state, round);
    }
    keccak_f1600_final_round(state);
}


inline
void build_seed(
    __constant ulong const* const restrict header,
    ulong* const restrict state,
    uint4* const restrict seed,
    ulong nonce)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 4u; ++i)
    {
        state[i] = header[i];
    }

    state[4] = nonce;
    state[5] = 1ul;
    state[6] = 0ul;
    state[7] = 0ul;
    state[8] = 0x8000000000000000ul;

    __attribute__((opencl_unroll_hint))
    for (uint i = 9u; i < 25u; ++i)
    {
        state[i] = 0ul;
    }

    keccak_f1600_first(state);
    seed[0] = toU4(state[0], state[1]);
    seed[1] = toU4(state[2], state[3]);
    seed[2] = toU4(state[4], state[5]);
    seed[3] = toU4(state[6], state[7]);
}


inline
void load_item(
    __global uint4 const* const restrict dag,
    uint4* const restrict matrix,
    uint const word0,
    size_t const thread_lane_id,
    uint const index_gap,
    uint const value_reference)
{
    uint start_index;
    start_index = fnv1_u32(index_gap ^ word0, value_reference);
    start_index %= DAG_NUMBER_ITEM;
    start_index *= 8u;

    uint4 const item = dag[start_index + thread_lane_id];
    fnv1_u4(matrix, item);
}


#define __LOAD_ITEM(value_reference, gap)           \
    if (index_lane_id == thread_lane_id)            \
    {                                               \
        swapper[thread_group_id] = value_reference; \
    }                                               \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    load_item(                                      \
        dag,                                        \
        &matrix,                                    \
        word0,                                      \
        thread_lane_id,                             \
        index_gap + gap,                            \
        swapper[thread_group_id]);


__kernel
void ethash_search(
    __global uint4 const* const restrict dag,
    __global t_result* const restrict result,
    __constant ulong const* const restrict header,
    ulong start_nonce,
    ulong boundary)
{
    __local uint swapper[LEN_SWAPPER];
    __local uint4 hashes[LEN_HASHES];
    __local uint words0[LEN_WORD0];
    __local uint reduces[LEN_REDUCE];

    uint4 seed[LEN_SEED];
    ulong state[LEN_STATE];

    // Indique l'index du thread actuel.
    size_t const thread_id = get_global_id(1) * GROUP_SIZE + get_global_id(0);

    // On troncate l'ID pour avoir GROUP_SIZE en MAX.
    size_t const thread_id_max = thread_id % GROUP_SIZE;

    // Indique la LANE du thread
    size_t const thread_lane_id = thread_id_max % LANE_PARALLEL;

    // Indique le group au quel appartient le thread.
    size_t const thread_group_id = thread_id_max / LANE_PARALLEL;

    // Indique l'index du group de @hash au quel ce thread appartient.
    size_t const index_group_hash = thread_group_id * LEN_SEED;

    // Indique l'index de @hash a prendre en compte pour ce thread.
    size_t const index_lane_hash = index_group_hash + (thread_lane_id % LEN_SEED);

    // Indique l'index de @word0 a prendre en compte pour ce thread.
    size_t const index_group_word0 = thread_group_id * LANE_PARALLEL;

    // Indique l'index de @reduce a utiliser pour ce thread.
    size_t const index_reduce = thread_group_id * LANE_PARALLEL;

    // Indique la valeur du nonce que l'on calcule pour ce thread.
    ulong const nonce = start_nonce + thread_id;

    // On creer le @seed utiliser par ce thread.
    build_seed(header, state, seed, nonce);

    // On assign le word0 correspond a son thread / nonce.
    words0[thread_id_max] = seed[0].x;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(1)))
    for (size_t lane_id = 0ul; lane_id < LANE_PARALLEL; ++lane_id)
    {
        // @lane_id correspond a notre thread lane.
        // Alors on set la variable local @hash a partir de nos donne local.
        if (lane_id == thread_lane_id)
        {
            __attribute__((opencl_unroll_hint))
            for (uint i = 0u; i < LEN_SEED; ++i)
            {
                uint const gap = index_group_hash + i;
                hashes[gap].x = seed[i].x;
                hashes[gap].y = seed[i].y;
                hashes[gap].z = seed[i].z;
                hashes[gap].w = seed[i].w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        uint4 matrix = hashes[index_lane_hash];

        // On recupere le word0 de la LANE que l'on veux calculer.
        uint const index_word0 = index_group_word0 + lane_id;
        uint const word0 = words0[index_word0];

        // On charge les items du @dag et on le hash avec @matrix.
        __attribute__((opencl_unroll_hint(1)))
        for (uint index_loop = 0u; index_loop < 16u; ++index_loop)
        {
            uint const index_gap = index_loop * 4u;
            uint const index_lane_id = index_loop % LANE_PARALLEL;
            __LOAD_ITEM(matrix.x, 0u);
            __LOAD_ITEM(matrix.y, 1u);
            __LOAD_ITEM(matrix.z, 2u);
            __LOAD_ITEM(matrix.w, 3u);
        }

        // On reduit la matrice @matrix en un uint32
        reduces[thread_id_max] = fnv1_reduce_u4(&matrix);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane_id == thread_lane_id)
        {
            uint4 shuffle_1;
            shuffle_1.x = reduces[index_reduce];
            shuffle_1.y = reduces[index_reduce + 1u];
            shuffle_1.z = reduces[index_reduce + 2u];
            shuffle_1.w = reduces[index_reduce + 3u];
    
            uint4 shuffle_2;
            shuffle_2.x = reduces[index_reduce + 4u];
            shuffle_2.y = reduces[index_reduce + 5u];
            shuffle_2.z = reduces[index_reduce + 6u];
            shuffle_2.w = reduces[index_reduce + 7u];

            state[8]  = (((ulong)shuffle_1.y) << 32) | shuffle_1.x;
            state[9]  = (((ulong)shuffle_1.w) << 32) | shuffle_1.z;
            state[10] = (((ulong)shuffle_2.y) << 32) | shuffle_2.x;
            state[11] = (((ulong)shuffle_2.w) << 32) | shuffle_2.z;
        }
    }

    // On effectue le dernier SHA-256 sur @state.
    keccak_f1600_final(state);

    // On check si on a une reponse potentiel.
    ulong const bytes_result = as_ulong(as_uchar8(state[0]).s76543210);
    if (bytes_result <= boundary)
    {
        result->found = true;
        uint const index = atomic_inc(&result->count);
        if (4u > index)
        {
            result->nonces[index] = nonce;
        }
    }
}
