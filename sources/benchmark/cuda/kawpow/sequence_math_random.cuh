#pragma once


// period      = 321799
// COUNT_CACHE = 11
// COUNT_MATH  = 18
__device__ __forceinline__
void sequence_math_random(
    uint32_t const* __restrict__ const header_dag,
    uint32_t* __restrict__ const mix,
    uint4 const* __restrict__ entries)
{
    uint32_t dag_offset;
    uint32_t data;
    // iter[0] merge 3
    dag_offset = mix[13] & 4095u;
    mix[17] = ror_u32(mix[17], 24) ^ header_dag[dag_offset];
    // iter[0] sel_math 7
    data = mix[22] | mix[13];
    // iter[0] sel_merge 1
    mix[22] = (mix[22] ^ data) * 33;
    // iter[1] merge 3
    dag_offset = mix[2] & 4095u;
    mix[11] = ror_u32(mix[11], 14) ^ header_dag[dag_offset];
    // iter[1] sel_math 6
    data = mix[16] & mix[3];
    // iter[1] sel_merge 2
    mix[31] = rol_u32(mix[31], 28) ^ data;
    // iter[2] merge 2
    dag_offset = mix[30] & 4095u;
    mix[28] = rol_u32(mix[28], 17) ^ header_dag[dag_offset];
    // iter[2] sel_math 7
    data = mix[30] | mix[29];
    // iter[2] sel_merge 2
    mix[9] = rol_u32(mix[9], 27) ^ data;
    // iter[3] merge 0
    dag_offset = mix[15] & 4095u;
    mix[1] = (mix[1] * 33) + header_dag[dag_offset];
    // iter[3] sel_math 7
    data = mix[8] | mix[30];
    // iter[3] sel_merge 1
    mix[5] = (mix[5] ^ data) * 33;
    // iter[4] merge 1
    dag_offset = mix[14] & 4095u;
    mix[8] = (mix[8] ^ header_dag[dag_offset]) * 33;
    // iter[4] sel_math 8
    data = mix[1] ^ mix[29];
    // iter[4] sel_merge 0
    mix[21] = (mix[21] * 33) + data;
    // iter[5] merge 2
    dag_offset = mix[23] & 4095u;
    mix[13] = rol_u32(mix[13], 17) ^ header_dag[dag_offset];
    // iter[5] sel_math 7
    data = mix[16] | mix[27];
    // iter[5] sel_merge 0
    mix[30] = (mix[30] * 33) + data;
    // iter[6] merge 2
    dag_offset = mix[7] & 4095u;
    mix[14] = rol_u32(mix[14], 6) ^ header_dag[dag_offset];
    // iter[6] sel_math 2
    data = __umulhi(mix[29], mix[19]);
    // iter[6] sel_merge 0
    mix[29] = (mix[29] * 33) + data;
    // iter[7] merge 3
    dag_offset = mix[16] & 4095u;
    mix[20] = ror_u32(mix[20], 20) ^ header_dag[dag_offset];
    // iter[7] sel_math 5
    data = ror_u32(mix[18], mix[3]);
    // iter[7] sel_merge 0
    mix[4] = (mix[4] * 33) + data;
    // iter[8] merge 2
    dag_offset = mix[8] & 4095u;
    mix[12] = rol_u32(mix[12], 18) ^ header_dag[dag_offset];
    // iter[8] sel_math 1
    data = mix[30] * mix[24];
    // iter[8] sel_merge 1
    mix[26] = (mix[26] ^ data) * 33;
    // iter[9] merge 1
    dag_offset = mix[28] & 4095u;
    mix[6] = (mix[6] ^ header_dag[dag_offset]) * 33;
    // iter[9] sel_math 2
    data = __umulhi(mix[7], mix[1]);
    // iter[9] sel_merge 0
    mix[7] = (mix[7] * 33) + data;
    // iter[10] merge 0
    dag_offset = mix[22] & 4095u;
    mix[2] = (mix[2] * 33) + header_dag[dag_offset];
    // iter[10] sel_math 7
    data = mix[17] | mix[27];
    // iter[10] sel_merge 3
    mix[23] = ror_u32(mix[23], 1) ^ data;
    // iter[11] sel_math 5
    data = ror_u32(mix[4], mix[19]);
    // iter[11] sel_merge 2
    mix[19] = rol_u32(mix[19], 11) ^ data;
    // iter[12] sel_math 3
    data = min(mix[17], mix[20]);
    // iter[12] sel_merge 1
    mix[24] = (mix[24] ^ data) * 33;
    // iter[13] sel_math 2
    data = __umulhi(mix[23], mix[15]);
    // iter[13] sel_merge 3
    mix[10] = ror_u32(mix[10], 23) ^ data;
    // iter[14] sel_math 3
    data = min(mix[23], mix[0]);
    // iter[14] sel_merge 2
    mix[0] = rol_u32(mix[0], 10) ^ data;
    // iter[15] sel_math 4
    data = rol_u32(mix[30], mix[28]);
    // iter[15] sel_merge 3
    mix[25] = ror_u32(mix[25], 12) ^ data;
    // iter[16] sel_math 8
    data = mix[28] ^ mix[21];
    // iter[16] sel_merge 3
    mix[18] = ror_u32(mix[18], 13) ^ data;
    // iter[17] sel_math 6
    data = mix[29] & mix[31];
    // iter[17] sel_merge 2
    mix[3] = rol_u32(mix[3], 13) ^ data;
    // iter[0] merge_entries 2
    mix[0] = rol_u32(mix[0], 31) ^ entries->x;
    // iter[1] merge_entries 3
    mix[27] = ror_u32(mix[27], 21) ^ entries->y;
    // iter[2] merge_entries 2
    mix[16] = rol_u32(mix[16], 27) ^ entries->z;
    // iter[3] merge_entries 1
    mix[15] = (mix[15] ^ entries->w) * 33;
}


// period      = 321799
// COUNT_CACHE = 11
// COUNT_MATH  = 18
__device__ __forceinline__
void sequence_math_random_cache_only(
    uint32_t const* __restrict__ const header_dag,
    uint32_t* __restrict__ const mix,
    uint4 const* __restrict__ entries)
{
    uint32_t dag_offset;
    uint32_t data;
    // iter[0] merge 3
    dag_offset = mix[13] & 4095u;
    mix[17] = ror_u32(mix[17], 24) ^ __ldg(&header_dag[dag_offset]);
    // iter[0] sel_math 7
    data = mix[22] | mix[13];
    // iter[0] sel_merge 1
    mix[22] = (mix[22] ^ data) * 33;
    // iter[1] merge 3
    dag_offset = mix[2] & 4095u;
    mix[11] = ror_u32(mix[11], 14) ^ __ldg(&header_dag[dag_offset]);
    // iter[1] sel_math 6
    data = mix[16] & mix[3];
    // iter[1] sel_merge 2
    mix[31] = rol_u32(mix[31], 28) ^ data;
    // iter[2] merge 2
    dag_offset = mix[30] & 4095u;
    mix[28] = rol_u32(mix[28], 17) ^ __ldg(&header_dag[dag_offset]);
    // iter[2] sel_math 7
    data = mix[30] | mix[29];
    // iter[2] sel_merge 2
    mix[9] = rol_u32(mix[9], 27) ^ data;
    // iter[3] merge 0
    dag_offset = mix[15] & 4095u;
    mix[1] = (mix[1] * 33) + __ldg(&header_dag[dag_offset]);
    // iter[3] sel_math 7
    data = mix[8] | mix[30];
    // iter[3] sel_merge 1
    mix[5] = (mix[5] ^ data) * 33;
    // iter[4] merge 1
    dag_offset = mix[14] & 4095u;
    mix[8] = (mix[8] ^ __ldg(&header_dag[dag_offset])) * 33;
    // iter[4] sel_math 8
    data = mix[1] ^ mix[29];
    // iter[4] sel_merge 0
    mix[21] = (mix[21] * 33) + data;
    // iter[5] merge 2
    dag_offset = mix[23] & 4095u;
    mix[13] = rol_u32(mix[13], 17) ^ __ldg(&header_dag[dag_offset]);
    // iter[5] sel_math 7
    data = mix[16] | mix[27];
    // iter[5] sel_merge 0
    mix[30] = (mix[30] * 33) + data;
    // iter[6] merge 2
    dag_offset = mix[7] & 4095u;
    mix[14] = rol_u32(mix[14], 6) ^ __ldg(&header_dag[dag_offset]);
    // iter[6] sel_math 2
    data = __umulhi(mix[29], mix[19]);
    // iter[6] sel_merge 0
    mix[29] = (mix[29] * 33) + data;
    // iter[7] merge 3
    dag_offset = mix[16] & 4095u;
    mix[20] = ror_u32(mix[20], 20) ^ __ldg(&header_dag[dag_offset]);
    // iter[7] sel_math 5
    data = ror_u32(mix[18], mix[3]);
    // iter[7] sel_merge 0
    mix[4] = (mix[4] * 33) + data;
    // iter[8] merge 2
    dag_offset = mix[8] & 4095u;
    mix[12] = rol_u32(mix[12], 18) ^ __ldg(&header_dag[dag_offset]);
    // iter[8] sel_math 1
    data = mix[30] * mix[24];
    // iter[8] sel_merge 1
    mix[26] = (mix[26] ^ data) * 33;
    // iter[9] merge 1
    dag_offset = mix[28] & 4095u;
    mix[6] = (mix[6] ^ __ldg(&header_dag[dag_offset])) * 33;
    // iter[9] sel_math 2
    data = __umulhi(mix[7], mix[1]);
    // iter[9] sel_merge 0
    mix[7] = (mix[7] * 33) + data;
    // iter[10] merge 0
    dag_offset = mix[22] & 4095u;
    mix[2] = (mix[2] * 33) + __ldg(&header_dag[dag_offset]);
    // iter[10] sel_math 7
    data = mix[17] | mix[27];
    // iter[10] sel_merge 3
    mix[23] = ror_u32(mix[23], 1) ^ data;
    // iter[11] sel_math 5
    data = ror_u32(mix[4], mix[19]);
    // iter[11] sel_merge 2
    mix[19] = rol_u32(mix[19], 11) ^ data;
    // iter[12] sel_math 3
    data = min(mix[17], mix[20]);
    // iter[12] sel_merge 1
    mix[24] = (mix[24] ^ data) * 33;
    // iter[13] sel_math 2
    data = __umulhi(mix[23], mix[15]);
    // iter[13] sel_merge 3
    mix[10] = ror_u32(mix[10], 23) ^ data;
    // iter[14] sel_math 3
    data = min(mix[23], mix[0]);
    // iter[14] sel_merge 2
    mix[0] = rol_u32(mix[0], 10) ^ data;
    // iter[15] sel_math 4
    data = rol_u32(mix[30], mix[28]);
    // iter[15] sel_merge 3
    mix[25] = ror_u32(mix[25], 12) ^ data;
    // iter[16] sel_math 8
    data = mix[28] ^ mix[21];
    // iter[16] sel_merge 3
    mix[18] = ror_u32(mix[18], 13) ^ data;
    // iter[17] sel_math 6
    data = mix[29] & mix[31];
    // iter[17] sel_merge 2
    mix[3] = rol_u32(mix[3], 13) ^ data;
    // iter[0] merge_entries 2
    mix[0] = rol_u32(mix[0], 31) ^ entries->x;
    // iter[1] merge_entries 3
    mix[27] = ror_u32(mix[27], 21) ^ entries->y;
    // iter[2] merge_entries 2
    mix[16] = rol_u32(mix[16], 27) ^ entries->z;
    // iter[3] merge_entries 1
    mix[15] = (mix[15] ^ entries->w) * 33;
}
