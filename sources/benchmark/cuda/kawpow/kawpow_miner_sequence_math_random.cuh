#pragma once


// Inner loop for prog_seed 321799
__device__ __forceinline__ void progPowLoop(
    const uint32_t loop,
    uint32_t mix[PROGPOW_REGS],
    const dag_t *g_dag,
    const uint32_t c_dag[PROGPOW_CACHE_WORDS],
    const bool hack_false)
{
    dag_t data_dag;
    uint32_t offset, data;
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES-1);
    // global load
    offset = SHFL(mix[0], loop%PROGPOW_LANES, PROGPOW_LANES);
    offset %= PROGPOW_DAG_ELEMENTS;
    offset = offset * PROGPOW_LANES + ((lane_id ^ loop) % PROGPOW_LANES);
    data_dag = g_dag[offset];
    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();
    // cache load 0
    offset = mix[13] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[17] = ROTR32(mix[17], 24) ^ data;
    // random math 0
    data = mix[22] | mix[13];
    mix[22] = (mix[22] ^ data) * 33;
    // cache load 1
    offset = mix[2] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[11] = ROTR32(mix[11], 14) ^ data;
    // random math 1
    data = mix[16] & mix[3];
    mix[31] = ROTL32(mix[31], 28) ^ data;
    // cache load 2
    offset = mix[30] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[28] = ROTL32(mix[28], 17) ^ data;
    // random math 2
    data = mix[30] | mix[29];
    mix[9] = ROTL32(mix[9], 27) ^ data;
    // cache load 3
    offset = mix[15] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[1] = (mix[1] * 33) + data;
    // random math 3
    data = mix[8] | mix[30];
    mix[5] = (mix[5] ^ data) * 33;
    // cache load 4
    offset = mix[14] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[8] = (mix[8] ^ data) * 33;
    // random math 4
    data = mix[1] ^ mix[29];
    mix[21] = (mix[21] * 33) + data;
    // cache load 5
    offset = mix[23] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[13] = ROTL32(mix[13], 17) ^ data;
    // random math 5
    data = mix[16] | mix[27];
    mix[30] = (mix[30] * 33) + data;
    // cache load 6
    offset = mix[7] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[14] = ROTL32(mix[14], 6) ^ data;
    // random math 6
    data = mul_hi(mix[29], mix[19]);
    mix[29] = (mix[29] * 33) + data;
    // cache load 7
    offset = mix[16] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[20] = ROTR32(mix[20], 20) ^ data;
    // random math 7
    data = ROTR32(mix[18], mix[3] % 32);
    mix[4] = (mix[4] * 33) + data;
    // cache load 8
    offset = mix[8] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[12] = ROTL32(mix[12], 18) ^ data;
    // random math 8
    data = mix[30] * mix[24];
    mix[26] = (mix[26] ^ data) * 33;
    // cache load 9
    offset = mix[28] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[6] = (mix[6] ^ data) * 33;
    // random math 9
    data = mul_hi(mix[7], mix[1]);
    mix[7] = (mix[7] * 33) + data;
    // cache load 10
    offset = mix[22] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[2] = (mix[2] * 33) + data;
    // random math 10
    data = mix[17] | mix[27];
    mix[23] = ROTR32(mix[23], 1) ^ data;
    // random math 11
    data = ROTR32(mix[4], mix[19] % 32);
    mix[19] = ROTL32(mix[19], 11) ^ data;
    // random math 12
    data = minumun(mix[17], mix[20]);
    mix[24] = (mix[24] ^ data) * 33;
    // random math 13
    data = mul_hi(mix[23], mix[15]);
    mix[10] = ROTR32(mix[10], 23) ^ data;
    // random math 14
    data = minumun(mix[23], mix[0]);
    mix[0] = ROTL32(mix[0], 10) ^ data;
    // random math 15
    data = ROTL32(mix[30], mix[28] % 32);
    mix[25] = ROTR32(mix[25], 12) ^ data;
    // random math 16
    data = mix[28] ^ mix[21];
    mix[18] = ROTR32(mix[18], 13) ^ data;
    // random math 17
    data = mix[29] & mix[31];
    mix[3] = ROTL32(mix[3], 13) ^ data;
    // consume global load data
    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();
    mix[0] = ROTL32(mix[0], 31) ^ data_dag.s[0];
    mix[27] = ROTR32(mix[27], 21) ^ data_dag.s[1];
    mix[16] = ROTL32(mix[16], 27) ^ data_dag.s[2];
    mix[15] = (mix[15] ^ data_dag.s[3]) * 33;
}
