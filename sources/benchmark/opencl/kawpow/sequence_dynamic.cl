// period 321799
inline
void sequence_dynamic(
    uint const* const restrict header_dag, 
    uint* const restrict hash, 
    uint4 entries)
{
        uint dag_offset;
        uint data;
        // iter[0] merge 3
        dag_offset = hash[13] & 4095u;
        hash[17] = ror_u32(hash[17], 24) ^ header_dag[dag_offset];
        // iter[0] sel_math 7
        data = hash[22] | hash[13];
        // iter[0] sel_merge 1
        hash[22] = (hash[22] ^ data) * 33;
        // iter[1] merge 3
        dag_offset = hash[2] & 4095u;
        hash[11] = ror_u32(hash[11], 14) ^ header_dag[dag_offset];
        // iter[1] sel_math 6
        data = hash[16] & hash[3];
        // iter[1] sel_merge 2
        hash[31] = rol_u32(hash[31], 28) ^ data;
        // iter[2] merge 2
        dag_offset = hash[30] & 4095u;
        hash[28] = rol_u32(hash[28], 17) ^ header_dag[dag_offset];
        // iter[2] sel_math 7
        data = hash[30] | hash[29];
        // iter[2] sel_merge 2
        hash[9] = rol_u32(hash[9], 27) ^ data;
        // iter[3] merge 0
        dag_offset = hash[15] & 4095u;
        hash[1] = (hash[1] * 33) + header_dag[dag_offset];
        // iter[3] sel_math 7
        data = hash[8] | hash[30];
        // iter[3] sel_merge 1
        hash[5] = (hash[5] ^ data) * 33;
        // iter[4] merge 1
        dag_offset = hash[14] & 4095u;
        hash[8] = (hash[8] ^ header_dag[dag_offset]) * 33;
        // iter[4] sel_math 8
        data = hash[1] ^ hash[29];
        // iter[4] sel_merge 0
        hash[21] = (hash[21] * 33) + data;
        // iter[5] merge 2
        dag_offset = hash[23] & 4095u;
        hash[13] = rol_u32(hash[13], 17) ^ header_dag[dag_offset];
        // iter[5] sel_math 7
        data = hash[16] | hash[27];
        // iter[5] sel_merge 0
        hash[30] = (hash[30] * 33) + data;
        // iter[6] merge 2
        dag_offset = hash[7] & 4095u;
        hash[14] = rol_u32(hash[14], 6) ^ header_dag[dag_offset];
        // iter[6] sel_math 2
        data = mul_hi(hash[29], hash[19]);
        // iter[6] sel_merge 0
        hash[29] = (hash[29] * 33) + data;
        // iter[7] merge 3
        dag_offset = hash[16] & 4095u;
        hash[20] = ror_u32(hash[20], 20) ^ header_dag[dag_offset];
        // iter[7] sel_math 5
        data = ror_u32(hash[18], hash[3]);
        // iter[7] sel_merge 0
        hash[4] = (hash[4] * 33) + data;
        // iter[8] merge 2
        dag_offset = hash[8] & 4095u;
        hash[12] = rol_u32(hash[12], 18) ^ header_dag[dag_offset];
        // iter[8] sel_math 1
        data = hash[30] * hash[24];
        // iter[8] sel_merge 1
        hash[26] = (hash[26] ^ data) * 33;
        // iter[9] merge 1
        dag_offset = hash[28] & 4095u;
        hash[6] = (hash[6] ^ header_dag[dag_offset]) * 33;
        // iter[9] sel_math 2
        data = mul_hi(hash[7], hash[1]);
        // iter[9] sel_merge 0
        hash[7] = (hash[7] * 33) + data;
        // iter[10] merge 0
        dag_offset = hash[22] & 4095u;
        hash[2] = (hash[2] * 33) + header_dag[dag_offset];
        // iter[10] sel_math 7
        data = hash[17] | hash[27];
        // iter[10] sel_merge 3
        hash[23] = ror_u32(hash[23], 1) ^ data;
        // iter[11] sel_math 5
        data = ror_u32(hash[4], hash[19]);
        // iter[11] sel_merge 2
        hash[19] = rol_u32(hash[19], 11) ^ data;
        // iter[12] sel_math 3
        data = min(hash[17], hash[20]);
        // iter[12] sel_merge 1
        hash[24] = (hash[24] ^ data) * 33;
        // iter[13] sel_math 2
        data = mul_hi(hash[23], hash[15]);
        // iter[13] sel_merge 3
        hash[10] = ror_u32(hash[10], 23) ^ data;
        // iter[14] sel_math 3
        data = min(hash[23], hash[0]);
        // iter[14] sel_merge 2
        hash[0] = rol_u32(hash[0], 10) ^ data;
        // iter[15] sel_math 4
        data = rol_u32(hash[30], hash[28]);
        // iter[15] sel_merge 3
        hash[25] = ror_u32(hash[25], 12) ^ data;
        // iter[16] sel_math 8
        data = hash[28] ^ hash[21];
        // iter[16] sel_merge 3
        hash[18] = ror_u32(hash[18], 13) ^ data;
        // iter[17] sel_math 6
        data = hash[29] & hash[31];
        // iter[17] sel_merge 2
        hash[3] = rol_u32(hash[3], 13) ^ data;
        // iter[0] merge_entries 2
        hash[0] = rol_u32(hash[0], 31) ^ entries.x;
        // iter[1] merge_entries 3
        hash[27] = ror_u32(hash[27], 21) ^ entries.y;
        // iter[2] merge_entries 2
        hash[16] = rol_u32(hash[16], 27) ^ entries.z;
        // iter[3] merge_entries 1
        hash[15] = (hash[15] ^ entries.w) * 33;
}
