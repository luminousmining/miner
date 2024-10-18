inline
ulong xor5(
    ulong* const state,
    uint const i)
{
    return   state[i]
           ^ state[i + 5u]
           ^ state[i + 10u]
           ^ state[i + 15u]
           ^ state[i + 20u];
}
