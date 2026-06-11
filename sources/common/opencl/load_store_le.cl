inline
ulong load_le_u64(uchar const* p)
{
    ulong v = 0;
    for (int b = 0; b < 8; ++b)
    {
        v |= ((ulong)p[b]) << (8 * b);
    }
    return v;
}


inline
void store_le_u256(ulong const* state, uchar* out)
{
    for (int w = 0; w < 4; ++w)
    {
        for (int b = 0; b < 8; ++b)
        {
            out[w * 8 + b] = (uchar)((state[w] >> (8 * b)) & 0xFF);
        }
    }
}
