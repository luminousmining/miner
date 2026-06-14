inline
ulong rol_u64(
    ulong const value,
    uint const offset)
{
    return (value << offset) | (value >> (64 - offset));
}


inline
ulong ror_u64(
    ulong const value,
    uint const offset)
{
    return ((value >> offset) ^ (value << (64 - offset)));
}


inline
uint rol_u32(uint x, uint n)
{
    return rotate((x), (uint)(n));
}


inline
uint ror_u32(uint x, uint n)
{
    return rotate((x), (uint)(32 - n));
}


inline
uint bswap32(uint const x)
{
    return as_uint(as_uchar4(x).s3210);
}


inline
ulong bswap64(ulong const x)
{
    return as_ulong(as_uchar8(x).s76543210);
}
