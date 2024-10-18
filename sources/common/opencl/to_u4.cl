inline
uint4 toU4(
    ulong const a,
    ulong const b)
{
    uint4 v;

    v.x = a;
    v.y = a >> 32;
    v.z = b;
    v.w = b >> 32;

    return v;
}
