#define FNV1_OFFSET 0x811c9dc5
#define FNV1_PRIME 0x01000193u


inline
uint fnv1_u32(
    uint const u,
    uint const v)
{
    return (u * 0x01000193u) ^ v;
}

inline
void fnv1_v4_from_v4(
    uint4* const src,
    uint4 const* const from)
{
    src->x = fnv1_u32(src->x, from->x);
    src->y = fnv1_u32(src->y, from->y);
    src->z = fnv1_u32(src->z, from->z);
    src->w = fnv1_u32(src->w, from->w);
}


inline
void fnv1_u4(
    uint4* const restrict dst,
    uint4 const src)
{
    dst->x = fnv1_u32(dst->x, src.x);
    dst->y = fnv1_u32(dst->y, src.y);
    dst->z = fnv1_u32(dst->z, src.z);
    dst->w = fnv1_u32(dst->w, src.w);
}


inline
uint fnv1_reduce_u4(
    uint4 const* const v)
{
    return fnv1_u32(fnv1_u32(fnv1_u32(v->x, v->y), v->z), v->w);
}


inline
uint4 fnv1_concat(
    uint4 const* const v1,
    uint4 const* const v2,
    uint4 const* const v3,
    uint4 const* const v4)
{
    uint4 r;
    r.x = fnv1_reduce_u4(v1);
    r.y = fnv1_reduce_u4(v2);
    r.z = fnv1_reduce_u4(v3);
    r.w = fnv1_reduce_u4(v4);
    return r;
}


inline
uint fnv1a_u32(uint u, uint v)
{
    return (u ^ v) * FNV1_PRIME;
}
