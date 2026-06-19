// Result buffer shared with the host (mirrors algo::kheavyhash::Result in
// sources/algo/kheavyhash/result.hpp). MAX_RESULT is overridable by the host
// kernel generator (addDefine); the fallback keeps standalone builds valid.
#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

typedef struct __attribute__((aligned(8)))
{
    uchar found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;


inline void publishHit(__global Result* result, ulong const nonce)
{
    uint const idx = atomic_inc(&result->count);
    result->found = 1;
    if (idx < MAX_RESULT)
    {
        result->nonces[idx] = nonce;
    }
}
