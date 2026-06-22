typedef struct __attribute__((aligned(8)))
{
    uint  found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;


inline
void publishHit(__global Result* result, ulong const nonce)
{
    uint const idx = atomic_inc(&result->count);
    result->found = 1;
    if (idx < MAX_RESULT)
    {
        result->nonces[idx] = nonce;
    }
}
