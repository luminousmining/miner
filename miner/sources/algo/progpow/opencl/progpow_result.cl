typedef struct __attribute__ ((aligned (16))) s_result
{
    bool found;
    uint count;
    ulong nonces[4];
    uint  hash[4][8];
} t_result;
