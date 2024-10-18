#define reverseBytesInt(input,output)                                          \
{                                                                              \
    void * p = &input;                                                         \
    uchar4 bytesr = ((uchar4 *)p)[0].wzyx;                                     \
    output = *((uint *)&bytesr);                                               \
}


#define fn_Add(Val1, Val2, cv, Result,ret)                                     \
{                                                                              \
    ulong tmp = (ulong)Val1 + (ulong)Val2 + (ulong)cv;                         \
    Result = tmp;                                                              \
    ret = tmp >> 32;                                                           \
}


__constant
ulong ivals[8] =
{
    0x6A09E667F2BDC928, 0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};
