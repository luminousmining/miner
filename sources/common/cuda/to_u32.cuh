
__device__ __forceinline__
void toU32FromU64(
    uint32_t* const hash,
    uint64_t const v0,
    uint64_t const v1)
{
    asm volatile
    (
        "mov.b64 {%0,%1},%2;\n"
        : "=r"(hash[0]),
          "=r"(hash[1])
        : "l"(v0)
    );
    asm volatile
    (
        "mov.b64 {%0,%1},%2;\n"
        : "=r"(hash[2]),
          "=r"(hash[3])
        : "l"(v1)
    );
}
