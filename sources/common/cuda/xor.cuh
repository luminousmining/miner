#pragma once


__device__ __forceinline__
uint64_t xor5(
    uint64_t const* const arr,
    uint32_t const i)
{
    uint64_t result;
    asm
    (
        "  lop3.b64 %0, %1, %2, %3, 0x96;" // 0x96 = XOR: arr[i] ^ arr[i+5] ^ arr[i+10]
        "  lop3.b64 %0, %0, %4, %5, 0x96;" // 0x96 = XOR: result ^ arr[i+15] ^ arr[i+20]
        : "=l"(result)       // %0
        : "l"(arr[i]),       // %1
          "l"(arr[i + 5u]),  // %2
          "l"(arr[i + 10u]), // %3
          "l"(arr[i + 15u]), // %4
          "l"(arr[i + 20u])  // %5
    ); // arr[i] ^ arr[i + 5u] ^ arr[i + 10u] ^ arr[i + 15u] ^ arr[i + 20u]

    return result;

    // return arr[i]
    //      ^ arr[i + 5u]
    //      ^ arr[i + 10u]
    //      ^ arr[i + 15u]
    //      ^ arr[i + 20u];
}


__device__ __forceinline__
uint64_t xor5(
    uint32_t const* const arr,
    uint32_t const i)
{
    return arr[i]
         ^ arr[i + 5u]
         ^ arr[i + 10u]
         ^ arr[i + 15u]
         ^ arr[i + 20u];
}
