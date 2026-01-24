#pragma once


__device__ __forceinline__
uint64_t xor5(
    uint64_t const* const arr,
    uint32_t const i)
{
    // TODO: Replace xor with lop3
    uint64_t result;
    asm
    (
        " xor.b64 %0, %1, %2;\n" // arr[i] ^ arr[i + 5u]
        " xor.b64 %0, %0, %3;\n" // result ^ arr[i + 10u]
        " xor.b64 %0, %0, %4;\n" // result ^ arr[i + 15u]
        " xor.b64 %0, %0, %5;\n" // result ^ arr[i + 20u]
        : "=l"(result)           // %0: result
        : "l"(arr[i]),           // %1: arr[i]
          "l"(arr[i + 5u]),      // %2: arr[i + 5u]
          "l"(arr[i + 10u]),     // %3: arr[i + 10u]
          "l"(arr[i + 15u]),     // %4: arr[i + 15u]
          "l"(arr[i + 20u])      // %5: arr[i + 20u]
    ); // arr[i] ^ arr[i + 5u] ^ arr[i + 10u] ^ arr[i + 15u] ^ arr[i + 20u]
    return result;
}


__device__ __forceinline__
uint32_t xor5(
    uint32_t const* const arr,
    uint32_t const i)
{
    // TODO: Replace xor with lop3
    uint32_t result;
    asm
    (
        " xor.b32 %0, %1, %2;\n" // arr[i] ^ arr[i + 5u]
        " xor.b32 %0, %0, %3;\n" // result ^ arr[i + 10u]
        " xor.b32 %0, %0, %4;\n" // result ^ arr[i + 15u]
        " xor.b32 %0, %0, %5;\n" // result ^ arr[i + 20u]
        : "=r"(result)           // %0: result
        : "r"(arr[i]),           // %1: arr[i]
          "r"(arr[i + 5u]),      // %2: arr[i + 5u]
          "r"(arr[i + 10u]),     // %3: arr[i + 10u]
          "r"(arr[i + 15u]),     // %4: arr[i + 15u]
          "r"(arr[i + 20u])      // %5: arr[i + 20u]
    ); // arr[i] ^ arr[i + 5u] ^ arr[i + 10u] ^ arr[i + 15u] ^ arr[i + 20u]
    return result;
}
