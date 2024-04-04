#pragma once


#define PRINT_BUFFER(identifier, buffer, length)                               \
    {                                                                          \
        printf("[%s]", identifier);                                            \
        print_buffer(buffer, length);                                          \
        printf("\n");                                                          \
    }


#define THD_PRINT_BUFFER(identifier, buffer, length)                           \
    {                                                                          \
        uint32_t const tidTmp{ (blockIdx.x * blockDim.x) + threadIdx.x };      \
        thread_print_buffer(identifier, tidTmp, buffer, length);               \
    }


__forceinline__ __device__
void print_buffer(
    uint32_t const* const buffer,
    uint32_t const length)
{
    printf("(uint32_t) ->");
    for (int i = 0; i < length; ++i)
    {
        if (0u != buffer[i])
        {
            printf("%#08x%s", buffer[i], (i + 1) < length ? ", " : " ");
        }
        else
        {
            printf("0x00000000%s", (i + 1) < length ? ", " : " ");
        }
        if (i > 0 && (i + 1) % 4 == 0)
        {
            printf("\n");
        }
    }
}


__forceinline__ __device__
void print_buffer(
    uint8_t const* const buffer,
    uint32_t const length)
{
    printf("(uint8_t) ->\n");
    for (int i = 0; i < length; ++i)
    {
        if (0u != buffer[i])
        {
            printf("%#02x%s", buffer[i], (i + 1) < length ? ", " : " ");
        }
        else
        {
            printf("0x00%s", (i + 1) < length ? ", " : " ");
        }
        if (i > 0 && (i + 1) % 4 == 0)
        {
            printf("\n");
        }
    }
}


__forceinline__ __device__
void thread_print_buffer(
    char const* __restrict__ const identifier,
    uint32_t const threadId,
    uint64_t const* __restrict__ const buffer,
    uint32_t const length)
{
    for (int i = 0; i < length; ++i)
    {
        if (0 != buffer[i])
        {
            printf("[%u]-[%s][%d] -> 0x%llux\n",
                threadId,
                identifier,
                i,
                buffer[i]);
        }
        else
        {
            printf("[%u]-[%s][%d] -> 0x0000000000000000\n",
                threadId,
                identifier,
                i);
        }
    }
}


__forceinline__ __device__
void thread_print_buffer(
    char const* __restrict__ const identifier,
    uint32_t const threadId,
    uint32_t const* __restrict__ const buffer,
    uint32_t const length)
{
    for (int i = 0; i < length; ++i)
    {
        if (0 != buffer[i])
        {
            printf("[%u]-[%s][%d] -> %#08x\n",
                threadId,
                identifier,
                i,
                buffer[i]);
        }
        else
        {
            printf("[%u]-[%s][%d] -> 0x00000000\n",
                threadId,
                identifier,
                i);
        }
    }
}


__forceinline__ __device__
void thread_print_buffer(
    char const* __restrict__ const identifier,
    uint32_t const threadId,
    uint8_t const* __restrict__ const buffer,
    uint32_t const length)
{
    printf("(uint8_t) ->\n");
    for (int i = 0; i < length; ++i)
    {
        if (0 != buffer[i])
        {
            printf("[%u]-[%s][%d] -> %#02x\n",
                threadId,
                identifier,
                i,
                buffer[i]);
        }
        else
        {
            printf("[%u]-[%s][%d] -> 0x00\n",
                threadId,
                identifier,
                i);
        }
    }
}
