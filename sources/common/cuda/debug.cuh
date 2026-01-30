#pragma once

#include <stdio.h>



#define PRINT_BUFFER(identifier, buffer, length)                               \
    {                                                                          \
        printf("[%s]", identifier);                                            \
        print_buffer(buffer, length);                                          \
        printf("\n");                                                          \
    }


#define THD_PRINT_BUFFER(identifier, buffer, length)                           \
    {                                                                          \
        uint32_t const debug_tid{ (blockIdx.x * blockDim.x) + threadIdx.x };   \
        thread_print_buffer(identifier, debug_tid, buffer, length);            \
    }


#define PRINT_TRACE(thread_id_target)                                          \
    {                                                                          \
        printf                                                                 \
        (                                                                      \
            "[%s][%d]: tid(%u)\n",                                             \
            __FUNCTION__,                                                      \
            __LINE__,                                                          \
            thread_id_target                                                   \
        );                                                                     \
    }


#define PRINT_TRACE_IF(thread_id_target)                                       \
    {                                                                          \
        uint32_t const debug_tid{ (blockIdx.x * blockDim.x) + threadIdx.x };   \
        if (debug_tid == thread_id_target)                                     \
        {                                                                      \
            PRINT_TRACE(debug_tid);                                            \
        }                                                                      \
    }


template<typename T>
__forceinline__ __device__
void print_buffer(
    T const* const buffer,
    uint32_t const length)
{
    ///////////////////////////////////////////////////////////////////////////
    if constexpr (std::is_floating_point_v<T>)
    {
        printf("(float %zubits) ->\n", sizeof(T) * 8);
    }
    else if constexpr (sizeof(T) == 8)
    {
        printf("(64bits) ->\n");
    }
    else if constexpr (sizeof(T) == 4)
    {
        printf("(32bits) ->\n");
    }
    else if constexpr (sizeof(T) == 2)
    {
        printf("(16bits) ->\n");
    }
    else if constexpr (sizeof(T) == 1)
    {
        printf("(8bits) ->\n");
    }

    ///////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0u; i < length; ++i)
    {
        if (0 != buffer[i])
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                printf
                (
                    "%f%s",
                    (double)buffer[i],
                    (i + 1u) < length ? ", " : " "
                );
            }
            else
            {
                printf
                (
                    "0x%0*llx%s",
                    (int32_t)(sizeof(T) * 2),
                    (uint64_t)buffer[i],
                    (i + 1u) < length ? ", " : " "
                );
            }
        }
        else
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                printf
                (
                    "0.0%s",
                    (i + 1u) < length ? ", " : " "
                );
            }
            else
            {
                printf
                (
                    "0x%0*llx%s",
                    (int32_t)(sizeof(T) * 2),
                    0ull,
                    (i + 1u) < length ? ", " : " "
                );
            }
        }

        if (i > 0 && (i + 1) % 4 == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}


template<typename T>
__forceinline__ __device__
void thread_print_buffer(
    char const* __restrict__ const identifier,
    uint32_t const threadId,
    T const* __restrict__ const buffer,
    uint32_t const length)
{
    for (uint32_t i = 0u; i < length; ++i)
    {
        if (0 != buffer[i])
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                printf
                (
                    "[%u]-[%s][%d] -> %f\n",
                    threadId,
                    identifier,
                    i,
                    (double)buffer[i]
                );
            }
            else if constexpr (std::is_signed_v<T>)
            {
                printf
                (
                    "[%u]-[%s][%d] -> 0x%0*llx (%lld)\n",
                    threadId,
                    identifier,
                    i,
                    (int32_t)(sizeof(T) * 2),
                    (uint64_t)buffer[i],
                    (long long)buffer[i]
                );
            }
            else
            {
                printf
                (
                    "[%u]-[%s][%d] -> 0x%0*llx\n",
                    threadId,
                    identifier,
                    i,
                    (int32_t)(sizeof(T) * 2),
                    (uint64_t)buffer[i]
                );
            }
        }
        else
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                printf
                (
                    "[%u]-[%s][%d] -> 0.0\n",
                    threadId,
                    identifier,
                    i
                );
            }
            else
            {
                printf
                (
                    "[%u]-[%s][%d] -> 0x%0*llx\n",
                    threadId,
                    identifier,
                    i,
                    (int32_t)(sizeof(T) * 2),
                    0ull
                );
            }
        }
    }
}
