#pragma once

#if !defined(__LIB_CUDA)
#include <memory>
#include <boost/thread.hpp>
#endif

#include <common/error/cuda_error.hpp>
#include <common/tool/tool_memory_trace.hpp>


#define UNIQUE_LOCK(mtxName) boost::unique_lock<boost::mutex> lock{ mtxName };
#define UNIQUE_LOCK_NAME(mtxName, lockName) boost::unique_lock<boost::mutex> lockName{ mtxName };

#define SAFE_DELETE(ptr)                                                                                               \
    {                                                                                                                  \
        delete ptr;                                                                                                    \
        ptr = nullptr;                                                                                                 \
    }

#define SAFE_DELETE_ARRAY(ptr)                                                                                         \
    {                                                                                                                  \
        delete[] ptr;                                                                                                  \
        ptr = nullptr;                                                                                                 \
    }

#define IS_NULL(function)                                                                                              \
    if (nullptr == (function)) [[unlikely]]                                                                            \
    {                                                                                                                  \
        logErr() << "(" << __FUNCTION__ << ":" << __LINE__ << ")"                                                      \
                 << "(" << #function << ")"                                                                            \
                 << " is nullptr";                                                                                     \
        return false;                                                                                                  \
    }

#define CU_ALLOC(src, size)                                                                                            \
    CUDA_ER(cudaMalloc((void**)src, size));                                                                            \
    if (nullptr != src)                                                                                                \
    {                                                                                                                  \
        if (nullptr != *src)                                                                                           \
        {                                                                                                              \
            TOOL_MEMORY_ALLOC(*src, size);                                                                             \
        }                                                                                                              \
    }

#define CU_CALLOC(src, size)                                                                                           \
    CUDA_ER(cudaMalloc((void**)src, size));                                                                            \
    if (nullptr != src)                                                                                                \
    {                                                                                                                  \
        CUDA_ER(cudaMemset((void*)*src, 0, size));                                                                     \
        if (nullptr != *src)                                                                                           \
        {                                                                                                              \
            TOOL_MEMORY_ALLOC(*src, size);                                                                             \
        }                                                                                                              \
    }


#define CU_ALLOC_HOST(src, size)                                                                                       \
    CUDA_ER(cudaMallocHost((void**)src, size, 0));                                                                     \
    if (nullptr != src)                                                                                                \
    {                                                                                                                  \
        if (nullptr != *src)                                                                                           \
        {                                                                                                              \
            TOOL_MEMORY_ALLOC(*src, size);                                                                             \
        }                                                                                                              \
    }

#define CU_SAFE_DELETE(ptr)                                                                                            \
    if (nullptr != ptr)                                                                                                \
    {                                                                                                                  \
        CUDA_ER(cudaFree(ptr));                                                                                        \
        TOOL_MEMORY_FREE(ptr);                                                                                         \
        ptr = nullptr;                                                                                                 \
    }

#define CU_SAFE_DELETE_HOST(ptr)                                                                                       \
    if (nullptr != ptr)                                                                                                \
    {                                                                                                                  \
        CUDA_ER(cudaFreeHost(ptr));                                                                                    \
        TOOL_MEMORY_FREE(ptr);                                                                                         \
        ptr = nullptr;                                                                                                 \
    }

#define NEW(type) new (std::nothrow) type

#define NEW_ARRAY(type, size) new (std::nothrow) type[size]

// Shared-ownership allocation routed through a single macro so the memory-tracing
// tooling has one place to hook (mirrors NEW/NEW_ARRAY). Variadic so constructor
// arguments forward through: NEW_SHARED(Type) or NEW_SHARED(Type, arg0, arg1).
#define NEW_SHARED(type, ...) std::make_shared<type>(__VA_ARGS__)


namespace common
{
    template<typename T>
    inline T max_limit(T const value, T const maximun)
    {
        return value <= maximun ? value : maximun;
    }


    template<typename T>
    inline T min_limit(T const value, T const minimun)
    {
        return value >= minimun ? value : minimun;
    }

    template<typename T>
    inline void swap(T* a, T* b)
    {
        T const tmp{ *a };
        *a = *b;
        *b = tmp;
    }
}
