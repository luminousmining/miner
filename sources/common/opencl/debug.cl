#ifndef COMMON_OPENCL_DEBUG
#define COMMON_OPENCL_DEBUG


#include "kernel/common/grid.cl"


///////////////////////////////////////////////////////////////////////////////
// PRINT NUMBER
///////////////////////////////////////////////////////////////////////////////

#define PRINT_U8(name, value) \
    printf(#name "(%u): 0x%02lx | %u\n", (get_thread_id()), (value), (uint)(value));


#define PRINT_U16(name, value) \
    printf(#name "(%u): 0x%04lx | %u\n", (get_thread_id()), (value), (uint)(value));


#define PRINT_U32(name, value) \
    printf(#name "(%u): 0x%08lx | %u \n", (get_thread_id()), (value), (value));


#define PRINT_U64(name, value) \
    printf(#name "(%u): 0x%016lx | %lu\n", (get_thread_id()), (value), (value));


#define PRINT_U8_IF(name, target, value)    \
    if (get_thread_id() == (target))        \
    {                                       \
        PRINT_U8(name, value)               \
    }                                       \


#define PRINT_U16_IF(name, target, value)   \
    if (get_thread_id() == (target))        \
    {                                       \
        PRINT_U16(name, value)              \
    }                                       \


#define PRINT_U32_IF(name, target, value)   \
    if (get_thread_id() == (target))        \
    {                                       \
        PRINT_U32(name, value)              \
    }                                       \



#define PRINT_U64_IF(name, target, value)   \
    if (get_thread_id() == (target))        \
    {                                       \
        PRINT_U64(name, value)              \
    }                                       \


///////////////////////////////////////////////////////////////////////////////
// PRINT BUFFER
///////////////////////////////////////////////////////////////////////////////


#define PRINT_BUFFER_U32(name, buff, size)                                  \
    printf(#name "(%u)\n", get_thread_id());                                \
    for (uint i = 0; i < (size); ++i)                                       \
    {                                                                       \
        printf("(%u): buff[%u] = 0x%08x", get_thread_id(), i, (buff)[i]);   \
    }                                                                       \
    printf("\n");


#define PRINT_BUFFER_U64(name, buff, size)                                  \
    printf(#name "(%u)\n", get_thread_id());                                \
    for (uint i = 0; i < (size); ++i)                                       \
    {                                                                       \
        printf("(%u): buff[%u] = 0x%016lx", get_thread_id(), i, (buff)[i]); \
    }                                                                       \
    printf("\n");


#define PRINT_BUFFER_U32_IF(name, target, buff, size)   \
    if (get_thread_id() == (target))                    \
    {                                                   \
        PRINT_BUFFER_U32(name, buff, size)              \
    } 


#define PRINT_BUFFER_U64_IF(name, target, buff, size)   \
    if (get_thread_id == (target))                      \
    {                                                   \
        PRINT_BUFFER_U64(name, buff, size)              \
    }


#endif // COMMON_OPENCL_DEBUG
