#pragma once

#if defined(CUDA_ENABLE)


#define NVML_CALL(function)                                                    \
    {                                                                          \
        nvmlReturn_t const nvmlError{ (function) };                            \
        if (nvmlReturn_t::NVML_SUCCESS != nvmlError)                           \
        {                                                                      \
            logErr()                                                           \
                << "[" << static_cast<int32_t>(nvmlError) << "]"               \
                << "(" << __FUNCTION__ << ":" << __LINE__ << ")"               \
                << "(" << #function << ":"                                     \
                << nvmlErrorString(nvmlError);                                 \
        }                                                                      \
    }


#define NVML_ER(function)                                                      \
    {                                                                          \
        nvmlReturn_t const nvmlError{ (function) };                            \
        if (nvmlReturn_t::NVML_SUCCESS != nvmlError)                           \
        {                                                                      \
            logErr()                                                           \
                << "[" << static_cast<int32_t>(nvmlError) << "]"               \
                << "(" << __FUNCTION__ << ":" << __LINE__ << ")"               \
                << "(" << #function << ":"                                     \
                << nvmlErrorString(nvmlError);                                 \
                return false;                                                  \
        }                                                                      \
    }

#endif
