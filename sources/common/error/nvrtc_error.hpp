#pragma once

#if defined(CUDA_ENABLE)

#include <nvrtc.h>

#include <common/custom.hpp>
#include <common/log/log.hpp>


#define NVRTC_ER(function)\
    {                                                                          \
        nvrtcResult const nvrtcCodeError{ (function) };                        \
        if (NVRTC_SUCCESS != nvrtcCodeError)                                   \
        {                                                                      \
            logErr() << "[" << static_cast<int>(nvrtcCodeError) << "]"         \
                << "(" << #function << ")"                                     \
                << nvrtcGetErrorString(nvrtcCodeError);                        \
            return false;                                                      \
        }                                                                      \
    }


#define NVRTC_BUILD(function)                                                  \
    {                                                                          \
        nvrtcResult const codeError { (function) };                            \
        if (NVRTC_SUCCESS != codeError)                                        \
        {                                                                      \
            size_t logSize { 0ull };                                           \
            NVRTC_ER(nvrtcGetProgramLogSize(cuProgram, &logSize));             \
            char* log { new char[logSize] };                                   \
            NVRTC_ER(nvrtcGetProgramLog(cuProgram, log));                      \
            logErr() << "Error: " << kernelName << " : " << log;               \
            SAFE_DELETE_ARRAY(log);                                            \
            return false;                                                      \
        }                                                                      \
    }

#endif
