#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <common/log/log.hpp>


#define CUDA_ER(function)                                                      \
    {                                                                          \
        cudaError_t const cuCodeError { (function) };                          \
        if (cudaSuccess != cuCodeError)                                        \
        {                                                                      \
            logErr()                                                           \
                << "[" << cuCodeError << "]"                                   \
                << "(" << __FUNCTION__ << ")"                                  \
                << "(" << #function << ":" << __LINE__ << ")"                  \
                << "{" << cudaGetErrorString(cuCodeError) << "}";              \
            return false;                                                      \
        }                                                                      \
    }


#define CU_ER(function)                                                        \
    {                                                                          \
        CUresult const cuCodeError{ (function) };                              \
        if (CUresult::CUDA_SUCCESS != cuCodeError)                             \
        {                                                                      \
            char const* msg;                                                   \
            cuGetErrorString(cuCodeError, &msg);                               \
            logErr()                                                           \
                << #function                                                   \
                << " - codeError[" << cuCodeError << "]: "                     \
                << msg;                                                        \
            return false;                                                      \
        }                                                                      \
    }
