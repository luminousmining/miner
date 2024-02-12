#pragma once

#include <boost/thread.hpp>

#include <common/error/cuda_error.hpp>


#define UNIQUE_LOCK(mtxName)\
    boost::unique_lock<boost::mutex> lock{ mtxName };


#define UNIQUE_LOCK_NAME(mtxName, lockName)\
    boost::unique_lock<boost::mutex> lockName{ mtxName };


#define SAFE_DELETE(ptr)\
    if (nullptr != ptr)\
    {\
        delete ptr;\
        ptr = nullptr;\
    }


#define SAFE_DELETE_ARRAY(ptr)\
    if (nullptr != ptr)\
    {\
        delete[] ptr;\
        ptr = nullptr;\
    }

#define IS_NULL(function)\
    if (nullptr == (function))\
    {\
        logErr()\
            << "(" << __FUNCTION__ << ":" << __LINE__ << ")"\
            << "(" << #function << ")"\
            << " is nullptr";\
        return false;\
    }

#define CU_SAFE_DELETE(ptr)\
    if (nullptr != ptr)\
    {\
        CUDA_ER(cudaFree(ptr));\
    }

#define CU_SAFE_DELETE_HOST(ptr)\
    if (nullptr != ptr)\
    {\
        CUDA_ER(cudaFreeHost(ptr));\
    }

namespace common
{
    template<typename T>
    inline
    void swap(T* a, T* b)
    {
        T const tmp { *a };
        *a = *b;
        *b = tmp;
    }
}
