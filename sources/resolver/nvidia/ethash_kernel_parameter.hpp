#pragma once


#include <algo/ethash/result.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace ethash
        {
            struct KernelParameters
            {
                uint32_t*             seedCache{ nullptr };
                uint32_t*             lightCache{ nullptr };
                uint32_t*             dagCache{ nullptr };
                algo::ethash::Result* resultCache{ nullptr };
            };
        }
    }
}
