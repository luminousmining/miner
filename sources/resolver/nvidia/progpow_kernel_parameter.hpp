#pragma once


#include <algo/progpow/result.hpp>


namespace resolver
{
    namespace nvidia
    {
        namespace progpow
        {
            struct KernelParameters
            {
                uint32_t* lightCache { nullptr };
                uint32_t* dagCache { nullptr };
                uint32_t* headerCache { nullptr };
                algo::progpow::Result* resultCache { nullptr };
            };
        }
    }
}
