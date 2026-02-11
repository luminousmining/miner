#pragma once

#include <common/error/cuda_error.hpp>


namespace resolver
{
    namespace tests
    {
        struct Properties
        {
            uint32_t       cuIndex{ 0u };
            CUdevice       cuDevice;
            CUcontext      cuContext{ nullptr };
            cudaStream_t   cuStream{ nullptr };
            cudaDeviceProp cuProperties{};
        };

        inline
        bool cleanUpCuda(resolver::tests::Properties& properties)
        {
            if (nullptr != properties.cuContext)
            {
                CUDA_ER(cudaDeviceReset());
            }
            return true;
        }

        inline
        bool initializeCuda(resolver::tests::Properties& properties,
                                   uint32_t const index = 0u)
        {
            if (false == cleanUpCuda(properties))
            {
                return false;
            }

            properties.cuIndex = index;

            CU_ER(cuInit(0));
            CU_ER(cuDeviceGet(&properties.cuDevice, properties.cuIndex));
            CU_ER(cuCtxCreate(&properties.cuContext, nullptr, CU_CTX_SCHED_BLOCKING_SYNC, properties.cuDevice));
            CUDA_ER(cudaStreamCreateWithFlags(&properties.cuStream, cudaStreamNonBlocking));
            CUDA_ER(cudaGetDeviceProperties(&properties.cuProperties, properties.cuIndex));

            logInfo() << "Device [" << properties.cuProperties.name << "] selected!";

            return true;
        }
    }
}
