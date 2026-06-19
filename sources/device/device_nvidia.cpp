#if defined(CUDA_ENABLE)

#include <common/custom.hpp>
#include <device/device.hpp>
#include <resolver/nvidia/autolykos_v2.hpp>
#include <resolver/nvidia/blake3.hpp>
#include <resolver/nvidia/etchash.hpp>
#include <resolver/nvidia/ethash.hpp>
#include <resolver/nvidia/evrprogpow.hpp>
#include <resolver/nvidia/firopow.hpp>
#include <resolver/nvidia/kawpow.hpp>
#include <resolver/nvidia/kheavyhash.hpp>
#include <resolver/nvidia/meowpow.hpp>
#include <resolver/nvidia/progpow.hpp>
#include <resolver/nvidia/progpow_quai.hpp>


void device::Device::setResolverNvidia(algo::ALGORITHM const algorithm)
{
    switch (algorithm)
    {
        case algo::ALGORITHM::SHA256:
        {
            break;
        }
        case algo::ALGORITHM::ETHASH:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaEthash);
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaEtchash);
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        case algo::ALGORITHM::PROGPOWZ:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaProgPOW);
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaProgpowQuai);
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaKawPOW);
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaMeowPOW);
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaFiroPOW);
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaEvrprogPOW);
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaAutolykosV2);
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaBlake3);
            break;
        }
        case algo::ALGORITHM::KHEAVYHASH:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaKHeavyHash);
            break;
        }
        case algo::ALGORITHM::UNKNOWN:
        {
            break;
        }
    }
}

#endif
