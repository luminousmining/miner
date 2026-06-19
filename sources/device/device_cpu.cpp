#if defined(CPU_ENABLE)

#include <common/custom.hpp>
#include <device/device.hpp>
#include <resolver/cpu/blake3.hpp>


void device::Device::setResolverCpu(algo::ALGORITHM const algorithm)
{
    switch (algorithm)
    {
        case algo::ALGORITHM::BLAKE3:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverCpuBlake3);
            break;
        }
        case algo::ALGORITHM::SHA256:
        case algo::ALGORITHM::ETHASH:
        case algo::ALGORITHM::ETCHASH:
        case algo::ALGORITHM::PROGPOW:
        case algo::ALGORITHM::PROGPOWQUAI:
        case algo::ALGORITHM::PROGPOWZ:
        case algo::ALGORITHM::KAWPOW:
        case algo::ALGORITHM::MEOWPOW:
        case algo::ALGORITHM::FIROPOW:
        case algo::ALGORITHM::EVRPROGPOW:
        case algo::ALGORITHM::AUTOLYKOS_V2:
        case algo::ALGORITHM::KHEAVYHASH:
        case algo::ALGORITHM::UNKNOWN:
        {
            break;
        }
    }
}

#endif
