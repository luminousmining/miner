#if defined(AMD_ENABLE)

#include <common/custom.hpp>
#include <device/device.hpp>
#include <resolver/amd/autolykos_v2.hpp>
#include <resolver/amd/blake3.hpp>
#include <resolver/amd/etchash.hpp>
#include <resolver/amd/ethash.hpp>
#include <resolver/amd/evrprogpow.hpp>
#include <resolver/amd/firopow.hpp>
#include <resolver/amd/kawpow.hpp>
#include <resolver/amd/kheavyhash.hpp>
#include <resolver/amd/meowpow.hpp>
#include <resolver/amd/progpow.hpp>
#include <resolver/amd/progpow_quai.hpp>


void device::Device::setResolverAmd(algo::ALGORITHM const algorithm)
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
            resolver = NEW(resolver::ResolverAmdEthash);
            break;
        }
        case algo::ALGORITHM::ETCHASH:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdEtchash);
            break;
        }
        case algo::ALGORITHM::PROGPOW:
        case algo::ALGORITHM::PROGPOWZ:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdProgPOW);
            break;
        }
        case algo::ALGORITHM::PROGPOWQUAI:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdProgpowQuai);
            break;
        }
        case algo::ALGORITHM::KAWPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdKawPOW);
            break;
        }
        case algo::ALGORITHM::MEOWPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdMeowPOW);
            break;
        }
        case algo::ALGORITHM::FIROPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdFiroPOW);
            break;
        }
        case algo::ALGORITHM::EVRPROGPOW:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdEvrprogPOW);
            break;
        }
        case algo::ALGORITHM::AUTOLYKOS_V2:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdAutolykosV2);
            break;
        }
        case algo::ALGORITHM::BLAKE3:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdBlake3);
            break;
        }
        case algo::ALGORITHM::KHEAVYHASH:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdKHeavyHash);
            break;
        }
        case algo::ALGORITHM::UNKNOWN:
        {
            break;
        }
    }
}

#endif
