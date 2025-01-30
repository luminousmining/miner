#pragma once

#include <algo/algo_type.hpp>
#include <stratum/autolykos_v2.hpp>
#include <stratum/blake3.hpp>
#include <stratum/etchash.hpp>
#include <stratum/ethash.hpp>
#include <stratum/evrprogpow.hpp>
#include <stratum/firopow.hpp>
#include <stratum/kawpow.hpp>
#include <stratum/meowpow.hpp>
#include <stratum/progpow_quai.hpp>
#include <stratum/progpow_z.hpp>
#include <stratum/progpow.hpp>
#include <stratum/sha256.hpp>
#include <stratum/smart_mining.hpp>


namespace stratum
{
    stratum::Stratum* NewStratum(algo::ALGORITHM const algorithm);
}
