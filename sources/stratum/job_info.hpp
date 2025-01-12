#pragma once

#include <string>

#include <algo/hash.hpp>


namespace stratum
{
    struct StratumJobInfo
    {
        // Common
        int32_t        epoch{ -1 };
        algo::hash256  jobID{};
        algo::hash256  headerHash{};
        algo::hash256  seedHash{};
        algo::hash256  boundary{};
        uint64_t       nonce{ 0ull };
        uint64_t       startNonce{ 0ull };
        uint64_t       extraNonce{ 0ull };
        uint64_t       gapNonce{ 0x1ull };
        uint64_t       blockNumber{ 0ull };
        uint64_t       period{ 0xFFFFFFFFFFFFFFFFull };
        uint64_t       boundaryU64{ 0ull };
        uint32_t       targetBits{ 0u };
        bool           cleanJob{ false };
        std::string    jobIDStr{};

        // SHA56
        algo::hash1024 coinb1{};
        algo::hash2048 coinb2{};
        algo::hash256  merkletree[12]{};

        // ETHASH && PROGPOW
        uint32_t       extraNonceSize{ 0u };
        uint32_t       extraNonce2Size{ 0u };

        // BLAKE3
        algo::hash3072 headerBlob{};
        algo::hash256  targetBlob{};
        uint32_t       fromGroup { 0u };
        uint32_t       toGroup { 0u };

        StratumJobInfo(StratumJobInfo&& obj) = delete;
        StratumJobInfo& operator=(StratumJobInfo&& obj) = delete;

        StratumJobInfo() = default;
        ~StratumJobInfo() = default;

        StratumJobInfo(StratumJobInfo const& obj);
        StratumJobInfo& operator=(StratumJobInfo const& obj);

        void copy(StratumJobInfo const& obj);
    };
}