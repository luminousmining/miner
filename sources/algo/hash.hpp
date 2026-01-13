#pragma once


#include <cstdint>


namespace algo
{
    union u_hash256
    {
        uint64_t word64[4];
        uint32_t word32[8];
        uint8_t  ubytes[32];
        int8_t   bytes[32];
    };

    union u_hash512
    {
        uint64_t word64[8];
        uint32_t word32[16];
        uint8_t  ubytes[64];
        int8_t   bytes[64];
    };

    union u_hash800
    {
        uint32_t word32[25];
        uint8_t  ubytes[100];
        uint8_t  bytes[100];
    };

    union u_hash1024
    {
        union u_hash512 h512[2];
        uint64_t        word64[16];
        uint32_t        word32[32];
        uint8_t         ubytes[128];
        int8_t          bytes[128];
    };

    union u_hash2048
    {
        uint64_t   word64[32];
        uint32_t   word32[64];
        uint8_t    ubytes[256];
        int8_t     bytes[256];
    };

    union u_hash3072
    {
        uint64_t   word64[48];
        uint32_t   word32[96];
        uint8_t    ubytes[384];
        int8_t     bytes[384];
    };

    union u_hash4096
    {
        uint64_t   word64[64];
        uint32_t   word32[128];
        uint8_t    ubytes[512];
        int8_t     bytes[512];
    };

    using hash256  = union u_hash256;
    using hash512  = union u_hash512;
    using hash800  = union u_hash800;
    using hash1024 = union u_hash1024;
    using hash2048 = union u_hash2048;
    using hash3072 = union u_hash3072;
    using hash4096 = union u_hash4096;

    constexpr uint64_t LEN_HASH_256          { sizeof(algo::hash256)                    };
    constexpr uint64_t LEN_HASH_256_WORD_8   { sizeof(algo::hash256) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_256_WORD_32  { sizeof(algo::hash256) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_256_WORD_64  { sizeof(algo::hash256) / sizeof(uint64_t) };

    constexpr uint64_t LEN_HASH_512          { sizeof(algo::hash512)                    };
    constexpr uint64_t LEN_HASH_512_WORD_8   { sizeof(algo::hash512) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_512_WORD_32  { sizeof(algo::hash512) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_512_WORD_64  { sizeof(algo::hash512) / sizeof(uint64_t) };

    constexpr uint64_t LEN_HASH_800          { sizeof(algo::hash800)                    };
    constexpr uint64_t LEN_HASH_800_WORD_32  { sizeof(algo::hash800) / sizeof(uint32_t) };

    constexpr uint64_t LEN_HASH_1024         { sizeof(algo::hash1024)                    };
    constexpr uint64_t LEN_HASH_1024_WORD_8  { sizeof(algo::hash1024) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_1024_WORD_32 { sizeof(algo::hash1024) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_1024_WORD_64 { sizeof(algo::hash1024) / sizeof(uint64_t) };

    constexpr uint64_t LEN_HASH_2048         { sizeof(algo::hash2048)                    };
    constexpr uint64_t LEN_HASH_2048_WORD_8  { sizeof(algo::hash2048) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_2048_WORD_32 { sizeof(algo::hash2048) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_2048_WORD_64 { sizeof(algo::hash2048) / sizeof(uint64_t) };

    constexpr uint64_t LEN_HASH_3072         { sizeof(algo::hash3072)                    };
    constexpr uint64_t LEN_HASH_3072_WORD_8  { sizeof(algo::hash3072) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_3072_WORD_32 { sizeof(algo::hash3072) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_3072_WORD_64 { sizeof(algo::hash3072) / sizeof(uint64_t) };

    constexpr uint64_t LEN_HASH_4096         { sizeof(algo::hash4096)                    };
    constexpr uint64_t LEN_HASH_4096_WORD_8  { sizeof(algo::hash4096) / sizeof(uint8_t)  };
    constexpr uint64_t LEN_HASH_4096_WORD_32 { sizeof(algo::hash4096) / sizeof(uint32_t) };
    constexpr uint64_t LEN_HASH_4096_WORD_64 { sizeof(algo::hash4096) / sizeof(uint64_t) };
}
