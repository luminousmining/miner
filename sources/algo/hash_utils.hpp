#pragma once


#include <string>
#include <utility>

#include <algo/hash.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>


namespace algo
{
    enum class HASH_SHIFT : uint8_t
    {
        RIGHT,
        LEFT,
        NOTHING
    };

    uint64_t       toUINT64(algo::hash256 const& hash);
    algo::hash256  toHash256(uint8_t const* bytes);
    algo::hash256  toHash256(std::string const& str);
    algo::hash256  toHash256(double const value);
    algo::hash256  toHash256_v2(double const value);
    algo::hash1024 toHash1024(std::string const& str);


    template <typename T>
    inline
    std::string toHex(
        T const& rhs,
        bool const withZero = true)
    {
        static const char* base{ "0123456789abcdef" };
        auto const length{ sizeof(rhs.ubytes) };
        std::string str;

        str.reserve(length * 2);

        for (uint32_t i{ 0u }; i < length; ++i)
        {
            auto const byte{ rhs.ubytes[i] };
            str.push_back(base[byte >> 4]);
            str.push_back(base[byte & 0xf]);
        }

        if (false == withZero)
        {
            int32_t i{ 0u };
            int32_t size{ cast32(str.size()) };
            while (i < size && '0' == str.at(i))
            {
                ++i;
            }
            str.assign(str.substr(i));
        }

        return str;
    }

    template<typename T>
    inline
    bool isHashEmpty(T const& hash)
    {
        uint32_t const length { sizeof(T) };
        for (uint32_t i { 0u }; i < length; ++i)
        {
            if (hash.bytes[i] != 0)
            {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    inline
    bool isEqual(T const& lhs, T const& rhs)
    {
        uint32_t const length { sizeof(T) };
        for (uint32_t i{ 0u }; i < length; ++i)
        {
            if (lhs.bytes[i] != rhs.bytes[i])
            {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    inline
    void copyHash(T& dst, T const& src)
    {
        uint32_t const length { sizeof(T) };
        for (uint32_t i{ 0u }; i < length; ++i)
        {
            dst.bytes[i] = src.bytes[i];
        }
    }


    template<typename T>
    inline
    T toHash(std::string const& str,
             algo::HASH_SHIFT shift = algo::HASH_SHIFT::NOTHING)
    {
        T hash{};
        std::string hex{ str };

        size_t const lengthU8 { (sizeof(T) / sizeof(uint8_t)) * 2u };

        if (algo::HASH_SHIFT::LEFT == shift)
        {
            for (size_t i { hex.size() }; i < lengthU8; ++i)
            {
                hex += "0";
            }
        }
        else if (algo::HASH_SHIFT::RIGHT == shift)
        {
            // TODO : add shifting right
        }

        int64_t index{ cast64(hex.length()) };

        if ((hex.length() % 2) != 0)
        {
            ++index;
            hex.push_back('0');
        }

        int64_t pos { cast64(sizeof(T)) - 1ll };

        while (index > 0 && pos > 0)
        {
            std::string strHex;
            strHex += hex.at(index - 2ll);
            strHex += hex.at(index - 1ll);

            uint32_t const valHex{ castU32(std::stoul(strHex, nullptr, 16)) };
            uint8_t byte{ castU8(valHex) };

            hash.ubytes[pos] = byte;

            index -= 2ll;
            --pos;
        }

        return hash;
    }

    template<typename HashTo, typename HashFrom>
    inline
    HashTo toHash2(HashFrom const& src)
    {
        HashTo dst{};

        uint32_t const len { (sizeof(HashTo) < sizeof(HashFrom) ? sizeof(HashTo) : sizeof(HashFrom)) };

        for (uint32_t i { 0u }; i < len; ++i)
        {
            dst.ubytes[i] = src.ubytes[i];
        }

        return dst;
    }


    template<typename T>
    inline
    T decimalToHash(std::string const& input)
    {
        ////////////////////////////////////////////////////////////////////////
        T hash{};

        uint32_t const lenght{ castU32(input.size()) };
        uint32_t* fs{ NEW_ARRAY(uint32_t, lenght) };
        if (nullptr == fs)
        {
            return hash;
        }

        uint32_t ts[sizeof(T) + 10]{ 1u };
        uint32_t accs[sizeof(T) + 10]{ 0u };

        uint32_t tmp;
        uint32_t rem;
        uint32_t ip;

        ////////////////////////////////////////////////////////////////////////
        uint32_t k{ 0u };
        for (int32_t i { cast32(lenght) - 1 }; i >= 0; --i)
        {
            auto value = castU32(input[i] - '0');
            fs[k++] = value;
        }

        ////////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < lenght; ++i)
        {
            for (uint32_t j{ 0u }; j < algo::LEN_HASH_512; ++j)
            {
                accs[j] += ts[j] * fs[i];

                tmp = accs[j];
                rem = 0;
                ip = j;

                do
                {
                    rem = tmp >> 4;
                    accs[ip++] = tmp - (rem << 4);
                    accs[ip] += rem;
                    tmp = accs[ip];
                }
                while (tmp >= 16);
            }

            for (uint32_t j{ 0u }; j < algo::LEN_HASH_512; ++j)
            {
                ts[j] *= 10;
            }

            for (uint32_t j{ 0u }; j < algo::LEN_HASH_512; ++j)
            {
                tmp = ts[j];
                rem = 0;
                ip = j;

                do
                {
                    rem = tmp >> 4;
                    ts[ip++] = tmp - (rem << 4);
                    ts[ip] += rem;
                    tmp = ts[ip];
                }
                while (tmp >= 16);
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // cppcheck-suppress internalAstError
        for (int32_t i = cast32(sizeof(T)) - 1; i >= 0; --i)
        {
            auto const value
            {
                (accs[i] < 10)
                    ? castU8(accs[i] + '0')
                    : castU8(accs[i] + 'A' - 0xA)
            };
            hash.ubytes[sizeof(T) - 1u - i] = value;
        }

        ////////////////////////////////////////////////////////////////////////
        SAFE_DELETE_ARRAY(fs);
        return hash;
    }

    template<typename T>
    inline
    T toLittleEndian(T const& input)
    {
        T hash{};

        uint32_t const lenght{ sizeof(T) / sizeof(uint8_t) };
        for (uint32_t i{ 0u }; i < lenght; ++i)
        {
            uint32_t const index { lenght - i };
            hash.ubytes[i >> 1] |=
            (
                (
                    (input.ubytes[index - 1] >= 'A')
                        ? input.ubytes[index - 1] - 'A' + 0xA
                        : input.ubytes[index - 1] - '0'
                )
                & 0xF
            )
            << (((i & 1)) << 2);
        }

        return hash;
    }

    template<typename T>
    inline
    T toBigEndian(T const& input)
    {
        T hash{};

        uint32_t const lenght{ sizeof(T) / sizeof(uint8_t) };
        for (uint32_t i{ 0u }; i < lenght; ++i)
        {
            uint32_t const index { lenght - i };
            hash.ubytes[i >> 1] |=
            (
                (
                    (input.ubytes[index - 1] >= 'A')
                        ? input.ubytes[index - 1] - 'A' + 0xA
                        : input.ubytes[index - 1] - '0'
                )
                & 0xF
            )
            << ((!(i & 1)) << 2);
        }

        return hash;
    }
}
