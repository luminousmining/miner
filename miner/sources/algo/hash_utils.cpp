#include <cmath>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/algorithm/string.hpp>

#include <algo/hash_utils.hpp>


uint64_t algo::toUINT64(
    algo::hash256 const& hash)
{
    std::string const hashHex{ algo::toHex(hash) };
    std::string hex{};
    for (uint32_t i{ 0u }; i < 16u; ++i)
    {
        hex += hashHex.at(i);
    }
    char* end{ nullptr };
    return std::strtoull(hex.c_str(), &end, 16);
}


algo::hash256 algo::toHash256(
    uint8_t const* bytes)
{
    algo::hash256 hash{};
    memcpy(&hash, bytes, algo::LEN_HASH_256);
    return hash;
}


algo::hash256 algo::toHash256(
    std::string const& str)
{
    algo::hash256 hash{};
    std::string hex{ str };
    size_t index{ hex.length() };

    if ((hex.length() % 2) != 0)
    {
        ++index;
        hex.push_back('0');
    }

    uint64_t pos{ algo::LEN_HASH_256 - 1 };

    while (index > 0u)
    {
        std::string strHex;
        strHex += hex.at(index - 2);
        strHex += hex.at(index - 1);

        auto valHex{ std::stoul(strHex, nullptr, 16) };
        auto byte{ castU8(valHex) };

        hash.ubytes[pos] = byte;

        index -= 2;
        --pos;
    }

    return hash;
}


algo::hash256 algo::toHash256(
    double const valueOriginal)
{
    using namespace boost::multiprecision;
    using BigInteger = boost::multiprecision::cpp_int;

    static BigInteger const base { "0xffff0000000000000000000000000000000000000000000000000000" };

    BigInteger product{};

    if (0.0 == fabs(valueOriginal))
    {
        product = BigInteger("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
    else
    {
        double const value{ 1.0 / valueOriginal };

        BigInteger const idiff{ value };
        product = base * idiff;

        std::string const stringDifficulty{ boost::lexical_cast<std::string>(value) };
        size_t const lengthDifficulty { stringDifficulty.length() };
        size_t const offset { stringDifficulty.find('.') };

        if (offset != std::string::npos)
        {
            // Number of decimal places
            size_t const precision { (lengthDifficulty - 1u) - offset };

            // Effective sequence of decimal places
            std::string decimals { stringDifficulty.substr(offset + 1u) };

            // Strip leading zeroes. If a string begins with
            // 0 or 0x boost parser considers it hex
            decimals = decimals.erase(0, decimals.find_first_not_of('0'));

            // Build up the divisor as string - just in case
            // parser does some implicit conversion with 10^precision
            std::string decimalDivisor { "1" };
            decimalDivisor.resize(precision + 1u, '0');

            // This is the multiplier for decimalsthe decimal part
            BigInteger const multiplier { decimals };

            // This is the divisor for the decimal part
            BigInteger const divisor { decimalDivisor };

            BigInteger decimalproduct{  base * multiplier };
            decimalproduct /= divisor;

            // Add the computed decimal part
            // to product
            product += decimalproduct;
        }
    }

    // Normalize to 64 chars hex with "0x" prefix
    std::stringstream ss;
    ss  << std::setw(64)
        << std::setfill('0')
        << std::hex
        << product;

    std::string target{ ss.str() };
    boost::algorithm::to_lower(target);

    return toHash256(target);
}


algo::hash1024 algo::toHash1024(
    std::string const& str)
{
    algo::hash1024 hash{};
    std::string hex{ str };
    int64_t index{ cast64(hex.length()) };

    if ((hex.length() % 2) != 0)
    {
        ++index;
        hex.push_back('0');
    }

    int64_t pos{ cast64(algo::LEN_HASH_1024) - 1ll };

    while (index > 0 && pos > 0)
    {
        std::string strHex{};
        strHex += hex.at(index - 2);
        strHex += hex.at(index - 1);

        uint32_t const valHex{ castU32(std::stoul(strHex, nullptr, 16)) };
        uint8_t const byte{ castU8(valHex) };

        hash.ubytes[pos] = byte;

        index -= 2;
        --pos;
    }

    return hash;
}
