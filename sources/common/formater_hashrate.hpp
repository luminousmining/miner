#pragma once

#include <string>
#include <sstream>
#include <iomanip>


namespace common
{
    inline
    std::string getHashrateSuffix(uint32_t const unit)
    {
        using namespace std::string_literals;
        switch (unit)
        {
            case 0u: return "H"s;
            case 1u: return "KH"s;
            case 2u: return "MH"s;
            case 3u: return "GH"s;
            case 4u: return "TH"s;
            case 5u: return "PH"s;
            case 6u: return "EH"s;
            case 7u: return "ZH"s;
        }
        return "H";
    }


    inline
    std::string hashrateToString(double hashes, uint32_t unit = 0)
    {
        while (unit < 7 && hashes > castFloat(1.e3))
        {
            hashes /= 1.e3;
            ++unit;
        }

        std::stringstream ss;
        ss << std::fixed;
        ss << std::setprecision(2);
        ss << hashes;
        ss << getHashrateSuffix(unit);

        return ss.str();
    }
}
