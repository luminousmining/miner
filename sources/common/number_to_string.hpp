#pragma once


namespace common
{
    std::string doubleToString(double const value, int32_t const precission = 2)
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(precission) << value;
        return os.str();
    }
}
