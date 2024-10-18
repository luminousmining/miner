#pragma once


#include <cstdlib>


namespace common
{
    inline
    char* getEnv(char const* variableName)
    {
        return std::getenv(variableName);
    }
}
