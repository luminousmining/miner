#pragma once

#include <string>
#include <concepts>
#include <cstdlib>

#if defined(WIN32)
    #include <stdlib.h>
#endif


namespace common
{
    inline
    char* getEnv(char const* variableName)
    {
        return std::getenv(variableName);
    }


    inline
    void setVarEnv(std::string const& variable, std::string const& value)
    {
        if (   true == variable.empty()
            || true == value.empty())
        {
            return;
        }

#if defined(WIN32)
        if (0 != _putenv_s(variable.c_str(), value.c_str()))
        {
            logErr() << "Can not set varibale environment [" << variable << "] = [" << value << "]";
            return;
        }
#elif defined(__linux__)
        if (0 != setenv(variable.c_str(), value.c_str(), 0))
        {
            logErr() << "Can not set varibale environment [" << variable << "] = [" << value << "]";
            return;
        }
#endif
        logInfo() << "Variable environment setted [" << variable << "] = [" << value << "]";
    }


    template<std::integral T>
    inline void setVarEnv(std::string const& variable, T value)
    {
        common::setVarEnv(variable, std::to_string(value));
    }
}
