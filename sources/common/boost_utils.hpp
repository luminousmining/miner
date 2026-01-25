#pragma once


#include <boost/exception/diagnostic_information.hpp>
#include <boost/json.hpp>

#include <common/cast.hpp>
#include <common/log/log.hpp>


namespace common
{
    template<typename T>
    inline
    T boostJsonGetNumber(boost::json::value const& value)
    {
        try
        {
            if (true == value.is_double())
            {
                return static_cast<T>(value.as_double());
            }
            else if (true == value.is_uint64())
            {
                return static_cast<T>(value.as_uint64());
            }
        }
        catch(boost::exception const& e)
        {
            logErr() << diagnostic_information(e);
            return T{0};
        }

        int64_t const result = value.as_int64();
        return static_cast<T>(result);
    }

    template<typename T>
    inline
    T boostJsonGetNumber(boost::json::value const& obj,
                         std::string const& name)
    {
        return common::boostJsonGetNumber<T>(obj.at(name));
    }

    bool boostJsonContains(boost::json::object const& obj,
                           std::string const& keyName);
    std::string boostGetString(boost::json::object const& obj,
                               std::string const& keyName);
    std::string boostGetString(boost::json::array const& obj,
                               uint32_t const index);
}
