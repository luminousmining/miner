#pragma once


#include <boost/exception/diagnostic_information.hpp>
#include <boost/json.hpp>

#include <common/cast.hpp>
#include <common/log/log.hpp>


namespace common
{
    template<typename T>
    T boostJsonGetNumber(boost::json::value const& v)
    {
        try
        {
            if (v.is_double())
            {
                return static_cast<T>(v.as_double());
            }
            else if (v.is_uint64())
            {
                return static_cast<T>(v.as_uint64());
            }
            return static_cast<T>(v.as_int64());
        }
        catch(boost::exception const& e)
        {
            logErr() << diagnostic_information(e);
            return T{0};
        }
    }
}
