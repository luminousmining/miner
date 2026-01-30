#pragma once

#if defined(AMD_ENABLE)

#include <common/log/log.hpp>


char const* openclShowError(cl_int err);


#define OPENCL_EXCEPTION_ERROR_SHOW(function, clErr)\
    {\
        logErr()\
            << "(" << #function << ":" << __LINE__ << ")"\
            << "{" << openclShowError(clErr.err()) << " - " << clErr.what() << "}";\
    }

#define OPENCL_CALL(function)\
    cl_int const clCodeError{ (function) };\
    if (CL_SUCCESS != clCodeError)\
    { \
        { \
            logErr()\
                << "(" << __FUNCTION__ << ":" << __LINE__ << ")"\
                << "(" << #function << ")"\
                << "[" << clCodeError << "]: "\
                << openclShowError(clCodeError); \
        }\
        return false;\
    }

#define OPENCL_CATCH(action)\
    try\
    {\
        action;\
    }\
    catch(cl::Error const& clErr)\
    {\
        OPENCL_EXCEPTION_ERROR_SHOW(action, clErr);\
    }


#if defined(CL_HPP_ENABLE_EXCEPTIONS)
#define OPENCL_ER(function)\
    try\
    {\
        OPENCL_CALL(function);\
    }\
    catch(cl::Error const& clErr)\
    {\
        OPENCL_EXCEPTION_ERROR_SHOW(function, clErr);\
    }

#endif //defined(CL_HPP_ENABLE_EXCEPTIONS)


#if !defined(CL_HPP_ENABLE_EXCEPTIONS)
#define OPENCL_ER(function)\
    {\
        OPENCL_CALL(function);\
    }
#endif //!defined(CL_HPP_ENABLE_EXCEPTIONS)


#endif // AMD_ENABLE
