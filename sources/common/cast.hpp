#pragma once

#include <chrono>
#include <cstddef>

#if !defined(__LIB_CUDA) && defined(AMD_ENABLE)
#include <CL/opencl.hpp>
#endif // !__LIB_CUDA && AMD_ENABLE

////////////////////////////////////////////////////////////////////////////////
#define castU2(value) static_cast<unsigned short>(value)
#define castU8(value)  static_cast<uint8_t>(value)
#define castU16(value) static_cast<uint16_t>(value)
#define castU32(value) static_cast<uint32_t>(value)
#define castU64(value) static_cast<uint64_t>(value)
////////////////////////////////////////////////////////////////////////////////
#define cast2(value) static_cast<short>(value)
#define cast8(value)  static_cast<int8_t>(value)
#define cast16(value) static_cast<int16_t>(value)
#define cast32(value) static_cast<int32_t>(value)
#define cast64(value) static_cast<int64_t>(value)
////////////////////////////////////////////////////////////////////////////////
#define castUL(value) static_cast<unsigned long>(value)
#define castLL(value) static_cast<long long>(value)
////////////////////////////////////////////////////////////////////////////////
#define castSize(value)   static_cast<size_t>(value)
#define castDouble(value) static_cast<double>(value)
#define castFloat(value)  static_cast<float>(value)
#define castBool(value)   static_cast<bool>(value)
////////////////////////////////////////////////////////////////////////////////
#define castVOIDP(value)  reinterpret_cast<void*>(value)
#define castVOIDPP(value) reinterpret_cast<void**>(value)


#if !defined(__LIB_CUDA)
////////////////////////////////////////////////////////////////////////////////
#define castCLU8(value)  static_cast<cl_uchar>(value)
#define castCLU16(value) static_cast<cl_ushort>(value)
#define castCLU32(value) static_cast<cl_uint>(value)
#define castCLU64(value) static_cast<cl_ulong>(value)
////////////////////////////////////////////////////////////////////////////////
#define castCLU8_4(value)  static_cast<cl_uchar4>(value)
#define castCLU16_4(value) static_cast<cl_ushort4>(value)
#define castCLU32_4(value) static_cast<cl_uint4>(value)
#define castCLU64_4(value) static_cast<cl_ulong4>(value)
////////////////////////////////////////////////////////////////////////////////
#define castCL8(value)  static_cast<cl_char>(value)
#define castCL16(value) static_cast<cl_short>(value)
#define castCL32(value) static_cast<cl_int>(value)
#define castCL64(value) static_cast<cl_long>(value)
////////////////////////////////////////////////////////////////////////////////
#define castCL8_4(value)  static_cast<cl_char4>(value)
#define castCL16_4(value) static_cast<cl_short4>(value)
#define castCL32_4(value) static_cast<cl_int4>(value)
#define castCL64_4(value) static_cast<cl_long4>(value)
////////////////////////////////////////////////////////////////////////////////
#define castCLFloat(value)  static_cast<cl_float>(value)
#define castCLDouble(value) static_cast<cl_double>(value)
////////////////////////////////////////////////////////////////////////////////
#define castCLFloat_4(value)  static_cast<cl_float4>(value)
#define castCLDouble_4(value) static_cast<cl_double4>(value)
////////////////////////////////////////////////////////////////////////////////
#define castDuration(to, value) std::chrono::duration_cast<to>(value)
#define castNs(value)   castDuration(std::chrono::nanoseconds,  value)
#define castUs(value)   castDuration(std::chrono::microseconds, value)
#define castMs(value)   castDuration(std::chrono::milliseconds, value)
#define castSec(value)  castDuration(std::chrono::seconds,      value)
#define castMin(value)  castDuration(std::chrono::minutes,      value)
#define castHour(value) castDuration(std::chrono::hours,        value)
#define castDay(value)  castDuration(std::chrono::days,         value)
////////////////////////////////////////////////////////////////////////////////
#define castPtrHash512(value) reinterpret_cast<algo::hash512*>(value)
#endif // !__LIB_CUDA
