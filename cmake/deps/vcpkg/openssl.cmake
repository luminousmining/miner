set(OPENSSL_USE_STATIC_LIBS TRUE)
set(OPEN_SSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)

# LM_WIN_CROSS: presence of OPENSSL_WIN_ROOT means prebuilt MSVC OpenSSL libs are
# staged by the Docker windows-cross image. That preset resolves Boost via vcpkg
# (USE_VCPKG=ON) but cannot cross-build OpenSSL to windows-msvc from Linux, so
# OpenSSL is supplied out-of-band and must be wired up by hand here -- a plain
# find_package would not see it. Fall back to vcpkg-resolved OpenSSL otherwise.
if (DEFINED ENV{OPENSSL_WIN_ROOT})
    set(LM_WIN_CROSS ON)
    set(_ssl "$ENV{OPENSSL_WIN_ROOT}")
    add_library(OpenSSL::Crypto STATIC IMPORTED)
    add_library(OpenSSL::SSL    STATIC IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION "${_ssl}/lib/libcrypto.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${_ssl}/include")
    set_target_properties(OpenSSL::SSL PROPERTIES
        IMPORTED_LOCATION "${_ssl}/lib/libssl.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${_ssl}/include")
else()
    set(LM_WIN_CROSS OFF)
    find_package(OpenSSL REQUIRED COMPONENTS SSL Crypto)
endif()
