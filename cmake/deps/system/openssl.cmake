if (WIN32)
    set(OPENSSL_ROOT_DIR "C:\\Program Files\\OpenSSL")
endif()
set(OPENSSL_USE_STATIC_LIBS TRUE)
set(OPEN_SSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)

# LM_WIN_CROSS: presence of OPENSSL_WIN_ROOT means prebuilt MSVC OpenSSL libs
# are staged by the Docker windows-cross image (vcpkg can't cross-build OpenSSL).
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
    find_package(OpenSSL 1.1.1 REQUIRED COMPONENTS SSL Crypto)
endif()
