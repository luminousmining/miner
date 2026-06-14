# Build — Windows (from scratch, native toolchain)

This guide builds the miner natively on Windows with Visual Studio and the GPU SDKs
and libraries installed manually on the host.

## Libraries

- cuda 13.1
- OpenSSL 1.1.1
- boost 1.91.0
- OpenCL 3.0.19

## Requirements

- Visual Studio 2022
- Windows SDK 10.0.22621.0
- CMake >= 3.22.4

## Install

cmake :
https://github.com/Kitware/CMake/releases/tag/v3.22.4

cuda :
https://developer.nvidia.com/cuda-13-1-0-download-archive

boost :
https://archives.boost.io/release/1.91.0/source/boost_1_91_0.zip
```bat
bootstrap.bat
b2.exe debug release
b2.exe install --prefix=C:\\Boost
```

opencl :
```bat
git clone  https://github.com/KhronosGroup/OpenCL-SDK.git
cd OpenCL-SDK
git fetch --all
git checkout tags/v2025.07.23
git submodule init
git submodule update
if not exist build_opencl mkdir build_opencl
cd build_opencl
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DOPENCL_SDK_BUILD_SAMPLES=ON -DOPENCL_SDK_TEST_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=C:/OpenCL
cmake --build . --target install
cd ..
```

openssl :
Install [perl](https://github.com/openssl/openssl/blob/master/NOTES-PERL.md)
```bat
git clone https://github.com/openssl/openssl.git
cd openssl
git fetch --all
git checkout tags/OpenSSL_1_1_1t
perl Configure VC-WIN64A
```
Open `Visual Studio Developer Command Promp x86_x64` with privileges !!!
```bat
nmake
nmake install
```

## Build

```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
