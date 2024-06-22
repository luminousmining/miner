# Requirements
  
## Libraires
- cuda 12.5 => Windows
- cuda 12.0 => Ubuntu
- OpenSSL 1.1.1
- boost 1.85.0
- OpenCL 3.0.15
  
### Windows
- Visual Studio 2022
- Windows SDK 10.0.22621.0
- CMake >= 3.22.4

### Install
cmake: https://github.com/Kitware/CMake/releases/tag/v3.22.4
cuda : https://developer.nvidia.com/cuda-12-5-0-download-archive
boost : https://boostorg.jfrog.io/artifactory/main/release/1.85.0/source/boost_1_85_0.zip
```bat
bootstrap.bat
b2.exe release
b2.exe debug
b2.exe install --prefix=C:\\Boost
```
opencl :  https://github.com/KhronosGroup/OpenCL-SDK.git 
```bat
git fetch --all
git checkout tags/v2023.04.17
git submodule init
git submodule update
if not exist build_opencl mkdir build_opencl
cd build_opencl
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF^ -DBUILD_DOCS=OFF^ -DBUILD_EXAMPLES=OFF^ -DBUILD_TESTS=OFF^ -DOPENCL_SDK_BUILD_SAMPLES=ON^ -DOPENCL_SDK_TEST_SAMPLES=OFF^ -DCMAKE_INSTALL_PREFIX=C:/OpenCL
cmake --build . --target install
cd ..
```
openssl : https://github.com/openssl/openssl.git  
Need Perl :https://github.com/openssl/openssl/blob/master/NOTES-PERL.md 
```bat
git fetch --all
git checkout tags/OpenSSL_1_1_1t
perl Configure VC-WIN64A
```
Open `Visual Studio Developer Command Promp x86_x64` with privileges !!!
```bat
nmake
nmake install
```
  
### Linux
- Linux: clang++ 10
- CMake >= 3.22

# Platforms
  
## Windows

```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
  
## Linux
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
