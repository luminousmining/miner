# Requirements
  
## Libraires
- cuda 12.3 => Windows
- cuda 12.0 => Ubuntu
- OpenSSL 1.1.1
- boost 1.83.0
- OpenCL 3.0.15
  
## Windows
- Visual Studio 2022
- SDK 10.0.20348.0
  
## Linux
- Linux: clang++ 10
  
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
