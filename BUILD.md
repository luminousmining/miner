# Requirements

> **Just want a binary?** See [Docker (no local toolchain)](#docker-no-local-toolchain)
> below — it builds every variant in containers and needs none of the manually
> installed libraries/compilers in this section.

## Libraires
- cuda 13.1 => Windows
- cuda 13.1 => Ubuntu
- OpenSSL 1.1.1
- boost 1.91.0
- OpenCL 3.0.19
  
### Windows
- Visual Studio 2022
- Windows SDK 10.0.22621.0
- CMake >= 3.22.4

### Install
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
  
### Linux
- clang++ == 11
- CMake >= 3.22

### Install
cmake :
```sh
wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh --no-check-certificate
sudo mv cmake-3.26.4-linux-x86_64.sh /opt/cmake-3.26.4-linux-x86_64.sh
sudo chmod +x /opt/cmake-3.26.4-linux-x86_64.sh
sudo /opt/cmake-3.26.4-linux-x86_64.sh
sudo cp -r cmake-3.26.4-linux-x86_64 /opt/
sudo rm -rf cmake-3.26.4-linux-x86_64
sudo ln -s /opt/cmake-3.26.4-linux-x86_64/bin/* /usr/local/bin
```
  
compiler :
```sh
sudo apt install -y build-essential libstdc++-12-dev libc++abi-12-dev gnutls-dev cppcheck checkinstall clang-15 libx11-dev
```
  
openssl :
```sh
git clone https://github.com/openssl/openssl.git
cd openssl
./Configure
make
sudo make install
```
  
opencl :
```sh
git clone https://github.com/KhronosGroup/OpenCL-SDK.git
cd OpenCL-SDK
git fetch --all
git checkout tags/v2025.07.23
git submodule init
git submodule update
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --build . --target install
```
  
cuda :
```sh
wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
```
  
boost :
```sh
wget --no-check-certificate https://archives.boost.io/release/1.91.0/source/boost_1_91_0.tar.gz
tar -xvf boost_1_91_0.tar.gz
cd boost_1_91_0
./bootstrap.sh --prefix=/usr/local
./b2 debug release -j$(nproc)
sudo ./b2 install
```

gpu performance api:
```sh
wget https://github.com/GPUOpen-Tools/gpu_performance_api/releases/download/v4.3-tag/GPUPerfAPI-Linux-4.3.0.2.tgz
tar -xvf GPUPerfAPI-Linux-4.3.0.2.tgz
mv 4_4 gpu_performance_api
```
  
# Platforms

## Docker (no local toolchain)

Every variant builds in Docker with **no local compilers or SDKs** — only Docker
(BuildKit) is required. Windows binaries are **cross-compiled** from a Linux container
(clang-cl + xwin), so Docker stays in **Linux container mode** for all targets; there is
no engine-mode switch and no Windows host needed.

Two Dockerfiles, selected by a `GPU` build-arg (`amd` | `nvidia` | `both`, default
`both`):

| Dockerfile | Output | Backends |
|---|---|---|
| `docker/Dockerfile.windows-cross` | `miner.exe` (PE32+) | AMD (OpenCL), NVIDIA (CUDA), or **both in one binary** |
| `docker/Dockerfile.linux` | `miner` (ELF) | AMD, NVIDIA, or both |

`GPU=amd` uses a lean `ubuntu:24.04` base; `GPU=nvidia`/`both` use
`nvidia/cuda:13.1.2-devel-ubuntu24.04`.

### Helper script (PowerShell)
```powershell
scripts/docker-build.ps1 -Os windows-cross -Gpu both   # combined AMD+NVIDIA miner.exe
scripts/docker-build.ps1 -Os linux         -Gpu amd    # AMD-only ELF
scripts/docker-build.ps1 -Os all           -Gpu both   # both OSes
```
Binaries are extracted to `dist/<os>-<gpu>/` (e.g. `dist/windows-cross-both/`).

### Direct docker build
```sh
# Windows, combined AMD+NVIDIA  ->  dist/windows-cross-both/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.windows-cross \
    --build-arg GPU=both --target artifact -o dist/windows-cross-both .

# Linux, AMD only  ->  dist/linux-amd/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.linux \
    --build-arg GPU=amd --target artifact -o dist/linux-amd .
```
The artifact contains `miner[.exe]`, the OpenCL `kernel/` directory, and (on Windows)
the required OpenSSL + CUDA runtime DLLs. The combined `miner.exe` runs on a host that
has only one vendor's GPU — it probes for the NVIDIA driver and skips NVIDIA cleanly when
absent; pass `--nvidia=false` or `--amd=false` to force a single backend.

## Windows (native toolchain)
```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
  
## Linux (native toolchain)
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
